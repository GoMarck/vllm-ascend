#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
from collections.abc import Iterator
from typing import Any

import requests
import torch
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

_RFORK_EXTRA_TENSOR_ATTRS = (
    # W8A8 static postprocess creates these as plain Tensor attributes.
    "aclnn_input_scale_reciprocal",
    "aclnn_input_offset",
    # W8A8 dynamic postprocess may materialize runtime-only tensors.
    "weight_fp32",
    "weight_scale_fp32",
    "weight_1",
    "weight_2",
    "weight_1_scale",
    "weight_2_scale",
    "weight_1_scale_fp32",
    "weight_2_scale_fp32",
    "weight_1_offset",
    "weight_2_offset",
    "w13_weight_scale_fp32",
    "fused_w1_scale",
    "fused_w2_scale",
)

_RFORK_EXTRA_TENSOR_LIST_ATTRS = (
    # W8A8/W4A8 MoE EPLB postprocess stores per-expert runtime tensors in lists.
    "w13_weight_list",
    "w2_weight_list",
    "w13_weight_scale_list",
    "w2_weight_scale_list",
    "w13_weight_scale_fp32_list",
    "w13_scale_bias_list",
    "w2_scale_bias_list",
    "fused_w1_scale_list",
    "fused_w2_scale_list",
)


def _join_tensor_name(module_prefix: str, tensor_name: str) -> str:
    return f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name


def _iter_named_rfork_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    seen_names: set[str] = set()

    for name, tensor in model.named_parameters():
        if tensor.numel() == 0:
            continue
        seen_names.add(name)
        yield name, tensor

    for module_prefix, module in model.named_modules():
        for attr_name in _RFORK_EXTRA_TENSOR_ATTRS:
            tensor = getattr(module, attr_name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            tensor_name = _join_tensor_name(module_prefix, attr_name)
            if tensor_name in seen_names:
                continue
            seen_names.add(tensor_name)
            yield tensor_name, tensor

        for attr_name in _RFORK_EXTRA_TENSOR_LIST_ATTRS:
            tensors = getattr(module, attr_name, None)
            if not isinstance(tensors, (list, tuple)):
                continue
            for idx, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    continue
                tensor_name = _join_tensor_name(module_prefix, f"{attr_name}.{idx}")
                if tensor_name in seen_names:
                    continue
                seen_names.add(tensor_name)
                yield tensor_name, tensor


def _get_tensor_ptrs_in_block(address: int, size: int, tensor_ptrs: set[int]) -> set[int]:
    block_end = address + size
    return {ptr for ptr in tensor_ptrs if address <= ptr < block_end}


class RForkTransferBackend:
    def __init__(self):
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from yr.datasystem import TransferEngine  # type: ignore[import-not-found]
        except ImportError as e:
            err_msg = (
                "Failed to import TransferEngine from yr.datasystem. "
                "Please install @yuanrong-datasystem/transfer_engine."
            )
            logger.error(err_msg)
            raise ImportError(err_msg) from e

        transfer_engine = TransferEngine()
        local_hostname = join_host_port(get_ip(), get_open_port())
        ret = transfer_engine.initialize(local_hostname, "ascend", f"npu:{torch.npu.current_device()}")
        if ret.is_error():
            err_msg = (
                f"TransferEngine initialization failed: "
                f"initialize({local_hostname}, 'ascend', "
                f"'npu:{int(torch.npu.current_device())}') -> {ret.to_string()}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.rfork_transfer_engine = transfer_engine
        self.rfork_transfer_engine_session_id = local_hostname
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _get_transfer_engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def register_memory_region(self, model):
        transfer_engine = self._get_transfer_engine()
        start_reg_mr_tic = time.time()

        weight_mr_dict = {}
        tensor_ptrs: set[int] = set()
        for name, tensor in _iter_named_rfork_tensors(model):
            weight_mr_dict[name] = (
                tensor.data_ptr(),
                tensor.numel(),
                tensor.element_size(),
            )
            tensor_ptrs.add(tensor.data_ptr())

        memory_snapshot = torch.npu.memory.memory_snapshot()
        weight_blocks_for_reg_mr = []
        covered_tensor_ptrs: set[int] = set()
        for segment in memory_snapshot:
            current_weight_block = None
            for block in segment.get("blocks", []):
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if address < 0 or size < 0 or state == "":
                    continue
                block_tensor_ptrs = set()
                if state == "active_allocated":
                    block_tensor_ptrs = _get_tensor_ptrs_in_block(address, size, tensor_ptrs)
                if block_tensor_ptrs:
                    covered_tensor_ptrs.update(block_tensor_ptrs)
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
            if current_weight_block is not None:
                weight_blocks_for_reg_mr.append(current_weight_block)

        if tensor_ptrs != covered_tensor_ptrs:
            logger.error(
                "Failed to find NPU memory blocks for %d RFork tensors.",
                len(tensor_ptrs - covered_tensor_ptrs),
            )
            return False

        addresses, sizes = zip(*weight_blocks_for_reg_mr) if weight_blocks_for_reg_mr else ((), ())
        ret = transfer_engine.batch_register_memory(addresses, sizes)
        if ret.is_error():
            logger.error(
                "batch_register_memory failed for %d blocks, ret: %s",
                len(weight_blocks_for_reg_mr),
                ret.to_string(),
            )
            return False

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        logger.info(
            "register_memory_region time: %.4fs, tensors: %d, blocks: %d",
            time.time() - start_reg_mr_tic,
            len(weight_mr_dict),
            len(weight_blocks_for_reg_mr),
        )
        return True

    def unregister_memory_region(self) -> bool:
        transfer_engine = self._get_transfer_engine()
        if not self.registered_weight_blocks:
            self.rfork_transfer_engine_weights_info_dict = None
            return True

        start_unreg_mr_tic = time.time()
        ret = transfer_engine.batch_unregister_memory([address for address, _ in self.registered_weight_blocks])
        if ret.is_error():
            logger.error(
                "batch_unregister_memory failed for %d blocks, ret: %s",
                len(self.registered_weight_blocks),
                ret.to_string(),
            )
            return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        logger.info(
            "unregister_memory_region time: %.4fs",
            time.time() - start_unreg_mr_tic,
        )
        return True

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
    ):
        transfer_engine = self._get_transfer_engine()
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        seed_session_id, seed_weight_info = get_remote_instance_transfer_engine_info(seed_url, local_seed_key)
        if seed_session_id is None or seed_weight_info is None:
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in _iter_named_rfork_tensors(model):
            weight_info = seed_weight_info.get(name, None)
            if weight_info is None:
                logger.error("Cannot find weight info for %s.", name)
                return False

            seed_ptr, seed_len, seed_size = weight_info
            if seed_len != tensor.numel() or seed_size != tensor.element_size():
                logger.error(
                    "Weight info mismatch for %s, expected (%s, %s), got (%s, %s)",
                    name,
                    seed_len,
                    seed_size,
                    tensor.numel(),
                    tensor.element_size(),
                )
                return False

            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())

        start_transfer_tic = time.time()
        ret = transfer_engine.batch_transfer_sync_read(
            seed_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if ret.is_error():
            logger.error(
                "Failed to transfer weights from remote instance seed_ip=%s, seed_port=%s, ret=%s",
                seed_instance_ip,
                seed_instance_service_port,
                ret.to_string(),
            )
            return False

        logger.info("transfer weights time: %.4fs", time.time() - start_transfer_tic)
        return True


def get_remote_instance_transfer_engine_info(seed_url: str, local_seed_key: str):
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
        )
        if response.status_code != 200:
            logger.error(
                "GET %s/get_rfork_transfer_engine_info failed: %s",
                seed_url,
                response.status_code,
            )
            return None, None

        data = response.json()
        info = data.get("rfork_transfer_engine_info", None)
        if info is not None and isinstance(info, list) and len(info) == 2:
            return info[0], info[1]

        logger.error(
            "Failed to get rfork_transfer_engine_info in response from %s.",
            seed_url,
        )
        return None, None
    except Exception as e:
        logger.error("Exception getting transfer engine info from %s: %s", seed_url, e)
        return None, None
