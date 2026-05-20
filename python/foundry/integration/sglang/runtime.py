# SPDX-License-Identifier: Apache-2.0
"""Runtime state and VMM setup for the Foundry SGLang integration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from foundry import ops as cge
from foundry.allocation_region import parse_size
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    compute_workspace_rank,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
)

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    sglang_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    memory_pool_config: dict = field(default_factory=dict)
    final_alloc_offset: int = 0


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    rank: int = 0
    loaded_graphs: dict = field(default_factory=dict)


_state: CUDAGraphExtensionState | None = None
_final_alloc_offset: int = 0


def get_state() -> CUDAGraphExtensionState | None:
    return _state


def _workspace_dir() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_dir


def _workspace_root() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_root


def create_warmup_state(memory_pool_config: dict | None = None) -> WarmupState:
    try:
        from sglang.version import __version__ as sglang_version
    except Exception:
        sglang_version = "unknown"

    props = torch.cuda.get_device_properties(0)
    return WarmupState(
        sglang_version=sglang_version,
        timestamp=datetime.now().isoformat(),
        cuda_version=torch.version.cuda or "unknown",
        gpu_name=props.name,
        gpu_total_memory=props.total_memory,
        memory_pool_config=memory_pool_config or {},
    )


def save_warmup_state(state: WarmupState) -> None:
    workspace_root = _workspace_root()
    if workspace_root is None:
        return
    ext_state = get_state()
    path = os.path.join(workspace_root, "warmup_state.json")
    if ext_state is not None and ext_state.rank != 0 and os.path.exists(path):
        return
    os.makedirs(workspace_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    logger.info("[Foundry] Saved SGLang warmup state to %s", path)


def load_warmup_state() -> WarmupState:
    workspace_root = _workspace_root()
    if workspace_root is None:
        raise RuntimeError("Foundry workspace_root is not initialized")
    path = os.path.join(workspace_root, "warmup_state.json")
    if not os.path.exists(path):
        raise RuntimeError(f"Foundry warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


def setup_graph_extension(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> None:
    """Set up the VMM region before SGLang initializes NCCL/process groups."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return

    t0 = time.perf_counter()
    rank = compute_workspace_rank(server_args, tp_rank, pp_rank, dp_rank)
    Path(cfg.workspace_root).mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    logger.info("[Foundry] SGLang rank=%d workspace_dir=%s", rank, workspace_dir)

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        cge.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            raise RuntimeError(
                f"Foundry workspace for rank {rank} does not exist: {workspace_dir}"
            )
        cge.load_cuda_modules_and_libraries(str(workspace_dir))

    region_size = parse_size(cfg.region_size)
    cge.set_allocation_region(cfg.base_addr, region_size)
    _ = torch._C._cuda_getCurrentBlasHandle()
    _state = CUDAGraphExtensionState(rank=rank)
    logger.info(
        "[Foundry] SGLang graph extension setup completed in %.3f s",
        time.perf_counter() - t0,
    )


def skip_to_scratch_boundary() -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = cge.get_current_alloc_offset()
    if current > scratch:
        logger.warning(
            "[Foundry] Current allocation offset %d exceeds scratch size %d",
            current,
            scratch,
        )
        return
    cge.set_current_alloc_offset(scratch)
    logger.info("[Foundry] SGLang skipped allocator to scratch boundary %d", scratch)


def capture_final_alloc_offset() -> int:
    global _final_alloc_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return 0
    _final_alloc_offset = cge.get_current_alloc_offset()
    if cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        with open(path, "w") as f:
            json.dump({"final_alloc_offset": _final_alloc_offset}, f)
    if cfg.workspace_root is not None:
        warmup_state_path = os.path.join(cfg.workspace_root, "warmup_state.json")
        if os.path.exists(warmup_state_path):
            state = load_warmup_state()
            state.final_alloc_offset = _final_alloc_offset
            save_warmup_state(state)
    logger.info("[Foundry] SGLang final_alloc_offset=%d", _final_alloc_offset)
    return _final_alloc_offset


def preallocate_for_load_mode() -> None:
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return
    final = 0
    if cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        if os.path.exists(path):
            with open(path) as f:
                final = json.load(f).get("final_alloc_offset", 0)
    if final <= 0:
        final = load_warmup_state().final_alloc_offset
    remaining = final - cge.get_current_alloc_offset()
    if remaining > 0:
        cge.preallocate_region(remaining)


def log_alloc_offset(label: str) -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    offset = cge.get_current_alloc_offset()
    logger.info(
        "[Foundry] SGLang alloc_offset[%s]=%d (%.2f MB)",
        label,
        offset,
        offset / (1024 * 1024),
    )


def setup_ld_preload_env() -> None:
    current = os.environ.get("LD_PRELOAD", "")
    for path in (get_hook_library_path(), get_nvshmem_host_path()):
        if path and path not in current:
            current = f"{path}:{current}" if current else path
    if current:
        os.environ["LD_PRELOAD"] = current
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["CGE_MODE"] = mode.value
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())
