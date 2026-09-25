# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state and VMM setup for the Foundry SGLang integration."""

from __future__ import annotations

import contextlib
import gc
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
    get_verbs_udev_wait_shim_path,
)
from foundry.integration.sglang.nvtx import nvtx_traced

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    sglang_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    memory_pool_config: dict = field(default_factory=dict)
    # Runtime-context overrides the memory-pool resolver made on SAVE
    # ([source, {field: value}] entries, e.g. the Mamba cache size of hybrid
    # models); LOAD skips the resolver and replays them instead.
    context_overrides: list = field(default_factory=list)
    final_alloc_offset: int = 0


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    rank: int = 0
    loaded_graphs: dict = field(default_factory=dict)
    # Set by the first runner capture (prefill when prefill graphs are on,
    # decode otherwise): the pre-capture bootstraps and the layout start run
    # there once, and LOAD's preallocation right after them.
    layout_started: bool = False
    preallocated: bool = False
    # Whether this process restored (LOAD) prefill graphs.
    prefill_graphs_restored: bool = False


_state: CUDAGraphExtensionState | None = None

# Per-rank record of the deterministic allocation layout, written at the end
# of SAVE and consumed by LOAD's preallocation (see record_region_layout).
_LAYOUT_FILE = "region_layout.json"
_layout_start_offset: int | None = None
_final_alloc_offset: int = 0


def get_state() -> CUDAGraphExtensionState | None:
    return _state


def _workspace_dir() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_dir


def _workspace_root() -> str | None:
    cfg = get_config()
    return None if cfg is None else cfg.workspace_root


def create_warmup_state(
    memory_pool_config: dict | None = None, context_overrides: list | None = None
) -> WarmupState:
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
        context_overrides=list(context_overrides or []),
    )


def save_warmup_state(state: WarmupState) -> None:
    """Write the rank's warmup state into its own workspace dir, plus the
    workspace-root copy (written once, by whichever rank gets there first).

    The memory pool config is per rank: ranks on the same node can see
    different free memory (the launcher / DP controller context sits on the
    first visible GPU), so a single shared file forced the wrong KV pool size
    onto the other rank on LOAD and its restored graphs read a pool shifted by
    that difference (elastic-EP recover, 2026-09-07)."""
    workspace_root = _workspace_root()
    if workspace_root is None:
        return
    payload = asdict(state)
    rank_dir = _workspace_dir()
    if rank_dir is not None:
        os.makedirs(rank_dir, exist_ok=True)
        rank_path = os.path.join(rank_dir, "warmup_state.json")
        with open(rank_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("[Foundry] Saved SGLang warmup state to %s", rank_path)
    ext_state = get_state()
    path = os.path.join(workspace_root, "warmup_state.json")
    if ext_state is not None and ext_state.rank != 0 and os.path.exists(path):
        return
    os.makedirs(workspace_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[Foundry] Saved SGLang warmup state to %s", path)


def load_warmup_state() -> WarmupState:
    workspace_root = _workspace_root()
    if workspace_root is None:
        raise RuntimeError("Foundry workspace_root is not initialized")
    path = os.path.join(workspace_root, "warmup_state.json")
    rank_dir = _workspace_dir()
    if rank_dir is not None and os.path.exists(os.path.join(rank_dir, "warmup_state.json")):
        path = os.path.join(rank_dir, "warmup_state.json")  # per-rank state wins
    if not os.path.exists(path):
        raise RuntimeError(f"Foundry warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


@nvtx_traced("foundry.setup_graph_ext (binary restore)")
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
    t_bin = 0.0
    if cfg.mode == CUDAGraphExtensionMode.LOAD:
        cge.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            raise RuntimeError(f"Foundry workspace for rank {rank} does not exist: {workspace_dir}")
        t1 = time.perf_counter()
        cge.load_cuda_modules_and_libraries(str(workspace_dir))
        t_bin = time.perf_counter() - t1

    region_size = parse_size(cfg.region_size)
    t1 = time.perf_counter()
    cge.set_allocation_region(cfg.base_addr, region_size)
    t_region = time.perf_counter() - t1
    t1 = time.perf_counter()
    _ = torch._C._cuda_getCurrentBlasHandle()
    t_blas = time.perf_counter() - t1
    _state = CUDAGraphExtensionState(rank=rank)
    logger.info(
        "[Foundry] SGLang graph extension setup completed in %.3f s "
        "(binary restore %.3f s, region reserve %.3f s, cuBLAS handle %.3f s)",
        time.perf_counter() - t0,
        t_bin,
        t_region,
        t_blas,
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


@contextlib.contextmanager
def allocation_region_suspended():
    """Run a block whose device allocations must not touch the deterministic
    layout: inside it the hook passes cuMemAlloc through to the driver (plain
    allocations outside the region, cursor untouched). On exit the torch cache
    is emptied before the region is resumed, so no cached segment from outside
    the region can later be handed to an allocation that belongs inside it."""
    cge.stop_allocation_region()
    try:
        yield
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        cge.resume_allocation_region()


def mark_layout_start() -> None:
    """Begin the deterministic layout; the same sequence point on SAVE and LOAD.

    Both modes first return torch's cached-but-free blocks to the driver, so
    the caching allocator is in the same state (empty) when the layout begins
    and every later allocation takes the same cuMemAlloc path on both sides.
    SAVE then records the cursor as the layout start. Everything SAVE allocates
    after this point is either replayed at recorded absolute offsets (graph
    memory) or allocated at the same offset on both sides (the FlashInfer
    metadata pre-pass); the pre-capture bootstraps run before it on both modes,
    so LOAD's cursor equals the recorded start (a difference is a divergence
    and is logged as a warning by preallocate_for_load_mode)."""
    global _layout_start_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.mode != CUDAGraphExtensionMode.SAVE:
        return
    _layout_start_offset = cge.get_current_alloc_offset()
    logger.info("[Foundry] SGLang layout start offset=%d", _layout_start_offset)


@nvtx_traced("foundry.save.record_layout")
def record_region_layout() -> int:
    """SAVE: persist the rank's allocation layout for LOAD.

    ``final_alloc_offset`` is the cursor watermark; ``live_ranges`` are the
    (offset, size) ranges still mapped at this point. LOAD backs only the live
    ranges, so memory SAVE allocated and freed inside the span costs LOAD
    nothing."""
    global _final_alloc_offset
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return 0
    _final_alloc_offset = cge.get_current_alloc_offset()
    if cfg.workspace_dir is not None:
        ranges = cge.get_live_region_ranges()
        layout = {
            "start_offset": _layout_start_offset,
            "final_alloc_offset": _final_alloc_offset,
            "live_ranges": ranges,
        }
        with open(os.path.join(cfg.workspace_dir, _LAYOUT_FILE), "w") as f:
            json.dump(layout, f)
        logger.info(
            "[Foundry] SGLang region layout: %d live ranges (%.0f MB) in the region, "
            "%.0f MB of them above the layout start; span to the watermark %.0f MB",
            len(ranges),
            sum(r[1] for r in ranges) / 2**20,
            sum(r[1] for r in ranges if r[0] >= (_layout_start_offset or 0)) / 2**20,
            (_final_alloc_offset - (_layout_start_offset or 0)) / 2**20,
        )
    if cfg.workspace_root is not None:
        warmup_state_path = os.path.join(cfg.workspace_root, "warmup_state.json")
        if os.path.exists(warmup_state_path):
            state = load_warmup_state()
            state.final_alloc_offset = _final_alloc_offset
            save_warmup_state(state)
    logger.info("[Foundry] SGLang final_alloc_offset=%d", _final_alloc_offset)
    if cfg.workspace_dir is not None and os.environ.get("FOUNDRY_SGLANG_CHECK_ARCHIVE") == "1":
        # Opt-in (reads every .cugraph): report graph pointers that LOAD will not have.
        from foundry.integration.sglang import archive_check

        archive_check.report(
            archive_check.check_rank_archive(
                cfg.workspace_dir, cfg.base_addr, parse_size(cfg.region_size)
            )
        )
    return _final_alloc_offset


@nvtx_traced("foundry.memory_restore")
def preallocate_for_load_mode() -> None:
    """LOAD: back the recorded layout ahead of the graph restore.

    Maps, in one step, every range that was live at the end of SAVE between
    LOAD's current cursor and the watermark, then checks the cursor against
    SAVE's recorded layout start (equal by construction; a difference means
    the allocation sequences diverged and is logged, moving the cursor up when
    LOAD is behind). Allocations inside the span are pointer bumps; a hole is
    backed on demand by the hook and logged, since it too means the sequences
    diverged."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return
    if cfg.workspace_dir is None:
        raise RuntimeError("Foundry workspace_dir is not initialized")
    path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry region layout not found: {path} (re-save the archive with this version)"
        )
    with open(path) as f:
        layout = json.load(f)
    final = int(layout["final_alloc_offset"])
    start = layout.get("start_offset")
    current = cge.get_current_alloc_offset()
    if final > current:
        ranges = [
            (int(off), int(size))
            for off, size in layout.get("live_ranges", [])
            if off + size > current and off < final
        ]
        t0 = time.perf_counter()
        ok = cge.preallocate_ranges(ranges, final)
        logger.info(
            "[Foundry] LOAD memory restore: %d live ranges (%.0f MB) mapped in %.1f ms",
            len(ranges),
            sum(size for _, size in ranges) / (1 << 20),
            1000 * (time.perf_counter() - t0),
        )
        if not ok:
            free, total = torch.cuda.mem_get_info()
            raise RuntimeError(
                f"Foundry LOAD could not back the {(final - current) / 2**20:.0f} MB layout span "
                f"({free / 2**20:.0f} MB of {total / 2**20:.0f} MB free): LOAD keeps the recorded "
                "kernel images resident and maps the span in one step, so it needs more headroom "
                "than SAVE. Lower --mem-fraction-static (for SAVE and LOAD alike) or capture "
                "fewer graphs."
            )
    if start is None:
        return
    if current == start:
        return
    # The two cursors coincide by construction; a difference means the SAVE and
    # LOAD allocation sequences diverged before the layout start. Moving up
    # keeps later allocations at SAVE's offsets; being ahead cannot be repaired.
    if current < start:
        cge.set_current_alloc_offset(start)
    logger.warning(
        "[Foundry] SGLang LOAD cursor %d != SAVE's layout start %d (%s): the allocation "
        "sequences differ before the graph restore",
        current,
        start,
        "moved up to it" if current < start else "ahead of it, not moved",
    )


def log_alloc_offset(label: str) -> None:
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    offset = cge.get_current_alloc_offset()
    # Free device memory alongside the region offset: LOAD ends with less free
    # memory than SAVE at identical offsets (resident kernel images, reserved
    # ranges SAVE had already freed), and this trail is how that gap is measured.
    try:
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        free_mb = float("nan")
    logger.info(
        "[Foundry] SGLang alloc_offset[%s]=%d (%.2f MB) free=%.0f MB",
        label,
        offset,
        offset / (1024 * 1024),
        free_mb,
    )


def setup_ld_preload_env() -> None:
    current = os.environ.get("LD_PRELOAD", "")
    # Prepend, never overwrite: the launcher's own LD_PRELOAD entries stay.
    for path in (get_hook_library_path(), get_nvshmem_host_path(), get_verbs_udev_wait_shim_path()):
        if path and path not in current:
            current = f"{path}:{current}" if current else path
    if current:
        os.environ["LD_PRELOAD"] = current
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value
        # NCCL registers user buffers of graph-captured collectives (above a
        # size threshold) and the kernels then read peers' remote addresses
        # from an array NCCL fills on the host at capture time. A restored
        # graph replays those kernels without the registration, so the array
        # holds garbage at LOAD (illegal address in the DP-attention all-gather
        # at bs>=4 with NCCL 2.30). Keep every size on the unregistered path.
        os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")
        os.environ.setdefault("NCCL_LOCAL_REGISTER", "0")
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())
