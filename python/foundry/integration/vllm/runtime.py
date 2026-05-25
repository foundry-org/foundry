# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime state, VMM memory ops, and LD_PRELOAD env setup for the
foundry vLLM integration.

Merged from former ``state.py`` + ``lifecycle.py`` + ``memory.py`` +
``distributed.py``. All process-local globals and the primitives that
touch foundry's C extension live here.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import vllm
from vllm.logger import init_logger

from foundry import ops as fops
from foundry.allocation_region import parse_size
from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    _compute_rank_from_parallel_config,
    get_config,
    get_graph_extension_mode,
    get_hook_library_path,
    get_nvshmem_host_path,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

log = init_logger("vllm.foundry.runtime")


# ---------------------------------------------------------------------------
# Persistent state (WarmupState on disk + process-local globals)
# ---------------------------------------------------------------------------


@dataclass
class WarmupState:
    vllm_version: str = ""
    timestamp: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_total_memory: int = 0
    available_gpu_memory: list[int] = field(default_factory=list)
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0
    gpu_memory_utilization: float = 0.0
    final_alloc_offset: int = 0


_final_alloc_offset: int = 0


def set_final_alloc_offset(v: int) -> None:
    global _final_alloc_offset
    _final_alloc_offset = v


def get_final_alloc_offset() -> int:
    return _final_alloc_offset


def create_warmup_state() -> WarmupState:
    props = torch.cuda.get_device_properties(0)
    return WarmupState(
        vllm_version=vllm.__version__,
        timestamp=datetime.now().isoformat(),
        cuda_version=torch.version.cuda or "unknown",
        gpu_name=props.name,
        gpu_total_memory=props.total_memory,
    )


def save_warmup_state(workspace_root: str, state: WarmupState) -> None:
    path = os.path.join(workspace_root, "warmup_state.json")
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    log.info("Saved warmup state to %s", path)


def load_warmup_state(workspace_root: str) -> WarmupState:
    path = os.path.join(workspace_root, "warmup_state.json")
    if not os.path.exists(path):
        raise RuntimeError(f"Warmup state file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    valid = set(WarmupState.__dataclass_fields__.keys())
    return WarmupState(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Worker-scope state (per-process, set by setup_graph_extension)
# ---------------------------------------------------------------------------


@dataclass
class CUDAGraphExtensionState:
    capture_index: int = 0
    # LOAD: strong-reference table keyed by GraphIdentifier. The upstream
    # CUDAGraphWrapper stores `entry.output = weak_ref_tensors(output)`, so
    # without a strong ref here the VMM-backed output tensors returned by
    # `FoundryCUDAGraph.finish_one_graph_load` get GC'd between captures and
    # the next graph replay writes to freed memory → CUDA illegal memory
    # access. Do not remove.
    loaded_graphs: dict = field(default_factory=dict)


_state: CUDAGraphExtensionState | None = None


def get_state() -> CUDAGraphExtensionState | None:
    return _state


# ---------------------------------------------------------------------------
# VMM region setup (called from patched init_worker_distributed_environment)
# ---------------------------------------------------------------------------


def setup_graph_extension(parallel_config: ParallelConfig) -> None:
    """VMM region + (LOAD only) CUDA modules. Called BEFORE NCCL."""
    global _state
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return

    t_total = time.perf_counter()
    rank = _compute_rank_from_parallel_config(parallel_config)
    workspace_dir = Path(cfg.workspace_root) / f"rank_{rank}"
    cfg.workspace_dir = str(workspace_dir)
    log.info("[foundry] rank=%d workspace_dir=%s", rank, workspace_dir)

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

    elif cfg.mode == CUDAGraphExtensionMode.LOAD:
        fops.set_skip_fatbin_processing(True)
        if not workspace_dir.exists():
            # rank_0 fallback is only safe for single-rank setups.
            # Under TP/PP/DP > 1 each rank has its own device addresses,
            # so loading rank 0's graphs on rank N would produce wrong
            # pointers and likely crash on first replay.
            is_single_rank = (
                parallel_config.tensor_parallel_size == 1
                and parallel_config.pipeline_parallel_size == 1
                and parallel_config.data_parallel_size == 1
                and parallel_config.nnodes == 1
            )
            fallback = Path(cfg.workspace_root) / "rank_0"
            if is_single_rank and fallback.exists():
                log.warning(
                    "Workspace %s missing; falling back to %s (safe only for single-rank setups)",
                    workspace_dir,
                    fallback,
                )
                cfg.workspace_dir = str(fallback)
            else:
                raise RuntimeError(
                    f"Workspace for rank {rank} ({workspace_dir}) does not "
                    f"exist. For multi-rank LOAD, re-run SAVE with the same "
                    f"parallelism topology."
                )
        log.info("[foundry] LOAD: loading CUDA modules from %s", cfg.workspace_dir)
        t = time.perf_counter()
        fops.load_cuda_modules_and_libraries(cfg.workspace_dir)
        log.info(
            "[foundry TIMING] load_cuda_modules_and_libraries: %.3f s", time.perf_counter() - t
        )

    # set_allocation_region MUST come after load_cuda_modules (which does
    # cuCtxCreate).
    region_size = parse_size(cfg.region_size)
    log.info("[foundry] Setting up VMM region at %s size %s", hex(cfg.base_addr), cfg.region_size)
    fops.set_allocation_region(cfg.base_addr, region_size)
    # NOTE(liuxs): workspace init is in prepare_graph_capture
    # # Bring cuBLAS handle up so its tiny bring-up lands in scratch.
    # _ = torch._C._cuda_getCurrentBlasHandle()

    _state = CUDAGraphExtensionState()
    log.info("[foundry TIMING] setup_graph_extension total: %.3f s", time.perf_counter() - t_total)


def skip_to_scratch_boundary() -> None:
    """Advance allocator past scratch region. Called AFTER NCCL."""
    cfg = get_config()
    if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
        return
    scratch = parse_size(cfg.scratch_space_size)
    current = fops.get_current_alloc_offset()
    if current > scratch:
        log.warning(
            "[foundry] Current offset %d > scratch_space_size %d. Increase scratch_space_size.",
            current,
            scratch,
        )
        return
    log.info("[foundry] Skip to scratch boundary: %d bytes (current was %d)", scratch, current)
    fops.set_current_alloc_offset(scratch)


def capture_final_alloc_offset() -> int:
    """Record per-rank watermark after all graph captures on SAVE."""
    cfg = get_config()
    offset = fops.get_current_alloc_offset()
    set_final_alloc_offset(offset)
    log.info(
        "[foundry] Captured final_alloc_offset: %d bytes (%.2f GB)", offset, offset / (1024**3)
    )
    if cfg is not None and cfg.workspace_dir is not None:
        path = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        with open(path, "w") as f:
            json.dump({"final_alloc_offset": offset}, f)
        log.info("[foundry] Saved per-rank final_alloc_offset to %s", path)
    return offset


# ---------------------------------------------------------------------------
# VMM preallocation (LOAD only)
# ---------------------------------------------------------------------------


def preallocate_for_load_mode() -> None:
    """One cuMemCreate + cuMemMap covering the deterministic region."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    # Per-rank file preferred (EP), fall back to shared warmup state.
    final = 0
    if cfg.workspace_dir is not None:
        p = os.path.join(cfg.workspace_dir, "final_alloc_offset.json")
        if os.path.exists(p):
            with open(p) as f:
                final = json.load(f).get("final_alloc_offset", 0)
    if final <= 0 and cfg.workspace_root is not None:
        final = load_warmup_state(cfg.workspace_root).final_alloc_offset
    if final <= 0:
        return

    current = fops.get_current_alloc_offset()
    remaining = final - current
    if remaining <= 0:
        log.info("[foundry] LOAD: no preallocation needed")
        return

    log.info("[foundry] LOAD: preallocating %d bytes (%.2f GB)", remaining, remaining / (1024**3))
    t = time.perf_counter()
    if fops.preallocate_region(remaining):
        log.info("[foundry TIMING] preallocate_region: %.3f s", time.perf_counter() - t)
    else:
        log.warning("[foundry] preallocate_region failed for %d bytes", remaining)


# ---------------------------------------------------------------------------
# Graph-capture workspace pinning (SAVE + LOAD)
# ---------------------------------------------------------------------------


def _preallocate_attention_workspaces(metadata_builders: list | None) -> None:
    try:
        from vllm.v1.attention.backends.flashinfer import (
            FlashInferMetadataBuilder,
            _get_trtllm_gen_workspace_buffer,
        )
    except ImportError:
        return
    _get_trtllm_gen_workspace_buffer()
    if metadata_builders:
        for b in metadata_builders:
            if isinstance(b, FlashInferMetadataBuilder):
                b._get_workspace_buffer()


def _eager_inductor_lazy_init(device: torch.device) -> None:
    """Pre-fire ``torch._inductor.fx_passes.joint_graph.lazy_init`` outside
    any CUDA-graph capture window.

    The first in-capture compile inside ``capture_model`` triggers
    ``joint_graph.lazy_init`` which calls ``_sfdp_init`` — that builds SDPA
    pattern templates by copying CPU constants onto the device, which is
    illegal inside a capture window:

        RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        graph capture unless the CPU tensor is pinned.

    Pre-warming retires the lazy init before capture starts.

    Why only ``joint_graph`` and not the other ``lazy_init`` functions in
    ``torch._inductor.fx_passes.*``: ``post_grad.lazy_init`` and friends
    are also ``@init_once_fakemode`` (``@functools.cache``-keyed on
    ``input_device``), but inductor's own ``post_grad_passes`` calls them
    with **no argument** (``input_device=None``). If we pre-warm with the
    indexed ``cuda:N`` device the cache keys don't match, both calls run,
    and they re-register the same patterns:

        torch._inductor.exc.InductorError: Duplicate pattern: amax_default

    Only ``joint_graph.lazy_init`` is dangerous to leave for capture-time
    (its body actually allocates on the device); the others just register
    patterns and are safe to leave to inductor.
    """
    try:
        from torch._inductor.fx_passes.joint_graph import lazy_init
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init not importable: %s", e)
        return
    try:
        lazy_init(device)
    except Exception as e:
        log.debug("[foundry] joint_graph.lazy_init(%s) skipped: %s", device, e)


def prepare_graph_capture(
    metadata_builders: list | None = None,
    device: torch.device | None = None,
) -> None:
    """Force cuBLAS + attention workspaces to land at deterministic offsets,
    and pre-warm torch._inductor's pattern-matcher lazy init so the first
    in-capture compile doesn't try to copy CPU tensors onto the device.

    ``device`` is the model_runner's device; passing it lets the inductor
    lazy-init cache key match the in-capture compile path.
    """
    import foundry as foundry_pkg

    if device is not None:
        _eager_inductor_lazy_init(device)
    fops.preallocate_cublas_workspaces()
    _preallocate_attention_workspaces(metadata_builders)
    if foundry_pkg.graph.default_capture_stream is None:
        foundry_pkg.graph.default_capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(foundry_pkg.graph.default_capture_stream):
        fops.preallocate_cublas_workspaces()
        _preallocate_attention_workspaces(metadata_builders)


# ---------------------------------------------------------------------------
# Subprocess env setup (LD_PRELOAD + FOUNDRY_MODE consumed by foundry/csrc/hook.cpp)
# ---------------------------------------------------------------------------


def setup_ld_preload_env() -> None:
    """Mutate os.environ so child processes inherit the foundry hook.

    Idempotent, non-restoring — keeps the mutation for the parent's
    lifetime. Called from patched subprocess-spawn sites.

    Also stamps ``FOUNDRY_SPAWN_T0_NS`` right before the child fork so
    the child can log ``spawn → install_hooks`` wall time (dominated by
    Python + torch + vllm cold import with LD_PRELOAD'd hook).
    """
    current = os.environ.get("LD_PRELOAD", "")
    hook = get_hook_library_path()
    if hook and hook not in current:
        current = f"{hook}:{current}" if current else hook
    nvs = get_nvshmem_host_path()
    if nvs and nvs not in current:
        current = f"{nvs}:{current}" if current else nvs
    if current:
        os.environ["LD_PRELOAD"] = current

    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value

    # Overwrite every spawn site — child reads it on entry.
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())

    log.info("[foundry] LD_PRELOAD set (mode=%s)", mode.value)
