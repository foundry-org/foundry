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
    attention_dp_active,
    compute_workspace_rank,
    dp_replica,
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
    # FlashInfer's tuned tactics of this rank (the content of SGLang's autotune
    # cache file), installed by LOAD before capture; None: default tactics.
    flashinfer_tactics: dict | None = None


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
    # Decode-type capture scopes restored so far (LOAD; one per kind, e.g.
    # target_verify, draft_decode, draft_extend under EAGLE).
    decode_scopes_done: int = 0


_state: CUDAGraphExtensionState | None = None

# Per-rank record of the deterministic allocation layout, written at the end
# of SAVE and consumed by LOAD's preallocation (see record_region_layout).
_LAYOUT_FILE = "region_layout.json"
# Version of the allocation sequence between the layout start and the end of
# the last decode-type capture scope, recorded in _LAYOUT_FILE on SAVE and
# required on LOAD.
# 2: both modes run sglang's own decode capture loop and substitute only the
# capture (1, unmarked: a FlashInfer metadata pre-pass before the loop, which
# allocates the per-bs wrappers in another order; LOAD cannot replay it).
CAPTURE_LOOP_VERSION = 2
_layout_start_offset: int | None = None
# LOAD with a share manifest: (canonical rank, [(kernel, param, offset, width, value)]),
# set by hooks.before_parallel_init after the manifest check.
_archive_share: tuple[int, list] | None = None
_final_alloc_offset: int = 0
# Written by a SAVE that fell back (api.deactivate) into its rank's workspace
# and the workspace root; LOAD falls back when it finds either.
_INVALID_FILE = "invalid.json"


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
    if cfg.mode == CUDAGraphExtensionMode.LOAD and _archive_share is not None:
        # Archive share: restore the canonical rank's archive; the "= rank" sites
        # are written into the param bytes as each .cugraph is read.
        canonical, patches = _archive_share
        workspace_dir = Path(cfg.workspace_root) / f"rank_{canonical}"
        cge.set_param_patches(patches)
        logger.info(
            "[Foundry] archive share: rank %d restores rank_%d with %d '= rank' patch site(s)",
            rank,
            canonical,
            len(patches),
        )
    cfg.workspace_dir = str(workspace_dir)
    logger.info(
        "[Foundry] SGLang rank=%d device=cuda:%d CUDA_VISIBLE_DEVICES=%s",
        rank,
        torch.cuda.current_device(),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    logger.info(
        "[Foundry] SGLang rank=%d (tp_rank=%d pp_rank=%d dp_rank=%s replica=%d "
        "attention_dp=%s) workspace_dir=%s",
        rank,
        tp_rank,
        pp_rank,
        dp_rank,
        dp_replica(server_args, dp_rank),
        attention_dp_active(server_args),
        workspace_dir,
    )

    if cfg.mode == CUDAGraphExtensionMode.SAVE:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        from foundry.integration.sglang import share

        share.write_rank_config(str(workspace_dir), rank, server_args, torch.cuda.current_device())
        # A marker of an earlier SAVE; this SAVE writes its own only after the
        # ranks agreed at a capture scope, which every rank reaches after this.
        (Path(cfg.workspace_root) / _INVALID_FILE).unlink(missing_ok=True)
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
    if os.environ.get("FOUNDRY_DIAG_MEMHIST") == "1":
        # DIAGNOSTIC ONLY (eep-archive-share study): caching-allocator history with Python stacks
        # from here to the end of SAVE, dumped by record_region_layout (memhist.pickle).
        torch.cuda.memory._record_memory_history(max_entries=1000000, stacks="python")
    if os.environ.get("FOUNDRY_DIAG_NIXL_OUTSIDE_REGION") == "1":
        _diag_nixl_outside_region(cfg.mode, str(workspace_dir))
    logger.info(
        "[Foundry] SGLang graph extension setup completed in %.3f s "
        "(binary restore %.3f s, region reserve %.3f s, cuBLAS handle %.3f s)",
        time.perf_counter() - t0,
        t_bin,
        t_region,
        t_blas,
    )


def _diag_nixl_outside_region(mode, workspace_dir: str) -> None:
    """DIAGNOSTIC ONLY (NIXL EP study): build the NIXL EP buffer with the region
    suspended (its cuMemAlloc fallback otherwise lands in a region chunk that UCX
    cuda_ipc cannot map) and log the driver-placed addresses for SAVE vs LOAD.
    Both modes capture a probe graph (_diag_nixl_probe_params); SAVE stores its
    params in <rank>/nixl_diag.json. LOAD, when the rank dir holds nixl_reloc.json
    (sites made offline from the SAVE archive), moves every listed param word by
    the probe's LOAD - SAVE difference of the same kernel and param index. A single
    delta is wrong: NIXL's allocations do not keep their relative placement."""
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    orig = NixlEPBuffer.get_nixl_buffer.__func__

    def get_nixl_buffer(cls, *args, **kwargs):
        if cls._state().buffer is not None:
            return orig(cls, *args, **kwargs)
        with allocation_region_suspended():
            buf = orig(cls, *args, **kwargs)
            probe = _diag_nixl_probe_params(cls._state())
        rdma = buf.get_local_buffer_tensor(torch.uint8, use_rdma_buffer=True)
        base = rdma.data_ptr()
        logger.info(
            "[Foundry] DIAG nixl buffer outside region: rdma_buffer=0x%x size=%d probe=%s",
            base,
            rdma.numel(),
            {k: [hex(v) for v in vs] for k, vs in probe.items()},
        )
        if mode == CUDAGraphExtensionMode.SAVE:
            with open(os.path.join(workspace_dir, "nixl_diag.json"), "w") as f:
                json.dump({"rdma_base": base, "rdma_size": rdma.numel(), "probe": probe}, f)
        reloc = os.path.join(workspace_dir, "nixl_reloc.json")
        if mode == CUDAGraphExtensionMode.LOAD and os.path.exists(reloc):
            with open(reloc) as f:
                d = json.load(f)
            with open(os.path.join(workspace_dir, "nixl_diag.json")) as f:
                saved_probe = json.load(f)["probe"]
            patches = []
            for g, k, pi, off, v in d["sites"]:
                kind = _diag_nixl_kind(k)
                delta = probe[kind][pi] - saved_probe[kind][pi]
                patches.append((g, k, pi, off, 8, (v + delta) & 0xFFFFFFFFFFFFFFFF))
            if _archive_share is not None:
                patches += _archive_share[1]  # set_param_patches replaces, keep the rank sites
            cge.set_param_patches(patches)
            deltas = {
                kind: [hex(a - b) for a, b in zip(probe[kind], saved_probe[kind]) if a != b]
                for kind in probe
            }
            logger.info(
                "[Foundry] DIAG nixl relocation over %d site(s); probe deltas %s",
                len(patches),
                deltas,
            )
        return buf

    NixlEPBuffer.get_nixl_buffer = classmethod(get_nixl_buffer)


def _diag_nixl_kind(kernel_name: str) -> str | None:
    """DIAGNOSTIC ONLY: the NIXL EP kernel a relocation site belongs to."""
    for marker, kind in (
        ("8dispatch", "dispatch"),
        ("7combine", "combine"),
        ("query_mask_buffer", "query"),
    ):
        if marker in kernel_name:
            return kind
    return None


def _diag_nixl_probe_params(state) -> dict:
    """DIAGNOSTIC ONLY: capture (never replay) one NIXL dispatch + combine with
    the engine's buffer and return each kernel's 8-byte params by index; run with
    the region suspended so the probe's tensors and graph stay out of the layout."""
    import ctypes

    from nixl_ep import topk_idx_t

    buf = state.buffer
    n_ranks = state.max_ranks  # NIXL rank_bound = num_experts / num_local_experts, as the engine
    hidden, max_tok = state.hidden_size, state.num_max_dispatch_tokens_per_rank
    n_exp = state.num_local_experts * n_ranks
    x = torch.zeros(1, hidden, dtype=torch.bfloat16, device="cuda")
    idx = torch.arange(8, dtype=topk_idx_t, device="cuda").view(1, 8)
    w = torch.ones(1, 8, dtype=torch.float32, device="cuda")
    y = torch.zeros(
        state.num_local_experts, max_tok * n_ranks, hidden, dtype=torch.bfloat16, device="cuda"
    )
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        _, _, handle, _, hook = buf.dispatch(
            x, idx, max_tok, n_exp, use_fp8=True, async_finish=False, return_recv_hook=True
        )
        hook()
        _, _, hook2 = buf.combine(y, idx, w, handle, async_finish=False, return_recv_hook=True)
        hook2()
        buf.query_mask_buffer(torch.zeros(n_ranks, dtype=torch.int32, device="cuda"))
    lib = ctypes.CDLL("libcuda.so.1")

    class KParams(ctypes.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2
        _fields_ = (
            [("func", ctypes.c_void_p)]
            + [(f, ctypes.c_uint) for f in ["gx", "gy", "gz", "bx", "by", "bz", "smem"]]
            + [
                ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
                ("extra", ctypes.c_void_p),
                ("kern", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
            ]
        )

    graph = ctypes.c_void_p(int(g.raw_cuda_graph()))
    n = ctypes.c_size_t(0)
    lib.cuGraphGetNodes(graph, None, ctypes.byref(n))
    nodes = (ctypes.c_void_p * n.value)()
    lib.cuGraphGetNodes(graph, nodes, ctypes.byref(n))
    out = {}
    for node in nodes:
        t = ctypes.c_int(-1)
        lib.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(t))
        if t.value != 0:
            continue
        p = KParams()
        lib.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p))
        name = ctypes.c_char_p()
        if p.func:
            lib.cuFuncGetName(ctypes.byref(name), ctypes.c_void_p(p.func))
            info, h = lib.cuFuncGetParamInfo, p.func
        else:
            lib.cuKernelGetName(ctypes.byref(name), ctypes.c_void_p(p.kern))
            info, h = lib.cuKernelGetParamInfo, p.kern
        nm = (name.value or b"").decode()
        kind = _diag_nixl_kind(nm)
        if kind is None or kind in out:
            continue
        vals = []
        for i in range(64):
            off, sz = ctypes.c_size_t(0), ctypes.c_size_t(0)
            if (
                info(ctypes.c_void_p(h), ctypes.c_size_t(i), ctypes.byref(off), ctypes.byref(sz))
                != 0
            ):
                break
            raw = ctypes.string_at(p.kernelParams[i], sz.value)
            vals.append(int.from_bytes(raw[:8].ljust(8, b"\0"), "little") if sz.value == 8 else 0)
        out[kind] = vals
    del g
    return out


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


_gc_held = False


def hold_cyclic_gc() -> None:
    """Keep Python's cyclic collector off until the layout starts.

    Model loading leaves tensors in reference cycles (EAGLE3: two draft linear
    weights, 110 MB). The collector frees them when its allocation counters
    trip, which the weight-loading threads move from run to run, so a later
    allocation (the draft's RoPE cache) takes the freed block on some runs and
    another one on others, and LOAD's buffers miss the addresses SAVE's graphs
    hold. With the collector off, cycles are freed only by the explicit
    collections both modes run at the same points (memory-pool resolution,
    layout start)."""
    global _gc_held
    if gc.isenabled():
        gc.disable()
        _gc_held = True


def release_cyclic_gc() -> None:
    global _gc_held
    if _gc_held:
        gc.enable()
        _gc_held = False


def mark_archive_invalid(step: str, reason: str) -> None:
    """SAVE fell back: mark the rank's workspace and the workspace root so that
    a LOAD of this archive falls back too (:func:`archive_invalid_marker`)."""
    cfg = get_config()
    payload = {"step": step, "reason": reason, "timestamp": datetime.now().isoformat()}
    for directory in (cfg.workspace_dir, cfg.workspace_root):
        if directory is None:
            continue
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, _INVALID_FILE), "w") as f:
            json.dump(payload, f, indent=2)
    logger.warning(
        "[Foundry] SAVE archive %s marked invalid (%s)", cfg.workspace_root, _INVALID_FILE
    )


def archive_invalid_marker(rank: int) -> tuple[str, str, str] | None:
    """LOAD: (path, step, reason) of the invalid marker of the workspace root or
    of ``rank``'s workspace, or None."""
    cfg = get_config()
    for directory in (cfg.workspace_root, os.path.join(cfg.workspace_root, f"rank_{rank}")):
        path = os.path.join(directory, _INVALID_FILE)
        if os.path.exists(path):
            with open(path) as f:
                marker = json.load(f)
            return path, marker["step"], marker["reason"]
    return None


def leave_graph_extension() -> None:
    """After a fallback in a scheduler past setup: later allocations of this
    thread go to the driver (the region's live allocations stay valid and are
    freed through the hook as before) and the hook stops dumping kernel
    binaries."""
    cge.stop_allocation_region()
    cge.set_skip_fatbin_processing(True)


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
    release_cyclic_gc()
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
            "capture_loop_version": CAPTURE_LOOP_VERSION,
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
    if os.environ.get("FOUNDRY_DIAG_MEMHIST") == "1" and cfg.workspace_dir is not None:
        # DIAGNOSTIC ONLY (eep-archive-share study)
        import pickle

        with open(os.path.join(cfg.workspace_dir, "memhist.pickle"), "wb") as f:
            pickle.dump(torch.cuda.memory._snapshot(), f)
    if cfg.workspace_dir is not None and os.environ.get("FOUNDRY_SGLANG_CHECK_ARCHIVE") == "1":
        # Opt-in (reads every .cugraph): report graph pointers that LOAD will not have.
        from foundry.integration.sglang import archive_check

        archive_check.report(
            archive_check.check_rank_archive(
                cfg.workspace_dir, cfg.base_addr, parse_size(cfg.region_size)
            )
        )
    return _final_alloc_offset


def set_archive_share(canonical: int, patches: list) -> None:
    global _archive_share
    _archive_share = (canonical, patches)


def archive_share_active() -> bool:
    return _archive_share is not None


def check_capture_loop_version() -> None:
    """LOAD, at setup (before the weights load): refuse an archive whose
    allocation sequence this code cannot replay (see CAPTURE_LOOP_VERSION)."""
    cfg = get_config()
    if cfg is None or cfg.mode != CUDAGraphExtensionMode.LOAD or cfg.workspace_dir is None:
        return
    path = os.path.join(cfg.workspace_dir, _LAYOUT_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry region layout not found: {path} (re-SAVE with the current code)"
        )
    with open(path) as f:
        version = json.load(f).get("capture_loop_version")
    if version != CAPTURE_LOOP_VERSION:
        raise RuntimeError(
            f"Foundry archive {cfg.workspace_dir} has capture_loop_version={version}, this "
            f"code needs {CAPTURE_LOOP_VERSION}: re-SAVE with the current code"
        )


@nvtx_traced("foundry.memory_restore")
def preallocate_for_load_mode() -> None:
    """LOAD: back the recorded layout ahead of the graph restore.

    Maps, in one step, every range that was live at the end of SAVE between
    LOAD's current cursor and the watermark, then checks the cursor against
    SAVE's recorded layout start (equal by construction; a difference means
    the allocation sequences diverged and is logged, moving the cursor up when
    LOAD is behind). Allocations inside the span are pointer bumps. A range
    SAVE allocated and freed again before the end is not in the layout: when
    LOAD's replay of the same sequence allocates it, the hook backs it on
    demand and logs it at INFO (the same ranges on every LOAD of an archive,
    e.g. the segments sglang's ``empty_cache`` releases between the EAGLE
    draft scopes)."""
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
    if os.environ.get("FOUNDRY_DIAG_MEMHIST_DUMP_AT") == label and cfg.workspace_dir is not None:
        # DIAGNOSTIC ONLY: allocator history up to this checkpoint (needs FOUNDRY_DIAG_MEMHIST=1).
        import pickle

        with open(os.path.join(cfg.workspace_dir, f"memhist_{label}.pickle"), "wb") as f:
            pickle.dump(torch.cuda.memory._snapshot(), f)
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


def _set_subprocess_preload_env() -> None:
    """LD_PRELOAD (prepended, never overwritten: the launcher's own entries
    stay), FOUNDRY_MODE (read by the hook library) and the spawn timestamp."""
    current = os.environ.get("LD_PRELOAD", "")
    for path in (get_hook_library_path(), get_nvshmem_host_path(), get_verbs_udev_wait_shim_path()):
        if path and path not in current:
            current = f"{path}:{current}" if current else path
    if current:
        os.environ["LD_PRELOAD"] = current
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        os.environ["FOUNDRY_MODE"] = mode.value
    os.environ["FOUNDRY_SPAWN_T0_NS"] = str(time.perf_counter_ns())


_SUBPROCESS_ENV_KEYS = ("LD_PRELOAD", "FOUNDRY_MODE", "FOUNDRY_SPAWN_T0_NS")


def _start_resource_tracker() -> None:
    """Start the launcher's multiprocessing resource tracker before the
    preload window: under the spawn start method Python starts it lazily in
    the first ``Process.start()``, which would otherwise be inside the window
    and give the tracker the hook too."""
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()


@contextlib.contextmanager
def scoped_environ(values: dict[str, str]):
    """``values`` in ``os.environ`` for the block (a child spawned inside
    copies them), the previous values restored afterwards."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def subprocess_preload_env():
    """``api.configure_subprocess``: the children's preload environment, set
    only around ``Process.start()`` (the child copies ``os.environ`` at that
    point) and restored afterwards, so the launcher's other processes do not
    load the hook."""
    _start_resource_tracker()
    saved = {key: os.environ.get(key) for key in _SUBPROCESS_ENV_KEYS}
    try:
        _set_subprocess_preload_env()
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
