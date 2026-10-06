# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SAVE warm-up forwards in a private MemPool, between two preparation passes.

torch >= 2.14's dynamo refuses to compile while a stream is capturing, so the
first call of every ``torch.compile``d helper must happen before the capture,
as SGLang's own two warm-up forwards per shape do. Their allocations must not
change the deterministic layout LOAD replays (LOAD runs no forward). Each
runner's capture loop therefore runs twice (:func:`run_capture_loop`):

- pass A, preparation: SGLang's loop prepares every shape as usual (dummy
  batch, attention metadata, wrappers); Foundry's ``capture_one`` runs, on
  SAVE only, the two warm-up forwards inside ONE private
  ``torch.cuda.MemPool`` with the hook's allocation region stopped (so they
  never move the deterministic cursor, checked per shape), and produces no
  graph. LOAD runs the same preparation with no forward. After the pass the
  pool must hold no live block; it is released (MemPool destructor) and no
  allocator segment may still carry its id.
- pass B, capture: the loop runs again; ``capture_one`` captures (SAVE, no
  warm-ups) or restores (LOAD) each graph, as in a single-pass run.

Persistent resources first created by a forward (cuBLAS workspace, global
kernel-argument caches, lazily built per-layer buffers) would otherwise land
in the private pool and be retained: :func:`bootstrap_persistent_resources`
creates them in the normal allocation domain on both modes before pass A.
After pass B, :func:`audit_graph_pointers` fails SAVE when a saved graph
references the private pool's former address ranges.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from foundry import ops as cge
from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)

logger = logging.getLogger(__name__)

PREPARE = "prepare"
CAPTURE = "capture"


class _Loop:
    """State of the capture loop being run by :func:`run_capture_loop`."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.phase = PREPARE
        self.pool: torch.cuda.MemPool | None = None
        self.pool_ranges: list[tuple[int, int]] = []
        self.warmed_shapes = 0


_loop: _Loop | None = None


def current_phase() -> str | None:
    """PREPARE or CAPTURE inside :func:`run_capture_loop`, None outside it
    (a single-pass capture loop: capture_one captures or restores)."""
    return None if _loop is None else _loop.phase


# ---------------------------------------------------------------------------
# Persistent bootstrap
# ---------------------------------------------------------------------------


def _attn_backend(runner: Any) -> Any:
    return getattr(runner, "attn_backend", None) or runner.model_runner.attn_backend


def _flashinfer_decode_updater(attn_backend: Any) -> Any:
    """``indices_updater_decode`` of a FlashInfer backend, also when it is the
    full-attention child of a hybrid (linear-attention) wrapper."""
    for backend in (attn_backend, getattr(attn_backend, "full_attn_backend", None)):
        updater = getattr(backend, "indices_updater_decode", None)
        if updater is not None:
            return updater
    return None


def _dispatchers(runner: Any) -> dict[str, Any]:
    return {
        name: module.dispatcher
        for name, module in runner.model_runner.model.named_modules()
        if hasattr(module, "dispatcher")
    }


def bootstrap_persistent_resources(runner: Any) -> list[str]:
    """Create, on SAVE and LOAD alike and in the normal (recorded) allocation
    domain, the persistent device resources a first forward would otherwise
    create inside the private warm-up pool. Runs once per runner, at the head
    of pass A, on the capture stream.

    This is an audited list for the validated backends (FlashInfer and FA3
    attention, DeepGEMM / Triton MoE, standard and DeepEP dispatch), not a
    universal inventory: another backend may own another such resource. A
    missed one shows up as a block retained by the pool after pass A (SAVE
    fails, naming the block and, when found, its owner) or as a graph
    reference into the pool's former ranges (:func:`audit_graph_pointers`).
    Extend the list here; never free a retained block instead.

    Returns the names of the resources prepared (logged)."""
    device = torch.device("cuda", torch.cuda.current_device())
    prepared = []
    # 1. cuBLAS handle and workspace of the capture stream (per stream).
    torch.cuda.current_blas_handle()
    prepared.append("cublas_handle")
    # 2. FlashInfer's global ALiBi slopes buffer: FlashInfer creates it on the
    #    first decode plan, also for models without ALiBi.
    updater = _flashinfer_decode_updater(_attn_backend(runner))
    fi_utils = sys.modules.get("flashinfer.utils")
    if updater is not None and fi_utils is not None:
        fi_utils._get_cache_alibi_slopes_buf(updater.num_qo_heads, device)
        prepared.append("flashinfer_alibi_slopes")
    dispatchers = _dispatchers(runner)
    if dispatchers:
        # 3. MoE router's cached int32 placeholder (a kernel pointer argument).
        from sglang.kernels.ops.moe.moe_fused_gate import _dummy_i32

        _dummy_i32(device)
        prepared.append("moe_router_placeholder")
        # 4. EP > 1: StandardDispatcher's per-layer local expert map, built
        #    lazily by the first dispatch.
        n = 0
        for dispatcher in dispatchers.values():
            prepare = getattr(dispatcher, "prepare_local_expert_mapping", None)
            if (
                prepare is not None
                and getattr(dispatcher, "moe_ep_size", 1) > 1
                and not getattr(dispatcher, "skip_local_expert_mapping", True)
            ):
                prepare()
                n += 1
        if n:
            prepared.append(f"local_expert_mapping x{n}")
    return prepared


# ---------------------------------------------------------------------------
# Warm-up pool
# ---------------------------------------------------------------------------


def _pool_segments(pool: torch.cuda.MemPool) -> list[dict]:
    return pool.snapshot(include_traces=False)


def _owner_paths(runner: Any, ranges: list[tuple[int, int]], depth: int = 5) -> list[str]:
    """Attribute paths (from the runner, its attention backend, buffers and
    MoE dispatchers) of CUDA tensors whose storage lies in ``ranges``: the
    likely owner of a retained block. Best effort, depth-limited."""
    roots = {
        "runner": runner,
        "attn_backend": _attn_backend(runner),
        "buffers": getattr(runner, "buffers", None),
        **{f"dispatcher[{k}]": v for k, v in _dispatchers(runner).items()},
    }
    found: list[str] = []
    seen: set[int] = set()

    def visit(obj: Any, path: str, left: int) -> None:
        if id(obj) in seen or len(found) >= 20:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and any(lo <= obj.data_ptr() < hi for lo, hi in ranges):
                found.append(path)
            return
        if left <= 0 or isinstance(obj, (str, bytes, int, float, bool, type(None), type)):
            return
        if isinstance(obj, dict):
            items = list(obj.items())[:200]
        elif isinstance(obj, (list, tuple)):
            items = list(enumerate(obj))[:200]
        elif isinstance(obj, torch.nn.Module) or callable(obj):
            return
        else:
            items = list(getattr(obj, "__dict__", {}).items())[:200]
        for key, value in items:
            if str(key) in ("model_runner", "model", "_cuda_graph_runner"):
                continue
            visit(value, f"{path}.{key}", left - 1)

    for name, obj in roots.items():
        if obj is not None:
            visit(obj, name, depth)
    return found


def warm_up(
    shape_key: Any,
    forward_fn: Callable[[], Any],
    post_warmup_hook: Callable[[], None] | None,
    tp_group: Any,
) -> None:
    """SAVE, pass A: SGLang's two warm-up forwards for one shape, as upstream
    ``FullCudaGraphBackend.capture_one`` runs them (synchronize, TP barrier,
    forward, drop the output, post-warm-up hook), inside the loop's private
    pool with the allocation region stopped. Upstream's graph-pool precarve
    measurement is not run (the precarve is pinned off) and its output-buffer
    reuse is not touched (Foundry packs the outputs itself)."""
    loop = _loop
    if loop.pool is None:
        loop.pool = torch.cuda.MemPool()
    cursor = cge.get_current_alloc_offset()
    cge.stop_allocation_region()
    try:
        with torch.cuda.use_mem_pool(loop.pool):
            for _ in range(2):
                torch.cuda.synchronize()
                if tp_group is not None:
                    tp_group.barrier()
                output = forward_fn()
                del output
                if post_warmup_hook is not None:
                    post_warmup_hook()
        torch.cuda.synchronize()
    finally:
        cge.resume_allocation_region()
    after = cge.get_current_alloc_offset()
    if after != cursor:
        raise RuntimeError(
            f"[Foundry] SAVE warm-up of {shape_key} moved the deterministic cursor "
            f"{cursor} -> {after}: an allocation bypassed the private pool"
        )
    loop.warmed_shapes += 1


def _release_pool(loop: _Loop) -> None:
    """SAVE, after pass A: require no live block in the pool, release it and
    require that no allocator segment still carries its id. A retained block
    is a hard error naming the block and, when found, its owner: it is a
    persistent resource missing from :func:`bootstrap_persistent_resources`
    (or created by a post-warm-up hook), never something to free here."""
    pool = loop.pool
    loop.pool = None
    if pool is None:
        logger.info("[Foundry] SAVE warm-up pool: no warm-up ran")
        return
    segments = _pool_segments(pool)
    reserved = sum(s["total_size"] for s in segments)
    allocated = sum(s["allocated_size"] for s in segments)
    active = sum(s["active_size"] for s in segments)
    loop.pool_ranges = [(s["address"], s["address"] + s["total_size"]) for s in segments]
    if allocated:
        live = [
            (b["address"], b["size"])
            for s in segments
            for b in s.get("blocks", [])
            if b.get("state") == "active_allocated"
        ]
        owners = _owner_paths(loop.runner, [(a, a + n) for a, n in live])
        raise RuntimeError(
            f"[Foundry] SAVE warm-up pool retains {allocated} allocated bytes after the "
            f"preparation pass ({len(live)} blocks: "
            + ", ".join(f"0x{a:x}+{n}" for a, n in live[:8])
            + f"); owners found: {owners or 'none'}. A persistent resource is created by a "
            "warm-up forward or a post-warm-up hook: create it in "
            "warmup_pool.bootstrap_persistent_resources instead"
        )
    pool_id = pool.id
    del pool
    gc.collect()
    remaining = [
        s for s in torch.cuda.memory_snapshot() if tuple(s.get("segment_pool_id", ())) == tuple(pool_id)
    ]
    logger.info(
        "[Foundry] SAVE warm-ups of %d shapes ran in a private pool: reserved %.1f MB, "
        "active %d bytes before release; released (MemPool destructor): %s",
        loop.warmed_shapes,
        reserved / 2**20,
        active,
        "no segment left" if not remaining else f"{len(remaining)} segments LEFT",
    )
    if remaining:
        raise RuntimeError(
            f"[Foundry] SAVE warm-up pool {pool_id} still owns {len(remaining)} allocator "
            "segments after its release"
        )


# ---------------------------------------------------------------------------
# Graph pointer audit
# ---------------------------------------------------------------------------

_KERNEL_ARG_KEYS = ("value_hex", "extra_argBuffer_hex")
_ADDRESS_KEYS = ("srcDevice", "dstDevice", "dst", "src", "devPtr")
# fused_qknorm_warp's QKNormParams (72 bytes): pointers at 0/8 (q/k) and 48/56
# (weights); the other words are scalars or padding that may look like one.
_QKNORM_POINTER_OFFSETS = (0, 8, 48, 56)


def audit_graph_pointers(files: list[Path], ranges: list[tuple[int, int]]) -> tuple[int, int]:
    """Scan saved graph JSONs' kernel-argument words and memcpy/memset
    addresses for pointers into ``ranges`` (the released warm-up pool's former
    segments); raise if any. Conservative, not a proof: a pointer reached by
    indirection is not seen. Returns (files, nodes) scanned."""
    hits: list[str] = []
    nodes = 0

    def inside(address: int) -> bool:
        return any(lo <= address < hi for lo, hi in ranges)

    def walk(obj: Any, path: str, function_name: str) -> None:
        nonlocal nodes
        if isinstance(obj, dict):
            function_name = obj.get("function_name", function_name)
            if isinstance(obj.get("type"), str) and obj["type"].endswith("Node"):
                nodes += 1
            for key, value in obj.items():
                if key in _KERNEL_ARG_KEYS and isinstance(value, str):
                    data = bytes.fromhex(value)
                    for offset in range(0, len(data) - 7, 8):
                        address = int.from_bytes(data[offset : offset + 8], "little")
                        if not inside(address):
                            continue
                        if (
                            "fused_qknorm_warp" in function_name
                            and "QKNormParams" in function_name
                            and len(data) == 72
                            and offset not in _QKNORM_POINTER_OFFSETS
                        ):
                            continue
                        hits.append(f"{path}.{key}+{offset}: 0x{address:x} ({function_name})")
                elif key in _ADDRESS_KEYS and isinstance(value, (str, int)):
                    try:
                        address = int(value, 0) if isinstance(value, str) else value
                    except ValueError:
                        continue
                    if inside(address):
                        hits.append(f"{path}.{key}: 0x{address:x}")
                else:
                    walk(value, f"{path}.{key}", function_name)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]", function_name)

    for path in files:
        walk(json.loads(path.read_text()), path.name, "")
    if files and not nodes:
        raise RuntimeError(f"[Foundry] SAVE pointer audit found no graph nodes in {len(files)} files")
    if hits:
        raise RuntimeError(
            f"[Foundry] SAVE: {len(hits)} captured-graph references into the released "
            "warm-up pool: " + "; ".join(hits[:10])
        )
    return len(files), nodes


def _graph_files(workspace_dir: str | None) -> set[Path]:
    if workspace_dir is None:
        return set()
    return set(Path(workspace_dir).glob("graph*.json"))


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------


def run_capture_loop(runner: Any, loop_fn: Callable[[], Any]) -> Any:
    """Run a runner's per-shape capture loop (``_capture_one_stream``, inside
    its capture session) as pass A (preparation; SAVE warm-ups in the private
    pool) then pass B (capture or restore). Without SAVE / LOAD, or when the
    integration is disabled with ``FOUNDRY_SGLANG_SINGLE_PASS=1``, the loop
    runs once."""
    global _loop
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE or os.environ.get("FOUNDRY_SGLANG_SINGLE_PASS") == "1":
        return loop_fn()
    if _loop is not None:
        raise RuntimeError("[Foundry] nested capture loops are not supported")
    _loop = loop = _Loop(runner)
    try:
        prepared = bootstrap_persistent_resources(runner)
        logger.info("[Foundry] persistent bootstrap (%s): %s", type(runner).__name__, prepared)
        rt.log_alloc_offset("after_persistent_bootstrap")
        loop_fn()
        # Same sequence point on both modes: cyclic garbage of the preparation
        # pass (deterministic domain) is collected before the capture pass.
        gc.collect()
        if mode == CUDAGraphExtensionMode.SAVE:
            _release_pool(loop)
        rt.log_alloc_offset("after_preparation_pass")
        loop.phase = CAPTURE
        cfg = get_config()
        before = _graph_files(cfg.workspace_dir if cfg else None)
        result = loop_fn()
        if mode == CUDAGraphExtensionMode.SAVE and loop.pool_ranges:
            new = sorted(_graph_files(cfg.workspace_dir if cfg else None) - before)
            n_files, n_nodes = audit_graph_pointers(new, loop.pool_ranges)
            logger.info(
                "[Foundry] SAVE pointer audit: %d graphs, %d nodes, no reference into the "
                "released warm-up pool (%d ranges)",
                n_files,
                n_nodes,
                len(loop.pool_ranges),
            )
        rt.log_alloc_offset("after_capture_pass")
        return result
    finally:
        _loop = None
