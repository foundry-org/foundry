# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SAVE warm-up forwards in a private MemPool, per shape, before its capture.

torch >= 2.14's dynamo refuses to compile while a stream is capturing, so the
first call of every ``torch.compile``d helper must happen before the capture,
as SGLang's own two warm-up forwards per shape do. Their allocations must not
move the deterministic layout LOAD replays (LOAD runs no forward), and they
must not share the default caching-allocator pool with the per-shape
preparation buffers the graphs reference: a buffer served from a cached
warm-up block on SAVE needs a fresh segment on LOAD. So, inside a runner's
capture loop (:func:`run_capture_loop`), ``capture_one`` on SAVE runs the two
warm-up forwards of each shape in ONE private ``torch.cuda.MemPool`` with the
hook's allocation region stopped (checked per shape), then captures the shape
as before. After the loop the pool must hold no live block; it is released
(MemPool destructor) and no allocator segment may still carry its id. LOAD
runs the same loop with no forward and restores each graph.

Persistent resources first created by a forward (cuBLAS workspaces, global
kernel-argument caches, lazily built per-layer buffers) would otherwise land
in the private pool and be retained: :func:`bootstrap_persistent_resources`
creates them in the normal allocation domain on both modes before the loop.
A retained block fails SAVE; ``FOUNDRY_SGLANG_WARM_POOL_TRACE=1`` adds the
stack that allocated it.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import torch

from foundry import ops as cge
from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
)

logger = logging.getLogger(__name__)


def pool_trace_enabled() -> bool:
    """``FOUNDRY_SGLANG_WARM_POOL_TRACE=1``: record allocation stacks (Python
    and C++) during the SAVE capture loop so that a block retained by the
    warm-up pool is reported with the stack that allocated it."""
    return os.environ.get("FOUNDRY_SGLANG_WARM_POOL_TRACE") == "1"


class _Loop:
    """State of the capture loop being run by :func:`run_capture_loop`."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.pool: torch.cuda.MemPool | None = None
        self.warmed_shapes = 0


_loop: _Loop | None = None


def active() -> bool:
    """True inside :func:`run_capture_loop` (SAVE warms each shape before
    capturing it); False in a single-pass loop (an sglang that does not call
    it: the first forward then happens inside the capture)."""
    return _loop is not None


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
    create inside the private warm-up pool. Runs once per runner, before its
    capture loop, on the capture stream.

    The list is what a graph references and no capture allocates: audited for
    the validated backends (FlashInfer and FA3 attention, DeepGEMM / Triton
    MoE, standard and DeepEP dispatch), not a universal inventory. A missed
    resource shows up as a block retained by the pool after the loop (SAVE
    fails, naming the block and, with ``FOUNDRY_SGLANG_WARM_POOL_TRACE=1``,
    the stack that allocated it). Extend the list here; never free a retained
    block instead.

    Returns the names of the resources prepared (logged)."""
    device = torch.device("cuda", torch.cuda.current_device())
    prepared = []
    # 1. cuBLAS handle and workspace of the capture stream. torch keeps one
    #    workspace per (handle, stream), so also for every side stream a model
    #    module holds, directly (Qwen3.5's GDN `alt_stream`) or in a list
    #    (DeepSeek-V4's `alt_streams`): the first GEMM there would create one
    #    in the warm-up pool.
    torch.cuda.current_blas_handle()
    prepared.append("cublas_handle")
    side_streams: dict[int, torch.cuda.Stream] = {}
    for module in runner.model_runner.model.modules():
        for value in vars(module).values():
            items = value if isinstance(value, (list, tuple)) else (value,)
            for item in items:
                if isinstance(item, torch.cuda.Stream):
                    side_streams[item.cuda_stream] = item
    for stream in side_streams.values():
        with torch.cuda.stream(stream):
            torch.cuda.current_blas_handle()
    if side_streams:
        prepared.append(f"cublas_handle(side streams x{len(side_streams)})")
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


def _block_stack(block: dict, limit: int = 30) -> str:
    """The recorded allocation stack of a snapshot block (empty unless
    :func:`pool_trace_enabled`), innermost frame first."""
    frames = block.get("frames") or []
    lines = [f"{f.get('filename')}:{f.get('line')} {f.get('name')}" for f in frames[:limit]]
    return "\n      ".join(lines)


def warm_up(
    shape_key: Any,
    forward_fn: Callable[[], Any],
    post_warmup_hook: Callable[[], None] | None,
    tp_group: Any,
) -> None:
    """SAVE, before capturing ``shape_key``: SGLang's two warm-up forwards for
    the shape, as upstream ``FullCudaGraphBackend.capture_one`` runs them
    (synchronize, TP barrier, forward, drop the output, post-warm-up hook),
    inside the loop's private pool with the allocation region stopped.
    Upstream's graph-pool precarve measurement is not run (the precarve is
    pinned off) and its output-buffer reuse is not touched (Foundry packs the
    outputs itself)."""
    loop = _loop
    if loop is None:
        raise RuntimeError("[Foundry] warm_up called outside run_capture_loop")
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
    """SAVE, after the loop: require no live block in the pool, release it and
    require that no allocator segment still carries its id. A retained block
    is a hard error naming the block (and its allocation stack under
    :func:`pool_trace_enabled`): a persistent resource missing from
    :func:`bootstrap_persistent_resources`, or created by a post-warm-up
    hook, never something to free here."""
    pool = loop.pool
    loop.pool = None
    if pool is None:
        logger.info("[Foundry] SAVE warm-up pool: no warm-up ran")
        return
    segments = pool.snapshot(include_traces=pool_trace_enabled())
    reserved = sum(s["total_size"] for s in segments)
    allocated = sum(s["allocated_size"] for s in segments)
    active_bytes = sum(s["active_size"] for s in segments)
    if allocated:
        live = [
            (b["address"], b["size"], _block_stack(b))
            for s in segments
            for b in s.get("blocks", [])
            if b.get("state") == "active_allocated"
        ]
        stacks = "".join(
            f"\n  block 0x{a:x}+{n} allocated at:\n      {st}" for a, n, st in live[:8] if st
        )
        raise RuntimeError(
            f"[Foundry] SAVE warm-up pool retains {allocated} allocated bytes after the "
            f"capture loop ({len(live)} blocks: "
            + ", ".join(f"0x{a:x}+{n}" for a, n, _ in live[:8])
            + "). A persistent resource is created by a warm-up forward or a "
            "post-warm-up hook: create it in warmup_pool.bootstrap_persistent_resources "
            "instead" + (stacks or " (set FOUNDRY_SGLANG_WARM_POOL_TRACE=1 for allocation stacks)")
        )
    pool_id = pool.id
    del pool
    gc.collect()
    remaining = [
        s
        for s in torch.cuda.memory_snapshot()
        if tuple(s.get("segment_pool_id", ())) == tuple(pool_id)
    ]
    logger.info(
        "[Foundry] SAVE warm-ups of %d shapes ran in a private pool: reserved %.1f MB, "
        "active %d bytes before release; released (MemPool destructor): %s",
        loop.warmed_shapes,
        reserved / 2**20,
        active_bytes,
        "no segment left" if not remaining else f"{len(remaining)} segments LEFT",
    )
    if remaining:
        raise RuntimeError(
            f"[Foundry] SAVE warm-up pool {pool_id} still owns {len(remaining)} allocator "
            "segments after its release"
        )


# ---------------------------------------------------------------------------
# The capture loop
# ---------------------------------------------------------------------------


def run_capture_loop(runner: Any, loop_fn: Callable[[], Any]) -> Any:
    """Run a runner's per-shape capture loop (``_capture_one_stream``, inside
    its capture session): persistent bootstrap, then the loop, in which
    ``capture_one`` on SAVE warms each shape in the private pool right before
    capturing it (LOAD restores); then, on SAVE, the pool is released and
    checked. Without SAVE / LOAD the loop runs as it is."""
    global _loop
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        return loop_fn()
    if _loop is not None:
        raise RuntimeError("[Foundry] nested capture loops are not supported")
    _loop = loop = _Loop(runner)
    tracing = mode == CUDAGraphExtensionMode.SAVE and pool_trace_enabled()
    if tracing:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="all")
    try:
        prepared = bootstrap_persistent_resources(runner)
        logger.info(
            "[Foundry] capture loop (%s); persistent bootstrap: %s",
            type(runner).__name__,
            prepared,
        )
        rt.log_alloc_offset("after_persistent_bootstrap")
        result = loop_fn()
        # Same sequence point on both modes: cyclic garbage of the loop
        # (deterministic domain) is collected before the pool check.
        gc.collect()
        if mode == CUDAGraphExtensionMode.SAVE:
            _release_pool(loop)
        rt.log_alloc_offset("after_capture_loop")
        return result
    finally:
        _loop = None
        if tracing:
            torch.cuda.memory._record_memory_history(enabled=None)
