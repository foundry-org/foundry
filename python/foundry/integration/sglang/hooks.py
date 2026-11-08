# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime call-site functions of the Foundry SGLang integration, called
from SGLang's own call sites through ``api.py``."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import logging
import threading
import time
from dataclasses import asdict

from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.api import Unsupported
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    compute_workspace_rank,
    get_config,
    get_graph_extension_mode,
)

logger = logging.getLogger(__name__)


def _workspace_ranks(parallel) -> tuple[int, int, int | None]:
    """(tp_rank, pp_rank, dp_rank) for the Foundry workspace rank.

    ``parallel`` is upstream's ``get_parallel()`` context (stamped by
    ``publish(ranks=...)`` before any process group exists).

    Only spawn-time ranks are read: the attention-DP ranks (``attn_dp_rank``,
    ``attn_tp_rank``) are computed in ``initialize_dp_attention``, inside the
    bring-up this runs ahead of. They are not needed either: attention-DP
    groups sit inside the TP world, so ``tp_rank`` already tells them apart.
    ``dp_rank`` is what the DP controller passed: the replica without
    attention DP, the attention-DP group with it;
    ``config.compute_workspace_rank`` takes the replica from it
    (``config.dp_replica``)."""
    return parallel.tp_rank, parallel.pp_rank, parallel.dp_rank


def _resolve_triton_key():
    """Triton's process-wide cache key function (``functools.lru_cache``d):
    ``triton.runtime.cache`` on current Triton, ``triton.compiler.compiler``
    on older releases."""
    try:
        from triton.runtime.cache import triton_key
    except ImportError:
        from triton.compiler.compiler import triton_key
    return triton_key


def _prewarm_triton_key() -> None:
    try:
        _resolve_triton_key()()
    except Exception as e:  # noqa: BLE001 - a prewarm must never fail the rank
        logger.debug("[Foundry] SGLang triton_key prewarm skipped: %r", e)


def _start_triton_key_prewarm() -> None:
    """Compute Triton's cache key (sha256 of libtriton and the triton.language
    sources, ~0.45 s, cached for the process) on a daemon thread, so it
    overlaps the weight load instead of landing on the first Triton launch
    inside the capture loop. hashlib releases the GIL on large buffers.
    Nothing waits for it: a later caller either hits the cache or computes
    the key itself, as without the prewarm."""
    threading.Thread(
        target=_prewarm_triton_key, name="foundry-triton-key-prewarm", daemon=True
    ).start()
    logger.info("[Foundry] SGLang triton_key prewarm started")


def _before_distributed_init(server_args, device: str, gpu_id: int, ranks) -> None:
    """Bind this rank's VMM region before any communicator exists. Same
    sequence point on SAVE and LOAD: the head of the process-group bring-up
    (``bootstrap.init_parallel_runtime``)."""
    mode = get_graph_extension_mode()
    # Bind this rank's CUDA device BEFORE reserving the VMM region. The
    # distributed bring-up calls set_device(gpu_id) itself, but foundry's
    # set_allocation_region (inside setup_graph_extension) reserves the region
    # on the *current* device. For DP rank > 0 the current device is still
    # cuda:0 at this point, so without setting it first the region lands on the
    # wrong GPU and the rank's later allocations fault with an async illegal
    # memory access (surfacing at the first Stream()/kernel). Single-GPU is
    # unaffected (gpu_id == 0).
    if device == "cuda":
        import torch

        torch.get_device_module(device).set_device(gpu_id)

    _start_triton_key_prewarm()

    tp_rank, pp_rank, dp_rank = ranks
    rt.setup_graph_extension(server_args, tp_rank=tp_rank, pp_rank=pp_rank, dp_rank=dp_rank)
    rt.log_alloc_offset("after_setup_graph_ext")
    if mode == CUDAGraphExtensionMode.LOAD:
        rt.check_capture_loop_version()
        # Grow the driver's graph-exec memory now, on a background thread,
        # so Phase 2's instantiates at the capture point reuse it (see
        # graph_ops.start_exec_pool_prewarm).
        from foundry.integration.sglang.graph_ops import start_exec_pool_prewarm

        start_exec_pool_prewarm()


# ---------------------------------------------------------------------------
# Call-site functions (api.py calls them from SGLang's own call sites).
# ---------------------------------------------------------------------------


def _waited(barrier):
    def waited_barrier(self, *args, **kwargs):
        work = barrier(self, *args, **kwargs)
        if work is not None:
            work.wait()
        return work

    return waited_barrier


_barrier_wait_installed = False


def _wait_on_barrier_works() -> None:
    """Make ``ProcessGroup.barrier()`` finish on the calling thread.

    The barrier is an async all-reduce of a 4-byte CUDA tensor that torch keeps
    alive until the returned Work is waited on. A caller that drops the Work
    (DeepEP v2's ``ElasticBuffer.__init__``) leaves the release to the NCCL
    watchdog thread, at a run-dependent moment: the freed block then is or is
    not a hole in torch's small pool when the graph layout allocates its host
    buffers, and LOAD's buffers miss the addresses SAVE's graphs hold (EAGLE3
    on a DeepEP v2 target: draft-decode index out of bounds, or a lower accept
    length). Waiting releases the tensor at the call, on both modes."""
    global _barrier_wait_installed
    if _barrier_wait_installed:
        return
    from torch._C import _distributed_c10d as c10d

    patched = []
    for name in ("ProcessGroup", "Backend"):
        cls = getattr(c10d, name, None)
        if cls is not None and "barrier" in vars(cls):
            cls.barrier = _waited(cls.barrier)
            patched.append(name)
    _barrier_wait_installed = True
    logger.info(
        "[Foundry] barrier works are waited on at the call (%s)", ", ".join(patched) or "none"
    )


def _disable_dynamo() -> None:
    """Scheduler process: besides the ``TORCHDYNAMO_DISABLE=1`` pin (read when
    ``torch.compile`` decorates), set ``torch._dynamo.config.disable``, which
    dynamo checks on every frame, so a helper decorated before the pin took
    effect does not compile inside a capture either."""
    import torch._dynamo

    if not torch._dynamo.config.disable:
        torch._dynamo.config.disable = True
        logger.info("[Foundry] torch._dynamo.config.disable = True (TORCHDYNAMO_DISABLE=1 pin)")


def deactivate(step: str, reason: str) -> None:
    """``api.deactivate``: SAVE in a scheduler past setup marks the archive
    invalid; the allocation region and the kernel-binary dump stop, the cyclic
    GC hold ends, and the mode becomes none, so every later call is a no-op."""
    cfg = get_config()
    if cfg is None:
        return
    if rt.get_state() is not None:
        if cfg.mode == CUDAGraphExtensionMode.SAVE:
            rt.mark_archive_invalid(step, reason)
        rt.leave_graph_extension()
    rt.release_cyclic_gc()
    cfg.mode = CUDAGraphExtensionMode.NONE


def log_init_summary() -> None:
    mode = get_graph_extension_mode()
    state = rt.get_state()
    if mode == CUDAGraphExtensionMode.NONE or state is None:
        return
    if mode == CUDAGraphExtensionMode.SAVE:
        done = f"{state.capture_index} graphs saved"
    else:
        done = f"{len(state.loaded_graphs)} graphs restored"
    logger.info(
        "[Foundry] init summary: cuda-graph persistence %s, %s (%s)",
        mode.value,
        done,
        get_config().workspace_dir,
    )


def _check_archive_valid(parallel) -> None:
    """LOAD, before anything is set up: an archive SAVE marked invalid (it fell
    back) is not restored. Every SAVE rank writes the root marker after the
    ranks agreed, so every LOAD rank sees one and they fall back together."""
    rank = compute_workspace_rank(parallel, *_workspace_ranks(parallel))
    marker = rt.archive_invalid_marker(rank)
    if marker is not None:
        path, step, reason = marker
        raise Unsupported(
            "archive", f"SAVE marked {path} invalid (unsupported at {step}: {reason})"
        )


def before_parallel_init(device: str) -> None:
    """Head of ``bootstrap.init_parallel_runtime``: bind this
    rank's VMM region before any communicator exists."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        return
    from sglang.srt.runtime_context import get_device, get_parallel

    parallel = get_parallel()
    if mode == CUDAGraphExtensionMode.LOAD:
        _check_archive_valid(parallel)
    _disable_dynamo()
    _wait_on_barrier_works()
    rt.hold_cyclic_gc()
    # get_parallel() answers from the published placement (resolved
    # widths and spawn ranks), unlike the raw record fields.
    _before_distributed_init(
        parallel,
        device,
        get_device().gpu_id,
        _workspace_ranks(parallel),
    )


# Context overrides the memory-pool resolver issued on SAVE
# (get_context().override, e.g. max_mamba_cache_size for hybrid
# Mamba/GDN/KDA models); the pool factories read them from the bags, so LOAD
# must replay them when it skips the resolver. Filled by
# record_memory_pool_overrides, persisted by record_memory_pool_config.
_resolve_overrides: list = []
_resolve_log_start: int | None = None


def _json_safe(value) -> bool:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    return False


def replay_saved_memory_pool_config():
    """Head of ``KVCacheConfigurator._resolve_memory_pool_config``.

    SAVE: remember where the runtime-context override log stands, and return
    None (the resolver runs). LOAD: return the saved ``MemoryPoolConfig``
    instead of profiling memory, so pool construction itself runs the SAME
    upstream code with the SAME sizes in both modes (identical VMM allocation
    trajectory); the caller returns it without running the resolver.

    Post-capture KV sizing (SGLANG_ENABLE_POST_CAPTURE_KV_SIZING) is not
    supported: it re-sizes the pool from post-capture free memory, which LOAD
    cannot reproduce (rejected at resolution)."""
    global _resolve_log_start
    mode = get_graph_extension_mode()
    if mode != CUDAGraphExtensionMode.NONE:
        rt.log_alloc_offset("before_memory_pool_config")
    if mode == CUDAGraphExtensionMode.SAVE:
        from sglang.srt.runtime_context import get_context

        _resolve_log_start = len(get_context().overrides_log())
        return None
    if mode != CUDAGraphExtensionMode.LOAD:
        return None
    import torch
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    state = rt.load_warmup_state()
    if not state.memory_pool_config:
        raise RuntimeError("Foundry LOAD requires memory_pool_config")
    # Mirror _profile_available_bytes' allocator side effects (gc +
    # empty_cache via get_available_gpu_memory). Without this, torch's
    # caching allocator retains segments that SAVE released, and later
    # allocations take a different cuMemAlloc path — drifting the VMM
    # cursor away from SAVE's recorded offsets.
    gc.collect()
    torch.cuda.empty_cache()
    valid = {f.name for f in dataclasses.fields(MemoryPoolConfig)}
    config = MemoryPoolConfig(**{k: v for k, v in state.memory_pool_config.items() if k in valid})
    if state.context_overrides:
        from sglang.srt.runtime_context import get_context

        for source, fields in state.context_overrides:
            get_context().override(f"foundry_replay:{source}", **fields)
    logger.info(
        "[Foundry] SGLang reused saved memory pool config (%d context overrides replayed)",
        len(state.context_overrides),
    )
    return config


def record_memory_pool_overrides() -> None:
    """End of ``_resolve_memory_pool_config`` (SAVE): keep the overrides the
    resolver issued since :func:`replay_saved_memory_pool_config`."""
    global _resolve_log_start
    if get_graph_extension_mode() != CUDAGraphExtensionMode.SAVE or _resolve_log_start is None:
        return
    from sglang.srt.runtime_context import get_context

    _resolve_overrides[:] = [
        [source, {k: v for k, v in fields.items() if _json_safe(v)}]
        for source, fields in get_context().overrides_log()[_resolve_log_start:]
    ]
    _resolve_log_start = None


def _save_memory_pool_state(memory_pool_config) -> None:
    state = rt.create_warmup_state(asdict(memory_pool_config), context_overrides=_resolve_overrides)
    rt.save_warmup_state(state)


def record_memory_pool_config(memory_pool_config) -> None:
    """End of ``_resolve_memory_pool_config`` (API 2.3), SAVE: keep the
    resolver's context overrides and write them with the resolved
    MemoryPoolConfig into ``warmup_state.json``. The configurator hands this
    config object unchanged to the pools and back to
    ``ModelRunner.memory_pool_config``."""
    if get_graph_extension_mode() != CUDAGraphExtensionMode.SAVE:
        return
    record_memory_pool_overrides()
    _save_memory_pool_state(memory_pool_config)


def _begin_graph_layout(model, mode, flashinfer_tactics: str | None = None) -> None:
    """Pre-capture bootstraps + layout start, once per process, at the first
    runner capture: the prefill runner's when prefill graphs are on (sglang
    captures prefill before decode), the decode runner's otherwise. Same
    sequence point on SAVE and LOAD. ``flashinfer_tactics``: the runner's
    FlashInfer tuned-tactics cache file (None: its kernels take none)."""
    state = rt.get_state()
    if state is None:
        raise RuntimeError("Foundry SGLang state is not initialized")
    if state.layout_started:
        return

    # Pre-capture bootstraps. Everything sglang initializes lazily on the
    # first eager forward that a capturing stream rejects, or that must exist
    # at the same addresses on both modes, is done here, at the same sequence
    # point on SAVE and LOAD; no model forward runs. The MoE dispatcher buffers
    # / expert mappings and the logits all-gather state are sglang's own: its
    # EagerRunner warmup (BaseRunner._prepare_capture_state) builds them before
    # this first scope, on both modes.
    from foundry.integration.sglang.graph_ops import (
        agree_unsupported,
        bootstrap_collective_connections,
        bootstrap_flashinfer_tactics,
        bootstrap_lazy_runtimes,
        capture_state_unsupported,
    )

    # The fallback point of the capture state: before any bootstrap, so a
    # fallback leaves no Foundry-specific state behind (host-only check).
    agree_unsupported("capture state", capture_state_unsupported(model))
    # 0. FlashInfer's tuned tactics (both modes, host-only): installed from
    #    SAVE's cache file / the archive, since the tuning pass itself is
    #    pinned off (its allocations are timing-dependent).
    bootstrap_flashinfer_tactics(flashinfer_tactics)
    # 1. NCCL communicators (both modes): their first collective allocates
    #    and connects, which capture rejects.
    rt.log_alloc_offset("before_collective_bootstrap")
    bootstrap_collective_connections()
    rt.log_alloc_offset("after_collective_bootstrap")
    # 2. SAVE only: DeepGEMM's one-time runtime init, which capture rejects
    #    (allocation-free; the region stays suspended around it). The model's
    #    JIT kernel loads then happen inside the captured forward.
    if mode == CUDAGraphExtensionMode.SAVE:
        with rt.allocation_region_suspended():
            bootstrap_lazy_runtimes()
        rt.log_alloc_offset("after_lazy_runtimes")
    # 3. Both modes, before the first eager extend: the FlashInfer SM90 GDN
    #    prefill kernels load from FlashInfer's .o cache instead of compiling
    #    8-10 s in every scheduler (host-side patch, see flashinfer_cute_cache).
    from foundry.integration.sglang import flashinfer_cute_cache

    if flashinfer_cute_cache.gdn_flashinfer_in_use():
        if flashinfer_cute_cache.enabled():
            status = flashinfer_cute_cache.install()
            if status:
                logger.info("[Foundry] FlashInfer CuTe-DSL cache: %s", status)
        else:
            logger.info(
                "[Foundry] FlashInfer CuTe-DSL cache: off (%s=0)",
                flashinfer_cute_cache.OPT_OUT_ENV,
            )

    # The deterministic layout begins here on both modes: same sequence
    # point, same (empty) caching-allocator state; LOAD's
    # preallocate_for_load_mode maps and replays from this point.
    rt.mark_layout_start()
    state.layout_started = True


def _preallocate_once() -> None:
    """LOAD: map the recorded layout once, at the first runner capture that
    restores graphs (it spans both phases' graph memory)."""
    state = rt.get_state()
    if state.preallocated:
        return
    rt.log_alloc_offset("before_preallocate")
    rt.preallocate_for_load_mode()
    rt.log_alloc_offset("after_preallocate")
    state.preallocated = True


# The capture scope being run: (phase, prefill request-slot count or None,
# decode-type scope kind or None).
_scope: tuple[str, int | None, str | None] | None = None


def capture_one(shape_key, forward_fn, *, pool, stream):
    """Replaces ``FullCudaGraphBackend.capture_one`` for one shape, SAVE or
    LOAD, inside :func:`capture_scope`; the runner's capture loop around it is
    SGLang's own. Returns ``(graph, output)`` for the caller to store in the
    backend. ``pool`` / ``stream``: the backend's graph pool and capture
    stream (SAVE)."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        raise RuntimeError("[Foundry] capture_one called without SAVE or LOAD")
    if _scope is None:
        raise RuntimeError("[Foundry] capture_one called outside capture_scope")
    req_slots = _scope[1]
    kind = _scope[2]
    variant = _graph_variant(shape_key)
    if mode == CUDAGraphExtensionMode.LOAD:
        if variant is not None:
            raise RuntimeError(
                f"[Foundry] shape {shape_key} restores a graph variant ({variant}); "
                "Foundry save/load does not support graph variants"
            )
        # LOAD, both runners: the upstream capture loop runs (see
        # capture_scope) and only the capture is replaced, by the archived
        # graph for this shape; its allocator events replay here, at the point
        # SAVE captured it, after the same per-shape eager work. No forward
        # runs on LOAD.
        from foundry.integration.sglang.graph_ops import (
            restore_next_decode_graph,
            restore_next_prefill_graph,
        )

        if req_slots is not None:
            graph, out = restore_next_prefill_graph(shape_key, req_slots)
        else:
            graph, out = restore_next_decode_graph(shape_key)
        # Every graph returned (SAVE's FoundryCUDAGraph, LOAD's restored
        # graphs) is a foundry ``ops.CUDAGraph``, which binds ``reset()``
        # (csrc/binding.cpp): upstream's FullCudaGraphBackend.cleanup() calls
        # ``graph.reset()`` on each.
        return graph, out

    # SAVE: capture directly, no warm-up forwards (their freed activations
    # would leave cached segments LOAD cannot reproduce). Dynamo is pinned off
    # (pins.ENV_PINS), as it refuses to trace inside a capture.
    from foundry.integration.sglang.graph_ops import save_graph

    graph, out = _save_capture(shape_key, variant, forward_fn, pool=pool, stream=stream)
    if req_slots is not None:
        save_graph(graph, out, shape_key, prefill_req_slots=req_slots)
    else:
        save_graph(graph, out, shape_key, kind=kind)
    return graph, out


# SAVE: the dispatchers and logits processors of the current scope's model,
# whose attributes every capture must leave as it found them (_save_capture).
_snapshot_objects: list = []
# SAVE: graphs discarded at the fallback. They stay alive: a reset graph that
# was the pool's only user drops the pool's use count to 0 while its blocks
# are live, and torch's next capture_begin into that pool then asserts
# (CUDACachingAllocator "use_count > 0"). One graph at most: after the
# fallback no Foundry capture follows.
_discarded_graphs: list = []


def _graph_variant(shape_key) -> str | None:
    """The ShapeKey field that makes this shape one of several graphs, as
    ``field=value``, or None: ``attention_variant`` (DSV4.1 candidate-indexer
    graphs on SM100, the HIP DSA dual graph) or ``logits_variant`` (sharded
    verify's vocab-shard logits). The archive holds one graph per shape."""
    for field in ("attention_variant", "logits_variant"):
        value = getattr(shape_key, field, None)
        if value is not None:
            return f"{field}={value!r}"
    return None


def _save_capture(shape_key, variant, forward_fn, *, pool, stream):
    """SAVE: capture one shape, unless it is unsupported on any rank: a
    graph variant, or the capture created an attribute of a
    dispatcher or logits processor. Such an attribute lives in graph-pool
    memory that later graphs and eager forwards would share, where a native
    capture builds it in its eager warmup; so the graph is discarded, the
    attributes are reset and :class:`Unsupported` lets SGLang capture this
    shape and the rest natively (the graphs captured before stay valid)."""
    from sglang.srt.model_executor.runner_utils.pool import graph_pool_capture_scope

    from foundry.integration.sglang.graph_ops import (
        agree_unsupported,
        attribute_snapshot,
        capture_graph,
        created_attributes,
        reset_created_attributes,
    )

    graph = out = None
    created = []
    if variant is not None:
        reason = f"shape {shape_key} is a graph variant ({variant})"
    else:
        before = attribute_snapshot(_snapshot_objects)
        with graph_pool_capture_scope():
            graph, out = capture_graph(pool, stream, forward_fn)
        created = created_attributes(_snapshot_objects, before)
        reason = None
        if created:
            counts: dict[str, int] = {}
            for obj, name, _ in created:
                key = f"{type(obj).__name__}.{name}"
                counts[key] = counts.get(key, 0) + 1
            names = ", ".join(
                key if n == 1 else f"{key} ({n} objects)" for key, n in counts.items()
            )
            reason = f"{names} created during graph capture"
    try:
        agree_unsupported("capture", reason)
    except Unsupported:
        reset_created_attributes(created)
        if graph is not None:
            _discarded_graphs.append(graph)
        raise
    return graph, out


@contextlib.contextmanager
def _prefill_capture_scope(model, req_slots: int, flashinfer_tactics: str | None = None):
    """Around ``PrefillCudaGraphRunner.capture`` (its body, SGLang's capture
    loop, runs inside on both modes). ``model``: the loaded model (the
    first scope checks its prebuilt state); ``req_slots``: the runner's fixed request-
    slot count, recorded with the prefill graphs. The prefill backend is
    full: the resolution step validated it (full or disabled)."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        yield
        return
    from sglang.srt.runtime_context import get_flags

    from foundry.integration.sglang.graph_ops import (
        finish_prefill_graph_restore,
        save_prefill_graph_state,
        start_prefill_graph_restore,
    )

    _begin_graph_layout(model, mode, flashinfer_tactics)
    dp_flags = get_flags().dp
    if mode == CUDAGraphExtensionMode.LOAD:
        _preallocate_once()
        record = start_prefill_graph_restore()
        # Latched by the DP gather helpers inside SAVE's captured forwards,
        # which LOAD does not run; replay reads it
        # (prefill_graph_tolerates_sum_len) to pick the DP padding mode. Set
        # before the capture loop so its log line matches SAVE's.
        if record.get("prefill_graph_has_dp_gather"):
            dp_flags.prefill_graph_has_dp_gather = True
        rt.log_alloc_offset("before_prefill_restore")
        # LOAD runs the upstream capture loop itself (as for decode): its
        # eager work around each capture (dummy batches, the attention
        # metadata planned per shape, the chunked-prefix buffers, the capture
        # session's pool) then allocates exactly as on SAVE, and capture_one
        # restores each graph where SAVE captured it.
        t_loop = time.perf_counter()
        yield
        finish_prefill_graph_restore(time.perf_counter() - t_loop)
        rt.log_alloc_offset("after_prefill_restore")
        return

    rt.log_alloc_offset("save_before_prefill_capture")
    yield
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        return  # fell back inside the scope (capture_one); the archive is invalid
    rt.log_alloc_offset("save_after_prefill_capture")
    save_prefill_graph_state(
        req_slots=req_slots,
        has_dp_gather=bool(dp_flags.prefill_graph_has_dp_gather),
    )


@contextlib.contextmanager
def _decode_capture_scope(model, kind: str, flashinfer_tactics: str | None = None):
    """Around ``DecodeCudaGraphRunner.capture`` and its subclasses' (its body,
    SGLang's capture loop, runs inside on both modes). ``model``: the
    runner's loaded model (the first scope checks its prebuilt
    state); ``kind``: which decode-type graph set the runner captures
    (``target_decode``; with speculative decoding ``target_verify``,
    ``draft_decode``, ``draft_extend``). A process runs one scope per kind,
    in a fixed order (EAGLE: target_verify, draft_decode, draft_extend); each
    kind's graphs are saved and restored on their own, in capture order. The
    decode backend is full and elastic-EP recapture is off: the resolution
    steps pinned and validated them; attention graph variants are rejected
    per shape in :func:`capture_one`."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        yield
        return

    # No-op after the first scope (the prefill runner's or an earlier
    # decode-type runner's). Both EAGLE models are loaded by then.
    _begin_graph_layout(model, mode, flashinfer_tactics)

    if mode == CUDAGraphExtensionMode.LOAD:
        from foundry.integration.sglang.graph_ops import (
            finish_decode_graph_restore,
            has_prefill_graphs,
            load_decode_graph_state,
            start_decode_graph_restore,
        )

        state = rt.get_state()
        if has_prefill_graphs() and not state.prefill_graphs_restored:
            # SAVE's layout includes the prefill runner and its graphs;
            # without them every later address would be off.
            raise RuntimeError(
                "Foundry archive holds prefill graphs but this LOAD runs without them: "
                "pass the same --cuda-graph-backend-prefill as SAVE"
            )
        # The decode-type scopes SAVE ran, in order: this LOAD must run the
        # same ones in the same order (their allocations are one sequence).
        expected = load_decode_graph_state()["scopes"]
        index = state.decode_scopes_done
        if index >= len(expected) or expected[index] != kind:
            raise RuntimeError(
                f"Foundry archive was saved with the decode-type capture scopes {expected}; "
                f"this LOAD runs {kind!r} as scope {index}. SAVE and LOAD must use the same "
                "speculative decoding settings."
            )
        last = index == len(expected) - 1

        # Already done at the first scope.
        _preallocate_once()
        start_decode_graph_restore(kind)
        rt.log_alloc_offset(f"before_{kind}_restore")
        # The single substitution point: sglang's own capture loop runs
        # (kernel warmup, buffer seeding, the capture session's pool and
        # graph_pool_id, and per shape the dummy batch, TBO / LoRA prep, the
        # attention metadata incl. FlashInfer's per-bs wrappers and
        # workspaces, deepep_adapter.capture()), in SAVE's order, and
        # capture_one restores each graph in place of capturing it. Every
        # host-side object is thus re-created by sglang's code and lands at
        # SAVE's address through the deterministic allocation replay.
        t_loop = time.perf_counter()
        yield
        finish_decode_graph_restore(time.perf_counter() - t_loop)
        state.decode_scopes_done += 1
        rt.log_alloc_offset(f"after_load_{kind}_graphs")
        if not last:
            # The next decode-type scope continues the allocation sequence
            # with the caching allocator as SAVE left it: no empty_cache here.
            return
        rt.log_alloc_offset("after_load_all_graphs")
        return

    # SAVE: the same upstream loop; capture_one captures and saves each graph.
    yield
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        return  # fell back inside the scope (capture_one); the archive is invalid
    rt.log_alloc_offset(f"save_after_{kind}_capture")

    from foundry.integration.sglang.graph_ops import (
        pack_fatbins,
        save_decode_graph_state,
        save_graph_manifest,
    )

    # SAVE does not know whether a later decode-type scope follows (EAGLE's
    # draft runners), so the archive is completed after every scope: the
    # decode record, manifest (template = first graph of a partition, stable
    # as graphs are added), binary pack and region layout are rewritten from
    # scratch each time and the last scope's write is the one LOAD reads. None
    # of it allocates device memory.
    save_decode_graph_state(kind)
    save_graph_manifest()
    pack_fatbins()
    rt.record_region_layout()


@contextlib.contextmanager
def capture_scope(
    phase: str,
    model,
    req_slots: int | None = None,
    kind: str | None = None,
    flashinfer_tactics: str | None = None,
):
    """Prefill or decode capture scope: ``phase`` is
    ``"prefill"`` (with the runner's ``req_slots``, which every
    :func:`capture_one` inside uses) or ``"decode"`` (with the decode-type
    runner's ``kind``; None means ``target_decode``, the integration API 2.0
    contract). ``flashinfer_tactics`` (2.2): the runner's FlashInfer
    tuned-tactics cache file, or None (see
    :func:`graph_ops.bootstrap_flashinfer_tactics`)."""
    global _scope
    if phase == "prefill":
        if req_slots is None:
            raise ValueError("[Foundry] capture_scope('prefill') needs req_slots")
        kind = None
        inner = _prefill_capture_scope(model, req_slots, flashinfer_tactics)
    elif phase == "decode":
        from foundry.integration.sglang.graph_ops import check_decode_kind

        req_slots = None
        kind = check_decode_kind(kind)
        inner = _decode_capture_scope(model, kind, flashinfer_tactics)
    else:
        raise ValueError(f"[Foundry] capture_scope: unknown phase {phase!r}")
    if _scope is not None:
        raise RuntimeError(f"[Foundry] capture_scope({phase!r}) inside capture_scope{_scope}")
    if get_graph_extension_mode() == CUDAGraphExtensionMode.SAVE:
        from foundry.integration.sglang.graph_ops import capture_state_objects

        _snapshot_objects[:] = capture_state_objects(model)
    _scope = (phase, req_slots, kind)
    try:
        with inner:
            yield
    finally:
        _scope = None
        _snapshot_objects.clear()


def _check_resolved_graph_config(server_args) -> None:
    """Launcher, before any scheduler is spawned: the record must have been
    resolved with Foundry's graph pins (decode ``full``, prefill ``full`` or
    ``disabled``). They are declared by :func:`api.pin_server_args`; a record
    resolved without them would start without them and SAVE / LOAD would
    disagree with the archive."""
    if server_args is None:
        return
    try:
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.model_executor.cuda_graph_config import Backend
    except ImportError:
        return
    cfg = resolution_result(server_args, "cuda_graph_config")
    if cfg is None:
        return
    decode = cfg.decode.backend
    prefill = cfg.prefill.backend
    if decode != Backend.FULL or prefill not in (Backend.FULL, Backend.DISABLED):
        raise RuntimeError(
            "[Foundry] the resolved CUDA-graph config is "
            f"decode={decode!r} prefill={prefill!r}; Foundry needs decode 'full' and "
            "prefill 'full' or 'disabled'. The server args were resolved without "
            "Foundry's pins (--cuda-graph-persistence)."
        )
