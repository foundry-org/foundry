# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime call-site functions of the Foundry SGLang integration, called
from SGLang's own call sites (dependency route, through ``api.py``), and the
plugin route's wrappers (``install``) that call them from patched SGLang
functions."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import logging
import os
import threading
import time
from dataclasses import asdict

from foundry.integration.sglang import runtime as rt
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
    get_workspace_root,
    load_graph_extension_config,
)

logger = logging.getLogger(__name__)
_INSTALLED = False


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


def install(cfg_path: str | None) -> None:
    """Install the runtime patches for the TOML at ``cfg_path`` (idempotent).
    Reached from the sglang plugin (``foundry.integration.sglang.plugin``) in
    every process that runs ``load_plugins()``."""
    global _INSTALLED
    if not cfg_path:
        return
    if _INSTALLED:
        return

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns:
        logger.info(
            "[Foundry] SGLang spawn -> install: %.1f ms",
            (time.perf_counter_ns() - int(t0_ns)) / 1e6,
        )

    load_graph_extension_config(cfg_path)
    logger.info(
        "[Foundry] SGLang hooks installing: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    from sglang.srt.distributed import bootstrap

    _patch_init_parallel_runtime(bootstrap)
    _patch_alloc_memory_pool()
    _patch_cuda_graph_capture()
    _patch_spawn_sites()

    _INSTALLED = True
    logger.info("[Foundry] SGLang hooks installed")


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


def _after_runner_distributed_init() -> None:
    """End of ModelRunner.init_torch_distributed: the groups
    exist and pre-model-load memory has been measured; everything allocated
    since the region was bound sits in the scratch space, and the cursor jumps
    to the scratch boundary so weight loading starts at the same offset on
    SAVE and LOAD."""
    rt.log_alloc_offset("after_init_torch_dist")
    rt.skip_to_scratch_boundary()
    rt.log_alloc_offset("after_scratch_skip")


# ---------------------------------------------------------------------------
# Call-site functions. The dependency route (api.py) calls them from SGLang's
# own call sites; the plugin route (install) wraps SGLang's functions with
# them. Both routes therefore run the same code at the same sequence points.
# ---------------------------------------------------------------------------


def _disable_dynamo() -> None:
    """Scheduler process: besides the ``TORCHDYNAMO_DISABLE=1`` pin (read when
    ``torch.compile`` decorates), set ``torch._dynamo.config.disable``, which
    dynamo checks on every frame, so a helper decorated before the pin took
    effect does not compile inside a capture either."""
    import torch._dynamo

    if not torch._dynamo.config.disable:
        torch._dynamo.config.disable = True
        logger.info("[Foundry] torch._dynamo.config.disable = True (TORCHDYNAMO_DISABLE=1 pin)")


def before_parallel_init(device: str) -> None:
    """Head of ``bootstrap.init_parallel_runtime``: bind this
    rank's VMM region before any communicator exists."""
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        return
    _disable_dynamo()
    from sglang.srt.runtime_context import get_device, get_parallel

    parallel = get_parallel()
    # get_parallel() answers from the published placement (resolved
    # widths and spawn ranks), unlike the raw record fields.
    _before_distributed_init(
        parallel,
        device,
        get_device().gpu_id,
        _workspace_ranks(parallel),
    )


def after_parallel_init() -> None:
    """End of ``bootstrap.init_parallel_runtime``."""
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        return
    rt.log_alloc_offset("after_init_parallel_runtime")


def after_runner_distributed_init(is_draft_worker: bool) -> None:
    """End of ``ModelRunner.init_torch_distributed``. Draft
    workers reuse the target's groups and never reach init_parallel_runtime;
    only the target runner closes the window."""
    if (
        get_graph_extension_mode() != CUDAGraphExtensionMode.NONE
        and not is_draft_worker
        and rt.get_state() is not None
    ):
        _after_runner_distributed_init()


# Context overrides the memory-pool resolver issued on SAVE
# (get_context().override, e.g. max_mamba_cache_size for hybrid
# Mamba/GDN/KDA models); the pool factories read them from the bags, so LOAD
# must replay them when it skips the resolver. Filled by
# record_memory_pool_overrides, persisted by after_alloc_memory_pool.
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


def before_alloc_memory_pool(is_draft_worker: bool) -> None:
    """Head of ``ModelRunner.alloc_memory_pool``. Draft-worker pools reuse the
    target's resolved config upstream and are left alone."""
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE or is_draft_worker:
        return
    rt.log_alloc_offset("before_init_memory_pool")


def after_alloc_memory_pool(memory_pool_config, is_draft_worker: bool) -> None:
    """End of ``ModelRunner.alloc_memory_pool``: SAVE records the resolved
    MemoryPoolConfig and the resolver's context overrides."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE or is_draft_worker:
        return
    rt.log_alloc_offset("after_init_memory_pool")
    if mode == CUDAGraphExtensionMode.SAVE:
        state = rt.create_warmup_state(
            asdict(memory_pool_config), context_overrides=_resolve_overrides
        )
        rt.save_warmup_state(state)


def _begin_graph_layout(model, mode) -> None:
    """Pre-capture bootstraps + layout start, once per process, at the first
    runner capture: the prefill runner's when prefill graphs are on (sglang
    captures prefill before decode), the decode runner's otherwise. Same
    sequence point on SAVE and LOAD."""
    state = rt.get_state()
    if state is None:
        raise RuntimeError("Foundry SGLang state is not initialized")
    if state.layout_started:
        return

    # Pre-capture bootstraps. Everything sglang initializes lazily on the
    # first eager forward that a capturing stream rejects, or that must exist
    # at the same addresses on both modes, is done here, at the same sequence
    # point on SAVE and LOAD; no model forward runs.
    from foundry.integration.sglang.graph_ops import (
        bootstrap_collective_connections,
        bootstrap_deepep_buffer,
        bootstrap_lazy_runtimes,
        bootstrap_logits_gatherer,
    )

    # 1. NCCL communicators (both modes): their first collective allocates
    #    and connects, which capture rejects.
    rt.log_alloc_offset("before_collective_bootstrap")
    bootstrap_collective_connections()
    rt.log_alloc_offset("after_collective_bootstrap")
    # 2. The logits all-gather's symmetric-memory state (both modes): built
    #    on a host with multicast by any eager forward before capture, and
    #    invisible to the hook (torch symmetric memory is not cudaMalloc).
    rt.log_alloc_offset("before_logits_gatherer")
    bootstrap_logits_gatherer(model)
    rt.log_alloc_offset("after_logits_gatherer")
    # 3. The DeepEP-family buffer (both modes; a no-op for other all-to-all
    #    backends): NVSHMEM runtime + symmetric heap, otherwise created inside
    #    the first captured forward, where deep_ep_cpp.Buffer(...) aborts.
    rt.log_alloc_offset("before_deepep_bootstrap")
    bootstrap_deepep_buffer(model)
    rt.log_alloc_offset("after_deepep_bootstrap")
    # 4. SAVE only, every model: the two one-time runtime initializations
    #    capture rejects (inductor's lazy init, DeepGEMM's runtime init), with
    #    the allocation region suspended so their transient tensors never move
    #    the deterministic cursor. The model's own compiles and JIT kernel
    #    loads then happen inside the captured forward.
    if mode == CUDAGraphExtensionMode.SAVE:
        with rt.allocation_region_suspended():
            bootstrap_lazy_runtimes()
        rt.log_alloc_offset("after_lazy_runtimes")
    # 5. Both modes, before the first eager extend: the FlashInfer SM90 GDN
    #    prefill kernels load from FlashInfer's .o cache instead of compiling
    #    8-10 s in every scheduler (host-side patch, see flashinfer_cute_cache).
    from foundry.integration.sglang import flashinfer_cute_cache

    if flashinfer_cute_cache.gdn_flashinfer_in_use():
        status = flashinfer_cute_cache.install()
        if status:
            logger.info("[Foundry] FlashInfer CuTe-DSL cache: %s", status)

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
    if getattr(shape_key, "attention_variant", None) is not None:
        # ShapeKey.attention_variant (e.g. DSV4.1 candidate-indexer graphs on
        # SM100, the HIP DSA dual graph): several graphs per shape, which the
        # archive cannot represent.
        raise RuntimeError(
            f"[Foundry] shape {shape_key} captures an attention graph variant "
            f"({shape_key.attention_variant!r}); Foundry save/load does not support "
            "graph variants"
        )
    if mode == CUDAGraphExtensionMode.LOAD:
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
    # (plugin.ENV_PINS), as it refuses to trace inside a capture.
    from sglang.srt.model_executor.runner_utils.pool import graph_pool_capture_scope

    from foundry.integration.sglang.graph_ops import capture_graph, save_graph

    with graph_pool_capture_scope():
        graph, out = capture_graph(pool, stream, forward_fn)
    if req_slots is not None:
        save_graph(graph, out, shape_key, prefill_req_slots=req_slots)
    else:
        save_graph(graph, out, shape_key, kind=kind)
    return graph, out


@contextlib.contextmanager
def _prefill_capture_scope(model, req_slots: int):
    """Around ``PrefillCudaGraphRunner.capture`` (its body, SGLang's capture
    loop, runs inside on both modes). ``model``: the loaded model (the
    bootstraps walk its modules); ``req_slots``: the runner's fixed request-
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

    _begin_graph_layout(model, mode)
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
    rt.log_alloc_offset("save_after_prefill_capture")
    save_prefill_graph_state(
        req_slots=req_slots,
        has_dp_gather=bool(dp_flags.prefill_graph_has_dp_gather),
    )


@contextlib.contextmanager
def _decode_capture_scope(model, kind: str):
    """Around ``DecodeCudaGraphRunner.capture`` and its subclasses' (its body,
    SGLang's capture loop, runs inside on both modes). ``model``: the
    runner's loaded model (the bootstraps of the first scope walk its
    modules); ``kind``: which decode-type graph set the runner captures
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
    _begin_graph_layout(model, mode)

    if mode == CUDAGraphExtensionMode.LOAD:
        import torch

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

        # Surrender torch-cached-but-free segments so later eager allocations
        # cannot reuse VA ranges that restored graphs may reference internally
        # (torch graph pools provide this protection on SAVE; LOAD rebuilds
        # graph memory at the driver level with no pool bookkeeping).
        torch.cuda.empty_cache()
        rt.log_alloc_offset("after_post_load_empty_cache")
        return

    # SAVE: the same upstream loop; capture_one captures and saves each graph.
    yield
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
def capture_scope(phase: str, model, req_slots: int | None = None, kind: str | None = None):
    """Prefill or decode capture scope (both routes): ``phase`` is
    ``"prefill"`` (with the runner's ``req_slots``, which every
    :func:`capture_one` inside uses) or ``"decode"`` (with the decode-type
    runner's ``kind``; None means ``target_decode``, the integration API 2.0
    contract)."""
    global _scope
    if phase == "prefill":
        if req_slots is None:
            raise ValueError("[Foundry] capture_scope('prefill') needs req_slots")
        kind = None
        inner = _prefill_capture_scope(model, req_slots)
    elif phase == "decode":
        from foundry.integration.sglang.graph_ops import check_decode_kind

        req_slots = None
        kind = check_decode_kind(kind)
        inner = _decode_capture_scope(model, kind)
    else:
        raise ValueError(f"[Foundry] capture_scope: unknown phase {phase!r}")
    if _scope is not None:
        raise RuntimeError(f"[Foundry] capture_scope({phase!r}) inside capture_scope{_scope}")
    _scope = (phase, req_slots, kind)
    try:
        with inner:
            yield
    finally:
        _scope = None


def shared_read_ends_override(runner, attn_backend, forward_mode):
    """LOAD-mode WAR barrier for ``DecodeCudaGraphRunner._resolve_shared_read_ends``.

    Restored graphs carry no usable in-graph shared-read marker
    (in_graph_metadata_prep_done stays None), and upstream's fallback for that
    case is PRE_REPLAY — which fences the scheduler's shared-buffer writes
    BEFORE the replay that still reads them (upstream's own TODO calls
    POST_REPLAY the sound one). Fence after replay instead. Returns None to
    keep upstream's answer. Called on every decode replay."""
    if (
        get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD
        or runner.in_graph_metadata_prep_done is not None
    ):
        return None
    from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds

    if attn_backend.shared_read_ends(forward_mode) is SharedReadEnds.IN_REPLAY:
        return SharedReadEnds.POST_REPLAY
    return None


# ---------------------------------------------------------------------------
# Plugin route: wrap SGLang's functions with the call-site functions above.
# ---------------------------------------------------------------------------


def _patch_init_parallel_runtime(bootstrap) -> None:
    from sglang.srt.model_executor import model_runner as mr

    orig = bootstrap.init_parallel_runtime

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig(*args, **kwargs)
        # Keyword-only upstream: (*, server_args, device, dist_port).
        from sglang.srt.runtime_context import get_device

        before_parallel_init(kwargs.get("device", get_device().device))
        result = orig(*args, **kwargs)
        after_parallel_init()
        return result

    # Scheduler.__init__ calls it as ``bootstrap.init_parallel_runtime``, so
    # the module attribute is the dispatch target.
    bootstrap.init_parallel_runtime = patched

    cls = mr.ModelRunner
    orig_runner_init = cls.init_torch_distributed

    @functools.wraps(orig_runner_init)
    def patched_runner_init(self, *args, **kwargs):
        result = orig_runner_init(self, *args, **kwargs)
        after_runner_distributed_init(self.is_draft_worker)
        return result

    cls.init_torch_distributed = patched_runner_init


def _patch_alloc_memory_pool() -> None:
    """Plugin route for begin/record_memory_pool_overrides and
    before/after_alloc_memory_pool."""
    from sglang.srt.mem_cache import kv_cache_configurator as kvc_mod
    from sglang.srt.model_executor import model_runner as mr

    orig_resolve = kvc_mod.KVCacheConfigurator._resolve_memory_pool_config

    @functools.wraps(orig_resolve)
    def patched_resolve(self, pre_model_load_memory):
        replayed = replay_saved_memory_pool_config()
        if replayed is not None:
            return replayed
        config = orig_resolve(self, pre_model_load_memory)
        record_memory_pool_overrides()
        return config

    kvc_mod.KVCacheConfigurator._resolve_memory_pool_config = patched_resolve

    cls = mr.ModelRunner
    orig_alloc = cls.alloc_memory_pool

    @functools.wraps(orig_alloc)
    def patched_alloc(self, *args, **kwargs):
        before_alloc_memory_pool(self.is_draft_worker)
        result = orig_alloc(self, *args, **kwargs)
        after_alloc_memory_pool(self.memory_pool_config, self.is_draft_worker)
        return result

    cls.alloc_memory_pool = patched_alloc


def _patch_cuda_graph_capture() -> None:
    from sglang.srt.model_executor.runner import (
        decode_cuda_graph_runner as dcgr,
    )
    from sglang.srt.model_executor.runner import (
        prefill_cuda_graph_runner as pcgr,
    )
    from sglang.srt.model_executor.runner_backend import (
        full_cuda_graph_backend as fcgb,
    )

    # Graph-pool borrow / precarve, the metadata glue graph and the
    # memory-saver graph context are pinned off by the plugin's environment
    # pins (plugin.ENV_PINS), applied before this process started.

    backend_cls = fcgb.FullCudaGraphBackend
    runner_cls = dcgr.DecodeCudaGraphRunner
    prefill_runner_cls = pcgr.PrefillCudaGraphRunner
    orig_capture_one = backend_cls.capture_one
    orig_capture = runner_cls.capture
    orig_prefill_capture = prefill_runner_cls.capture
    orig_resolve_ends = runner_cls._resolve_shared_read_ends

    @functools.wraps(orig_capture_one)
    def patched_capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig_capture_one(
                self,
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
        graph, out = capture_one(
            shape_key, forward_fn, pool=self._pool, stream=self._capture_stream
        )
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    @functools.wraps(orig_prefill_capture)
    def patched_prefill_capture(self):
        if not self._is_full_backend:
            raise RuntimeError(
                "[Foundry] prefill CUDA graphs are persisted for the full backend only, "
                f"got {self.prefill_backend_name!r}: use --cuda-graph-backend-prefill "
                "full or disabled"
            )
        with capture_scope("prefill", self.model_runner.model, self._capture_req_slots):
            return orig_prefill_capture(self)

    @functools.wraps(orig_capture)
    def patched(self):
        reject_unsupported_decode_runner(self)
        with capture_scope("decode", self.model_runner.model, kind=decode_runner_kind(self)):
            return orig_capture(self)

    @functools.wraps(orig_resolve_ends)
    def patched_resolve_ends(self, attn_backend, forward_mode):
        override = shared_read_ends_override(self, attn_backend, forward_mode)
        if override is not None:
            return override
        return orig_resolve_ends(self, attn_backend, forward_mode)

    backend_cls.capture_one = patched_capture_one
    runner_cls.capture = patched
    prefill_runner_cls.capture = patched_prefill_capture
    runner_cls._resolve_shared_read_ends = patched_resolve_ends

    if os.environ.get("FOUNDRY_SGLANG_CAPTURE_TRACE") == "1":
        # Diagnostic: log every shape the capture loop (SAVE and LOAD) asks
        # sglang for, and which one fails. sglang's per-shape metadata sizing runs inside
        # capture_one_shape before any foundry code for that shape, so a
        # failure there is state carried over from earlier shapes.
        orig_capture_one_shape = runner_cls.capture_one_shape

        @functools.wraps(orig_capture_one_shape)
        def traced_capture_one_shape(self, size, forward, *args, **kwargs):
            width = getattr(self, "captured_req_width", 1)
            logger.info(
                "[Foundry] capture shape size=%d num_tokens=%d ragged=%s max_bs=%s",
                size,
                size * width,
                getattr(self, "ragged_verify_mode", None),
                getattr(self, "max_bs", None),
            )
            t0 = time.perf_counter()
            try:
                result = orig_capture_one_shape(self, size, forward, *args, **kwargs)
            except Exception:
                logger.error("[Foundry] capture shape size=%d FAILED", size)
                raise
            logger.info(
                "[Foundry] capture shape size=%d done in %.1f ms",
                size,
                1000 * (time.perf_counter() - t0),
            )
            return result

        runner_cls.capture_one_shape = traced_capture_one_shape


def _patch_spawn_sites() -> None:
    try:
        from sglang.srt.entrypoints import engine as engine_mod
    except Exception:
        engine_mod = None

    if engine_mod is not None:
        engine_cls = engine_mod.Engine
        # Upstream calls it as ``cls._launch_scheduler_processes(...)`` and
        # subclasses override it (RayEngine): keep it a classmethod so a
        # subclass inheriting it still receives its own ``cls``.
        orig_launch = engine_cls.__dict__["_launch_scheduler_processes"].__func__

        @functools.wraps(orig_launch)
        def patched_launch(cls, *args, **kwargs):
            if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
                server_args = args[0] if args else kwargs.get("server_args")
                _check_resolved_graph_config(server_args)
                rt.setup_ld_preload_env(server_args)
            return orig_launch(cls, *args, **kwargs)

        engine_cls._launch_scheduler_processes = classmethod(patched_launch)

    try:
        from sglang.srt.managers import data_parallel_controller as dpc
    except Exception:
        dpc = None

    if dpc is not None:
        orig_start = dpc.DataParallelController.launch_tensor_parallel_group

        @functools.wraps(orig_start)
        def patched_start(self, *args, **kwargs):
            if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
                rt.setup_ld_preload_env()
            return orig_start(self, *args, **kwargs)

        dpc.DataParallelController.launch_tensor_parallel_group = patched_start


def _check_resolved_graph_config(server_args) -> None:
    """Launcher, before any scheduler is spawned: the record must have been
    resolved with Foundry's graph pins (decode ``full``, prefill ``full`` or
    ``disabled``). They are declared by the plugin's resolution hooks; a
    record resolved before the plugin loaded would otherwise
    start without them and SAVE / LOAD would disagree with the archive."""
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
            "Foundry's resolution hooks: load plugins (sglang.srt.plugins.load_plugins) "
            "before resolving them, or pass the Foundry config at launch."
        )


def decode_runner_kind(runner) -> str:
    """Plugin route: the decode-type scope kind of ``runner`` (a
    ``DecodeCudaGraphRunner`` or subclass), derived as SGLang's dependency
    route derives it at its call site (``DecodeCudaGraphRunner.capture``):
    ``<target|draft>_<verify|extend|forward mode>``."""
    role = "draft" if runner.model_runner.is_draft_worker else "target"
    mode = runner.capture_forward_mode
    if mode.is_target_verify():
        return f"{role}_verify"
    if mode.is_draft_extend_v2():
        return f"{role}_extend"
    return f"{role}_{mode.name.lower()}"


def reject_unsupported_decode_runner(runner) -> None:
    """Runtime guard at the first decode capture, both modes and layouts.

    Things Foundry's archive cannot represent and that cannot all be seen
    from the server args (they depend on the GPU or on the model config):
    - a decode backend other than ``FullCudaGraphBackend``;
    - attention graph variants (``ShapeKey.attention_variant``, e.g. DSV4.1
      candidate-indexer graphs on SM100, the HIP DSA dual graph);
    - elastic-EP CUDA-graph recapture (upstream #33723): a scale event drops
      and recaptures the decode graphs after startup, which LOAD cannot
      replay from the archive."""
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    backend = getattr(runner, "backend", None)
    if not isinstance(backend, FullCudaGraphBackend):
        raise RuntimeError(
            "[Foundry] decode CUDA graphs must use the full backend, got "
            f"{type(backend).__name__}; Foundry pins --cuda-graph-backend-decode full"
        )
    if getattr(runner, "attention_graph_variants", None) is not None:
        raise RuntimeError(
            "[Foundry] this model captures attention graph variants "
            f"({type(runner.attention_graph_variants).__name__}); Foundry save/load "
            "does not support graph variants"
        )
    model_runner = getattr(runner, "model_runner", None)
    elastic = getattr(model_runner, "_elastic_cuda_graph_enabled", None)
    if callable(elastic) and elastic():
        raise RuntimeError(
            "[Foundry] elastic-EP CUDA-graph recapture is enabled (elastic EP with "
            "--max-ep-size > tp size); Foundry cannot restore recaptured graphs"
        )
