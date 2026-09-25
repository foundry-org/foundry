# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the Foundry SGLang integration.

Targets the runner architecture introduced after sglang 0.5.16: cuda-graph
capture lives in per-phase runners (DecodeCudaGraphRunner) that delegate the
actual graph create/capture/replay to a pluggable backend
(FullCudaGraphBackend). Foundry only supports the `full` backend: sglang's
handle_graph_extension forces decode=full and keeps prefill disabled unless
full is requested explicitly (PrefillCudaGraphRunner, captured before decode).

Kernel warmup no longer needs a patch: BaseRunner.warmup() runs no model
forwards (workspace prealloc + autotune, which foundry disables) and executes
at the same sequence point on SAVE and LOAD (EagerRunner.__init__), so its
allocations are symmetric by construction.
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import logging
import os
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


def _ep_lazy_init_needed() -> bool:
    """True when a DeepEP-family all-to-all backend (DeepEP, DeepEP v2,
    Mooncake EP) is active, so pre-capture lazy
    init (NVSHMEM buffer, DeepGEMM JIT) must be warmed up outside stream capture."""
    try:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        backend = get_moe_a2a_backend()
        return backend.is_deepep() or backend.is_deepep_v2() or backend.is_mooncake()
    except Exception:
        return False


def _resolve_dp_rank(model_runner) -> int | None:
    """Foundry workspace rank derivation. ParallelState carries both the
    regular dp_rank and the dp-attention rank, so no recomputation is needed."""
    ps = model_runner.ps
    if model_runner.server_args.enable_dp_attention:
        return ps.attn_dp_rank
    return ps.dp_rank


def install_hooks(server_args) -> None:
    global _INSTALLED
    cfg_path = server_args.foundry_graph_extension_config_path
    if not cfg_path:
        return
    if _INSTALLED:
        return

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns:
        logger.info(
            "[Foundry] SGLang spawn -> install_hooks: %.1f ms",
            (time.perf_counter_ns() - int(t0_ns)) / 1e6,
        )

    load_graph_extension_config(cfg_path)
    logger.info(
        "[Foundry] SGLang hooks installing: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    _patch_init_torch_distributed()
    _patch_alloc_memory_pool()
    _patch_cuda_graph_capture()
    _patch_spawn_sites()

    _INSTALLED = True
    logger.info("[Foundry] SGLang hooks installed")


def _patch_init_torch_distributed() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.init_torch_distributed

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)

        # Bind this rank's CUDA device BEFORE reserving the VMM region.
        # init_torch_distributed (orig) calls set_device(self.gpu_id)
        # internally, but foundry's set_allocation_region (inside
        # setup_graph_extension) reserves the region on the *current* device.
        # For DP rank > 0 the current device is still cuda:0 at this point, so
        # without setting it first the region lands on the wrong GPU and the
        # rank's later allocations fault with an async illegal memory access
        # (surfacing at the first Stream()/kernel). Mirrors model_runner's own
        # set_device(self.gpu_id). Single-GPU is unaffected (gpu_id == 0).
        if self.device == "cuda":
            import torch

            torch.get_device_module(self.device).set_device(self.gpu_id)

        rt.setup_graph_extension(
            self.server_args,
            tp_rank=self.ps.tp_rank,
            pp_rank=self.ps.pp_rank,
            dp_rank=_resolve_dp_rank(self),
        )
        rt.log_alloc_offset("after_setup_graph_ext")
        if mode == CUDAGraphExtensionMode.LOAD and _early_graph_builds_enabled():
            # Start rebuilding the CUDA graphs now, on foundry's background
            # thread, so template builds and member instantiation overlap
            # torch-distributed init, weight loading and the memory-pool
            # setup instead of sitting on the critical path at the capture
            # point. Only CUDA graph objects are created here (no torch
            # allocations: allocator replay and output-tensor reconstruction
            # happen in finish_graph_loads at the capture point, and NVSHMEM
            # module init still precedes it), so the deterministic layout is
            # unchanged. Opt-in (FOUNDRY_SGLANG_EARLY_GRAPH_BUILDS=1): see
            # _early_graph_builds_enabled for why it is off by default.
            from foundry.integration.sglang.graph_ops import start_graph_builds

            start_graph_builds()
        result = orig(self, *args, **kwargs)
        rt.log_alloc_offset("after_init_torch_dist")
        rt.skip_to_scratch_boundary()
        rt.log_alloc_offset("after_scratch_skip")
        return result

    cls.init_torch_distributed = patched


def _early_graph_builds_enabled() -> bool:
    # Off by default: it does not work for graphs with memcpy/memset nodes.
    # cuGraphAddMemcpyNode validates its addresses when the node is added, and
    # at setup the buffers the decode graphs copy into (the graph runner's
    # static inputs, the KV pool) do not exist yet: Qwen3.5-122B-FP8 EP8 fails
    # with "cuGraphAddMemcpyNode FAILED for node 1604 with error 1". The
    # graph build therefore has to follow the engine's allocations, i.e. the
    # capture point, and instantiation (the bulk of the restore) cannot be
    # overlapped with model initialization this way. Kept as an opt-in for
    # graph sets without such nodes.
    return os.environ.get("FOUNDRY_SGLANG_EARLY_GRAPH_BUILDS", "0") == "1"


def _patch_alloc_memory_pool() -> None:
    """SAVE records the resolved MemoryPoolConfig; LOAD short-circuits the
    memory profiling (`_resolve_memory_pool_config`) to return the saved
    config, so pool construction itself runs the SAME upstream code with the
    SAME sizes in both modes — identical VMM allocation trajectory.

    Post-capture KV sizing (SGLANG_ENABLE_POST_CAPTURE_KV_SIZING) is not
    supported: it re-sizes the pool from post-capture free memory, which LOAD
    cannot reproduce. It is off by default.
    """
    from sglang.srt.mem_cache import kv_cache_configurator as kvc_mod
    from sglang.srt.model_executor import model_runner as mr
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    orig_resolve = kvc_mod.KVCacheConfigurator._resolve_memory_pool_config
    # Context overrides the resolver issued on SAVE (get_context().override,
    # e.g. max_mamba_cache_size for hybrid Mamba/GDN/KDA models); the pool
    # factories read them from the bags, so LOAD must replay them when it skips
    # the resolver. Filled by patched_resolve, persisted by patched_alloc.
    resolve_overrides: list = []

    def _json_safe(value):
        if isinstance(value, (bool, int, float, str)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(_json_safe(v) for v in value)
        return False

    @functools.wraps(orig_resolve)
    def patched_resolve(self, pre_model_load_memory):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            from sglang.srt.runtime_context import get_context

            before = len(get_context().overrides_log())
            config = orig_resolve(self, pre_model_load_memory)
            resolve_overrides[:] = [
                [source, {k: v for k, v in fields.items() if _json_safe(v)}]
                for source, fields in get_context().overrides_log()[before:]
            ]
            return config
        if mode != CUDAGraphExtensionMode.LOAD:
            return orig_resolve(self, pre_model_load_memory)
        import torch

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
        config = MemoryPoolConfig(
            **{k: v for k, v in state.memory_pool_config.items() if k in valid}
        )
        if state.context_overrides:
            from sglang.srt.runtime_context import get_context

            for source, fields in state.context_overrides:
                get_context().override(f"foundry_replay:{source}", **fields)
        logger.info(
            "[Foundry] SGLang reused saved memory pool config (%d context overrides replayed)",
            len(state.context_overrides),
        )
        return config

    kvc_mod.KVCacheConfigurator._resolve_memory_pool_config = patched_resolve

    cls = mr.ModelRunner
    orig_alloc = cls.alloc_memory_pool

    @functools.wraps(orig_alloc)
    def patched_alloc(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_alloc(self, *args, **kwargs)

        if self.is_draft_worker:
            # Draft-worker pools reuse the target's resolved config upstream.
            return orig_alloc(self, *args, **kwargs)

        rt.log_alloc_offset("before_init_memory_pool")
        result = orig_alloc(self, *args, **kwargs)
        rt.log_alloc_offset("after_init_memory_pool")
        if mode == CUDAGraphExtensionMode.SAVE:
            state = rt.create_warmup_state(
                asdict(self.memory_pool_config), context_overrides=resolve_overrides
            )
            rt.save_warmup_state(state)
        return result

    cls.alloc_memory_pool = patched_alloc


def _patch_cuda_graph_capture() -> None:
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner import (
        decode_cuda_graph_runner as dcgr,
    )
    from sglang.srt.model_executor.runner import (
        prefill_cuda_graph_runner as pcgr,
    )
    from sglang.srt.model_executor.runner_backend import (
        full_cuda_graph_backend as fcgb,
    )
    from sglang.srt.model_executor.runner_utils.pool import (
        disable_graph_pool_borrow,
        get_or_create_global_graph_memory_pool,
        graph_pool_capture_scope,
    )

    # Foundry manages graph storage at the driver level; borrowing free pool
    # extents (SGLANG_ENABLE_GRAPH_POOL_BORROW, default off) would hand out VA
    # ranges that restored graphs reference. Keep it off in foundry modes.
    disable_graph_pool_borrow("foundry graph save/load manages graph storage")

    backend_cls = fcgb.FullCudaGraphBackend
    runner_cls = dcgr.DecodeCudaGraphRunner
    prefill_runner_cls = pcgr.PrefillCudaGraphRunner
    orig_capture_one = backend_cls.capture_one
    orig_capture = runner_cls.capture
    orig_prefill_capture = prefill_runner_cls.capture

    def begin_graph_layout(runner, mode) -> None:
        """Pre-capture bootstraps + layout start, once per process, at the
        first runner capture: the prefill runner's when prefill graphs are on
        (sglang captures prefill before decode), the decode runner's
        otherwise. Same sequence point on SAVE and LOAD."""
        state = rt.get_state()
        if state is None:
            raise RuntimeError("Foundry SGLang state is not initialized")
        if state.layout_started:
            return

        # Pre-capture bootstraps. Everything sglang initializes lazily on the
        # first eager forward that a capturing stream rejects, or that must
        # exist at the same addresses on both modes, is done here, at the same
        # sequence point on SAVE and LOAD; no model forward runs.
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
        bootstrap_logits_gatherer(runner)
        rt.log_alloc_offset("after_logits_gatherer")
        # 3. The DeepEP buffer (both modes, DeepEP-family backends): NVSHMEM
        #    runtime + symmetric heap, otherwise created inside the first
        #    captured forward, where deep_ep_cpp.Buffer(...) aborts.
        if _ep_lazy_init_needed():
            rt.log_alloc_offset("before_deepep_bootstrap")
            bootstrap_deepep_buffer(runner)
            rt.log_alloc_offset("after_deepep_bootstrap")
        # 4. SAVE only, every model: the two one-time runtime initializations
        #    capture rejects (inductor's lazy init, DeepGEMM's runtime init),
        #    with the allocation region suspended so their transient tensors
        #    never move the deterministic cursor. The model's own compiles and
        #    JIT kernel loads then happen inside the captured forward.
        if mode == CUDAGraphExtensionMode.SAVE:
            with rt.allocation_region_suspended():
                bootstrap_lazy_runtimes()
            rt.log_alloc_offset("after_lazy_runtimes")

        # The deterministic layout begins here on both modes: same sequence
        # point, same (empty) caching-allocator state; LOAD's
        # preallocate_for_load_mode maps and replays from this point.
        rt.mark_layout_start()
        state.layout_started = True

    def preallocate_once() -> None:
        """LOAD: map the recorded layout once, at the first runner capture
        that restores graphs (it spans both phases' graph memory)."""
        state = rt.get_state()
        if state.preallocated:
            return
        rt.log_alloc_offset("before_preallocate")
        rt.preallocate_for_load_mode()
        rt.log_alloc_offset("after_preallocate")
        state.preallocated = True

    def prefill_req_slots(backend) -> int | None:
        """The prefill runner's fixed request-slot count when ``backend`` is
        that runner's, None for the decode runner's."""
        runner = backend._cuda_graph_runner
        if isinstance(runner, prefill_runner_cls):
            return runner._capture_req_slots
        return None

    @functools.wraps(orig_capture_one)
    def patched_capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        mode = get_graph_extension_mode()
        req_slots = prefill_req_slots(self) if mode != CUDAGraphExtensionMode.NONE else None
        if mode == CUDAGraphExtensionMode.LOAD and req_slots is not None:
            # LOAD, prefill: the upstream capture loop runs (see
            # patched_prefill_capture) and only the capture is replaced, by the
            # archived graph for this shape; its allocator events replay here,
            # at the point SAVE captured it.
            from foundry.integration.sglang.graph_ops import restore_next_prefill_graph

            graph, out = restore_next_prefill_graph(shape_key, req_slots)
            self._graphs[shape_key] = graph
            self._outputs[shape_key] = out
            return
        if mode != CUDAGraphExtensionMode.SAVE:
            return orig_capture_one(
                self,
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )

        # SAVE: suppress upstream's two pre-capture warmup forwards. Their
        # non-deterministic activation allocations would pollute the torch
        # caching allocator with freed segments that LOAD cannot reproduce —
        # causing cache-miss vs cache-hit asymmetry that drifts the VMM cursor
        # away from each saved ``start_base_addr``. JIT / lazy init still
        # happens inside the captured forward and is recorded as alloc events.
        from foundry.integration.sglang.graph_ops import (
            capture_graph,
            create_device_graph,
            save_graph,
        )

        graph = create_device_graph()
        with graph_pool_capture_scope():
            out = capture_graph(graph, self._pool, self._capture_stream, forward_fn)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        save_graph(graph, out, shape_key, prefill_req_slots=req_slots)

    @functools.wraps(orig_prefill_capture)
    def patched_prefill_capture(self):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_prefill_capture(self)
        if not self._is_full_backend:
            raise RuntimeError(
                "[Foundry] prefill CUDA graphs are persisted for the full backend only, got "
                f"{self.prefill_backend_name!r}: use --cuda-graph-backend-prefill full or disabled"
            )
        from sglang.srt.runtime_context import get_flags

        from foundry.integration.sglang.graph_ops import (
            finish_prefill_graph_restore,
            save_prefill_graph_state,
            start_prefill_graph_restore,
        )

        begin_graph_layout(self, mode)
        dp_flags = get_flags().dp
        if mode == CUDAGraphExtensionMode.LOAD:
            preallocate_once()
            record = start_prefill_graph_restore()
            # Latched by the DP gather helpers inside SAVE's captured forwards,
            # which LOAD does not run; replay reads it
            # (prefill_graph_tolerates_sum_len) to pick the DP padding mode.
            # Set before capture() so its log line matches SAVE's.
            if record.get("prefill_graph_has_dp_gather"):
                dp_flags.prefill_graph_has_dp_gather = True
            rt.log_alloc_offset("before_prefill_restore")
            # Unlike decode, LOAD runs the upstream capture loop itself: its
            # eager work around each capture (dummy batches, the attention
            # metadata planned per shape, the chunked-prefix buffers, the
            # capture session's pool) then allocates exactly as on SAVE, and
            # patched_capture_one restores each graph where SAVE captured it.
            result = orig_prefill_capture(self)
            finish_prefill_graph_restore()
            rt.log_alloc_offset("after_prefill_restore")
            return result

        rt.log_alloc_offset("save_before_prefill_capture")
        result = orig_prefill_capture(self)
        rt.log_alloc_offset("save_after_prefill_capture")
        save_prefill_graph_state(
            req_slots=self._capture_req_slots,
            has_dp_gather=bool(dp_flags.prefill_graph_has_dp_gather),
        )
        return result

    @functools.wraps(orig_capture)
    def patched(self):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_capture(self)

        # No-op when the prefill runner captured first.
        begin_graph_layout(self, mode)

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch

            from foundry.integration.sglang.graph_ops import (
                has_prefill_graphs,
                initialize_all_attention_metadata,
                load_all_graphs,
            )

            state = rt.get_state()
            if has_prefill_graphs() and not state.prefill_graphs_restored:
                # SAVE's layout includes the prefill runner and its graphs;
                # without them every later address would be off.
                raise RuntimeError(
                    "Foundry archive holds prefill graphs but this LOAD runs without them: "
                    "pass the same --cuda-graph-backend-prefill as SAVE"
                )

            # Kernel warmup normally runs at the top of orig capture(); it is
            # a no-op here when EagerRunner already ran it, but keep the call
            # so the sequence point matches SAVE exactly.
            self.warmup()

            # Set up the shared graph memory pool once — upstream does this in
            # backend.capture_session, which LOAD never enters. Runtime replay
            # also requires set_graph_pool_id so pynccl knows the pool.
            backend = self.backend
            if backend._pool is None:
                backend._pool = get_or_create_global_graph_memory_pool(self.device_module)
            set_graph_pool_id(backend._pool)

            # Already done at the prefill capture when prefill graphs are on.
            preallocate_once()

            # Mirror capture()'s buffer seeding so the metadata pre-pass plans
            # against the same values on both modes.
            self.buffers.seq_lens.fill_(self.seq_len_fill_value)
            self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)

            # Pre-pass (FlashInfer only): allocate every per-bs wrapper in
            # ``reversed(capture_bs)`` order, matching the order SAVE used.
            # SAVE's reuse shim makes the in-capture allocation half reuse
            # these, so the cursor sits at SAVE's ``start_base_addr_0`` when
            # graph load begins. fa3 etc. allocate their cuda-graph metadata
            # once in init_cuda_graph_state and need no pre-pass.
            use_fi_prepass = hasattr(self.attn_backend, "indices_updater_decode")
            if use_fi_prepass:
                initialize_all_attention_metadata(self)
            rt.log_alloc_offset("after_pre_init")

            load_all_graphs(self)
            rt.log_alloc_offset("after_load_all_graphs")

            # Surrender torch-cached-but-free segments so later eager
            # allocations cannot reuse VA ranges that restored graphs may
            # reference internally (torch graph pools provide this protection
            # on SAVE; LOAD rebuilds graph memory at the driver level with no
            # pool bookkeeping).
            torch.cuda.empty_cache()
            rt.log_alloc_offset("after_post_load_empty_cache")

            # Hand the loaded graphs to the backend under the ShapeKeys the
            # runner will look up at replay.
            for bs in self.capture_bs:
                if bs not in state.loaded_graphs:
                    raise RuntimeError(
                        f"Foundry archive has no graph for capture bs={bs}; "
                        "re-save with the current cuda_graph_bs settings"
                    )
                key = self._make_graph_key(
                    self._capture_graph_size(bs=bs, num_tokens=bs * self.captured_req_width)
                )
                graph, output = state.loaded_graphs[bs]
                backend._graphs[key] = graph
                backend._outputs[key] = output

            # Non-FlashInfer backends (e.g. fa3) populate per-bs decode
            # metadata — looked up at replay — inside the capture loop, which
            # LOAD replaces. Run AFTER load_all_graphs: fa3's metadata are
            # lightweight views over the fixed init_cuda_graph_state
            # workspace, not graph memory, so the cursor is unaffected.
            if not use_fi_prepass:
                initialize_all_attention_metadata(self)

            # Upstream sets the DeepEP adapter's captured mode in
            # deepep_adapter.capture() during the capture loop, which LOAD
            # replaces — without this, replay asserts on the first decode.
            self.deepep_adapter.capture(is_extend_in_batch=False)
            return None

        # SAVE
        attn_backend = self.attn_backend
        use_fi_prepass = hasattr(attn_backend, "indices_updater_decode")
        real_prepare = getattr(attn_backend, "_prepare_cuda_graph_metadata", None)

        def reuse_pre_pass_prepare(bs, num_tokens, forward_mode, spec_info):
            # The pre-pass already allocated the wrappers for this bs and
            # stored them in ``decode_cuda_graph_metadata`` /
            # ``prefill_cuda_graph_metadata`` — reuse them instead of
            # re-allocating (no second torch.empty for
            # ``_int_workspace_buffer``), keeping the VMM cursor deterministic
            # vs LOAD. The planner half (init_forward_metadata_out_graph) then
            # runs upstream against the reused wrappers.
            from sglang.srt.layers.attention.flashinfer_backend import (
                DecodeMetadata,
                PrefillMetadata,
            )

            if forward_mode.is_decode_or_idle():
                wrappers = attn_backend.decode_cuda_graph_metadata.get(bs)
                if wrappers is not None:
                    attn_backend.forward_metadata = DecodeMetadata(wrappers)
                    return
            elif (
                forward_mode.is_target_verify()
                or forward_mode.is_draft_extend()
                or forward_mode.is_dllm_extend()
            ):
                wrappers = attn_backend.prefill_cuda_graph_metadata.get(bs)
                if wrappers is not None:
                    attn_backend.forward_metadata = PrefillMetadata(
                        wrappers, forward_mode.is_dllm_extend(), False
                    )
                    return
            return real_prepare(bs, num_tokens, forward_mode, spec_info)

        if use_fi_prepass:
            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
            )

            # Mirror capture()'s buffer seeding (it happens inside
            # orig_capture, after this pre-pass would otherwise run).
            self.buffers.seq_lens.fill_(self.seq_len_fill_value)
            self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            rt.log_alloc_offset("save_before_pre_init")
            initialize_all_attention_metadata(self)
            rt.log_alloc_offset("save_after_pre_init")
            # Drop the pre-pass's last forward_metadata ref so popping the
            # dict entry doesn't keep the wrapper alive at refcount 1.
            attn_backend.forward_metadata = None
            attn_backend._prepare_cuda_graph_metadata = reuse_pre_pass_prepare
        try:
            result = orig_capture(self)
        finally:
            if use_fi_prepass:
                attn_backend._prepare_cuda_graph_metadata = real_prepare

        from foundry.integration.sglang.graph_ops import (
            pack_fatbins,
            save_graph_manifest,
        )

        save_graph_manifest()
        pack_fatbins()
        rt.record_region_layout()
        return result

    # LOAD-mode WAR barrier: restored graphs carry no usable in-graph
    # shared-read marker (in_graph_metadata_prep_done stays None), and
    # upstream's fallback for that case is PRE_REPLAY — which fences the
    # scheduler's shared-buffer writes BEFORE the replay that still reads
    # them (upstream's own TODO calls POST_REPLAY the sound one). Fence
    # after replay instead.
    from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds

    orig_resolve_ends = runner_cls._resolve_shared_read_ends

    @functools.wraps(orig_resolve_ends)
    def patched_resolve_ends(self, attn_backend, forward_mode):
        if (
            get_graph_extension_mode() == CUDAGraphExtensionMode.LOAD
            and self.in_graph_metadata_prep_done is None
            and attn_backend.shared_read_ends(forward_mode) is SharedReadEnds.IN_REPLAY
        ):
            return SharedReadEnds.POST_REPLAY
        return orig_resolve_ends(self, attn_backend, forward_mode)

    backend_cls.capture_one = patched_capture_one
    runner_cls.capture = patched
    prefill_runner_cls.capture = patched_prefill_capture
    runner_cls._resolve_shared_read_ends = patched_resolve_ends

    if os.environ.get("FOUNDRY_SGLANG_CAPTURE_TRACE") == "1":
        # Diagnostic: log every shape the SAVE capture loop asks sglang for,
        # and which one fails. sglang's per-shape metadata sizing runs inside
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
            try:
                return orig_capture_one_shape(self, size, forward, *args, **kwargs)
            except Exception:
                logger.error("[Foundry] capture shape size=%d FAILED", size)
                raise

        runner_cls.capture_one_shape = traced_capture_one_shape


def _patch_spawn_sites() -> None:
    try:
        from sglang.srt.entrypoints import engine as engine_mod
    except Exception:
        engine_mod = None

    if engine_mod is not None:
        orig_launch = engine_mod.Engine._launch_scheduler_processes

        @functools.wraps(orig_launch)
        def patched_launch(self, *args, **kwargs):
            if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
                rt.setup_ld_preload_env()
            return orig_launch(self, *args, **kwargs)

        engine_mod.Engine._launch_scheduler_processes = patched_launch

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
