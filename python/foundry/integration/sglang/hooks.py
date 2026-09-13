# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the Foundry SGLang integration.

Targets the runner architecture introduced after sglang 0.5.16: cuda-graph
capture lives in per-phase runners (DecodeCudaGraphRunner) that delegate the
actual graph create/capture/replay to a pluggable backend
(FullCudaGraphBackend). Foundry only supports the `full` decode backend;
the activation shim in sglang (foundry_shim.apply_server_args) forces
decode=full / prefill=disabled.

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
        result = orig(self, *args, **kwargs)
        rt.log_alloc_offset("after_init_torch_dist")
        rt.skip_to_scratch_boundary()
        rt.log_alloc_offset("after_scratch_skip")
        return result

    cls.init_torch_distributed = patched


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

    @functools.wraps(orig_resolve)
    def patched_resolve(self, pre_model_load_memory):
        if get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD:
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
        logger.info("[Foundry] SGLang reused saved memory pool config")
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
            state = rt.create_warmup_state(asdict(self.memory_pool_config))
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
    orig_capture_one = backend_cls.capture_one
    orig_capture = runner_cls.capture

    # When set, the capture machinery is being reused as a foundry-driven WARMUP
    # pass: run one real forward per shape (no warmup repeats, no graph capture,
    # no store) to trigger all of sglang's pre-capture lazy init — DeepEP
    # buffer, DeepGEMM per-shape JIT, etc. — that would otherwise fire inside
    # the captured stream and abort with "operation not permitted when stream is
    # capturing". See `_run_warmup_pass`.
    warmup_active = [False]

    @functools.wraps(orig_capture_one)
    def patched_capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        mode = get_graph_extension_mode()
        if warmup_active[0]:
            # Warmup pass: one real eager forward; nothing captured or stored.
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
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
        save_graph(graph, out, shape_key)

    def _warmup_pass_barrier():
        """File-based barrier before the SAVE warmup pass (no device allocations, so the SAVE/LOAD
        allocation sequences stay identical). The warmup forwards run the elastic-EP all-to-all;
        nodes that reach them tens of seconds apart trip the a2a fault timeout, which forced SAVE
        to use a long timeout. That timeout is a kernel argument baked into the captured graphs,
        so LOAD-restored ranks then hang for the long timeout on the first step after a real fault
        (~70 s to detect a fault with a 30 s SAVE timeout vs 2 s). With the barrier, SAVE can run
        with the serving timeout. FOUNDRY_WARMUP_BARRIER_DIR: directory shared by the participating
        processes; FOUNDRY_WARMUP_BARRIER_COUNT: number of ranks that run the warmup pass (default:
        torch world size)."""
        d = os.environ.get("FOUNDRY_WARMUP_BARRIER_DIR")
        if not d:
            return
        import torch.distributed as dist

        n = int(os.environ.get("FOUNDRY_WARMUP_BARRIER_COUNT", "0") or 0)
        if n <= 0 and dist.is_initialized():
            n = dist.get_world_size()
        if n <= 1:
            return
        rank = dist.get_rank() if dist.is_initialized() else os.getpid()
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"warmup_ready_{rank}"), "w").close()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 600:
            if len([f for f in os.listdir(d) if f.startswith("warmup_ready_")]) >= n:
                break
            time.sleep(0.2)
        logger.info(
            "[Foundry] warmup-pass barrier: %d ranks ready after %.1f s",
            n,
            time.perf_counter() - t0,
        )

    def _run_warmup_pass(self):
        """Foundry-driven pre-capture warmup for the DeepEP/EP path.

        Reuses the upstream capture loop with graph capture neutered (see
        ``warmup_active``) to run one real forward per ``capture_bs`` BEFORE the
        real capture — triggering every pre-capture lazy init sglang normally
        does in its (foundry-suppressed) warmup forwards: DeepEP dispatch
        combine paths, DeepGEMM per-shape JIT compile, etc. SAVE-only: LOAD
        replays recorded allocations at absolute offsets and must not re-enter
        graph_capture() (it breaks the threaded finish_graph_loads with
        "invalid device context")."""
        _warmup_pass_barrier()
        warmup_active[0] = True
        t0 = time.perf_counter()
        try:
            orig_capture(self)
        finally:
            warmup_active[0] = False
        logger.info(
            "[Foundry] SGLang EP warmup pass (lazy-init) completed in %.3fs",
            time.perf_counter() - t0,
        )

    @functools.wraps(orig_capture)
    def patched(self):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_capture(self)

        # DeepEP/EP only (gated so the dense / single-GPU / DP paths are
        # untouched). bootstrap_deepep_buffer runs on both modes (cheap
        # singleton) so the DeepEP buffer is created at the same
        # allocation-sequence point on SAVE and LOAD — created lazily inside
        # the warmup pass on SAVE but by bootstrap on LOAD, its VMM address
        # differed between modes and one-sided NVSHMEM traffic + restored-graph
        # references then corrupted whatever tensor later reused that VA
        # (docs/sglang/known-issues.md).
        if _ep_lazy_init_needed():
            from foundry.integration.sglang.graph_ops import bootstrap_deepep_buffer

            rt.log_alloc_offset("before_deepep_bootstrap")
            bootstrap_deepep_buffer(self)
            rt.log_alloc_offset("after_deepep_bootstrap")
            if mode == CUDAGraphExtensionMode.SAVE:
                _run_warmup_pass(self)
                rt.log_alloc_offset("after_warmup_pass")

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch

            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
                load_all_graphs,
            )

            state = rt.get_state()
            if state is None:
                raise RuntimeError("Foundry SGLang state is not initialized")

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

            rt.log_alloc_offset("before_preallocate")
            rt.preallocate_for_load_mode()
            rt.log_alloc_offset("after_preallocate")

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
        rt.capture_final_alloc_offset()
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
    runner_cls._resolve_shared_read_ends = patched_resolve_ends


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
