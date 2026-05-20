# SPDX-License-Identifier: Apache-2.0
"""Runtime monkey-patch installer for the Foundry SGLang integration."""

from __future__ import annotations

import functools
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


def _resolve_dp_rank(model_runner) -> int | None:
    dp_rank = getattr(model_runner, "dp_rank", None)
    if dp_rank is not None:
        return dp_rank

    server_args = model_runner.server_args
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

        _, _, dp_rank = compute_dp_attention_world_info(
            server_args.enable_dp_attention,
            model_runner.tp_rank,
            server_args.tp_size,
            server_args.dp_size,
            server_args.attn_cp_size,
        )
        return dp_rank

    if getattr(server_args, "dp_size", 1) > 1:
        raise RuntimeError(
            "Foundry SGLang integration cannot derive regular DP rank because "
            "ModelRunner.dp_rank is absent. Preserve the constructor dp_rank on "
            "ModelRunner before initializing torch distributed."
        )

    return None


def install_hooks(server_args) -> None:
    global _INSTALLED
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
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
    _patch_init_memory_pool()
    _patch_load_model()
    _patch_kernel_warmup()
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

        rt.setup_graph_extension(
            self.server_args,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=_resolve_dp_rank(self),
        )
        rt.log_alloc_offset("after_setup_graph_ext")
        result = orig(self, *args, **kwargs)
        rt.log_alloc_offset("after_init_torch_dist")
        rt.skip_to_scratch_boundary()
        rt.log_alloc_offset("after_scratch_skip")
        return result

    cls.init_torch_distributed = patched


def _patch_init_memory_pool() -> None:
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as kv_mixin
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    cls = kv_mixin.ModelRunnerKVCacheMixin
    orig = cls.init_memory_pool

    @functools.wraps(orig)
    def patched(self, pre_model_load_memory):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, pre_model_load_memory)

        if mode == CUDAGraphExtensionMode.LOAD:
            import torch
            rt.log_alloc_offset("before_init_memory_pool")
            state = rt.load_warmup_state()
            if not state.memory_pool_config:
                raise RuntimeError("Foundry LOAD requires memory_pool_config")
            self.memory_pool_config = MemoryPoolConfig(**state.memory_pool_config)
            # Mirror SAVE's ``_resolve_memory_pool_config`` ->
            # ``get_available_gpu_memory(empty_cache=True)`` side
            # effect. Without this, torch's caching allocator retains
            # segments that SAVE released — causing the
            # attention-backend init below to take a different
            # cuMemAlloc path and drift the VMM cursor away from
            # SAVE's recorded ``start_base_addr``.
            torch.cuda.empty_cache()
            self._apply_memory_pool_config(self.memory_pool_config)
            rt.log_alloc_offset("after_init_memory_pool")
            logger.info("[Foundry] SGLang reused saved memory pool config")
            return None

        rt.log_alloc_offset("before_init_memory_pool")
        result = orig(self, pre_model_load_memory)
        rt.log_alloc_offset("after_init_memory_pool")
        state = rt.create_warmup_state(asdict(self.memory_pool_config))
        rt.save_warmup_state(state)
        return result

    cls.init_memory_pool = patched


def _patch_load_model() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.load_model

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        return orig(self, *args, **kwargs)

    cls.load_model = patched


def _patch_kernel_warmup() -> None:
    from sglang.srt.model_executor import model_runner as mr

    cls = mr.ModelRunner
    orig = cls.kernel_warmup

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)
        # Phase 1 keeps pre-graph model-forward warmups out of SAVE/LOAD.
        logger.info("[Foundry] SGLang kernel_warmup skipped in %s mode", mode.value)
        return None

    cls.kernel_warmup = patched


def _patch_cuda_graph_capture() -> None:
    from sglang.srt.model_executor import cuda_graph_runner as cgr

    cls = cgr.CudaGraphRunner
    orig_capture = cls.capture
    orig_create_device_graph = cls._create_device_graph
    orig_capture_graph = cls._capture_graph
    orig_capture_one_batch_size = cls.capture_one_batch_size

    @functools.wraps(orig_create_device_graph)
    def patched_create_device_graph(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import create_device_graph

            return create_device_graph()
        return orig_create_device_graph(self, *args, **kwargs)

    @functools.wraps(orig_capture_graph)
    def patched_capture_graph(self, graph, pool, stream, run_once_fn):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import capture_graph

            return capture_graph(graph, pool, stream, run_once_fn)
        return orig_capture_graph(self, graph, pool, stream, run_once_fn)

    @functools.wraps(orig_capture_one_batch_size)
    def patched_capture_one_batch_size(self, bs, forward, stream_idx=None):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.SAVE:
            # Suppress the two pre-capture warmup forwards
            # (cuda_graph_runner.py: ``for _ in range(2): run_once()``).
            # Their non-deterministic activation allocations would pollute
            # the torch caching allocator with freed segments that LOAD
            # cannot reproduce — causing the per-bs init's cache-miss vs
            # cache-hit asymmetry that drifts the VMM cursor away from
            # each saved ``start_base_addr``. JIT / autotune still happens
            # inside the graph capture (3rd run_once invocation) and is
            # recorded as alloc events, mirroring vLLM doc 04 §2.
            counter = [0]
            real_forward = forward

            def warmup_skipping_forward(*args, **kwargs):
                counter[0] += 1
                if counter[0] <= 2:
                    return None
                return real_forward(*args, **kwargs)

            forward = warmup_skipping_forward
        graph, output = orig_capture_one_batch_size(self, bs, forward, stream_idx)
        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import save_graph

            # Mirror the inline key shape upstream uses for self.graphs[key]
            # in `_capture_one_stream`. ``_make_graph_key`` and
            # ``get_capture_lora_variant`` were removed in sglang
            # commit ce2506e1c (record_nolora_graph deprecation).
            key = bs if stream_idx is None else f"{stream_idx}_{bs}"
            save_graph(graph, output, key)
        return graph, output

    @functools.wraps(orig_capture)
    def patched(self, *args, **kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.LOAD:
            from sglang.srt.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )
            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
                load_all_graphs,
            )

            state = rt.get_state()
            if state is None:
                raise RuntimeError("Foundry SGLang state is not initialized")
            # Set up the graph memory pool once — sglang shares one pool
            # across all captured graphs, and runtime replay also requires
            # it to be set so pynccl knows which pool the graph belongs to.
            if cgr.get_global_graph_memory_pool() is None:
                cgr.set_global_graph_memory_pool(self.device_module.graph_pool_handle())
            set_graph_pool_id(cgr.get_global_graph_memory_pool())
            rt.log_alloc_offset("before_preallocate")
            rt.preallocate_for_load_mode()
            rt.log_alloc_offset("after_preallocate")
            # Pre-pass: allocate every per-bs FlashInfer wrapper in
            # ``reversed(capture_bs)`` order, matching the order SAVE used.
            # SAVE's patched ``init_forward_metadata_capture_cuda_graph``
            # is idempotent so the inner per-iter call inside the SAVE
            # capture loop does not re-allocate. Same upfront allocation
            # sequence on both sides → cursor sits at SAVE's
            # ``start_base_addr_0`` when graph load begins.
            initialize_all_attention_metadata(self)
            rt.log_alloc_offset("after_pre_init")
            # Single ``start_graph_builds(all_paths)`` call so templates
            # and on-demand graphs link via ``shared_exec`` in the
            # manifest. ``finish_graph_loads`` replays alloc events
            # graph-by-graph in the same order SAVE captured them.
            load_all_graphs(self)
            rt.log_alloc_offset("after_load_all_graphs")
            self.graphs = {k: v[0] for k, v in state.loaded_graphs.items()}
            self.output_buffers = {k: v[1] for k, v in state.loaded_graphs.items()}
            return None

        if mode == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.sglang.graph_ops import (
                initialize_all_attention_metadata,
            )

            rt.log_alloc_offset("save_before_pre_init")
            # Pre-pass: allocate every per-bs FlashInfer wrapper up
            # front, in the same ``reversed(capture_bs)`` order LOAD
            # uses. Cursor advances by sum-of-int_workspace sizes.
            initialize_all_attention_metadata(self)
            rt.log_alloc_offset("save_after_pre_init")

            # Drop the pre-pass's last forward_metadata reference so
            # that bs's wrapper isn't kept alive by it — otherwise
            # popping the dict entry below leaves a refcount of 1 and
            # the inner init's reallocation can't reuse the segment.
            attn_backend = self.attn_backend
            attn_backend.forward_metadata = None

            real_init = attn_backend.init_forward_metadata_capture_cuda_graph

            def reuse_pre_pass_init(
                bs, num_tokens, req_pool_indices, seq_lens,
                encoder_lens, forward_mode, spec_info,
            ):
                # The pre-pass already allocated a wrapper for this
                # bs and stored it in
                # ``decode_cuda_graph_metadata`` /
                # ``prefill_cuda_graph_metadata``. Reuse it directly
                # — no second torch.empty for ``_int_workspace_buffer``.
                # Re-run the planner with the same buffer slices the
                # capture forward uses, then point
                # ``forward_metadata`` at the same wrappers. Same
                # plan call on LOAD via the symmetric pre-pass, so
                # the captured graph kernels reference VMM addresses
                # that LOAD's wrappers actually occupy.
                from sglang.srt.layers.attention.flashinfer_backend import (
                    DecodeMetadata,
                    PrefillMetadata,
                )

                if forward_mode.is_decode_or_idle():
                    wrappers = attn_backend.decode_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs, num_tokens, req_pool_indices, seq_lens,
                            encoder_lens, forward_mode, spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    attn_backend.indices_updater_decode.update(
                        req_pool_indices, seq_lens, seq_lens.cpu(),
                        seq_lens_sum, decode_wrappers=wrappers,
                        encoder_lens=encoder_lens, spec_info=spec_info,
                        fixed_split_size=None,
                        disable_split_kv=attn_backend.disable_cuda_graph_kv_split,
                    )
                    attn_backend.forward_metadata = DecodeMetadata(wrappers)
                    return
                if (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend()
                    or forward_mode.is_dllm_extend()
                ):
                    wrappers = attn_backend.prefill_cuda_graph_metadata.get(bs)
                    if wrappers is None:
                        return real_init(
                            bs, num_tokens, req_pool_indices, seq_lens,
                            encoder_lens, forward_mode, spec_info,
                        )
                    seq_lens_sum = seq_lens.sum().item()
                    use_ragged = forward_mode.is_dllm_extend()
                    prefix_lens = (
                        seq_lens - attn_backend.dllm_config.block_size
                        if forward_mode.is_dllm_extend()
                        else None
                    )
                    spec_info_arg = (
                        None if forward_mode.is_dllm_extend() else spec_info
                    )
                    attn_backend.indices_updater_prefill.update(
                        req_pool_indices, seq_lens, seq_lens.cpu(),
                        seq_lens_sum, prefix_lens=prefix_lens,
                        prefill_wrappers=wrappers, use_ragged=use_ragged,
                        encoder_lens=encoder_lens, spec_info=spec_info_arg,
                    )
                    attn_backend.forward_metadata = PrefillMetadata(
                        wrappers, use_ragged, False
                    )
                    return
                # Unknown mode — fall back to real init.
                return real_init(
                    bs, num_tokens, req_pool_indices, seq_lens,
                    encoder_lens, forward_mode, spec_info,
                )

            attn_backend.init_forward_metadata_capture_cuda_graph = reuse_pre_pass_init
            try:
                result = orig_capture(self, *args, **kwargs)
            finally:
                attn_backend.init_forward_metadata_capture_cuda_graph = real_init

            from foundry.integration.sglang.graph_ops import (
                pack_fatbins,
                save_graph_manifest,
            )

            save_graph_manifest()
            pack_fatbins()
            rt.capture_final_alloc_offset()
            return result

        return orig_capture(self, *args, **kwargs)

    cls._create_device_graph = patched_create_device_graph
    cls._capture_graph = patched_capture_graph
    cls.capture_one_batch_size = patched_capture_one_batch_size
    cls.capture = patched


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
