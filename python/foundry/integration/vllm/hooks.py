# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the foundry vLLM integration.

Pattern: wrap orig + insert foundry calls. For functions that must be
skipped on SAVE/LOAD (non-deterministic allocations), the patch replaces
the callee with a no-op dummy. Only ``CUDAGraphWrapper.__call__`` is
re-derived because it needs an inner block swap.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from foundry.integration.vllm import runtime as rt
from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
    get_workspace_root,
    load_graph_extension_config,
)

if TYPE_CHECKING:
    from vllm.config import CompilationConfig

log = init_logger("vllm.foundry.hooks")

_INSTALLED = False


def _quarantine_get_device_capability() -> None:
    """Serve ``torch.cuda.get_device_capability`` from NVML so flashinfer's
    module-level init can't trigger ``torch.cuda._lazy_init`` in the parent.

    If the parent ever initializes CUDA, vLLM's ``_maybe_force_spawn``
    overrides our forkserver choice to spawn — killing the fork-based
    fast-start path entirely. NVML reads the same (major, minor) tuple
    without creating a CUDA runtime context, so the parent stays
    fork-safe.
    """
    import torch

    try:
        import pynvml
    except ImportError:
        log.info(
            "[foundry] pynvml not available; cannot quarantine "
            "get_device_capability — parent may still init CUDA"
        )
        return
    orig = torch.cuda.get_device_capability
    _nvml_started = {"v": False}

    def _resolve_index(device) -> int:
        if device is None:
            return 0
        if isinstance(device, int):
            return device
        if hasattr(device, "index"):
            return device.index or 0
        return int(device)

    def patched(device=None):
        try:
            if not _nvml_started["v"]:
                pynvml.nvmlInit()
                _nvml_started["v"] = True
            idx = _resolve_index(device)
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            return pynvml.nvmlDeviceGetCudaComputeCapability(h)
        except Exception:
            return orig(device)

    torch.cuda.get_device_capability = patched
    log.info("[foundry] torch.cuda.get_device_capability → NVML (keeps parent fork-safe)")


def install_hooks(compilation_config: CompilationConfig) -> None:
    """Idempotent. Loads the TOML and installs all patches."""
    global _INSTALLED
    if _INSTALLED:
        return
    cfg_path = compilation_config.graph_extension_config_path
    if not cfg_path:
        return

    # Must run BEFORE anything that might import flashinfer. Cheap + safe to
    # run in every process: when called in the engine child (after fork),
    # CUDA init is already OK to happen, the patch just keeps returning the
    # correct capability without paying for it.
    _quarantine_get_device_capability()

    # Measure subprocess spawn cost (parent stamped FOUNDRY_SPAWN_T0_NS right
    # before fork; we are at the first foundry code that runs in the child).
    # This window is Python + torch + vllm cold import with the LD_PRELOAD'd
    # hook, plus vLLM's pre-install_hooks scaffolding — the biggest single
    # chunk of LOAD time on remote servers.
    t0_ns_str = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns_str:
        spawn_elapsed_ms = (time.perf_counter_ns() - int(t0_ns_str)) / 1e6
        log.info(
            "[foundry TIMING] subprocess spawn → install_hooks: %.1f ms "
            "(cold Python + torch + vllm import with LD_PRELOAD'd hook)",
            spawn_elapsed_ms,
        )

    load_graph_extension_config(cfg_path)
    log.info(
        "[foundry] install_hooks: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
        # NOTE(liuxs): vllm doesn't allow zero cudagraph_num_of_warmups, hardcode here
        compilation_config.cudagraph_num_of_warmups = 0
        log.info(
            "[foundry] compilation_config.cudagraph_num_of_warmups=0 "
            "(avoid asymmetric warmup forward through wrapper-fallthrough)"
        )
        # NOTE: Foundry only patches the V1 model runner
        # (``vllm.v1.worker.gpu_model_runner.GPUModelRunner``). vLLM's
        # ``VllmConfig.use_v2_model_runner`` property silently picks the
        # V2 runner for certain architectures (currently
        # ``Qwen3ForCausalLM`` per ``DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES``),
        # which our patches don't touch — so capture_model runs unhooked
        # and SAVE silently produces an empty archive. The serve scripts
        # in ``recipe/vllm/*.sh`` export ``VLLM_USE_V2_MODEL_RUNNER=0``
        # in their ``--save`` / ``--load`` branches to pin V1.

    _patch_init_worker_distributed_environment()
    _patch_kernel_warmup()
    _patch_dummy_runs()
    _patch_load_model()
    _patch_prepare_comm_buffer()
    _patch_capture_model()
    _patch_cuda_graph_wrapper_call()
    _patch_initialize_kv_caches()
    _patch_subprocess_spawn_sites()
    _patch_deepep()

    _INSTALLED = True
    log.info("[foundry] all patches installed")


# ---------------------------------------------------------------------------
# VMM region setup around NCCL.
# ---------------------------------------------------------------------------


def _patch_init_worker_distributed_environment() -> None:
    from vllm.v1.worker import gpu_worker as gw

    orig = gw.init_worker_distributed_environment

    @functools.wraps(orig)
    def patched(vllm_config, rank, distributed_init_method=None, local_rank=-1, backend="nccl"):
        active = get_graph_extension_mode() != CUDAGraphExtensionMode.NONE
        t0 = time.perf_counter()
        if active:
            rt.setup_graph_extension(vllm_config.parallel_config)
        result = orig(vllm_config, rank, distributed_init_method, local_rank, backend)
        if active:
            rt.skip_to_scratch_boundary()
            log.info(
                "[foundry TIMING] init_worker_distributed_environment total: "
                "%.3f s (setup_graph_extension + NCCL + skip_to_scratch)",
                time.perf_counter() - t0,
            )
        return result

    gw.init_worker_distributed_environment = patched


# ---------------------------------------------------------------------------
# Skip-with-dummy patches for non-deterministic functions.
# ---------------------------------------------------------------------------


def _patch_kernel_warmup() -> None:
    import vllm.v1.worker.gpu_worker as gw

    orig = gw.kernel_warmup

    @functools.wraps(orig)
    def patched(worker):
        if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
            return
        return orig(worker)

    gw.kernel_warmup = patched


def _patch_dummy_runs() -> None:
    """Skip ONLY the V1 sampler-warmup full forward inside
    ``Worker.compile_or_warm_up_model`` (gpu_worker.py:691). The sampler
    call right after (line 699) still runs on the zero hidden states so
    its workspaces get allocated before the first real inference. By the
    time it runs, ``_patch_capture_model`` has already called
    ``fops.stop_allocation_region()`` — so sampler workspaces land in
    the standard CUDA caching allocator, not the foundry VMM region.
    They don't need to be deterministic across SAVE/LOAD; only the
    captured-graph memory does.
    """
    from vllm.v1.worker import gpu_model_runner as gmr

    cls = gmr.GPUModelRunner
    orig_dummy = cls._dummy_run

    @functools.wraps(orig_dummy)
    def patched_dummy(self, *args, **kwargs):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig_dummy(self, *args, **kwargs)
        # Targeted skip for ONE call site: the V1 sampler-warmup full
        # forward at Worker.compile_or_warm_up_model — gpu_worker.py:691.
        # That call's unique signature is cudagraph_runtime_mode=NONE +
        # skip_eplb=True with no other "real-forward" flag set.
        #
        # Everything else passes through:
        # - capture (is_graph_capturing=True)
        # - capture warmup (force_attention=True for FULL; also dead since
        #   we force cudagraph_num_of_warmups=0)
        # - profile_run (is_profile=True)
        # - runtime DP empty-shard phantom from Worker.execute_dummy_batch
        #   (uniform_decode=True) — MUST run real forward so it joins the
        #   peer shard's coordinate_batch_across_dp all-reduce + EP all2all
        # - compile-warmup loop (no cudagraph_runtime_mode set; empty
        #   anyway under our cudagraph_capture_sizes config)
        from vllm.config import CUDAGraphMode

        if (
            kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.NONE
            and kwargs.get("skip_eplb") is True
            and not kwargs.get("is_profile")
            and not kwargs.get("is_graph_capturing")
            and not kwargs.get("force_attention")
            and not kwargs.get("uniform_decode")
        ):
            n = args[0] if args else kwargs.get("num_tokens", 1)
            d = self.model_config.get_hidden_size()
            hs = torch.zeros(n, d, device=self.device, dtype=self.model_config.dtype)
            return hs, hs
        return orig_dummy(self, *args, **kwargs)

    cls._dummy_run = patched_dummy


# ---------------------------------------------------------------------------
# GPUModelRunner.load_model: route model_loader.load_model through the
# foundry overlap helper. Uses a closure over the un-patched callable to
# avoid recursion. Lets upstream run untouched so it still wraps self.model
# with CUDAGraphWrapper / UBatchWrapper.
# ---------------------------------------------------------------------------


def _load_model_with_overlap(runner, do_original_load: Callable, model_loader) -> torch.nn.Module:
    """Unified preallocate → weight-load → start_graph_builds path for
    SAVE/LOAD across dense, unquantized MoE, and quantized MoE models.
    """
    # NOTE(liuxs): no split-load. Previous design pre-called
    # prepare_comm_buffer before preallocate based on a wrong assumption
    # that the two would collide on memory regions — they don't (NVSHMEM's
    # symmetric heap takes the large-alloc branch of cuMemAddressReserve
    # at hook.cpp:2523 and lands outside the foundry preallocated region).
    # Newer vLLM also uses modular kernels for quantized MoE which require
    # weights to be loaded before prepare_comm_buffer, making the
    # split-load path impossible to maintain anyway. All MoE variants now
    # share this unified dense path.
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        return do_original_load()

    if mode == CUDAGraphExtensionMode.LOAD:
        rt.preallocate_for_load_mode()
    result = do_original_load()
    if mode == CUDAGraphExtensionMode.LOAD:
        # Start graph template builds AFTER weight load completes.
        # Running them in parallel with weight load slowed weight load
        # down (driver contention) without recovering meaningful time
        # on the template-build side.
        from foundry.integration.vllm.graph_ops import start_graph_builds

        start_graph_builds()
    return result


def _patch_load_model() -> None:
    from vllm.v1.worker import gpu_model_runner as gmr

    cls = gmr.GPUModelRunner
    orig = cls.load_model

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)

        t0 = time.perf_counter()
        runner = self
        orig_get_loader = gmr.get_model_loader

        def patched_get_loader(load_config):
            loader = orig_get_loader(load_config)
            orig_loader_load = loader.load_model

            def do_original_load():
                return orig_loader_load(
                    vllm_config=runner.vllm_config, model_config=runner.model_config
                )

            def patched_loader_load(*_args, **_kwargs):
                return _load_model_with_overlap(runner, do_original_load, loader)

            loader.load_model = patched_loader_load
            return loader

        gmr.get_model_loader = patched_get_loader
        try:
            result = orig(self, *args, **kwargs)
        finally:
            gmr.get_model_loader = orig_get_loader

        log.info(
            "[foundry TIMING] GPUModelRunner.load_model total: %.3f s "
            "(preallocate + weight load + process_weights_after_loading "
            "+ start_graph_builds)",
            time.perf_counter() - t0,
        )
        return result

    cls.load_model = patched


# ---------------------------------------------------------------------------
# prepare_communication_buffer_for_model post-orig hook: flush pending
# NVSHMEM init for foundry-archived modules. DeepEP Buffer creation inside
# the orig bootstraps the NVSHMEM runtime, after which nvshmemx_cumodule_init
# can succeed for modules that were loaded earlier by
# fops.load_cuda_modules_and_libraries in setup_graph_extension.
# ---------------------------------------------------------------------------


def _patch_prepare_comm_buffer() -> None:
    from vllm.v1.worker import gpu_model_runner as gmr

    orig = gmr.prepare_communication_buffer_for_model

    @functools.wraps(orig)
    def patched(model):
        result = orig(model)
        # NOTE(liuxs): init_nvshmem must run AFTER prepare_comm_buffer (which
        # bootstraps the NVSHMEM runtime via DeepEP Buffer creation). Called
        # earlier, nvshmemx_cumodule_init silently returns 0 and replayed
        # graphs later fail with cuLaunchKernel errors.
        if get_graph_extension_mode() == CUDAGraphExtensionMode.LOAD:
            from foundry import ops as fops

            t = time.perf_counter()
            n = fops.init_nvshmem_for_loaded_modules()
            log.info(
                "[foundry TIMING] init_nvshmem_for_loaded_modules: %.3f s (%d modules)",
                time.perf_counter() - t,
                n,
            )
        return result

    gmr.prepare_communication_buffer_for_model = patched


# ---------------------------------------------------------------------------
# GPUModelRunner.capture_model:
#   pre-orig:  prepare_graph_capture (cuBLAS + attention workspaces land at
#              deterministic offsets)
#   orig:      runs on both SAVE and LOAD. On LOAD each CUDAGraphWrapper.__call__
#              dispatches to capture_or_load_graph → finish_one_graph_load.
#   post-orig: SAVE → save_graph_manifest + pack_fatbins + capture_final_alloc_offset
#              LOAD → unlock_workspace (allow inference growth)
#              both → stop_allocation_region (post-capture allocs use the
#                     standard CUDA allocator; only captured-graph memory
#                     needs the deterministic VMM region)
# ---------------------------------------------------------------------------


def _patch_capture_model() -> None:
    from vllm.v1.worker import gpu_model_runner as gmr

    cls = gmr.GPUModelRunner
    orig = cls.capture_model

    @functools.wraps(orig)
    def patched(self):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self)

        from foundry.integration.vllm.graph_ops import (
            pack_fatbins,
            save_graph_manifest,
        )

        builders = [
            b for ag_list in self.attn_groups for ag in ag_list for b in ag.metadata_builders
        ]
        rt.prepare_graph_capture(metadata_builders=builders, device=self.device)

        result = orig(self)

        if mode == CUDAGraphExtensionMode.SAVE:
            save_graph_manifest()
            pack_fatbins()
        elif mode == CUDAGraphExtensionMode.LOAD:
            # NOTE(liuxs): see if there is better way to handle it
            from vllm.v1.worker.workspace import unlock_workspace

            unlock_workspace()
            log.info("[foundry] unlocked workspace (allow growth at inference)")

        # Only memory referenced by captured CUDA graphs needs the
        # deterministic VMM region. Past this point (sampler warmup +
        # inference allocations) the VMM overhead just slows things
        # down, so switch back to the regular CUDA caching allocator.
        from foundry import ops as fops

        fops.stop_allocation_region()
        log.info("[foundry] stopped allocation region after capture_model")

        # Record the deterministic watermark on SAVE. We do this AFTER
        # stop_allocation_region so the value reflects only the VMM cursor
        # at end of capture — sampler warmup and any later allocations no
        # longer advance it. Was previously a post-hook on
        # compile_or_warm_up_model; same result, one fewer monkey-patch.
        if mode == CUDAGraphExtensionMode.SAVE:
            rt.capture_final_alloc_offset()

        return result

    cls.capture_model = patched


# ---------------------------------------------------------------------------
# CUDAGraphWrapper.__call__: re-derived to swap capture stanza with
# capture_or_load_graph. Only place re-derivation is necessary.
# ---------------------------------------------------------------------------


def _patch_cuda_graph_wrapper_call() -> None:
    from vllm.compilation import cuda_graph as cg_mod

    cls = cg_mod.CUDAGraphWrapper
    orig = cls.__call__

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig(self, *args, **kwargs)

        from contextlib import ExitStack
        from unittest.mock import patch as mock_patch

        from vllm.compilation.counter import compilation_counter
        from vllm.compilation.cuda_graph import CUDAGraphEntry
        from vllm.compilation.monitor import (
            validate_cudagraph_capturing_enabled,
        )
        from vllm.config import CUDAGraphMode
        from vllm.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import weak_ref_tensors

        from foundry.integration.vllm.graph_ops import capture_or_load_graph

        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)
        fc = get_forward_context()
        bd = fc.batch_descriptor
        rm = fc.cudagraph_runtime_mode
        if rm == CUDAGraphMode.NONE or rm != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        assert bd is not None
        if bd not in self.concrete_cudagraph_entries:
            self.concrete_cudagraph_entries[bd] = CUDAGraphEntry(batch_descriptor=bd)
        entry = self.concrete_cudagraph_entries[bd]

        if entry.cudagraph is None:
            if get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD:
                validate_cudagraph_capturing_enabled()
            entry.input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    stack.enter_context(mock_patch("gc.collect", lambda: None))
                    stack.enter_context(mock_patch("torch.accelerator.empty_cache", lambda: None))
                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())
                graph, output = capture_or_load_graph(
                    batch_descriptor=bd,
                    runnable=self.runnable,
                    weak_ref_output=self.cudagraph_options.weak_ref_output,
                    runtime_mode=self.runtime_mode,
                    graph_pool=self.graph_pool,
                    runnable_args=args,
                    runnable_kwargs=kwargs,
                )
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = graph
            compilation_counter.num_cudagraph_captured += 1
            return output

        entry.cudagraph.replay()
        return entry.output

    cls.__call__ = patched


# ---------------------------------------------------------------------------
# EngineCore._initialize_kv_caches: skip profile on SAVE-pass-2 / LOAD,
# write/update warmup_state on SAVE.
# ---------------------------------------------------------------------------


def _patch_initialize_kv_caches() -> None:
    import os

    from vllm.v1.engine import core as core_mod

    cls = core_mod.EngineCore
    orig = cls._initialize_kv_caches

    @functools.wraps(orig)
    def patched(self, vllm_config):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig(self, vllm_config)

        import time

        import vllm.envs as envs
        from vllm.v1.core.kv_cache_utils import (
            generate_scheduler_kv_cache_config,
            get_kv_cache_configs,
        )

        root = get_workspace_root()
        ws = None
        if root and os.path.exists(os.path.join(root, "warmup_state.json")):
            ws = rt.load_warmup_state(root)
            log.info(
                "[foundry] %s mode: warmup_state.json found, skipping KV profiling",
                mode.value.upper(),
            )

        specs = self.model_executor.get_kv_cache_specs()
        has_kv = any(s for s in specs)

        if ws is not None:
            available_mem = ws.available_gpu_memory
            num_gpu = ws.num_gpu_blocks
            num_cpu = ws.num_cpu_blocks
        elif mode == CUDAGraphExtensionMode.LOAD:
            raise RuntimeError("foundry LOAD requires warmup_state.json")
        elif has_kv:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                assert self.available_gpu_memory_for_kv_cache > 0
                available_mem = [self.available_gpu_memory_for_kv_cache] * len(specs)
            else:
                t = time.time()
                available_mem = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_mem[0]
                log.info("[foundry TIMING] Memory profiling: %.3f s", time.time() - t)
                log.info("[foundry] available_gpu_memory=%s", available_mem)
            num_gpu = num_cpu = 0
        else:
            available_mem = [0] * len(specs)
            num_gpu = num_cpu = 0

        assert len(specs) == len(available_mem)
        max_before = vllm_config.model_config.max_model_len
        cfgs = get_kv_cache_configs(vllm_config, specs, available_mem)
        if vllm_config.model_config.max_model_len != max_before:
            self.collective_rpc(
                "update_max_model_len",
                args=(vllm_config.model_config.max_model_len,),
            )
        sched = generate_scheduler_kv_cache_config(cfgs)
        if ws is None:
            num_gpu = sched.num_blocks
        elif sched.num_blocks != num_gpu:
            log.warning(
                "[foundry] num_gpu_blocks mismatch: loaded=%d computed=%d (using loaded)",
                num_gpu,
                sched.num_blocks,
            )
            sched = sched._replace(num_blocks=num_gpu)

        vllm_config.cache_config.num_gpu_blocks = num_gpu
        if sched.kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in sched.kv_cache_groups
            )
        vllm_config.validate_block_size()

        self.model_executor.initialize_from_config(cfgs)

        # data_parallel_index survives vLLM's "treat as DP=1" reset; use
        # it so only the DP-shard-0 EngineCore writes warmup_state.json.
        dp_shard = getattr(vllm_config.parallel_config, "data_parallel_index", 0) or 0
        if mode == CUDAGraphExtensionMode.SAVE and root is not None and dp_shard == 0:
            os.makedirs(root, exist_ok=True)
            if ws is not None:
                ws.final_alloc_offset = rt.get_final_alloc_offset()
                rt.save_warmup_state(root, ws)
                log.info(
                    "[foundry] Updated warmup state: final_alloc_offset=%d", ws.final_alloc_offset
                )
            else:
                new = rt.create_warmup_state()
                new.available_gpu_memory = available_mem
                new.num_gpu_blocks = num_gpu
                new.num_cpu_blocks = num_cpu
                new.gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
                new.final_alloc_offset = rt.get_final_alloc_offset()
                rt.save_warmup_state(root, new)
                log.info("[foundry] Saved warmup state: num_gpu_blocks=%d", num_gpu)

        return sched

    cls._initialize_kv_caches = patched


# ---------------------------------------------------------------------------
# Subprocess spawn sites: LD_PRELOAD injection.
# ---------------------------------------------------------------------------


def _patch_subprocess_spawn_sites() -> None:
    # Multiproc executor worker spawn.
    from vllm.v1.executor import multiproc_executor as me

    if hasattr(me.WorkerProc, "make_worker_process"):
        orig = me.WorkerProc.make_worker_process

        @functools.wraps(orig)
        def patched(*args, **kwargs):
            rt.setup_ld_preload_env()
            return orig(*args, **kwargs)

        me.WorkerProc.make_worker_process = staticmethod(patched)

    # EngineCore proc manager.
    from vllm.v1.engine import utils as eu

    orig_init = eu.CoreEngineProcManager.__init__

    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        rt.setup_ld_preload_env()
        return orig_init(self, *args, **kwargs)

    eu.CoreEngineProcManager.__init__ = patched_init


# ---------------------------------------------------------------------------
# DeepEP buffer kwargs: use_fabric=True, num_nvl_bytes=0 on any foundry mode.
# ---------------------------------------------------------------------------


def _patch_deepep() -> None:
    try:
        from vllm.distributed.device_communicators import all2all
    except Exception:
        return
    cls = getattr(all2all, "DeepEPLLAll2AllManager", None)
    if cls is None or not hasattr(cls, "_make_all2all_kwargs"):
        log.info("[foundry] DeepEP patch skipped (class/method not found)")
        return

    orig = cls._make_all2all_kwargs

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if isinstance(out, dict) and get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
            # Force VMM-fabric buffers
            # so the foundry cuMem hook tracks them, and disable the
            # NVLink buffer (LL-mode uses RDMA) to avoid fabric-cuMemCreate
            # collisions with preallocated region.
            out["use_fabric"] = True
            out["num_nvl_bytes"] = 0
        return out

    cls._make_all2all_kwargs = patched
