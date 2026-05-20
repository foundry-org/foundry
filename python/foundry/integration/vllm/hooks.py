# SPDX-License-Identifier: Apache-2.0
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
from typing import TYPE_CHECKING, Callable

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
        log.info("[foundry] pynvml not available; cannot quarantine "
                 "get_device_capability — parent may still init CUDA")
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
    log.info("[foundry] torch.cuda.get_device_capability → NVML "
             "(keeps parent fork-safe)")


def install_hooks(compilation_config: "CompilationConfig") -> None:
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
            "[CGE TIMING] subprocess spawn → install_hooks: %.1f ms "
            "(cold Python + torch + vllm import with LD_PRELOAD'd hook)",
            spawn_elapsed_ms,
        )

    load_graph_extension_config(cfg_path)
    log.info("[foundry] install_hooks: mode=%s workspace=%s",
             get_graph_extension_mode().value, get_workspace_root())

    _patch_init_worker_distributed_environment()
    _patch_kernel_warmup()
    _patch_dummy_runs()
    _patch_compile_or_warm_up_model()
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
    def patched(vllm_config, rank, distributed_init_method=None,
                local_rank=-1, backend="nccl"):
        active = get_graph_extension_mode() != CUDAGraphExtensionMode.NONE
        t0 = time.perf_counter()
        if active:
            rt.setup_graph_extension(vllm_config.parallel_config)
        result = orig(vllm_config, rank, distributed_init_method,
                      local_rank, backend)
        if active:
            rt.skip_to_scratch_boundary()
            log.info(
                "[CGE TIMING] init_worker_distributed_environment total: "
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
    """Skip only the non-deterministic pieces: the compile-warmup and
    sampler-warmup full-forward pass. The sampler itself still runs on
    dummy hidden states so its logits + workspace allocations land in
    the deterministic VMM region (matching old CGE behavior)."""
    from vllm.v1.worker import gpu_model_runner as gmr
    cls = gmr.GPUModelRunner
    orig_dummy = cls._dummy_run

    @functools.wraps(orig_dummy)
    def patched_dummy(self, *args, **kwargs):
        if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
            return orig_dummy(self, *args, **kwargs)
        # Let the real forward run for: memory profiling, capture warmup,
        # and capture itself. Skip compile-warmup loop and the sampler-
        # warmup full-forward (both produce non-deterministic allocs).
        if (kwargs.get("is_profile") or kwargs.get("is_graph_capturing")
                or kwargs.get("force_attention")):
            return orig_dummy(self, *args, **kwargs)
        n = args[0] if args else kwargs.get("num_tokens", 1)
        d = self.model_config.get_hidden_size()
        hs = torch.zeros(n, d, device=self.device,
                         dtype=self.model_config.dtype)
        return hs, hs

    cls._dummy_run = patched_dummy
    # NOTE: _dummy_sampler_run is intentionally NOT patched — it must
    # run with our dummy hidden states so logits + sampler workspaces
    # allocate inside the deterministic region before capture_final_alloc_offset.


# ---------------------------------------------------------------------------
# Worker.compile_or_warm_up_model: capture_final_alloc_offset on SAVE.
# ---------------------------------------------------------------------------


def _patch_compile_or_warm_up_model() -> None:
    from vllm.v1.worker import gpu_worker as gw
    cls = gw.Worker
    orig = cls.compile_or_warm_up_model

    @functools.wraps(orig)
    def patched(self):
        result = orig(self)
        if get_graph_extension_mode() == CUDAGraphExtensionMode.SAVE:
            rt.capture_final_alloc_offset()
        return result

    cls.compile_or_warm_up_model = patched


# ---------------------------------------------------------------------------
# GPUModelRunner.load_model: route model_loader.load_model through the
# foundry overlap helper. Uses a closure over the un-patched callable to
# avoid recursion. Lets upstream run untouched so it still wraps self.model
# with CUDAGraphWrapper / UBatchWrapper.
# ---------------------------------------------------------------------------


def _load_model_with_overlap(runner, do_original_load: Callable,
                             model_loader) -> torch.nn.Module:
    """Dense path: preallocate (LOAD) + start_graph_builds (LOAD) + load.
    MoE path (phase 2): delegates to moe.split_load_model."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        return do_original_load()

    # MoE split-load is phase 2; triggered by presence of moe_quant_metadata
    # in warmup_state.json.
    #
    # Only takes the split-load path for *quantized* MoE (quant_dtype set).
    # For unquantized MoE the new vLLM creates the modular kernel inside
    # process_weights_after_loading, and prepare_communication_buffer_for_model
    # calls UnquantizedFusedMoEMethod.maybe_make_prepare_finalize which now
    # hard-raises — so there is nothing useful to pre-init before weight load.
    # Dense-overlap path (start_graph_builds before load_weights) still gives
    # us the IO/GPU overlap without the split-load ceremony.
    moe_md = None
    root = get_workspace_root()
    if root:
        try:
            ws = rt.load_warmup_state(root)
            if ws.moe_quant_metadata and ws.moe_quant_metadata.get("quant_dtype"):
                moe_md = ws.moe_quant_metadata
        except (RuntimeError, FileNotFoundError):
            pass
    if moe_md is not None:
        from foundry.integration.vllm.moe import split_load_model
        return split_load_model(runner, model_loader, mode, moe_md)

    if mode == CUDAGraphExtensionMode.LOAD:
        rt.preallocate_for_load_mode()
        from foundry.integration.vllm.graph_ops import start_graph_builds
        start_graph_builds()
    return do_original_load()


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
                return orig_loader_load(vllm_config=runner.vllm_config,
                                        model_config=runner.model_config)

            def patched_loader_load(*_args, **_kwargs):
                return _load_model_with_overlap(runner, do_original_load,
                                                loader)

            loader.load_model = patched_loader_load
            return loader

        gmr.get_model_loader = patched_get_loader
        try:
            result = orig(self, *args, **kwargs)
        finally:
            gmr.get_model_loader = orig_get_loader

        # On SAVE, collect MoE quant metadata AFTER upstream load_model
        # (which ran prepare_communication_buffer_for_model and therefore
        # finalized the quant_method state). _initialize_kv_caches then
        # reads the process-local global and writes it into
        # warmup_state.json so pass 2 / LOAD can drive split-load.
        if get_graph_extension_mode() == CUDAGraphExtensionMode.SAVE:
            from foundry.integration.vllm.moe import collect_moe_quant_metadata
            try:
                m = runner.get_model() if hasattr(runner, "get_model") else runner.model
                collect_moe_quant_metadata(m)
            except Exception as e:
                log.info("[CGE] collect_moe_quant_metadata skipped: %s", e)
        log.info(
            "[CGE TIMING] GPUModelRunner.load_model total: %.3f s "
            "(preallocate + start_graph_builds + weight load + "
            "process_weights_after_loading)",
            time.perf_counter() - t0,
        )
        return result

    cls.load_model = patched


# ---------------------------------------------------------------------------
# prepare_communication_buffer_for_model suppressor: upstream load_model
# calls this after weight load; split_load_model already called it in
# step 2. Skip exactly one subsequent call when the flag is set.
# ---------------------------------------------------------------------------


def _patch_prepare_comm_buffer() -> None:
    from vllm.v1.worker import gpu_model_runner as gmr
    orig = gmr.prepare_communication_buffer_for_model

    @functools.wraps(orig)
    def patched(model):
        from foundry.integration.vllm.moe import should_suppress_prepare_comm
        if should_suppress_prepare_comm():
            log.info(
                "[CGE] prepare_communication_buffer_for_model skipped "
                "(split-load already prepared comm buffers)"
            )
            return
        return orig(model)

    gmr.prepare_communication_buffer_for_model = patched


# ---------------------------------------------------------------------------
# GPUModelRunner.capture_model: prepare_graph_capture before orig; on LOAD,
# replace orig with preload_all_graphs; on SAVE, post-orig save manifest +
# pack fatbins.
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

        from vllm.platforms import current_platform
        from foundry.integration.vllm.graph_ops import (
            pack_fatbins, preload_all_graphs, save_graph_manifest,
        )

        builders = [b for ag_list in self.attn_groups
                    for ag in ag_list for b in ag.metadata_builders]
        rt.prepare_graph_capture(metadata_builders=builders)

        if mode == CUDAGraphExtensionMode.LOAD:
            import gc
            gc.collect()
            torch.accelerator.empty_cache()
            preload_all_graphs(graph_pool=current_platform.graph_pool_handle())
            return 0

        # SAVE
        result = orig(self)
        save_graph_manifest()
        pack_fatbins()
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
            get_forward_context, is_forward_context_available,
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
            self.concrete_cudagraph_entries[bd] = CUDAGraphEntry(
                batch_descriptor=bd
            )
        entry = self.concrete_cudagraph_entries[bd]

        if entry.cudagraph is None:
            if get_graph_extension_mode() != CUDAGraphExtensionMode.LOAD:
                validate_cudagraph_capturing_enabled()
            entry.input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    stack.enter_context(mock_patch("gc.collect", lambda: None))
                    stack.enter_context(mock_patch(
                        "torch.accelerator.empty_cache", lambda: None))
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
            generate_scheduler_kv_cache_config, get_kv_cache_configs,
        )

        root = get_workspace_root()
        ws = None
        if root and os.path.exists(os.path.join(root, "warmup_state.json")):
            ws = rt.load_warmup_state(root)
            log.info("[CGE] %s mode: warmup_state.json found, skipping KV profiling",
                     mode.value.upper())

        specs = self.model_executor.get_kv_cache_specs()
        has_kv = any(s for s in specs)

        if ws is not None:
            available_mem = ws.available_gpu_memory
            num_gpu = ws.num_gpu_blocks
            num_cpu = ws.num_cpu_blocks
        elif mode == CUDAGraphExtensionMode.LOAD:
            raise RuntimeError("CGE LOAD requires warmup_state.json")
        elif has_kv:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                assert self.available_gpu_memory_for_kv_cache > 0
                available_mem = ([self.available_gpu_memory_for_kv_cache]
                                 * len(specs))
            else:
                t = time.time()
                available_mem = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_mem[0]
                log.info("[CGE TIMING] Memory profiling: %.3f s",
                         time.time() - t)
                log.info("[CGE] available_gpu_memory=%s", available_mem)
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
            log.warning("[CGE] num_gpu_blocks mismatch: loaded=%d computed=%d (using loaded)",
                        num_gpu, sched.num_blocks)
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
        dp_shard = getattr(
            vllm_config.parallel_config, "data_parallel_index", 0
        ) or 0
        if (mode == CUDAGraphExtensionMode.SAVE
                and root is not None and dp_shard == 0):
            os.makedirs(root, exist_ok=True)
            if ws is not None:
                ws.final_alloc_offset = rt.get_final_alloc_offset()
                moe = rt.get_moe_quant_metadata()
                if moe:
                    ws.moe_quant_metadata = moe
                rt.save_warmup_state(root, ws)
                log.info("[CGE] Updated warmup state: final_alloc_offset=%d",
                         ws.final_alloc_offset)
            else:
                new = rt.create_warmup_state()
                new.available_gpu_memory = available_mem
                new.num_gpu_blocks = num_gpu
                new.num_cpu_blocks = num_cpu
                new.gpu_memory_utilization = (
                    vllm_config.cache_config.gpu_memory_utilization
                )
                new.final_alloc_offset = rt.get_final_alloc_offset()
                moe = rt.get_moe_quant_metadata()
                if moe:
                    new.moe_quant_metadata = moe
                rt.save_warmup_state(root, new)
                log.info("[CGE] Saved warmup state: num_gpu_blocks=%d",
                         num_gpu)

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
        if (isinstance(out, dict)
                and get_graph_extension_mode() != CUDAGraphExtensionMode.NONE):
            # Mirrors cge.patch:2889-2906 + 2923: force VMM-fabric buffers
            # so the foundry cuMem hook tracks them, and disable the
            # NVLink buffer (LL-mode uses RDMA) to avoid fabric-cuMemCreate
            # collisions with preallocated region.
            out["use_fabric"] = True
            out["num_nvl_bytes"] = 0
        return out

    cls._make_all2all_kwargs = patched
