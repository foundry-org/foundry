# vLLM Hook Surface

Every monkey-patch `install_hooks(compilation_config)` installs.

## install_hooks order

```python
_quarantine_get_device_capability()       # NVML-based replacement; keeps parent fork-safe
load_graph_extension_config(cfg_path)

_patch_init_worker_distributed_environment()   # VMM region setup around NCCL
_patch_kernel_warmup()                          # no-op on SAVE/LOAD
_patch_dummy_runs()                             # dummy hidden-states on SAVE/LOAD
_patch_compile_or_warm_up_model()               # capture_final_alloc_offset; sampler warmup tweak
_patch_load_model()                             # dense overlap + MoE split-load orchestration
_patch_prepare_comm_buffer()                    # _comm_buffers_prepared gate
_patch_capture_model()                          # preload_all_graphs on LOAD
_patch_cuda_graph_wrapper_call()                # capture_or_load_graph swap
_patch_initialize_kv_caches()                   # WarmupState save/load
_patch_subprocess_spawn_sites()                 # LD_PRELOAD into child env before proc.start()
_patch_deepep()                                 # use_fabric=True on LOAD
```

## Pre-install: `_quarantine_get_device_capability`

Replaces `torch.cuda.get_device_capability` with an NVML-backed implementation so the parent process can query device capability without paying for a full CUDA context init. Important when the parent later uses `forkserver` — keeps the parent fork-safe.

## 1. `init_worker_distributed_environment`

```
rt.setup_graph_extension(vllm_config.parallel_config)
result = orig(...)                          # NCCL + comm bring-up
rt.skip_to_scratch_boundary()
```

Reserves the VMM region before NCCL; advances cursor to `cfg.scratch_space_size` after NCCL warmup. Same shape as the SGLang equivalent.

## 2. `kernel_warmup`

No-op on SAVE/LOAD. Skips FlashInfer autotune, CUDA-graph-private warmups, and other inductor cache-priming forwards whose allocations would diverge between SAVE and LOAD.

## 3. `_dummy_run`

The upstream `_dummy_run` runs a real forward for profile / warmup. Foundry's patch returns dummy hidden-state tensors when called without `is_profile=True`, `is_graph_capturing=True`, or `force_attention=True` — keeping the sampler-warmup path intact (downstream compute_logits + sampler workspaces still need to allocate) while killing the non-deterministic full forward.

## 4. `compile_or_warm_up_model`

- Skip the compile-warmup `_dummy_run` loop and the post-capture sampler full-forward on SAVE/LOAD.
- After `capture_model` finishes on SAVE: call `rt.capture_final_alloc_offset()` to record the deterministic watermark.

## 5. `load_model`

Replaces the inner `model_loader.load_model(...)` call with `_load_model_with_cge_overlap(model_loader)`. Two paths:

**Dense LOAD**:

```
rt.preallocate_for_load_mode()              # cuMemCreate+cuMemMap up to final_alloc_offset
start_graph_builds()                        # async; runs concurrently with weight IO
model_loader.load_model()                   # the slow part
# finish_graph_loads happens later in capture_model
```

**MoE LOAD** (split-load):

```
loader._initialize_model(...)               # build empty model
collect_moe_quant_metadata()                # may inject saved metadata
prepare_communication_buffer_for_model()    # NVSHMEM init
rt.preallocate_for_load_mode()
start_graph_builds()
loader.load_weights_and_postprocess()
reinit_moe_quant_config()                   # post-load
```

See [`moe-and-deepep.md`](moe-and-deepep.md).

**SAVE**:

- Dense: passthrough.
- MoE: `collect_moe_quant_metadata(m)` after `load_model` so the metadata gets serialized to `WarmupState`.

## 6. `prepare_communication_buffer_for_model`

Wraps with `if not _comm_buffers_prepared:` so the post-load comm-buffer init doesn't re-run on LOAD (it ran inside `_load_model_with_cge_overlap`).

## 7. `capture_model`

**SAVE**: call upstream `capture_model` (which captures graphs into our foundry-backed wrappers via the `CUDAGraphWrapper.__call__` patch), then `save_graph_manifest()` + `pack_fatbins()`.

**LOAD**: replace the upstream capture loop entirely with `preload_all_graphs(graph_pool=current_platform.graph_pool_handle())`. This calls `finish_graph_loads(pending)` to complete the already-started background graph builds, then populates `state.loaded_graphs` keyed by `BatchDescriptor`.

## 8. `CUDAGraphWrapper.__call__` (the load-bearing patch)

This is the only patch that re-derives an entire method body, because the upstream `__call__` has the `torch.cuda.CUDAGraph` capture stanza inline. The patch replaces that stanza with `capture_or_load_graph(...)`:

```python
if cge_mode == NONE:
    # upstream torch.cuda.CUDAGraph capture
elif cge_mode == SAVE:
    graph = FoundryCUDAGraph()
    with foundry_graph_ctx(graph, pool=graph_pool):
        output = runnable(...)
    graph.save(json_path, output)
elif cge_mode == LOAD:
    graph, output = state.loaded_graphs[identifier]
```

The patch preserves the upstream offloader sync (`get_offloader().sync_prev_onload()`) and honors `CUDAGraphOptions` / the validation gate (skipped on LOAD).

## 9. `_initialize_kv_caches`

**SAVE pass 1**: run upstream profiling + KV-block sizing; write the resolved values to `WarmupState`.

**SAVE pass 2 / LOAD**: skip the profiling forward. Read `WarmupState.available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks` and apply them directly. This is the equivalent of SGLang's `init_memory_pool` patch but for vLLM's KV-init phase.

## 10. Subprocess spawn sites

`_patch_subprocess_spawn_sites` wraps several call sites so `LD_PRELOAD` is injected into the child env immediately before `proc.start()`:

- `WorkerProc.make_worker_process` (MP executor)
- `CoreEngineProcManager.__init__` (engine-core proc spawn)
- `CoreEngineActorManager.__init__` (Ray actor manager env merge)
- Ray executor path (when present)

`setup_ld_preload_env()` (in `runtime.py`) prepends `libcuda_hook.so` (and optionally `libnvshmem_host.so`) to `os.environ["LD_PRELOAD"]`, sets `CGE_MODE`, and stamps `FOUNDRY_SPAWN_T0_NS` for spawn-to-install timing logs.

## 11. `DeepEPLLAll2AllManager._make_all2all_kwargs`

Forces `use_fabric=True` and `num_nvl_bytes=0` (RDMA-only) on any foundry mode. This ensures DeepEP's communicator buffer comes through the fabric path, where foundry can interpose. Phase-2 work; gated by MoE presence.

## Patch idiom

Almost every patch is wrap-and-call:

```python
orig = cls.method
@functools.wraps(orig)
def patched(self, *args, **kwargs):
    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        return orig(self, *args, **kwargs)
    # foundry pre-work
    result = orig(self, *args, **kwargs)
    # foundry post-work
    return result
cls.method = patched
```

Exception: `CUDAGraphWrapper.__call__` is re-derived (because the inner block to swap is inline).

## When foundry is absent

`vllm/compilation/foundry_shim.py` provides a no-op fallback `install_hooks` and a `CUDAGraphExtensionMode.NONE` stub. vLLM still builds and runs unchanged when foundry is not installed.
