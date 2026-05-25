# vLLM Hook Surface

Every monkey-patch `install_hooks(compilation_config)` installs.

## install_hooks order

```python
_quarantine_get_device_capability()       # NVML-based replacement; keeps parent fork-safe
load_graph_extension_config(cfg_path)
compilation_config.cudagraph_num_of_warmups = 0   # vLLM forces 1; we force 0

_patch_init_worker_distributed_environment()   # VMM region setup around NCCL
_patch_kernel_warmup()                          # no-op on SAVE/LOAD
_patch_dummy_runs()                             # signature-targeted skip of V1 sampler-warmup full forward
_patch_load_model()                             # _load_model_with_overlap (preallocate → load → start_graph_builds)
_patch_prepare_comm_buffer()                    # post-orig: init_nvshmem_for_loaded_modules on LOAD
_patch_capture_model()                          # prepare_graph_capture → orig → save/unlock → stop_allocation_region (+ capture_final_alloc_offset on SAVE)
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

The upstream `_dummy_run` runs a real forward for profile / warmup. Foundry's patch returns zero hidden-state tensors for **exactly one** caller — the V1 sampler-warmup full forward at `gpu_worker.py:691`, identified by its unique kwarg signature:

```python
if (
    kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.NONE
    and kwargs.get("skip_eplb") is True
    and not kwargs.get("is_profile")
    and not kwargs.get("is_graph_capturing")
    and not kwargs.get("force_attention")
    and not kwargs.get("uniform_decode")
):
    return zeros  # sampler-warmup only
return orig_dummy(self, *args, **kwargs)
```

Every other `_dummy_run` caller delegates to the original. The table below shows why this matters — most importantly, `Worker.execute_dummy_batch`'s DP empty-shard phantom call (`uniform_decode=True`) MUST reach the real forward so the empty shard joins `coordinate_batch_across_dp`'s CPU all-reduce and the EP all2all. An earlier coarser version of the patch short-circuited that call too and caused MoE+DP+concurrent bench to hang indefinitely with 0% GPU.

| Call site | Args | Patch behavior |
|---|---|---|
| `gpu_worker.py:592` compile-warmup loop | `(size, skip_eplb=True, remove_lora=False)` | pass through (no `cudagraph_runtime_mode=NONE`; under our config the loop is empty anyway) |
| `gpu_worker.py:691` V1 **sampler warmup** | `(num_tokens=max_num_reqs, skip_eplb=True, cudagraph_runtime_mode=NONE)` | **skip** (return zeros) |
| `gpu_worker.py:909` **DP empty-shard phantom** | `(num_tokens, uniform_decode=True)` | pass through (real forward → joins DP/EP collectives) |
| `gpu_model_runner.py:5799` profile_run | `is_profile=True` | pass through |
| `gpu_model_runner.py:6090` capture warmup (dead via num_warmups=0) | `force_attention=True/False` | pass through (via `force_attention`) |
| `gpu_model_runner.py:6101` capture | `is_graph_capturing=True` | pass through |

The sampler call at `gpu_worker.py:699` (`_dummy_sampler_run`) still runs on the zero hidden states. By the time it runs, `_patch_capture_model` has already called `fops.stop_allocation_region()` (§4), so the sampler's logits + workspace allocations go to the standard CUDA caching allocator — **not** the VMM region. They don't need to be deterministic across SAVE/LOAD, only the captured-graph memory does. The skip exists to save the cost of an extra real forward, not to keep workspaces in VMM.

`compile_or_warm_up_model` is **not** monkey-patched. It only ever needed a post-hook to call `rt.capture_final_alloc_offset()` on SAVE, and that call has been folded into `_patch_capture_model`'s post-hook — `stop_allocation_region()` freezes the VMM cursor before sampler warmup runs, so the watermark recorded inside capture_model is the same value the (now-deleted) compile_or_warm_up_model post-hook used to record.

## 4. `load_model`

Replaces the inner `model_loader.load_model(...)` call with `_load_model_with_overlap(...)`:

**LOAD** (unified across dense, unquantized MoE, quantized MoE):

```
rt.preallocate_for_load_mode()              # cuMemCreate+cuMemMap up to final_alloc_offset
model_loader.load_model()                   # the slow part
                                            #   - load_weights
                                            #   - process_weights_after_loading
                                            #   - prepare_communication_buffer_for_model
                                            #     (NVSHMEM bootstrap; our post-orig hook then
                                            #      flushes init_nvshmem_for_loaded_modules)
start_graph_builds()                        # async; runs concurrently with everything
                                            # between load_model returning and capture_model
                                            # (KV cache init, scheduler bring-up, etc.)
# finish_one_graph_load fires per batch later inside capture_model
```

Ordering note: an earlier design put `start_graph_builds()` *before* `model_loader.load_model()` so the background template builds would overlap with weight IO. In practice that was net-negative — the templates and weight load both hammer the CUDA driver, so the parallelism cost more than it saved. Running graph builds after `model_loader.load_model` lets them overlap with the cheaper post-load init phases instead.

**SAVE**: passthrough (vLLM's natural ordering).

Previous versions had a separate split-load path for quantized MoE that called `prepare_communication_buffer_for_model` before `preallocate_for_load_mode` with a lightweight quant-config inject. That dance was unnecessary — NVSHMEM's symmetric heap goes through the large-alloc branch of foundry's `cuMemAddressReserve` hook (`hook.cpp:2523`) and lands outside the foundry preallocated region, and newer vLLM uses modular kernels for quantized MoE so the pre-load quant inject is no longer relevant.

## 5. `prepare_communication_buffer_for_model`

Post-orig hook: on LOAD, flushes `init_nvshmem_for_loaded_modules()` immediately after the orig call bootstraps the NVSHMEM runtime (via DeepEP `Buffer` creation). `nvshmemx_cumodule_init` requires the runtime to be up, so this is the earliest correct call site for the pending NVSHMEM init list populated by `fops.load_cuda_modules_and_libraries` during `setup_graph_extension`.

## 6. `capture_model`

Both SAVE and LOAD share the same orchestration:

1. **Pre-orig:** `rt.prepare_graph_capture(metadata_builders=builders, device=self.device)` does three things, all *outside* any CUDA-graph capture window:
   - **`_eager_inductor_lazy_init(device)`** — calls `torch._inductor.fx_passes.joint_graph.lazy_init(device)` once. That `lazy_init` is `@init_once_fakemode` (i.e. `@functools.cache` keyed on `input_device`); its body invokes `_sfdp_init` which builds SDPA pattern templates by copying CPU constants onto the device. Without the pre-warm, the first in-capture compile triggers `_sfdp_init` *inside* `with torch.cuda.graph(...)` and the CPU→GPU copy raises `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture`. We pass `model_runner.device` because inductor's in-capture path keys on the indexed `cuda:N` device from `get_inputs_devices`; a non-indexed `torch.device("cuda")` would be a cache MISS and the in-capture call would still re-run the init.
     *Only `joint_graph.lazy_init` is pre-warmed.* The other three (`pre_grad`, `post_grad`, `freezing_patterns`) are also `@init_once_fakemode` but their bodies just call `register_replacement` for FX patterns, not CPU→GPU copies. Pre-warming them would actively hurt: inductor's `post_grad_passes` invokes `lazy_init()` (no argument, cache key `None`) at compile time, which is a different cache key from our `lazy_init(cuda:0)`. Both would run, both would `register_replacement("amax_default", …)`, and the second hits `torch._inductor.exc.InductorError: Duplicate pattern: amax_default`.
   - **`fops.preallocate_cublas_workspaces()`** so cuBLAS handles land at deterministic VMM offsets.
   - **`_preallocate_attention_workspaces(builders)`** — touch FlashInfer / TRT-LLM gen workspaces so they too land at deterministic VMM offsets.
2. **Orig:** the upstream capture loop runs unchanged. On LOAD, each `CUDAGraphWrapper.__call__` dispatches to `capture_or_load_graph` → `FoundryCUDAGraph.finish_one_graph_load(pending, index)` per batch (so the foundry VMM cursor walks the same trajectory it did on SAVE — allocations between captures interleave with replayed graph events in the same order on both sides).
3. **Post-orig (mode-specific):**
   - SAVE → `save_graph_manifest()` + `pack_fatbins()`
   - LOAD → `unlock_workspace()` so any inference-time growth past `final_alloc_offset` is allowed
4. **Post-orig (both modes, last step):** `fops.stop_allocation_region()`. Captured-graph memory needs the deterministic VMM region; runtime allocations (sampler warmup, inference workspaces, scheduler buffers) don't. After capture, allocations fall back to the standard CUDA caching allocator.
5. **Post-orig (SAVE only, last):** `rt.capture_final_alloc_offset()` — record the deterministic watermark. Runs after `stop_allocation_region`, so the value reflects only the VMM cursor at end of capture (later sampler-warmup allocations no longer advance it).

The capture loop is **not** wrapped in `torch.cuda.use_mem_pool(graph_pool)`. The wrapper's `set_graph_pool_id(...)` call (inside the patched `CUDAGraphWrapper.__call__`) is enough — pynccl_allocator routes captured-kernel allocations through the registered graph pool, and the between-capture allocations stay symmetric across SAVE 2 and LOAD because (a) `cudagraph_num_of_warmups=0` removes the asymmetric warmup forward, (b) the `_dummy_run` patch skips the same single site on both sides, and (c) `empty_cache()` inside `foundry/graph.py:CUDAGraph.__enter__` was removed (it was the primary `final_alloc_offset` inflator on vLLM, going from ~80 GB → ~194 GB on Qwen3-30B-A3B EP2 when it was present).

An earlier design used a `resume_allocation_region()` / `stop_allocation_region()` bracket *around the capture loop* to exclude between-capture allocations from the deterministic VMM region — that was wrong, since any tensor those allocations produced that was later referenced by a captured kernel ended up outside the region. The current design keeps the region ON across the entire loop and only stops it *after* the loop, when no captured kernel will reference anything new.

## 7. `CUDAGraphWrapper.__call__` (the load-bearing patch)

This is the only patch that re-derives an entire method body, because the upstream `__call__` has the `torch.cuda.CUDAGraph` capture stanza inline. The patch replaces that stanza with `capture_or_load_graph(...)`:

```python
if mode == NONE:
    # upstream torch.cuda.CUDAGraph capture
elif mode == SAVE:
    graph = FoundryCUDAGraph()
    with foundry_graph_ctx(graph, pool=graph_pool):
        output = runnable(...)
    graph.save(json_path, output)
elif mode == LOAD:
    # finish_one_graph_load(pending, index) replays this batch's allocator
    # events at the recorded VMM offsets and reconstructs output tensors.
    graph, output = FoundryCUDAGraph.finish_one_graph_load(
        _pending_graph_builds.pending,
        _pending_graph_builds.identifier_to_index[identifier],
    )
    # Strong-ref (graph, output) in process-global state so the weak-ref
    # that CUDAGraphWrapper stores on `entry.output` keeps a live target;
    # otherwise the next replay writes to GC'd VMM memory.
    state.loaded_graphs[identifier] = (graph, output)
```

The patch also preserves the upstream offloader sync (`get_offloader().sync_prev_onload()`) and honors `CUDAGraphOptions` / the validation gate (skipped on LOAD because `set_cudagraph_capturing_enabled(True)` is already set by orig). `set_graph_pool_id(self.graph_pool)` is called inside the patch so pynccl_allocator routes captured-kernel allocations through the right pool.

## 8. `_initialize_kv_caches`

**SAVE pass 1**: run upstream profiling + KV-block sizing; write the resolved values to `WarmupState`.

**SAVE pass 2 / LOAD**: skip the profiling forward. Read `WarmupState.available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks` and apply them directly. This is the equivalent of SGLang's `init_memory_pool` patch but for vLLM's KV-init phase.

## 9. Subprocess spawn sites

`_patch_subprocess_spawn_sites` wraps a couple of call sites so `LD_PRELOAD` is injected into the child env immediately before `proc.start()`:

- `WorkerProc.make_worker_process` (MP executor)
- `CoreEngineProcManager.__init__` (engine-core proc spawn)

`setup_ld_preload_env()` (in `runtime.py`) prepends `libcuda_hook.so` (and optionally `libnvshmem_host.so`) to `os.environ["LD_PRELOAD"]`, sets `FOUNDRY_MODE` (consumed by `foundry/csrc/hook.cpp` to enable early skip of fatbin processing on LOAD), and stamps `FOUNDRY_SPAWN_T0_NS` for spawn-to-install timing logs.

## 10. `DeepEPLLAll2AllManager._make_all2all_kwargs`

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
