# vLLM Integration Overview

Foundry persists vLLM's CUDA graphs to disk on SAVE and restores them on LOAD, skipping `torch.compile`, the inductor cache, kernel warmup, sampler warmup, and graph capture. The integration is the proving-ground for the foundry library; the SGLang integration is its sibling.

Validated on:

- Single-GPU Qwen3-1.7B / 4B / 14B
- DP > 1 (multi-GPU dense)
- Mixture-of-experts with DeepEP all-to-all (expert parallelism)

## Critical invariants (read this first)

Hard-won lessons. Each of these failure modes cost real debugging time. Future changes touching the integration must respect them.

### 1. No `torch.cuda.empty_cache()` between graph captures

Foundry's allocator offset is **monotonically increasing**. Freeing memory between captures unmaps the VMM range but does not rewind the cursor. Subsequent allocations advance the cursor further, inflating `final_alloc_offset` past the real working set (~194 GB observed instead of ~80 GB on Qwen3-30B-A3B EP2). Foundry's own `CUDAGraph.__enter__` had this exact bug and the call has been deleted at `foundry/python/foundry/graph.py:CUDAGraph.__enter__`.

**Don't introduce `empty_cache` (or any caching-allocator drain) inside the capture loop.**

### 2. `cudagraph_num_of_warmups` must be 0 in foundry mode

vLLM hard-overrides `compilation_config.cudagraph_num_of_warmups = 1` whenever cudagraph mode is enabled (`vllm/config/vllm.py:1052`), regardless of user config. The warmup call is `_dummy_run(cudagraph_runtime_mode=NONE, force_attention=True)` which goes through `CUDAGraphWrapper.__call__` with `rm == NONE` and short-circuits to `self.runnable(...)` — an *uncaptured* forward. That forward allocates transient buffers outside of any capture window, breaking VMM offset parity between SAVE and LOAD.

**`install_hooks` stomps the value back to 0** (`foundry/python/foundry/integration/vllm/hooks.py`). Don't undo this without re-introducing the asymmetry.

### 3. `init_nvshmem_for_loaded_modules` must run *after* `prepare_communication_buffer_for_model`

NVSHMEM-using modules loaded from the foundry archive are queued into `pending_nvshmem_init` by `fops.load_cuda_modules_and_libraries` in `setup_graph_extension`. They cannot be initialized via `nvshmemx_cumodule_init` until the NVSHMEM runtime is bootstrapped, which happens inside `prepare_communication_buffer_for_model` via DeepEP `Buffer` construction.

The init flush is done in `_patch_prepare_comm_buffer`'s **post-orig hook on LOAD**. Calling it earlier (e.g. at the end of `start_graph_builds`) silently no-ops — `nvshmemx_cumodule_init` returns 0 without doing anything, and replayed graphs later fail with `cuLaunchKernel` errors.

### 4. No split-load for MoE — use the unified preallocate → load path

A previous design had a separate `split_load_model` path for quantized MoE that called `prepare_communication_buffer_for_model` before `preallocate_for_load_mode`. This was based on the incorrect belief that prepare_comm_buffer's allocations would collide with the preallocated region.

In reality, NVSHMEM's symmetric heap goes through the **large-alloc branch** of foundry's `cuMemAddressReserve` hook (`hook.cpp:2523`) and lands in a separate VMM range outside the foundry preallocated region — no collision. Newer vLLM versions also use modular kernels for quantized MoE, which means weights must be fully loaded before `prepare_communication_buffer_for_model` can run (it now hard-raises pre-load). The split-load path is both **unnecessary and impossible** to maintain.

All MoE variants (quantized, unquantized) and dense models now share the same path: `preallocate_for_load_mode` → `do_original_load()` (weight load + `prepare_communication_buffer_for_model` + NVSHMEM init post-hook) → `start_graph_builds`. The background template builds were previously kicked off *before* `do_original_load` to overlap with weight IO; that was net-negative due to driver contention, so they now overlap with the cheaper post-load init work instead.

## How to use

### 1. Install foundry + vLLM

Follow `README.md`

### 2. Write a TOML config

```toml
# experimental/single-gpu/save_qwen_1.7b.toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive_qwen_1.7b"
scratch_space_size = "1024MB"
```

`load_qwen_1.7b.toml` mirrors the SAVE config with `mode = "load"`. Field semantics in [`memory-lifecycle.md`](memory-lifecycle.md).

### 3. Run SAVE twice, then LOAD

```bash
# SAVE pass 1 (profiles + captures)
rm -rf foundry_archive_qwen_1.7b
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait for "Application startup complete", SIGTERM

# SAVE pass 2 (deterministic; reuses profile from pass 1)
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait, SIGTERM

# LOAD
bash experimental/single-gpu/serve_qwen_1.7b.sh --load

# Query
bash experimental/query.sh 12000 Qwen/Qwen3-1.7B
```

Two SAVE passes are required (in contrast to SGLang's one-pass SAVE). The first pass runs vLLM's `memory_profiling` context manager, which does a real forward to measure peak memory. That forward's activations are caching-allocator-driven and non-reproducible across runs. SAVE pass 2 reads the persisted profile + KV-block sizing from `warmup_state.json` and re-allocates deterministically; LOAD does the same.

## What the integration does

vLLM's lifecycle splits across the engine-core process and worker processes. Foundry hooks both. The high-level shape:

**SAVE pass 1** (Worker.init_device → … → GPUModelRunner.capture_model):

1. `setup_graph_extension(parallel_config)` reserves the VMM region before NCCL.
2. NCCL bring-up runs in scratch space; cursor forced to scratch boundary.
3. `_initialize_kv_caches` runs `memory_profiling` (full forward).
4. `WarmupState` written with `available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks`.
5. Model weights → KV cache → graphs captured.
6. `capture_final_alloc_offset()` records the watermark.

**SAVE pass 2** (same code, but with warmup state present):

1. Same VMM setup as pass 1.
2. `_initialize_kv_caches` skips the profile forward and uses saved block counts.
3. Same deterministic allocation order ⇒ same `final_alloc_offset`.
4. Captured graphs are byte-identical to pass 1's (already on disk — pass 2 overwrites them).

**LOAD**:

1. VMM region setup same as SAVE.
2. `preallocate_for_load_mode` pre-maps the entire deterministic range.
3. `_initialize_kv_caches` reads saved state, skips profile forward.
4. `_load_model_with_overlap`: `do_original_load` runs first (weight IO + `process_weights_after_loading` + `prepare_communication_buffer_for_model`), then `start_graph_builds` kicks off background template construction so it overlaps with the cheaper post-load init work that follows (KV cache setup, scheduler bring-up). The order was reversed in an earlier design; running the templates against the weight load itself caused driver contention.
5. The post-orig hook on `prepare_communication_buffer_for_model` flushes `init_nvshmem_for_loaded_modules` once the NVSHMEM runtime is bootstrapped.
6. `capture_model` runs the upstream capture loop. `cudagraph_num_of_warmups` was already forced to 0 by `install_hooks`, so no asymmetric warmup forward fires.
7. `CUDAGraphWrapper.__call__` enters LOAD mode and calls `FoundryCUDAGraph.finish_one_graph_load(pending, index)` per batch, interleaving allocator-event replay with the loop's between-capture work so the VMM cursor walks the same trajectory it did on SAVE.
8. At end of `capture_model`, `fops.stop_allocation_region()` is called so all subsequent allocations (sampler warmup, scheduler, inference workspaces) go through the standard CUDA caching allocator. The VMM region only needs to hold memory referenced by captured graphs.
9. `compile_or_warm_up_model`'s sampler-warmup full forward is skipped via the kwarg-signature-targeted `_dummy_run` patch (returns zero hidden states for that one caller only); the compile-warmup loop is empty under our config. `_dummy_sampler_run` still runs on the zeros to populate sampler workspaces.

## Doc set

- [`overview.md`](overview.md) — this file
- [`direct-edits.md`](direct-edits.md) — every line that changed in `vllm/`
- [`hooks.md`](hooks.md) — every monkey-patch foundry installs and what it does on SAVE / LOAD
- [`memory-lifecycle.md`](memory-lifecycle.md) — VMM region setup, the SAVE↔LOAD allocation parity contract, two-pass SAVE rationale
- [`save-load-workflow.md`](save-load-workflow.md) — serve scripts, TOML schema, expected logs, validation checks
- [`memory-consistency.md`](memory-consistency.md) — what `memory_profiling` does that mandates two-pass SAVE, what's skipped on LOAD, and the silent-mismatch failure modes
- [`moe-and-deepep.md`](moe-and-deepep.md) — DeepEP `use_fabric=True` on LOAD, NVSHMEM init ordering

## Scope

| Supported | Notes |
|---|---|
| Dense decode (Qwen3, Llama, …) | primary test target |
| DP > 1 | multi-GPU dense |
| MoE with DeepEP | unified path; no split-load |
| `torch.compile` (full-graph) | runs on SAVE; `do_not_compile=True` on LOAD (faster startup; graphs replay via `CUDAGraphWrapper`) |

## Sibling: SGLang

The same shape but smaller surface area: see [`../sglang/overview.md`](../sglang/overview.md). Key differences are documented in [`../overview.md`](../overview.md#what-differs-between-the-two).
