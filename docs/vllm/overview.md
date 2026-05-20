# vLLM Integration Overview

Foundry persists vLLM's CUDA graphs to disk on SAVE and restores them on LOAD, skipping `torch.compile`, the inductor cache, kernel warmup, sampler warmup, and graph capture. The integration is the proving-ground for the foundry library; the SGLang integration is its sibling.

Validated on:

- Single-GPU Qwen3-1.7B / 4B / 14B
- DP > 1 (multi-GPU dense)
- Mixture-of-experts with DeepEP all-to-all (expert parallelism)

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
4. `WarmupState` written with `available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks`, and MoE quant metadata.
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
3. `_initialize_kv_caches` reads saved state.
4. `start_graph_builds(all_paths)` kicks off graph-template construction in a background thread, overlapping with weight load.
5. `load_model` waits on weight IO; foundry's `_load_model_with_cge_overlap` orchestrates the overlap.
6. `prepare_communication_buffer_for_model` is gated by `_comm_buffers_prepared` so it doesn't re-run.
7. `capture_model` is replaced with `preload_all_graphs(graph_pool)` which finalizes builds and reconstructs output tensors.
8. `CUDAGraphWrapper.__call__` enters LOAD mode and looks up the graph for the current `BatchDescriptor` in `state.loaded_graphs`.
9. `compile_or_warm_up_model`'s compile-warmup and `_dummy_run` loop are skipped. The sampler warmup runs on a dummy hidden-state tensor instead of a full forward.

## Doc set

- [`overview.md`](overview.md) — this file
- [`direct-edits.md`](direct-edits.md) — every line that changed in `vllm/`
- [`hooks.md`](hooks.md) — every monkey-patch foundry installs and what it does on SAVE / LOAD
- [`memory-lifecycle.md`](memory-lifecycle.md) — VMM region setup, the SAVE↔LOAD allocation parity contract, two-pass SAVE rationale
- [`save-load-workflow.md`](save-load-workflow.md) — serve scripts, TOML schema, expected logs, validation checks
- [`memory-consistency.md`](memory-consistency.md) — what `memory_profiling` does that mandates two-pass SAVE, what's skipped on LOAD, and the silent-mismatch failure modes
- [`moe-and-deepep.md`](moe-and-deepep.md) — MoE quant metadata collection / re-injection, DeepEP `use_fabric=True` on LOAD, NVSHMEM init ordering

## Scope

| Supported | Notes |
|---|---|
| Dense decode (Qwen3, Llama, …) | primary test target |
| DP > 1 | multi-GPU dense |
| MoE with DeepEP | EP-aware split load; needs `moe.py` |
| `torch.compile` (full-graph) | compile on SAVE; `do_not_compile=True` on LOAD |

## Sibling: SGLang

The same shape but smaller surface area: see [`../sglang/overview.md`](../sglang/overview.md). Key differences are documented in [`../overview.md`](../overview.md#what-differs-between-the-two).
