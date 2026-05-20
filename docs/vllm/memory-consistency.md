# vLLM Memory-Consistency Notes

vLLM's lifecycle has more non-deterministic paths than SGLang's. This doc enumerates each one, the rule the integration applies, and the failure mode you get if the rule breaks.

## The two-pass SAVE contract

**SAVE pass 1** runs the full upstream lifecycle including the profile forward. It writes `warmup_state.json` with the resolved KV-block sizing and MoE quant metadata; the captured graphs are functional but their `start_base_addr`s are not reproducible.

**SAVE pass 2** reads `warmup_state.json` and skips the profile forward. The lifecycle is now deterministic; graphs are re-captured with reproducible `start_base_addr`s, and `final_alloc_offset` is recorded.

**LOAD** reads `warmup_state.json`, preallocates the deterministic range, and restores the SAVE-pass-2 graphs.

Skipping pass 2 leaves SAVE-pass-1 graphs on disk. LOAD will either fail the cursor check or pass it and emit garbage — both have happened in our testing.

## Non-deterministic paths (and the rule each gets)

### 1. `_initialize_kv_caches` profile forward

Where: `vllm/v1/engine/core.py::EngineCore._initialize_kv_caches`.

What it does: runs `memory_profiling(...)` with a real `execute_model(...)` forward to measure peak memory, then divides available memory into KV blocks.

Rule:

- **SAVE pass 1**: run upstream profiling. Save `available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks`, `moe_quant_metadata` to `WarmupState`.
- **SAVE pass 2 / LOAD**: skip the profiling block. Use saved values for `num_gpu_blocks` / `num_cpu_blocks`.

Failure mode if broken: pass 2 / LOAD allocate different KV blocks than pass 1, drifting the cursor before capture begins.

### 2. Compile-warmup `_dummy_run` loop

Where: `vllm/v1/worker/gpu_worker.py::compile_or_warm_up_model`.

What it does: `for size in warmup_sizes: self.model_runner._dummy_run(size, ...)` to trigger inductor autotune and compile cache priming.

Rule: skip the whole loop on SAVE/LOAD. SAVE pass 1's compile cache is materialized through the actual graph capture (CUDAGraphWrapper runs inductor compile during the inner `capture_or_load_graph` call); the warmup loop is redundant and non-deterministic in its allocations.

### 3. `kernel_warmup`

Where: `vllm/v1/worker/gpu_worker.py::kernel_warmup`.

What it does: priming FlashInfer / various kernels.

Rule: no-op on SAVE/LOAD. Their internal allocations don't survive a SAVE/LOAD cycle, so reproducing them on LOAD is a waste — and any non-determinism is forbidden.

### 4. Full-forward sampler warmup

Where: `vllm/v1/worker/gpu_worker.py::compile_or_warm_up_model` (after `capture_model`).

What it does: `self.model_runner._dummy_run(...)` then `_dummy_sampler_run(last_hidden_states)` — a real forward followed by sampler warmup.

Rule: the **`_dummy_run` is replaced** with a dummy hidden-state tensor on SAVE/LOAD, but `_dummy_sampler_run` is **not** skipped. The sampler still allocates its workspaces (probs, topk-topp); we just don't need the forward.

Failure mode if `_dummy_sampler_run` is skipped: sampler workspace addresses on LOAD don't match SAVE's, and sampling kernels read unmapped memory.

### 5. Memory-history snapshotting / capture profiling

Off in foundry mode — `compilation_config.enable_profile_cuda_graph = False` etc.

### 6. NVSHMEM initialization

NVSHMEM modules are loaded on LOAD via `init_nvshmem_for_loaded_modules` (called from `preload_all_graphs`). This must happen before any captured graph that references NVSHMEM symbols is replayed.

## Silent mismatches

Cursor parity is necessary but not sufficient. The captured graph kernels reference raw pointers (e.g. into the FlashInfer workspace buffer). foundry's replay validates the cursor against `start_base_addr_X` for each graph, but it does not validate that those raw pointers resolve to the same data on LOAD as on SAVE.

The known silent-mismatch sources:

- **`prepare_communication_buffer_for_model` re-running on LOAD.** Gated by `_comm_buffers_prepared`; if the gate is bypassed, comm buffer addresses double-allocate at different offsets.
- **MoE quant metadata not collected on SAVE pass 1.** Without it, pass 2 / LOAD will re-derive a different quant config; the same MoE weights end up packed differently. Mitigated by `collect_moe_quant_metadata` in `_load_model_with_cge_overlap`.
- **DeepEP communicator buffers initialized in different mode.** `_patch_deepep` forces `use_fabric=True`; without it LOAD may set up the buffer in NVL mode while SAVE used fabric, breaking the captured kernels.

If you see correct cursors but garbage output, those three are the first suspects.

## Why three lifecycle phases need explicit "skip on LOAD" rules

Because we want to save time.

The integration's rule of thumb: **if upstream code does a CUDA allocation, that allocation must happen identically on SAVE pass 2 and LOAD, or it must be skipped on both**. 