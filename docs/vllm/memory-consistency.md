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

- **SAVE pass 1**: run upstream profiling. Save `available_gpu_memory`, `num_gpu_blocks`, `num_cpu_blocks` to `WarmupState`.
- **SAVE pass 2 / LOAD**: skip the profiling block. Use saved values for `num_gpu_blocks` / `num_cpu_blocks`.

Failure mode if broken: pass 2 / LOAD allocate different KV blocks than pass 1, drifting the cursor before capture begins.

### 2. Compile-warmup `_dummy_run` loop

Where: `vllm/v1/worker/gpu_worker.py::compile_or_warm_up_model`.

What it does: `for size in warmup_sizes: self.model_runner._dummy_run(size, ...)` to trigger inductor autotune and compile cache priming. `warmup_sizes` is derived from `compilation_config.compile_sizes` plus any `compile_ranges` not already covered by `cudagraph_capture_sizes`.

Rule: under the typical foundry config the loop iterates **zero times**, so no explicit skip is needed — `compile_sizes=[]`, and the single `compile_ranges_endpoints=[8192]` produces a range `[1, 8192]` that is already covered by `cudagraph_capture_sizes={1..512}`, so nothing gets appended to `warmup_sizes`. SAVE pass 1's compile cache is materialized through the actual graph capture (CUDAGraphWrapper runs inductor compile during the inner `capture_or_load_graph` call) instead.

If a user reconfigures `compile_sizes` so the loop becomes non-empty, those calls fall through to the real `_dummy_run` and SAVE/LOAD parity is the user's responsibility.

### 3. `kernel_warmup`

Where: `vllm/v1/worker/gpu_worker.py::kernel_warmup`.

What it does: priming FlashInfer / various kernels.

Rule: no-op on SAVE/LOAD. Their internal allocations don't survive a SAVE/LOAD cycle, so reproducing them on LOAD is a waste — and any non-determinism is forbidden.

### 4. Full-forward sampler warmup

Where: `vllm/v1/worker/gpu_worker.py::compile_or_warm_up_model` (after `capture_model`, line 691).

What it does: `self.model_runner._dummy_run(num_tokens=max_num_reqs, skip_eplb=True, cudagraph_runtime_mode=CUDAGraphMode.NONE)` (a real forward) followed by `_dummy_sampler_run(last_hidden_states)` (sampler warmup).

Rule: the **`_dummy_run` is replaced** with zero hidden states on SAVE/LOAD by the kwarg-signature match in `_patch_dummy_runs` — `cudagraph_runtime_mode=NONE` + `skip_eplb=True` + no other control flag uniquely identifies this caller. `_dummy_sampler_run` is **not** skipped; it runs on the zeros to populate sampler workspaces (probs, topk-topp).

Since `_patch_capture_model` calls `fops.stop_allocation_region()` before returning, the sampler workspace allocations created by this path land in the standard CUDA caching allocator, not the VMM region. That's fine — they don't need to be deterministic across SAVE/LOAD, only the captured-graph memory does.

Failure mode if `_dummy_sampler_run` is skipped entirely: sampler workspace not allocated at all, first real inference crashes when the sampler kernels are launched.

Failure mode if the `_dummy_run` patch is too broad — i.e. swallows other call sites that also lack `is_profile`/`is_graph_capturing`/`force_attention`: the DP empty-shard phantom at `gpu_worker.py:909` (`_dummy_run(num_tokens, uniform_decode=True)`) silently no-ops, the empty shard skips `coordinate_batch_across_dp`'s all-reduce, and the peer shard hangs forever on the CPU collective. See `claude-doc/invariants.md` §5.

### 5. Memory-history snapshotting / capture profiling

Off in foundry mode — `compilation_config.enable_profile_cuda_graph = False` etc.

### 6. `torch._inductor` pattern-matcher lazy init

Where: `torch._inductor.fx_passes.joint_graph.lazy_init` — fired the first time inductor compiles a graph.

What it does: `joint_graph.lazy_init` is `@init_once_fakemode` (`@functools.cache` keyed on `input_device`) and its body invokes `_sfdp_init` to build SDPA pattern templates. `_sfdp_init` constructs CPU constants and copies them onto the device — illegal inside a CUDA-graph capture window.

Rule: foundry's `prepare_graph_capture(metadata_builders, device)` (called pre-orig in `_patch_capture_model`) eagerly invokes `joint_graph.lazy_init(model_runner.device)`. Once retired, the in-capture compile path finds nothing left to initialize.

Why only `joint_graph` and not its three siblings in `torch._inductor.fx_passes.{pre_grad,post_grad,freezing_patterns}`: those bodies only call `register_replacement` for FX patterns; they don't allocate or copy device memory, so leaving them to inductor doesn't hurt. Pre-warming them actually breaks compile, because inductor's `post_grad_passes` invokes `lazy_init()` (no argument → `@functools.cache` key `None`) at compile time, which is a different cache key from our `lazy_init(cuda:0)`. Both bodies run, both try to `register_replacement("amax_default", …)`, and the second one fails:

```
torch._inductor.exc.InductorError: RuntimeError: Duplicate pattern: amax_default = ...
```

Failure mode if `joint_graph.lazy_init` pre-warm is broken — i.e. wrong device key (e.g. un-indexed `torch.device("cuda")` instead of `cuda:N`) or skipped entirely:

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```

raised from `_get_sfdp_patterns` inside the first in-capture compile. The cache key must match what inductor uses internally — pass the indexed device from `model_runner.device`.

### 7. NVSHMEM initialization

NVSHMEM modules are loaded on LOAD via `init_nvshmem_for_loaded_modules`, called from `_patch_prepare_comm_buffer`'s **post-orig hook** — immediately after `prepare_communication_buffer_for_model` bootstraps the NVSHMEM runtime via DeepEP `Buffer` construction. For dense models the pending list is empty and the call is a no-op.

Doing it earlier (inside `load_cuda_modules_and_libraries` or at the end of `start_graph_builds`) would be too early — the NVSHMEM runtime isn't bootstrapped yet, `nvshmemx_cumodule_init` silently returns 0, and replayed graphs later fail with `cuLaunchKernel` errors. See `claude-doc/invariants.md` §3.

## Silent mismatches

Cursor parity is necessary but not sufficient. The captured graph kernels reference raw pointers (e.g. into the FlashInfer workspace buffer). foundry's replay validates the cursor against `start_base_addr_X` for each graph, but it does not validate that those raw pointers resolve to the same data on LOAD as on SAVE.

The known silent-mismatch sources:

- **DeepEP communicator buffers initialized in different mode.** `_patch_deepep` forces `use_fabric=True`; without it LOAD may set up the buffer in NVL mode while SAVE used fabric, breaking the captured kernels.
- **NVSHMEM init fired before runtime is bootstrapped.** `init_nvshmem_for_loaded_modules()` must run after `prepare_communication_buffer_for_model` (which bootstraps the runtime via DeepEP `Buffer` creation). Called too early it silently no-ops, and captured kernels reference uninitialized NVSHMEM module state. Mitigated by the post-orig hook in `_patch_prepare_comm_buffer`.

If you see correct cursors but garbage output, those three are the first suspects.

## Why three lifecycle phases need explicit "skip on LOAD" rules

Because we want to save time.

The integration's rule of thumb: **if upstream code does a CUDA allocation, that allocation must happen identically on SAVE pass 2 and LOAD, or it must be skipped on both**.
