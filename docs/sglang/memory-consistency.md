# SGLang Memory-Consistency Post-Mortem

The hardest correctness problem in the SGLang integration was making LOAD produce the same output as SAVE. Cursors at every checkpoint matched, foundry's replay reported success, the graph launched cleanly — and the model emitted garbage tokens.

This doc walks every divergence we hit and the fix that landed. The common cause: anything that allocates on one path but not the other shifts the VMM cursor or changes the address an `_int_workspace_buffer` lands at, and the captured graph kernels read addresses fixed at SAVE time.

## The general invariant

**SAVE and LOAD must walk an identical VMM cursor trajectory.** Foundry validates `start_base_addr_X` for each captured graph; the validation only catches asymmetries up to the first divergence. Captured-kernel argument addresses (e.g. wrapper `_int_workspace_buffer.data_ptr()`) are **not** validated — they just point at whatever VMM offset SAVE recorded. If LOAD's allocator lands a different buffer at the same offset, kernels read it; if no allocation lands at the offset, kernels read the preallocated zero-initialized memory.

## Bug 1 — `CUDAGraph::capture_begin/end` toggling the VMM region

### Symptom

`replay_hook_events_from_json` errored on iter 2 with `Memory offset mismatch during replay`: current cursor 20 MB ahead of saved `start_base_addr_2`.

### Root cause

`foundry/csrc/CUDAGraph.cpp` used to bracket every capture with `resume_allocation_region()` / `stop_allocation_region()`. So on SAVE:

- iter 0 enters with region ON. bs=24 init allocates inside VMM; cursor advances.
- iter 0 `capture_end` → region OFF.
- iter 1 init runs with region OFF. Allocations bypass VMM, no cursor advance.
- iter 1 `capture_begin` → region ON; `start_base_addr_1` recorded.
- …

LOAD never calls `_capture_graph`, so nothing flipped the region off. LOAD's per-bs inits all advanced the cursor — past the saved `start_base_addr` values from iter 1 onward.

### Fix

Removed both calls in `CUDAGraph::capture_begin/end`. Only the hook recording window (`start_hook_record` / `end_hook_record`) brackets each capture. Region toggling is now each integration's responsibility, and both SGLang and vLLM keep the region ON across the whole capture loop.

(vLLM initially tried a Python-side `resume/stop_allocation_region` bracket around each capture to suppress the `final_alloc_offset` inflation seen on Qwen3-30B-A3B EP2 — ~194 GB observed vs ~80 GB actual. That was wrong: any tensor a between-capture allocation produced that was later referenced by a captured kernel would have ended up outside the deterministic region, so LOAD's preallocation couldn't recreate it at the right address. The actual root cause was the `empty_cache()` inside `foundry/graph.py:CUDAGraph.__enter__`, which dropped torch caching-allocator segments and forced subsequent inter-capture allocations to take a fresh `cuMemAlloc_v2` path. Removing it brought `final_alloc_offset` back down to ~80 GB. The vLLM integration now also wraps the whole capture loop in `torch.cuda.use_mem_pool(graph_pool)` so SAVE-2 ↔ LOAD caching state stays symmetric for the per-graph `finish_one_graph_load` replay path.)

## Bug 2 — Pre-capture warmup forwards on SAVE only

### Symptom

After bug 1: iter 2 still drifted 20 MB.

### Root cause

`CudaGraphRunner.capture_one_batch_size` runs:

```python
for _ in range(2):
    self.device_module.synchronize()
    self.model_runner.tp_group.barrier()
    run_once()
```

before `_capture_graph`. These forwards allocate activation tensors and immediately free them. Torch's caching allocator keeps the freed segments as cached free blocks. Later per-bs init allocations may hit or miss cache depending on segment fragmentation history — and the cache-hit pattern is not reproducible across SAVE and LOAD.

LOAD never runs these warmups → different cache state → different cache hit/miss decisions → cursor drift.

### Fix

In `patched_capture_one_batch_size` wrap `forward` with a 3-call counter; first two invocations return `None`, the third (inside `_capture_graph`) runs the real forward. The `synchronize` and `barrier` calls in the loop are harmless. JIT and autotune that the warmups normally trigger now run inside the captured graph and are recorded as alloc events — foundry replays them verbatim on LOAD.

## Bug 3 — Per-bs init outside graph capture

### Symptom

After bugs 1 and 2: cursors matched everywhere, no foundry-side error, but the model produced garbage tokens.

### Root cause

`init_forward_metadata_capture_cuda_graph(bs, ...)` is called by `capture_one_batch_size` **before** `_capture_graph`, so its allocations are not recorded in the captured graph's allocator events. Each call creates a new `BatchDecodeWithPagedKVCacheWrapper`; the wrapper's constructor allocates `_int_workspace_buffer` via `torch.empty(...)`. The captured graph kernels reference `_int_workspace_buffer.data_ptr()` directly.

For LOAD's wrapper to land at the same VMM address as SAVE's, the cursor at each iter's init time must match. With per-iter init alone, that depends on caching-allocator behavior, which is non-deterministic even after bug 2.

### Fix (two parts)

**Both sides**: pre-pass init for every bs in `reversed(capture_bs)` order before the capture / load loop. `initialize_all_attention_metadata` walks the bs list and calls `initialize_attention_metadata_for_bs` for each. Same call sequence on both sides → same cursor trajectory → same wrapper addresses.

**SAVE only**: the upstream `capture_one_batch_size` still calls the inner `init_forward_metadata_capture_cuda_graph(bs, ...)` per iter. Replaced it with `reuse_pre_pass_init` which:

- Looks up the existing wrapper from `decode_cuda_graph_metadata[bs]`.
- Re-runs `indices_updater_decode.update(...)` with the same buffer slices — idempotent write to the same `_int_workspace_buffer`.
- Sets `attn_backend.forward_metadata = DecodeMetadata(wrappers)` for the current iter.

No allocation. Captured graph references the pre-pass wrapper's address; LOAD's identical pre-pass produces a wrapper at the same address.

(Two earlier approaches didn't work: relying on torch's caching allocator to reuse a freed segment after popping the dict entry, and forcing GC. The caching allocator doesn't deterministically reuse segments when request sizes don't exactly match cached blocks. Explicit reuse via dict lookup is the only reliable path.)

The `attn_backend.forward_metadata = None` between the pre-pass and the patch install drops the last bs's `DecodeMetadata` reference — defensive, not strictly required by `reuse_pre_pass_init`.

## Bug 4 — Per-graph `start_graph_builds` broke template/on-demand linking

### Symptom

After bugs 1-3: cursors matched perfectly, no foundry replay error — but the runtime crashed at first inference with `Called CUDAGraph::replay without a preceding successful capture or load`.

### Root cause

`graph_manifest.json` groups captured graphs by topology. One graph per group is built as a **template** (full cuGraph + cuGraphExec instantiated); the rest are **on-demand** graphs that link to the template's `shared_exec` and patch in per-graph node updates at link time.

An earlier (interleaved) LOAD called `start_graph_builds([single_path])` once per graph. Every call had 1 input → 1 topology group → graph was its own template. But the manifest marks graphs 2-N as on-demand sharing graph 0's or graph 1's template. Those templates were loaded in earlier `start_graph_builds` calls and no longer in scope — so the on-demand graphs ended up with `shared_exec=null`. Runtime `CUDAGraph::replay` takes the on-demand path when `on_demand_data_` is set; with `shared_exec=null` it falls through to a `capture_ended_ || has_graph_exec_` check that errors.

### Fix

`load_all_graphs(self)` calls `start_graph_builds(all_paths)` exactly once, then `finish_graph_loads(pending)` once. All N graphs go through the manifest's template/on-demand linking in a single pass. `finish_graph_loads` then replays each graph's allocator events sequentially, walking the cursor through every `start_base_addr_X` in order.

## Bug 5 — `_resolve_memory_pool_config`'s hidden `empty_cache`

### Symptom

After bugs 1-4: cursors at `after_setup_graph_ext`, `after_init_torch_dist`, `after_scratch_skip`, `before_init_memory_pool`, and `after_init_memory_pool` all matched between SAVE and LOAD. But `save_before_pre_init` (29698 MB on SAVE) was 20 MB ahead of LOAD's `before_preallocate` (29678 MB).

### Root cause

SAVE's `init_memory_pool` calls `_resolve_memory_pool_config` → `_profile_available_bytes` → `get_available_gpu_memory(...)` (in `sglang/srt/utils/common.py`), which calls `torch.cuda.empty_cache()`. `empty_cache` releases torch caching-allocator segments back to the driver via `cuMemFree_v2`, which the foundry hook unmaps.

LOAD's `_patch_init_memory_pool` skipped upstream profiling entirely and went directly to `_apply_memory_pool_config` — no equivalent drain. Caching allocator on LOAD retained segments that SAVE released, so the subsequent attention-backend init (between `init_memory_pool` and `capture()`) took a different `cuMemAlloc` path on each side. 20 MB drift by the time `capture()` was reached.

### Fix

One line in `_patch_init_memory_pool`'s LOAD branch:

```python
torch.cuda.empty_cache()
self._apply_memory_pool_config(self.memory_pool_config)
```
