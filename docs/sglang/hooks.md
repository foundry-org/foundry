# SGLang Hook Surface

Every monkey-patch `install_hooks(server_args)` installs.

## Patches installed by `install_hooks`

In install order:

1. `_patch_init_torch_distributed`
2. `_patch_init_memory_pool`
3. `_patch_load_model`
4. `_patch_kernel_warmup`
5. `_patch_cuda_graph_capture`
6. `_patch_spawn_sites`

> Removed: an earlier `_patch_model_runner_init` wrapped `ModelRunner.__init__` to stash `dp_rank` on `self` (upstream didn't expose it as an attribute). Upstream sglang now does `self.dp_rank = dp_rank` in the constructor, so our wrapper is no longer needed; `_resolve_dp_rank` reads `self.dp_rank` directly.

### 1. `ModelRunner.init_torch_distributed`

Before upstream: `setup_graph_extension(...)` reserves the VMM region, loads cached fatbins (LOAD only via `load_cuda_modules_and_libraries`), and eagerly initializes the cuBLAS handle into scratch space.

After upstream: `skip_to_scratch_boundary()` forces the cursor to `cfg.scratch_space_size`.

```
[Foundry] SGLang alloc_offset[after_setup_graph_ext]=…
[Foundry] SGLang alloc_offset[after_init_torch_dist]=…
[Foundry] SGLang alloc_offset[after_scratch_skip]=…
```

### 2. `ModelRunnerKVCacheMixin.init_memory_pool`

- **SAVE**: call upstream `init_memory_pool` unchanged; after it returns, serialize the resolved `MemoryPoolConfig` (via `dataclasses.asdict`) into `warmup_state.json`.
- **LOAD**: skip upstream profiling. Load `MemoryPoolConfig` from `warmup_state.json`. Call `torch.cuda.empty_cache()` to mirror SAVE's `_resolve_memory_pool_config → get_available_gpu_memory(empty_cache=True)` side effect. Then call `_apply_memory_pool_config(config)` directly.

The `empty_cache()` mirror is load-bearing — see [`memory-consistency.md`](memory-consistency.md) §5.

### 3. `ModelRunner.load_model`

Currently a passthrough wrapper. Kept as a future hook point for a LOAD-time `start_graph_builds(all_paths)` overlap with weight IO; the current LOAD path is fast enough (~80 ms total) that overlap isn't needed.

### 4. `ModelRunner.kernel_warmup`

No-op on SAVE/LOAD:

```
[Foundry] SGLang kernel_warmup skipped in <mode> mode
```

The FlashInfer autotune path inside is shut off both by this no-op and by the forced `disable_flashinfer_autotune` flag.

### 5. `CudaGraphRunner` (the largest patch)

Four sub-patches:

#### 5a. `_create_device_graph`

On SAVE returns `foundry.CUDAGraph()` instead of `torch.cuda.CUDAGraph()`.

#### 5b. `_capture_graph`

On SAVE enters `foundry.graph(graph, pool=pool, stream=stream)` instead of `torch.cuda.graph(...)`, captures the `run_once` callable's allocations into foundry's hook event log.

#### 5c. `capture_one_batch_size`

On SAVE wraps the `forward` callable with a 3-call counter:

```python
counter = [0]; real_forward = forward
def warmup_skipping_forward(*args, **kwargs):
    counter[0] += 1
    if counter[0] <= 2:
        return None
    return real_forward(*args, **kwargs)
forward = warmup_skipping_forward
```

This suppresses the two pre-capture warmup forwards SGLang does in `for _ in range(2): run_once()`. Only the third invocation — inside `_capture_graph`'s graph capture context — runs the real forward. JIT and autotune allocations that those warmups would normally trigger now happen inside the captured graph and become foundry alloc events that replay verbatim on LOAD. (See `memory-consistency.md` §2.)

After upstream returns, calls `save_graph(graph, output, key)`. The key matches the inline shape upstream uses for `self.graphs[key]` in `_capture_one_stream`:

```python
key = bs if stream_idx is None else f"{stream_idx}_{bs}"
```

(Sglang removed the `_make_graph_key` / `get_capture_lora_variant` helpers in commit `ce2506e1c`, the same commit that deprecated `record_nolora_graph` dual MoE graph capture.)

#### 5d. `capture`

The outermost replacement.

**SAVE**:

```python
initialize_all_attention_metadata(self)             # pre-pass
attn_backend.forward_metadata = None
real_init = attn_backend.init_forward_metadata_capture_cuda_graph
attn_backend.init_forward_metadata_capture_cuda_graph = reuse_pre_pass_init
try:
    result = orig_capture(self, *args, **kwargs)    # upstream capture loop
finally:
    attn_backend.init_forward_metadata_capture_cuda_graph = real_init
save_graph_manifest()
pack_fatbins()
capture_final_alloc_offset()
```

`initialize_all_attention_metadata` walks `reversed(self.capture_bs)` and pre-allocates every per-bs FlashInfer wrapper. The wrappers are stored in `attn_backend.decode_cuda_graph_metadata[bs]`.

`reuse_pre_pass_init` is a drop-in replacement for the upstream inner init that runs inside `capture_one_batch_size`. For decode mode it:

1. Looks up the pre-pass wrapper from `decode_cuda_graph_metadata[bs]`.
2. Re-runs `indices_updater_decode.update(...)` with the same buffer slices (idempotent — writes plan info to the same `_int_workspace_buffer`).
3. Sets `attn_backend.forward_metadata = DecodeMetadata(wrappers)` so the captured forward sees the right metadata for this iter.

No allocation. The captured graph references the pre-pass wrapper's address; LOAD's pre-pass produces a wrapper at the same address.

**LOAD**:

```python
if cgr.get_global_graph_memory_pool() is None:
    cgr.set_global_graph_memory_pool(self.device_module.graph_pool_handle())
set_graph_pool_id(cgr.get_global_graph_memory_pool())
preallocate_for_load_mode()                         # cuMemCreate+cuMemMap up to final_alloc_offset
initialize_all_attention_metadata(self)             # pre-pass (same as SAVE)
load_all_graphs(self)                               # ONE start_graph_builds + finish_graph_loads
self.graphs = {k: v[0] for k, v in state.loaded_graphs.items()}
self.output_buffers = {k: v[1] for k, v in state.loaded_graphs.items()}
```

`load_all_graphs` calls `start_graph_builds(all_paths)` and `finish_graph_loads(pending)` exactly once each. This is required for the manifest's template + on-demand linking to work; per-graph `start_graph_builds([single_path])` calls would leave on-demand graphs without a `shared_exec`, and runtime replay would abort with `Called CUDAGraph::replay without a preceding successful capture or load`.

### 6. Spawn-site patches

Two parent-side wrappers:

```python
# Engine._launch_scheduler_processes
def patched_launch(self, *args, **kwargs):
    if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
        rt.setup_ld_preload_env()
    return orig_launch(self, *args, **kwargs)

# DataParallelController.launch_tensor_parallel_group
def patched_start(self, *args, **kwargs):
    if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
        rt.setup_ld_preload_env()
    return orig_start(self, *args, **kwargs)
```

`setup_ld_preload_env()` prepends `libcuda_hook.so` (and optionally `libnvshmem_host.so`) to `os.environ["LD_PRELOAD"]`, sets `CGE_MODE`, and records a wall-clock marker. All children spawned from these methods inherit the env.

## Patch idiom

All patches use the `wrap-and-call` idiom — short-circuit on `mode == NONE`, run foundry pre-work, call `orig`, run foundry post-work. The single exception is `capture()` on LOAD, which replaces the upstream method entirely (does not call `orig`).

## Install order

`install_hooks` calls the patch helpers in the listed order. Order doesn't matter for correctness here — every patch attaches to a different attribute — but spawn sites go last so the install-completion log line appears after every other patch has registered.
