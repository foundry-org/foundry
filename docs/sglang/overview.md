# SGLang Integration Overview

Foundry persists SGLang's `CudaGraphRunner` graphs to disk on SAVE and restores them on LOAD, skipping graph capture, kernel warmup, and the per-batch-size attention metadata setup costs.

Tested on single-GPU Qwen3-1.7B / 4B / 14B with the FlashInfer attention backend.

## How to use

### 1. Install foundry

```bash
pushd foundry && pip install -e . --no-build-isolation && popd
```

Or run from source; the serve scripts in `experimental/single-gpu/` already export `PYTHONPATH=…/foundry/python:…/sglang/python` so an editable install is not strictly required.

### 2. Write a TOML config

```toml
# experimental/single-gpu/save_qwen_1.7b.toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive_qwen_1.7b"
scratch_space_size = "1024MB"
```

The matching `load_qwen_1.7b.toml` just changes `mode = "load"`. See [`memory-lifecycle.md`](memory-lifecycle.md) for what each field controls.

### 3. Run SAVE, then LOAD

```bash
# SAVE
rm -rf foundry_archive_qwen_1.7b
bash experimental/single-gpu/serve_sglang_qwen_1.7b.sh --save
# Wait for "Application startup complete", then SIGTERM the server.

# LOAD
bash experimental/single-gpu/serve_sglang_qwen_1.7b.sh --load

# Query
curl -s http://0.0.0.0:12000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","prompt":"The capital of France is",
       "max_tokens":12,"temperature":0}'
# → "Paris. The capital of the United States is Washington, D"
```

A single SAVE pass is enough — SGLang doesn't run a profile-forward at startup, so there is no non-determinism that requires a second pass.

The scripts pre-load `libcuda_hook.so` from the foundry build via `LD_PRELOAD` and propagate it (and `CGE_MODE`) into every child process via `setup_ld_preload_env()`.

## What the integration does

SAVE:

1. `setup_graph_extension(...)` reserves a VMM region at `base_addr` and creates the per-rank workspace.
2. Distributed init / NCCL warmup runs in scratch space; the cursor is then forced to `scratch_space_size`.
3. Model weights, KV pool, and FlashInfer workspace buffers allocate inside the VMM region at byte-deterministic offsets.
4. `kernel_warmup` is a no-op.
5. `CudaGraphRunner.capture` runs a pre-pass that pre-allocates every per-bs FlashInfer wrapper, then enters the upstream capture loop with an idempotent inner-init shim (`reuse_pre_pass_init`) and a wrapper on `forward` that suppresses the two pre-capture warmup forwards.
6. Each captured graph is written to disk; a manifest groups topologically equivalent graphs.
7. The final VMM cursor is recorded as `final_alloc_offset`.

LOAD:

1. `setup_graph_extension(...)` restores the VMM region and replays captured fatbins into device code memory.
2. Distributed init runs as usual; the cursor advances to the same `scratch_space_size`.
3. Model weights and KV pool re-allocate at the same deterministic offsets. `init_memory_pool` reuses the saved `MemoryPoolConfig` (and calls `torch.cuda.empty_cache()` to mirror SAVE's `_resolve_memory_pool_config` side effect).
4. `CudaGraphRunner.capture` is replaced with: preallocate the entire deterministic range up to `final_alloc_offset`; run the same pre-pass init; call `start_graph_builds(all_paths) + finish_graph_loads(pending)` exactly once. All N graphs are loaded in one shot so the manifest's template/on-demand linking works.
5. `self.graphs` / `self.output_buffers` are populated from `state.loaded_graphs`; the rest of SGLang's serving path runs unchanged.

## Doc set

- [`overview.md`](overview.md) — this file
- [`direct-edits.md`](direct-edits.md) — every line that changed in `sglang/`
- [`hooks.md`](hooks.md) — every monkey-patch foundry installs and what it does on SAVE / LOAD
- [`memory-lifecycle.md`](memory-lifecycle.md) — VMM region setup, the SAVE↔LOAD allocation parity contract, and the `final_alloc_offset` watermark
- [`save-load-workflow.md`](save-load-workflow.md) — serve scripts, TOML schema, expected logs, validation checks
- [`memory-consistency.md`](memory-consistency.md) — the five known divergences that caused silent LOAD failures and how each was fixed

