# SGLang Integration Overview

Foundry persists SGLang's `CudaGraphRunner` graphs to disk on SAVE and restores them on LOAD, skipping graph capture, kernel warmup, and the per-batch-size attention metadata setup costs.

Tested on single-GPU Qwen3-1.7B / 4B / 14B and **data-parallel (DP=2)** Qwen3-1.7B with the FlashInfer attention backend.

## Parallelism

| Mode | Status | Notes |
|---|:---:|---|
| Single GPU | ✅ | Qwen3-1.7B / 4B / 14B |
| Data parallel (DP) | ✅ | One full replica per rank; validated DP=2. Requires the per-rank device binding (below) and `NCCL_CUMEM_ENABLE=0` / `NCCL_NVLS_ENABLE=0`. |
| Tensor parallel (TP) | 🚧 | Deterministic NCCL memory layout is under construction. |
| Expert parallel (DeepEP) | ✅ | Validated EP=2 on Qwen3-30B-A3B-FP8 (SAVE/SAVE2/LOAD/query); restored decode graphs match baseline throughput. See **Expert parallel** below. |

**Expert parallel (DeepEP).** EP runs DP-attention (each rank its own attention — no
NCCL all-reduce) + DeepEP for the MoE all-to-all (NVSHMEM, foundry-compatible). The
serve script is `recipe/sglang/serve_qwen3-30ba3bfp8_ep.sh
<ep_size> [--save|--load]` with: `--enable-dp-attention --moe-a2a-backend deepep
--deepep-mode low_latency --moe-runner-backend deep_gemm --attention-backend fa3
--disable-custom-all-reduce`. Required kernel builds in the env: `deep_ep` at sglang's
pinned commit (`9af0e0d`, not vLLM's), `sgl-deep-gemm>=0.1.2` (0.1.0 lacks
`m_grouped_bf16_gemm_nt_masked`), `flash-attn-3`. `fa3` is required because the
flashinfer ragged-prefill path has an off-by-one (`q.shape != qo_indptr`) under this
config. DeepEP low-latency caps dispatch at
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default 128); raise it (+ chunk
prefill) for larger batches and keep it identical across SAVE/LOAD. Foundry-specific
EP handling is in [`hooks.md`](hooks.md): a DeepEP buffer pre-capture bootstrap, a
SAVE-only warmup pass (triggers DeepGEMM JIT + buffer creation outside the capture
stream), `deepep_adapter` mode init on LOAD, the FlashInfer pre-pass gated off for
fa3, and a C++ fix binding the CUDA context on the graph-build pool workers.

**Per-rank device binding (DP/TP/EP).** Foundry's `set_allocation_region` binds the
VMM region to the CUDA device current at call time. Upstream sets the device
*inside* `init_torch_distributed`, which the integration wraps and front-runs, so
the hook explicitly calls `set_device(self.gpu_id)` before reserving the region —
otherwise rank > 0 reserves on `cuda:0` and faults. See [`hooks.md`](hooks.md) §1.
The DP serve script lives at `recipe/sglang/serve_qwen3-1.7b_dp.sh`
(`<dp_size> [--save|--load]`); pick GPUs with `CUDA_VISIBLE_DEVICES`.

## How to use

### 1. Install foundry

```bash
pushd foundry && pip install -e . --no-build-isolation && popd
```

An editable install (`pip install -e .`) is the supported path — the recipe serve scripts set no `PYTHONPATH` and rely on `foundry` being importable. To run straight from a source checkout, export `PYTHONPATH=…/foundry/python:…/sglang/python` yourself.

### 2. Write a TOML config

```toml
# recipe/sglang/foundry_save.toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive"
scratch_space_size = "1024MB"
```

The matching `foundry_load.toml` just changes `mode = "load"`. See [`memory-lifecycle.md`](memory-lifecycle.md) for what each field controls.

### 3. Run SAVE, then LOAD

```bash
# SAVE
rm -rf foundry_archive
bash recipe/sglang/serve_qwen3-mini.sh --save
# Wait for "Application startup complete", then SIGTERM the server.

# LOAD
bash recipe/sglang/serve_qwen3-mini.sh --load

# Query
curl -s http://0.0.0.0:12000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","prompt":"The capital of France is",
       "max_tokens":12,"temperature":0}'
# → "Paris. The capital of the United States is Washington, D"
```

A single SAVE pass is enough — SGLang doesn't run a profile-forward at startup, so there is no non-determinism that requires a second pass.

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
