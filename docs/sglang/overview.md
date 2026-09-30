# SGLang Integration Overview

Foundry persists SGLang's decode (and optionally prefill) CUDA graphs to disk on SAVE and restores them on LOAD, skipping graph capture, kernel warmup, and the per-batch-size attention metadata setup costs.

Foundry runs as an **SGLang plugin**: `pip install -e foundry` registers the entry point `foundry = "foundry_sglang_plugin:load"` in group `sglang.srt.plugins`, and `FOUNDRY_GRAPH_EXTENSION_CONFIG=<TOML>` in the launcher's environment switches it on. No SGLang change is needed (validated on upstream main `fa090f7755`, fork branch `foundry-plugin`). The plugin wraps three whitelisted resolution steps to pin the graph settings and installs the runtime patches in [`hooks.md`](hooks.md) in the launcher and in every scheduler process. The earlier in-tree route (fork flag `--foundry-graph-extension-config-path`, [`direct-edits.md`](direct-edits.md)) is superseded.

## Parallelism

| Mode | Status | Notes |
|---|:---:|---|
| Single GPU | ✅ | Qwen3-1.7B / 4B / 14B |
| Data parallel (DP) | ✅ | One full replica per rank; validated DP=2. Requires the per-rank device binding (below) and `NCCL_CUMEM_ENABLE=0` / `NCCL_NVLS_ENABLE=0`. |
| Tensor parallel (TP) | ✅ | torch symmetric-memory allreduce inside the decode graphs (`--enable-torch-symm-mem --disable-custom-all-reduce`); validated TP=2 (Qwen3-1.7B, Qwen3-32B) and TP=4 (Qwen3-32B). Without multicast (no IMEX channels) LOAD needs the optional sglang branch `symm-mem-no-multicast-fallback` (two-shot instead of disabling the communicator). |
| Expert parallel (DeepEP) | ✅ | DP attention + DeepEP low-latency, TP attention + EP, and attention-TP inside DP groups + EP; DeepEP v2. Validated up to EP8 (Qwen3.5-122B/397B, DeepSeek-V4-Flash, GLM, Inkling; see `recipe/sglang/README.md` **Validation**). See **Expert parallel** below. |

**Expert parallel (DeepEP).** The default EP recipe runs DP-attention + DeepEP for the
MoE all-to-all (NVSHMEM, foundry-compatible); TP attention + EP (`serve_qwen3-30ba3b_ep_tpattn.sh`,
the `tpepN` / `tpAepN` topologies of `serve_common.sh`) routes the attention all-reduce
through torch symmetric memory. The serve script is `recipe/sglang/serve_qwen3-30ba3b_ep.sh
<ep_size> [--save|--load]` with: `--enable-dp-attention --moe-a2a-backend deepep
--deepep-mode low_latency --moe-runner-backend deep_gemm --attention-backend fa3
--disable-custom-all-reduce`. The EP kernels (`sgl-deep-ep`, `sgl-deep-gemm`, fa3 inside
`sglang-kernel`) come as wheels with the sglang install. `fa3` is used because the
flashinfer ragged-prefill path has an off-by-one (`q.shape != qo_indptr`) under this
config. DeepEP low-latency caps dispatch at
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default 128); raise it (+ chunk
prefill) for larger batches and keep it identical across SAVE/LOAD. Foundry-specific
EP handling is in [`hooks.md`](hooks.md): pre-capture bootstraps (NCCL communicators,
the DeepEP / DeepEP v2 / Mooncake buffer, the logits all-gather state), a SAVE-side
runtime-init bootstrap (the two one-time inits capture rejects run outside the capture
stream; compiles and JIT loads happen inside it), and a
C++ fix binding the CUDA context on the graph-build pool workers.

**Per-rank device binding (DP/TP/EP).** Foundry's `set_allocation_region` binds the
VMM region to the CUDA device current at call time. Upstream sets the device
*inside* the distributed bring-up (`bootstrap.init_parallel_runtime` on main,
`ModelRunner.init_torch_distributed` on the fork base), which the integration wraps and front-runs, so
the hook explicitly calls `set_device(self.gpu_id)` before reserving the region —
otherwise rank > 0 reserves on `cuda:0` and faults. See [`hooks.md`](hooks.md) (patch 1).
The DP serve script lives at `recipe/sglang/serve_qwen3-1.7b_dp.sh`
(`<dp_size> [--save|--load]`); pick GPUs with `CUDA_VISIBLE_DEVICES`.

## How to use

### 1. Install foundry

```bash
pushd foundry && pip install -e . --no-build-isolation && popd
```

Install it into the venv that runs `sglang serve`. The install registers the plugin entry point; a `PYTHONPATH` source checkout does not, and SGLang then runs natively without an error even with `FOUNDRY_GRAPH_EXTENSION_CONFIG` set (as it does when `SGLANG_PLUGINS` is set without `foundry`). Check with `python -m foundry.integration.sglang.preflight --toml <toml> --save|--load`, which the recipe scripts run before every SAVE / LOAD.

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
# SAVE (the script exports FOUNDRY_GRAPH_EXTENSION_CONFIG=recipe/sglang/foundry_save.toml)
rm -rf foundry_archive
bash recipe/sglang/serve_qwen3-mini.sh --save
# Wait for "[Foundry verify] plugin active in N process(es)", then SIGTERM the server.

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

(This outline predates the move to per-phase runners; [`hooks.md`](hooks.md) lists the current patch points.)

SAVE:

1. `setup_graph_extension(...)` reserves a VMM region at `base_addr` and creates the per-rank workspace.
2. Distributed init / NCCL warmup runs in scratch space; the cursor is then forced to `scratch_space_size`.
3. Model weights, KV pool, and FlashInfer workspace buffers allocate inside the VMM region at byte-deterministic offsets.
4. `kernel_warmup` is a no-op.
5. `DecodeCudaGraphRunner.capture` runs the upstream capture loop unchanged (per-shape FlashInfer wrappers and other metadata allocate where sglang puts them); only `FullCudaGraphBackend.capture_one` is patched, to capture into a foundry graph without the two pre-capture warmup forwards.
6. Each captured graph is written to disk; a manifest groups topologically equivalent graphs.
7. The final VMM cursor is recorded as `final_alloc_offset` in `region_layout.json`, with `capture_loop_version = 2`.

LOAD:

1. `setup_graph_extension(...)` restores the VMM region and replays captured fatbins into device code memory; an archive without `capture_loop_version = 2` is refused here (re-SAVE).
2. Distributed init runs as usual; the cursor advances to the same `scratch_space_size`.
3. Model weights and KV pool re-allocate at the same deterministic offsets. `init_memory_pool` reuses the saved `MemoryPoolConfig` (and calls `torch.cuda.empty_cache()` to mirror SAVE's `_resolve_memory_pool_config` side effect).
4. `DecodeCudaGraphRunner.capture` preallocates the deterministic range up to `final_alloc_offset`, starts all decode graph builds in one `start_graph_builds(all_paths)` call (the manifest's template/on-demand linking needs one call), then runs the same upstream capture loop as SAVE. The patched `capture_one` takes the next archived graph for each shape (`finish_one_graph_load`: its allocator events replay where SAVE captured it) instead of capturing; `forward_fn` never runs.
5. The restored graphs sit in the backend's `_graphs` / `_outputs` under the loop's own shape keys; the rest of SGLang's serving path runs unchanged.

## Doc set

- [`overview.md`](overview.md) — this file
- [`direct-edits.md`](direct-edits.md) — the superseded in-tree route's edits to `sglang/` (not needed with the plugin)
- [`hooks.md`](hooks.md) — every monkey-patch foundry installs and what it does on SAVE / LOAD
- [`memory-lifecycle.md`](memory-lifecycle.md) — VMM region setup, the SAVE↔LOAD allocation parity contract, and the `final_alloc_offset` watermark
- [`save-load-workflow.md`](save-load-workflow.md) — serve scripts, TOML schema, expected logs, validation checks
- [`memory-consistency.md`](memory-consistency.md) — the five known divergences that caused silent LOAD failures and how each was fixed
