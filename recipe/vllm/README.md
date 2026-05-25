# Foundry recipe — vLLM

End-to-end serve scripts for SAVE / LOAD of CUDA graphs through the foundry vLLM integration. All scripts in this directory share the same pair of foundry TOML files (`foundry_save.toml` / `foundry_load.toml`) — pick a script for your model + parallelism, run `--save` twice (vLLM requires a two-pass save for deterministic memory layout), then `--load`, then query.

## Files in this directory

```
recipe/vllm/
├── README.md                         # this file
├── foundry_save.toml                 # shared SAVE config (workspace_root = "foundry_archive")
├── foundry_load.toml                 # shared LOAD config (same workspace_root)
├── serve_qwen3-mini.sh               # Qwen3-1.7B           single GPU
├── serve_qwen3-14b_dp.sh             # Qwen3-14B            data parallel
├── serve_qwen3-30ba3b_ep.sh          # Qwen3-30B-A3B (MoE)  expert parallel (DeepEP)
└── serve_qwen3-30ba3bfp8_ep.sh       # Qwen3-30B-A3B FP8    expert parallel (DeepGEMM)
```

Every script accepts the same trailing `--save` / `--load` flag. Scripts that scale across GPUs take the parallel-size as the first positional argument:

```bash
bash serve_qwen3-mini.sh              [--save|--load]
bash serve_qwen3-14b_dp.sh        <dp_size> [--save|--load]
bash serve_qwen3-30ba3b_ep.sh     <ep_size> [--save|--load]
bash serve_qwen3-30ba3bfp8_ep.sh  <ep_size> [--save|--load]
```

Because the two TOMLs are shared (single `workspace_root = "foundry_archive"`), one archive is written per host and switching between scripts re-uses or overwrites it — run a fresh `rm -rf foundry_archive` whenever you change models or topology before SAVE pass 1.

## Installation

The recipes assume the standard workspace layout — `vllm/` (the foundry-org vLLM fork) and `foundry/` (this repo) live as siblings under a single working directory:

```
<workspace>/
├── foundry/                # this repo (foundry-org/foundry)
│   ├── python/foundry/     # `pip install -e .` builds libcuda_hook.so here
│   ├── recipe/vllm/        # <-- you are here
│   └── ...
└── vllm/                   # foundry-org/vllm fork (with direct edits applied)
```

### 1. Create a clean Python env

```bash
conda create -p venv/ python=3.12
conda activate venv/
conda install -c conda-forge boost-cpp boost     # foundry C++ deps
pip install cmake                                 # must be cmake 4.0.0+; re-enter the venv to verify
pip install uv
```

### 2. Install vLLM (do this BEFORE foundry — vLLM may overwrite torch)

```bash
git clone https://github.com/foundry-org/vllm.git
cd vllm
git checkout foundry
VLLM_USE_PRECOMPILED=1 uv pip install --editable . \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130
cd ..
```

Verify the install with `uv pip list`:

```
nvidia-cublas    13.1.0.3
torch            2.11.0(+cu130)
vllm             0.1.dev15646+g040974074.precompiled
```

### 3. Install Foundry

```bash
git clone https://github.com/foundry-org/foundry.git
cd foundry
git checkout bump
uv pip install -e . --no-build-isolation
uv pip install pytest
pytest tests/
cd ..
```

If `pytest` fails with `ImportError: libboost_json.so.1.85.0: cannot open shared object file`, the conda boost path is missing from `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Or make it a venv hook so every activate picks it up:
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
  > $CONDA_PREFIX/etc/conda/activate.d/ld_path.sh
```

### 4. (EP scripts only) Build NVSHMEM for DeepEP

The EP scripts depend on DeepEP, which needs `libnvshmem_host.so` pre-loaded alongside the foundry hook. Build it from the vLLM repo's helper:

```bash
cd vllm/tools/ep_kernels
# Hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# Blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
cd ../../..
```

That will compile and install DeepEP module and put its shared lib in `vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem/lib/libnvshmem_host.so`

Then uncomment the `nvshmem_host_path=` in `foundry_save.toml` / `foundry_load.toml` and points to your `libnvshmem_host.so`.

## Workflow

vLLM's `_initialize_kv_caches` does a real-forward memory-profile pass on the first run, so SAVE must run **twice** for a deterministic memory layout:

```bash
# 0. Fresh start
rm -rf foundry_archive

# 1. SAVE pass 1 — runs memory_profiling (real forward), captures graphs
bash serve_qwen3-mini.sh --save
# wait for "Application startup complete", then Ctrl-C

# 2. SAVE pass 2 — skips memory profiling, re-captures graphs at
#    deterministic VMM offsets
bash serve_qwen3-mini.sh --save
# wait for "Application startup complete", then Ctrl-C

# 3. LOAD — preallocates the same VMM range, reloads modules, replays graphs
bash serve_qwen3-mini.sh --load
# leave running

# 4. Query (separate shell)
bash ../../../experimental/query.sh 12000 Qwen/Qwen3-1.7B
```

For DP / EP, prepend the parallel size:

```bash
rm -rf foundry_archive
bash serve_qwen3-30ba3b_ep.sh 2 --save     # SAVE pass 1, EP=2
bash serve_qwen3-30ba3b_ep.sh 2 --save     # SAVE pass 2
bash serve_qwen3-30ba3b_ep.sh 2 --load
bash ../../../experimental/query.sh 12000 Qwen/Qwen3-30B-A3B
```

## Archive layout

Every SAVE writes into `foundry_archive/` next to the script's working directory:

```
foundry_archive/
├── warmup_state.json              # shared: KV-block sizing + final_alloc_offset
└── rank_<N>/
    ├── graph_*.json + .cugraph    # one pair per captured graph
    ├── graph_manifest.json        # topology groups + template assignments
    ├── fatbin_image_packed.img    # packed CUDA modules
    ├── fatbin_entrypoint_packed.txt
    └── final_alloc_offset.json    # per-rank VMM watermark
```

For TP / DP / EP each rank gets its own `rank_<N>/` subdirectory; `warmup_state.json` is written by rank 0 only.

## What the scripts configure for you

All scripts, when invoked with `--save` or `--load`, export one env-var override in that branch (baseline-vLLM runs leave it unset):

- `VLLM_USE_V2_MODEL_RUNNER=0` — pin the V1 model runner. Foundry's monkey-patches are on `vllm.v1.worker.gpu_model_runner.GPUModelRunner`; vLLM's `VllmConfig.use_v2_model_runner` property silently routes architectures in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (currently `Qwen3ForCausalLM`) to `vllm.v1.worker.gpu.model_runner` (V2), which our patches don't touch. Without this pin, `capture_model` runs unhooked, no archive is written, and SAVE silently produces an empty workspace.

The two EP scripts (`*_ep.sh`) on top of the basic flags also set:

- `--enable-expert-parallel --all2all-backend deepep_low_latency`.
- `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` — disable NCCL fast paths (CUMEM P2P, NVLS multicast) whose driver-capability flags the foundry VMM region doesn't carry. Scoped to the `--save` / `--load` branches only; baseline-vLLM runs leave them unset so NCCL picks its normal fast path.
- `VLLM_USE_FLASHINFER_SAMPLER=1` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` — required for foundry-compatible MoE behavior (set unconditionally; harmless on baseline).
- `--max-num-batched-tokens` capped at a value satisfying the DeepEP constraint below.
- **No manual `LD_PRELOAD`.** The scripts do not export `LD_PRELOAD` themselves. Foundry's `setup_ld_preload_env` runs at every worker spawn (`WorkerProc.make_worker_process`, `CoreEngineProcManager.__init__`) and prepends `libcuda_hook.so` / `libnvshmem_host.so` to the child env using the paths declared in the TOML config. Baseline runs don't need either preloaded by the shell.

### DeepEP `nvshmem_qp_depth` constraint

DeepEP's `low_latency_dispatch` (`deep_ep/buffer.py:602`) asserts:

```
nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
```

`num_max_dispatch_tokens_per_rank` comes from `moe.max_num_tokens`, which post-rebase points at `scheduler_config.max_num_batched_tokens`. The default `NVSHMEM_QP_DEPTH=1024` therefore caps `--max-num-batched-tokens` at **511**. Two ways to stay green:

1. **Keep `--max-num-batched-tokens` small** (e.g. 256 — matches the largest captured graph in `CUDAGRAPH_CAPTURE_SIZES`). The EP scripts default to this.
2. **Raise the QP depth** at process startup: `export NVSHMEM_QP_DEPTH=$(( (MAX_NUM_BATCHED_TOKENS + 1) * 2 ))`. Pick this if you want chunked-prefill batches larger than the cudagraph capture size.

## Notes & conventions

- All scripts pin `--compilation-config.cudagraph_mode FULL_DECODE_ONLY` and capture all sizes `1..max-num-batched-tokens`.
- All scripts use port `12000` and host `0.0.0.0`.
- The shared TOMLs use `base_addr = 0x600000000000` and `region_size = "256GB"`. If you serve a much larger model, bump `region_size` in both TOMLs (the values must match across SAVE and LOAD).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[HOOK] ERROR: Reserved address … != requested base 0x600000000000` | The VMM region base collided with another allocation. Re-run; this is non-deterministic and the second run usually succeeds. |
| `[HOOK] WARNING: Neither nvshmemx_cumodule_init nor nvshmemx_culibrary_init found` | `libnvshmem_host.so` is not in `LD_PRELOAD`. EP scripts need both `libcuda_hook.so` *and* `libnvshmem_host.so` preloaded. |
| EP graph replay fails with `illegal memory access` | Almost always `libnvshmem_host.so` isn't preloaded. |
| `[foundry] LOAD requires warmup_state.json` raised on LOAD | SAVE pass 1 didn't write `warmup_state.json`. Make sure SAVE finishes "Application startup complete" before you Ctrl-C, and re-run pass 1, then pass 2. |
| `AssertionError ... nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2` in `deep_ep/buffer.py:602` | `max_num_batched_tokens > 511` with default `NVSHMEM_QP_DEPTH=1024`. Lower `--max-num-batched-tokens` or `export NVSHMEM_QP_DEPTH` higher — see the constraint above. |
| `[TIMING] NCCL init: 122.992 s` on Foundry SAVE | Foundry set `LD_PRELOAD` to apply cuda driver hook which extends module load time, but it won't affect LOAD because all modules are preloaded. |
