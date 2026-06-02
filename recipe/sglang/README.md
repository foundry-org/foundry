# Foundry recipe — SGLang

End-to-end serve scripts for SAVE / LOAD of CUDA graphs through the foundry SGLang
integration. All scripts in this directory share the same pair of foundry TOML files
(`foundry_save.toml` / `foundry_load.toml`) — pick a script for your model + parallelism,
run `--save`, then `--load`, then query. The integration code is in
[`../../python/foundry/integration/sglang/`](../../python/foundry/integration/sglang/);
design notes are under [`../../docs/sglang/`](../../docs/sglang/).

## Files in this directory

```
recipe/sglang/
├── README.md                       # this file
├── foundry_save.toml               # shared SAVE config (workspace_root = "foundry_archive")
├── foundry_load.toml               # shared LOAD config (same workspace_root)
├── serve_qwen3-mini.sh             # Qwen3-1.7B           single GPU
├── serve_qwen3-1.7b_dp.sh          # Qwen3-1.7B           data parallel
└── serve_qwen3-30ba3bfp8_ep.sh     # Qwen3-30B-A3B FP8    expert parallel (DeepEP)
```

Every script accepts the same trailing `--save` / `--load` flag. Scripts that scale
across GPUs take the parallel-size as the first positional argument:

```bash
bash serve_qwen3-mini.sh                       [--save|--load]
bash serve_qwen3-1.7b_dp.sh        <dp_size>   [--save|--load]
bash serve_qwen3-30ba3bfp8_ep.sh   <ep_size>   [--save|--load]
```

A single SAVE pass is enough — SGLang has no startup profile-forward, so there is no
non-determinism that requires a two-pass save (unlike the vLLM recipe).

Because the two TOMLs are shared (single `workspace_root = "foundry_archive"`), one
archive is written per host; run a fresh `rm -rf foundry_archive` whenever you change
model or topology before SAVE.

| Mode | Script | Model | Notes |
|---|---|---|---|
| Single GPU | `serve_qwen3-mini.sh` | Qwen3-1.7B | FlashInfer backend |
| Data parallel | `serve_qwen3-1.7b_dp.sh` | Qwen3-1.7B | one full replica/rank; `NCCL_CUMEM_ENABLE=0`/`NCCL_NVLS_ENABLE=0` |
| Expert parallel | `serve_qwen3-30ba3bfp8_ep.sh` | Qwen3-30B-A3B-FP8 | DP-attention + DeepEP; fa3 backend |

## Installation

The recipes assume `foundry` and the SGLang fork are **pip-installed** (editable is
fine) so both import without any `PYTHONPATH`, and foundry's spawn-site patch
auto-detects `libcuda_hook.so` from its install — the scripts set no `LD_PRELOAD`
themselves. The standard workspace layout has `foundry/` (this repo) and `sglang/`
(the foundry-org SGLang fork) as siblings:

```
<workspace>/
├── foundry/                # this repo
│   ├── python/foundry/     # `pip install -e .` builds libcuda_hook.so here
│   ├── recipe/sglang/      # <-- you are here
│   └── ...
└── sglang/                 # foundry-org/sglang fork (with direct edits applied)
```

Use a dedicated env, kept separate from the vLLM env so kernel pins don't clash:

```bash
conda create -p venv/ python=3.12
conda activate venv/
conda install -c conda-forge boost-cpp boost          # foundry C++ deps

# in-tree sglang fork, editable
pip install -e sglang/python --extra-index-url https://download.pytorch.org/whl/cu130
# foundry
pushd foundry && pip install -e . --no-build-isolation && popd
```

`libcuda_hook.so` finds boost via a baked rpath; if it can't, add the conda lib dir to
`LD_LIBRARY_PATH` (`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`).

## Run (single GPU / DP)

```bash
# single GPU
rm -rf foundry_archive
bash serve_qwen3-mini.sh --save     # wait for "Application startup complete", then SIGTERM
bash serve_qwen3-mini.sh --load     # leave running

# query (separate shell)
curl -s http://0.0.0.0:12000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","prompt":"The capital of France is","max_tokens":12,"temperature":0}'

# data parallel (pick GPUs with CUDA_VISIBLE_DEVICES)
rm -rf foundry_archive
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_dp.sh 2 --save
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_dp.sh 2 --load
```

## Run (expert parallel / DeepEP)

EP needs three kernel packages — all SGLang-native, no vLLM involved:

- **NVSHMEM** — already in the env. cu13 `torch` pulls the `nvidia-nvshmem-cuXX`
  wheel as a dependency (`libnvshmem_host.so.3` under `site-packages/nvidia/nvshmem/lib/`).
  Foundry auto-detects it from the wheel (just like `libcuda_hook.so`) and the
  spawn-site patches preload it into each worker — no manual path, no TOML field.
- **DeepEP** @ `9af0e0d` — SGLang's pin. Build via SGLang's own installer
  `sglang/scripts/ci/cuda/ci_install_deepep.sh` (it `git checkout`s exactly `9af0e0d`
  and builds against the NVSHMEM wheel above). For a single node you can skip the
  script's gdrcopy/RDMA apt steps and just build the wheel:

  ```bash
  git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP
  git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
  TORCH_CUDA_ARCH_LIST="9.0" python setup.py install   # Hopper; "9.0;10.0" for Blackwell
  cd ..
  ```

- **`sgl-deep-gemm >= 0.1.2`** (0.1.0 lacks `m_grouped_bf16_gemm_nt_masked`) and
  **`flash-attn-3`** (the fa3 attention backend — flashinfer's ragged-prefill path
  has an off-by-one under this config).

```bash
rm -rf foundry_archive
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep.sh 2 --save
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep.sh 2 --load
curl -s http://0.0.0.0:12000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-30B-A3B-FP8","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

The EP script sets `--enable-dp-attention --moe-a2a-backend deepep --deepep-mode
low_latency --moe-runner-backend deep_gemm --attention-backend fa3
--disable-custom-all-reduce` and `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`.
DeepEP low-latency caps dispatch at that per-rank token count (and asserts
`(n+1)*2 <= NVSHMEM_QP_DEPTH`); keep it and `--chunked-prefill-size` identical between
SAVE and LOAD so the captured graphs match.

## Archive layout

```
foundry_archive/
├── warmup_state.json              # KV-block sizing + MemoryPoolConfig (rank 0)
└── rank_<N>/
    ├── graph_*.json + .cugraph    # one pair per captured graph
    ├── graph_manifest.json        # topology groups + template assignments
    ├── fatbin_image_packed.img    # packed CUDA modules
    └── final_alloc_offset.json    # per-rank VMM watermark
```

For DP / EP each rank gets its own `rank_<N>/`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Reserved address … != requested base 0x600000000000` | VMM base collided with another allocation. Re-run; non-deterministic, the next run usually succeeds. |
| EP replay `illegal memory access` / `nvshmemx_cumodule_init not found` | `libnvshmem_host.so.3` not preloaded — foundry couldn't auto-detect the `nvidia-nvshmem` wheel. Confirm it's installed (`pip show nvidia-nvshmem-cu13`), or set `nvshmem_host_path` in both TOMLs. |
| `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2` | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` too high for `NVSHMEM_QP_DEPTH`; lower it or raise the QP depth. |
