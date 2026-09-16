# Foundry recipe — SGLang

End-to-end serve scripts for SAVE / LOAD of CUDA graphs through the foundry SGLang
integration.

Current target: the fork's **`foundry`** branch at `6272eb04c5` — upstream `main`
`03ea13a545` (2026-09-16) plus one 8-file integration commit; it pins **torch 2.13.0+cu130**,
sglang-kernel 0.4.7, sgl-deep-ep 0.1.2, sgl-deep-gemm 0.2.0 and ships the whole kernel stack as
wheels. The previous pairing (foundry v0.0.3 with `f1d688e52`) is kept as branch `foundry-0.0.3`;
the 0.0.2-era integration on `foundry-0.0.2`. Foundry `dev >= ac6104f` builds against torch 2.11 and
2.13 alike (version-guarded csrc). Validated on this pairing (see **Validation**):
single GPU, DP=2, TP=2, EP=2 with DeepEP low-latency and with DeepEP v2 —
save/load, TPOT and greedy-output parity against plain SGLang. The 0.0.2-era
`foundry-0.0.2` branch (~0.5.12 base, torch 2.11) still works with foundry `dev` —
the differences called out below are marked with the branch they apply to. All scripts in this directory share the same pair of foundry TOML files
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
├── serve_qwen3-1.7b_tp.sh          # Qwen3-1.7B           tensor parallel (symm-mem allreduce)
├── serve_qwen3-1.7b_dp.sh          # Qwen3-1.7B           data parallel
├── serve_qwen3-30ba3b_ep.sh        # Qwen3-30B-A3B (MoE)  expert parallel (DeepEP + DP-attention)
├── serve_qwen3-30ba3b_ep_tpattn.sh # Qwen3-30B-A3B (MoE)  expert parallel, TP attention (symm-mem allreduce)
└── serve_qwen3-30ba3bfp8_ep_v2.sh  # Qwen3-30B-A3B-FP8    expert parallel, DeepEP v2 (NCCL symmetric windows)
```

Every script accepts the same trailing `--save` / `--load` flag. Scripts that scale
across GPUs take the parallel-size as the first positional argument:

```bash
bash serve_qwen3-mini.sh                       [--save|--load]
bash serve_qwen3-1.7b_tp.sh        <tp_size>   [--save|--load]
bash serve_qwen3-1.7b_dp.sh        <dp_size>   [--save|--load]
bash serve_qwen3-30ba3b_ep.sh          <ep_size>   [--save|--load]
bash serve_qwen3-30ba3b_ep_tpattn.sh   <ep_size>   [--save|--load]
bash serve_qwen3-30ba3bfp8_ep_v2.sh    <ep_size>   [--save|--load]
```

Without `--save`/`--load` a script runs plain SGLang (the baseline). `SGL_EXTRA_ARGS`
appends extra `sglang serve` flags verbatim, e.g.
`SGL_EXTRA_ARGS="--cuda-graph-backend-prefill disabled"` to give a baseline the same
prefill-graph policy as a foundry LOAD (foundry restores decode graphs only).

The scripts use `--cuda-graph-max-bs-decode`; the pre-0.5.18 alias `--cuda-graph-max-bs` was
removed upstream, so they no longer run on the 0.0.2-era fork branch as-is.

A single SAVE pass is enough — SGLang has no startup profile-forward, so there is no
non-determinism that requires a two-pass save (unlike the vLLM recipe).

Because the two TOMLs are shared (single `workspace_root = "foundry_archive"`), one
archive is written per host; run a fresh `rm -rf foundry_archive` whenever you change
model or topology before SAVE.

| Mode | Script | Model | Notes |
|---|---|---|---|
| Single GPU | `serve_qwen3-mini.sh` | Qwen3-1.7B | FlashInfer backend |
| Tensor parallel | `serve_qwen3-1.7b_tp.sh` | Qwen3-1.7B | torch symm-mem allreduce (`--enable-torch-symm-mem --disable-custom-all-reduce`); mirrors the vLLM TP recipe |
| Data parallel | `serve_qwen3-1.7b_dp.sh` | Qwen3-1.7B | one full replica/rank; `NCCL_CUMEM_ENABLE=0`/`NCCL_NVLS_ENABLE=0` |
| Expert parallel | `serve_qwen3-30ba3b_ep.sh` | Qwen3-30B-A3B | DP-attention + DeepEP; fa3 backend; `SGL_MODEL=Qwen/Qwen3-30B-A3B-FP8` for FP8 |
| Expert parallel, TP attention | `serve_qwen3-30ba3b_ep_tpattn.sh` | Qwen3-30B-A3B | symm-mem allreduce + DeepEP (vLLM-shaped EP); needs the `foundry` branch's per-phase cuda-graph flags |
| Expert parallel, DeepEP v2 | `serve_qwen3-30ba3bfp8_ep_v2.sh` | Qwen3-30B-A3B-FP8 | NCCL symmetric windows + GIN instead of NVSHMEM; needs NCCL >= 2.30.7 (see below) |

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

Use a dedicated env, kept separate from the vLLM env so kernel pins don't clash
(the `foundry` branch pins torch 2.13; a torch-2.11 env cannot run it — sglang-kernel
0.4.6+ is built against the torch 2.13 C++ ABI):

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130

# in-tree sglang fork (branch foundry), editable — this pulls the FULL
# kernel stack as wheels: flashinfer 0.6.18, sglang-kernel 0.4.6.post1,
# sgl-deep-ep, sgl-deep-gemm, flash-attn-4. No hand-built kernels remain
# (fa3 now lives inside sglang-kernel as sgl_kernel.flash_attn).
pip install -e sglang/python

# flashinfer's cubin/jit-cache wheels lag on PyPI — take them from flashinfer's
# own index, versions matching flashinfer-python exactly:
pip install "flashinfer-cubin==0.6.18" --index-url https://flashinfer.ai/whl
pip install "flashinfer-jit-cache==0.6.18" --index-url https://flashinfer.ai/whl/cu130

# foundry build deps (boost from conda/system; cmake+ninja can come from pip)
pip install "cmake>=4.0" ninja wheel pytest
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

# tensor parallel (symm-mem allreduce inside the decode graphs)
rm -rf foundry_archive
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_tp.sh 2 --save
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_tp.sh 2 --load
```

TP notes: custom all-reduce (IPC-buffer registration per graph) and in-graph
pynccl are both replay paths foundry does not support; the TP script disables
them and enables `--enable-torch-symm-mem`, so every decode-graph allreduce is a
`symm_mem.two_shot_all_reduce_` (TP=2 on Hopper) on the persistent symmetric
buffer foundry places deterministically. On hosts without usable multicast (no
IMEX channels), the `foundry` fork keeps the communicator enabled on the
two-shot path — upstream sglang would silently fall back to in-graph NCCL,
which breaks LOAD. If a load aborts with `TorchSymmMemCommunicator ...
communicator is not available` in the log, the allreduce fell back to NCCL and
the archive is not replayable.

## Run (expert parallel / DeepEP)

On the `foundry` branch the EP kernel stack is entirely wheel-provided by the sglang
install above (`sgl-deep-ep`, `sgl-deep-gemm`; fa3 inside `sglang-kernel`) — there is
nothing to build. Two things still matter:

- **NVSHMEM** — already in the env. cu13 `torch` pulls the `nvidia-nvshmem-cuXX`
  wheel as a dependency (`libnvshmem_host.so.3` under `site-packages/nvidia/nvshmem/lib/`).
  Foundry auto-detects it from the wheel (just like `libcuda_hook.so`) and the
  spawn-site patches preload it into each worker — no manual path, no TOML field.
- **NVSHMEM host/device versions must match.** The `sgl-deep-ep` wheel statically
  embeds its NVSHMEM *device* library; the preloaded *host* library must be the same
  version. The auto-detected `nvidia-nvshmem` wheel satisfies this. Overriding
  `nvshmem_host_path` in the TOMLs with a lib from another NVSHMEM build (e.g. an
  old vLLM ep_kernels workspace) aborts every rank at DeepEP init with
  `NVSHMEM device library version does not match with NVSHMEM host library version`.

(`foundry-0.0.2` branch only: DeepEP @ `9af0e0d`, `sgl-deep-gemm >= 0.1.2` and
`flash-attn-3` were hand-built — see that branch's README.)

```bash
rm -rf foundry_archive
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3b_ep.sh 2 --save
CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3b_ep.sh 2 --load
curl -s http://0.0.0.0:12000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-30B-A3B","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

**EP with TP attention (symm-mem allreduce)** — `serve_qwen3-30ba3b_ep_tpattn.sh`.
The default EP recipe uses DP-attention, which needs no allreduce in the decode
graphs. This variant mirrors the vLLM EP topology instead: TP attention with its
allreduce routed through torch symm-mem (`--enable-torch-symm-mem`, custom AR
off) plus `--cuda-graph-backend-prefill disabled` — the prefill-graph disable
matters even for baseline runs of this topology, because without DP-attention
every rank dispatches the full prefill chunk and prefill-graph capture trips
DeepEP's `num_max_dispatch_tokens_per_rank` assert. `foundry` branch
only (uses the per-phase cuda-graph flags).

The EP script sets `--enable-dp-attention --moe-a2a-backend deepep --deepep-mode
low_latency --moe-runner-backend deep_gemm --attention-backend fa3
--disable-custom-all-reduce` and `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`.
DeepEP low-latency caps dispatch at that per-rank token count (and asserts
`(n+1)*2 <= NVSHMEM_QP_DEPTH`); keep it and `--chunked-prefill-size` identical between
SAVE and LOAD so the captured graphs match.

## Validation

Every recipe in this directory was run as shipped on 2026-09-05 (8×H100 host, 2 GPUs
per multi-GPU run, foundry v0.0.3, sglang fork branch `foundry` at `f1d688e52`, CUDA 13.3
compat library, NCCL 2.30.7)
through `experimental/recipe_validate.sh`: SAVE, then plain SGLang twice (the noise
control; `SGL_EXTRA_ARGS="--cuda-graph-backend-prefill disabled"` so it skips prefill
graphs like a LOAD), then LOAD. Per engine: seconds to `/health`, sglang's own
`cuda_graph decode` timing (capture on SAVE / baseline, restore on LOAD), a TPOT sweep
(`experimental/bench_sglang.sh`, 3 runs, 64 output tokens) and 32 fixed greedy prompts
at concurrency 1 / 8 / 32 compared word-for-word.

| Recipe | Model | Graphs | Capture → restore | To `/health`: SGLang / LOAD | TPOT p50 LOAD vs SGLang (bs 1 / 8 / 32 / 128) |
|---|---|---:|---:|---:|---|
| `serve_qwen3-mini.sh` | Qwen3-1.7B | 52 | 3.3 s → 0.43 s | 27.1 s / 25.1 s | -1.0 / -0.5 / -0.3 / -1.8 % |
| `serve_qwen3-1.7b_dp.sh 2` | Qwen3-1.7B | 52 | 3.2 s → 0.36 s | 33.1 s / 33.1 s | -1.5 / +0.3 / -0.6 / +1.4 % |
| `serve_qwen3-1.7b_tp.sh 2` | Qwen3-1.7B | 20 | 2.5 s → 0.29 s | 27.1 s / 27.1 s | -1.3 / -1.2 / +0.5 / +0.4 % |
| `serve_qwen3-30ba3b_ep.sh 2` | Qwen3-30B-A3B | 20 | 6.0 s → 0.62 s | 43.1 s / 43.1 s | +0.5 / +0.0 / +0.4 / +0.3 % |
| `serve_qwen3-30ba3bfp8_ep_v2.sh 2` | Qwen3-30B-A3B-FP8 | 20 | 7.0 s → 0.42 s | 43.1 s / 41.2 s | +0.7 / +0.7 / +0.5 / -0.5 % |

Greedy completions: identical to plain SGLang for every recipe at concurrency 1 and,
except one prompt each on dp/tp at concurrency 32 and the MoE recipes at 8/32, at
higher concurrency too; in those cells the two plain-SGLang runs disagree with each
other by the same amount (batch composition changes accumulation order), so the
restored graphs sit inside SGLang's own run-to-run noise.

With the recipes' default graph sets (20–52 decode graphs) capture is only a few
seconds, so time-to-health is dominated by weight loading and the differences above
are small; with all 256 decode graphs (`--cuda-graph-max-bs-decode 256 --disable-cuda-graph-padding`)
restore saves 25–50 s per engine start (see the top-level README's Performance table).

## DeepEP v2 (NCCL)

`serve_qwen3-30ba3bfp8_ep_v2.sh <ep_size> [--save|--load]` runs the MoE
all-to-all over DeepEP v2 (`--moe-a2a-backend deepep_v2`), i.e. NCCL
symmetric-memory windows and NCCL GIN (GDAKI/DOCA) rather than NVSHMEM.
Prototype status: validated on H100 EP=2/EP=4 with all 256 decode graphs.

Extra requirements:

```bash
# sgl-deep-ep's ElasticBuffer is compiled against NCCL 2.30.7 (torch pins 2.29.7)
pip install --no-deps nvidia-nccl-cu13==2.30.7
# 2.30.7 is a cuda13.3 build: on a 580.x (CUDA 13.0) host driver add the
# forward-compat library and put it first on LD_LIBRARY_PATH
apt-get install cuda-compat-13-3 && export LD_LIBRARY_PATH=/usr/local/cuda-13.3/compat:$LD_LIBRARY_PATH
```

Any later `pip install -e` of foundry re-resolves torch's NCCL pin; use
`--no-deps` or re-pin 2.30.7 afterwards.

What foundry does for v2 (all automatic): creates the `ElasticBuffer` at the
same pre-capture point on SAVE and LOAD (`_bootstrap_deepep_v2_buffer`),
reports success for `cuPointerSetAttribute(SYNC_MEMOPS)` on region memory
(DOCA requires it), and sets `NCCL_GRAPH_REGISTER=0`/`NCCL_LOCAL_REGISTER=0`
so no collective in a captured graph depends on registration state that a
restored graph cannot replay. Do not force `NCCL_CUMEM_ENABLE=0` with v2.

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
| `NVSHMEM device library version does not match with NVSHMEM host library version`, then segfault | A custom `nvshmem_host_path` in the TOMLs points at a different NVSHMEM build than the one inside the `sgl-deep-ep` wheel. Remove the override; foundry's auto-detected `nvidia-nvshmem` wheel matches. |
| `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2` | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` too high for `NVSHMEM_QP_DEPTH`; lower it or raise the QP depth. |
| TP/EP LOAD aborts `[HOOK] ERROR: cuLinkAddData failed for segment 0 with error 209` (SAVE may log the same during pre-link); DP unaffected | Driver too old for the CUDA version NCCL was built with. Foundry device-links NCCL's kernel library on reload, and the driver's linker rejects fatbins from a newer toolkit even though `cuLibraryLoadData` (plain sglang) accepts them. Compare `nvidia-smi` (driver CUDA version) against the `+cudaX.Y` in `NCCL version 2.29.7+cuda13.2` in the worker log — the torch 2.13 cu130 wheel pins a 13.2-built NCCL, so a 13.0 driver (580.x) fails while 595.x works. After installing NCCL 2.30.7 (a cuda13.3 build) for DeepEP v2, *every* recipe, TP and EP included, needs the CUDA 13.3 compat library (or a 13.3-capable driver): with the 13.2 compat lib, the TP SAVE logs `cuLinkAddData failed ... 209 during pre-link` for NCCL's 128-segment device library and the LOAD then aborts. Fix: upgrade the driver, or install NVIDIA's forward-compat package (`apt-get install cuda-compat-13-2`, then `LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:$LD_LIBRARY_PATH`). Downgrading NCCL to a 13.0 build is not an option: torch 2.13 needs `ncclCommResume` (>= 2.29). |
