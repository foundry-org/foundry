# vLLM Save / Load Workflow

## Serve scripts

`experimental/single-gpu/serve_qwen_<size>.sh` for 1.7B / 4B / 14B (single-GPU). DP / EP scripts under `experimental/data-parallel/` and `experimental/expert-parallel/`.

Key knobs (1.7B example):

```bash
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.6

FOUNDRY_HOOK="$(cd "$(dirname …)/../../foundry/python/foundry" && pwd)/libcuda_hook.so"
if [[ -f "$FOUNDRY_HOOK" ]]; then
    export LD_PRELOAD="${FOUNDRY_HOOK}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export VLLM_WORKER_MULTIPROC_METHOD=forkserver

vllm serve "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enable-prefix-caching \
    --max-num-seqs 512 \
    --attention-config.backend FLASH_ATTN \
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
    --compilation-config.graph_extension_config_path "$FOUNDRY_TOML"
```

> **Note:** Don't pass `--compilation-config.cudagraph_num_of_warmups N`. vLLM's `VllmConfig.__post_init__` hard-overrides this to `1` whenever cudagraph mode is enabled (`vllm/config/vllm.py:1052`), ignoring whatever you pass. Foundry then stomps it back to `0` inside `install_hooks` because the warmup `_dummy_run(cudagraph_runtime_mode=NONE, force_attention=True)` would invoke `CUDAGraphWrapper.runnable()` outside the capture window and break SAVE/LOAD allocation parity. See [`overview.md`](overview.md#2-cudagraph_num_of_warmups-must-be-0-in-foundry-mode).
>
> `VLLM_USE_AOT_COMPILE` defaults to `False` in `vllm/envs.py` and is not touched by `install_hooks`. If a future vLLM bumps the default to `True`, we'd need to force-set it back (an AOT cache-hit forward would otherwise produce non-deterministic allocations between SAVE and LOAD).

### Why pre-launch `LD_PRELOAD`

`libcuda_hook.so` must be loaded **before** any CUDA call so its interposers (`cuMemAlloc_v2`, `cuModuleLoadData`, …) run in every process. The integration also re-sets `LD_PRELOAD` via `setup_ld_preload_env()` before each `proc.start()`, but pre-loading from the shell protects the parent process and any helper threads.

### Why `VLLM_WORKER_MULTIPROC_METHOD=forkserver`

`forkserver` pre-imports `vllm` once and forks workers from that warmed-up parent — much faster cold start. Works with foundry because:

- `libcuda_hook.so` is in `LD_PRELOAD` so the forkserver itself has it.
- `torch.cuda.get_device_capability` is NVML-backed (foundry's `_quarantine_get_device_capability` patch) so the parent stays CUDA-clean.
- `cudaSetDevice` resolves through `RTLD_DEFAULT+RTLD_NEXT` in forkserver children where libcudart has `RTLD_LOCAL` scope.

## TOML configs

Same shape as SGLang:

```toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive_qwen_1.7b"
scratch_space_size = "1024MB"
```

## Run sequence (single-GPU)

```bash
cd /path/to/foundry-org

# 1. Clean any prior workspace
rm -rf foundry_archive_qwen_1.7b

# 2. SAVE pass 1 (profiles + captures)
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait for "Application startup complete", Ctrl-C / SIGTERM

# 3. SAVE pass 2 (deterministic; reuses saved profile)
bash experimental/single-gpu/serve_qwen_1.7b.sh --save
# wait, SIGTERM

# 4. LOAD
bash experimental/single-gpu/serve_qwen_1.7b.sh --load

# 5. Query
bash experimental/query.sh 12000 Qwen/Qwen3-1.7B
```

Always run two SAVE passes. Pass 1 writes `warmup_state.json` and a non-deterministic set of captured graphs; pass 2 overwrites the graphs with deterministic ones. Skipping pass 2 will produce a LOAD that appears to succeed but fails the cursor check or, worse, succeeds the cursor check and emits garbage.

## DP / EP

`experimental/data-parallel/serve.sh` and `experimental/expert-parallel/serve.sh` extend the same pattern with the per-rank flags vLLM needs (`--data-parallel-size`, `--enable-expert-parallel`, etc.). Each rank writes to its own `rank_{N}/` subdirectory under `workspace_root`; `warmup_state.json` stays at the workspace root and is written by rank 0.

## Expected logs

SAVE (success):

```
[foundry] install_hooks: mode=save workspace=foundry_archive_qwen_1.7b
[foundry TIMING] subprocess spawn → install_hooks: ... ms
[foundry] all patches installed
…
[foundry] start_graph_builds returned in 0.xs (building in background)   ← on LOAD only
[foundry TIMING] init_nvshmem_for_loaded_modules: 0.xs (N modules)       ← on LOAD only
(per-graph replay happens inline during capture_model on LOAD)       ← no separate "finish_graph_loads" log line
…
[foundry] capture_final_alloc_offset = … bytes                       ← on SAVE
INFO:     Application startup complete.
```

LOAD (success):

```
[foundry] install_hooks: mode=load workspace=foundry_archive_qwen_1.7b
…
[foundry] preallocate_for_load_mode = … MB
[foundry] start_graph_builds returned in 0.xs (building in background)
[foundry TIMING] init_nvshmem_for_loaded_modules: 0.xs (N modules)
…
(per-graph replay happens inline during capture_model — no separate log line)
…
INFO:     Application startup complete.
```

## Validation

A correct LOAD:

1. `final_alloc_offset` from SAVE pass 2 matches the LOAD-side post-replay cursor (foundry validates this internally).
2. `query.sh` returns coherent text. If cursors match but output is garbage, see [`memory-consistency.md`](memory-consistency.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[HOOK] ERROR: Memory offset mismatch during replay` | An asymmetric allocation between SAVE pass 2 and LOAD. The watermark logs identify which lifecycle phase. |
| LOAD aborts at first inference with `Called CUDAGraph::replay without a preceding successful capture or load` | On-demand graph lost its template link. Should not happen — `start_graph_builds` loads all paths in a single C++ call; if it does, check `_patch_capture_model` and that `_pending_graph_builds.identifier_to_index` covers every BatchDescriptor visited by the capture loop. |
| `_initialize_kv_caches` complains about missing fields on LOAD | `warmup_state.json` not written by SAVE pass 1. Re-run pass 1, then pass 2. |
| Hangs in NCCL bring-up | `setup_graph_extension` may have set up the VMM region after NCCL already touched memory. Verify `_patch_init_worker_distributed_environment` is wrapping the correct upstream symbol (vLLM renames sometimes). |
| Workers crash with `module not found` | foundry not installed in worker venv; pip install editable in the same env, or use the serve script's PYTHONPATH approach. |

## What to clean between runs

Always `rm -rf workspace_root` before SAVE pass 1.

Pass 2 reads `warmup_state.json` and overwrites graphs — no separate clean needed between passes.

LOAD never writes to the workspace.
