# SGLang Save / Load Workflow

## Serve scripts

Key knobs (1.7B example):

```bash
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.6

# No LD_PRELOAD / PYTHONPATH here: foundry + the sglang fork are pip-installed, and
# foundry's setup_ld_preload_env auto-detects libcuda_hook.so and LD_PRELOADs it
# into every worker at spawn time. (Running from a source checkout instead? Export
# PYTHONPATH=.../foundry/python:.../sglang/python yourself.)

sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs 512 \
    --foundry-graph-extension-config-path "$FOUNDRY_TOML"
```

`--cuda-graph-max-bs 512` is the closest analogue to vLLM's `--max-num-seqs 512` — it drives `capture_bs` to span a similar range of decode batch sizes (52 batch sizes from 1 → 512).

## TOML configs

`recipe/sglang/foundry_save.toml`:

```toml
mode = "save"
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive"
scratch_space_size = "1024MB"
```

`foundry_load.toml` is identical except `mode = "load"`. See [`memory-lifecycle.md`](memory-lifecycle.md) for the field semantics.

## Run sequence

```bash
cd /path/to/foundry/recipe/sglang

# 1. Clean any prior workspace
rm -rf foundry_archive

# 2. SAVE
bash serve_qwen3-mini.sh --save
# wait for "Application startup complete", then Ctrl-C / SIGTERM

# 3. LOAD
bash serve_qwen3-mini.sh --load
# wait for "Application startup complete"

# 4. Query
curl -s http://0.0.0.0:12000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","prompt":"The capital of France is",
       "max_tokens":12,"temperature":0}'
```

For Qwen3-1.7B, expected output:

```
" Paris. The capital of the United States is Washington, D"
```

A single SAVE pass is sufficient. SGLang does not run a profile-forward at startup (only `_profile_available_bytes` samples free GPU memory), so there is no allocation non-determinism that a second SAVE pass would need to mask.

## Per-rank workspace layout

```
foundry_archive/
  warmup_state.json                       # shared; MemoryPoolConfig + final_alloc_offset
  rank_0/
    graph_{0..N-1}_FULL_t{bs}_r{bs}_UX_pcN.json       # one per captured graph
    graph_{0..N-1}_FULL_t{bs}_r{bs}_UX_pcN.cugraph    # binary cuGraph blob
    graph_manifest.json                   # topology groups for template + on-demand linking
    final_alloc_offset.json               # per-rank VMM watermark
    fatbin_image_packed.img               # packed kernel fatbins
    fatbin_entrypoint_packed.txt          # fatbin entry-point index
```

Filename scheme (defined in `graph_ops._graph_filename`):

```
graph_{state.capture_index}_FULL_t{bs}_r{bs}_UX_pcN.json
```

`state.capture_index` increments per `save_graph` call so files sort in SAVE-time order. `_GRAPH_FILENAME_RE` in `graph_ops.py` parses them on LOAD.

## Expected logs

SAVE (success):

```
[Foundry] SGLang hooks installing: mode=save workspace=foundry_archive_qwen_1.7b
[Foundry] SGLang hooks installed
Foundry SGLang integration activated from .../save_qwen_1.7b.toml
[Foundry] SGLang rank=0 workspace_dir=foundry_archive_qwen_1.7b/rank_0
[Foundry] SGLang graph extension setup completed in 0.x s
[Foundry] SGLang skipped allocator to scratch boundary 1073741824
[Foundry] SGLang reused saved memory pool config       ← only on LOAD
[Foundry] SGLang kernel_warmup skipped in save mode
…
[Foundry] Saved SGLang CUDA graph graph_0_FULL_t512_r512_UX_pcN.json key=512
…
[Foundry] Saved SGLang CUDA graph graph_51_FULL_t1_r1_UX_pcN.json key=1
[foundry] Saved graph_manifest.json: 9 topology groups (9 templates, 43 on-demand, 43 stripped)
[Foundry] Saved SGLang warmup state to foundry_archive_qwen_1.7b/warmup_state.json
[Foundry] SGLang final_alloc_offset=22785556480
…
INFO:     Application startup complete.
```

LOAD (success):

```
[Foundry] SGLang hooks installing: mode=load workspace=foundry_archive_qwen_1.7b
[Foundry] SGLang hooks installed
[Foundry] SGLang graph extension setup completed in 0.x s
[Foundry] SGLang reused saved memory pool config
[Foundry] SGLang kernel_warmup skipped in load mode
[Foundry] SGLang alloc_offset[before_preallocate]=… (… MB)
[Foundry] SGLang alloc_offset[after_preallocate]=… (… MB)
[Foundry] SGLang alloc_offset[after_pre_init]=… (… MB)
[CGE] Using graph_manifest.json (9 topology groups)
[CGE] Phase 1: 52 graphs parsed in 0.x ms, 9 topologies, 4 threads, 52 binary + 0 json
[CGE BUILD] Template 0 (...): N nodes, done in X.X ms
…
[CGE] Phase 2: 9 templates + 43 on-demand = 52 graphs built in xx.x ms
[CGE] finish_graph_loads: 52 graphs, xx.x ms
[Foundry] Loaded 52 SGLang graphs in 0.0x s
[Foundry] SGLang alloc_offset[after_load_all_graphs]=22785556480 (… MB)
…
INFO:     Application startup complete.
```

The `after_load_all_graphs` value **must** equal SAVE's `final_alloc_offset`. If it doesn't, see [`memory-consistency.md`](memory-consistency.md).
