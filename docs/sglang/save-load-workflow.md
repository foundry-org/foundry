# SGLang Save / Load Workflow

## Serve scripts

Key knobs (1.7B example):

```bash
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.6

# No LD_PRELOAD / PYTHONPATH here: sglang (with the Foundry adapter) and foundry
# (sglang[foundry]) are pip-installed in one venv, and Foundry sets LD_PRELOAD
# around every worker at spawn time (libcuda_hook.so auto-detected).

# Foundry is switched on by sglang's own flags; the mode is the flag's, not the TOML's.
sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs-decode 512 \
    --cuda-graph-persistence save \
    --cuda-graph-persistence-config recipe/sglang/foundry.toml
```

The recipe scripts are exactly this shape: `--save` / `--load` appends the two
`--cuda-graph-persistence` arguments above, no mode runs plain SGLang. If Foundry is not
installed (or its integration API major mismatches), SGLang stops at startup with the
reason instead of running natively; see
[`recipe/sglang/README.md`](../../recipe/sglang/README.md#activation---cuda-graph-persistence).

On SAVE / LOAD, Foundry's resolution steps pin the configs that select unreplayable state
(NCCL registration / cuMem / NVLS, DeepGEMM precompile, and the rest) and stop the launch
if one was set to the contrary ([`overview.md`](overview.md#what-foundry-pins-and-why));
the multi-GPU scripts pass the collective flags (`--disable-custom-all-reduce --enable-torch-symm-mem`)
in every mode so baseline, SAVE and LOAD capture the same graph set, and all scripts source `set_env.sh`.

`--cuda-graph-max-bs-decode 512` is the closest analogue to vLLM's `--max-num-seqs 512` — it drives `capture_bs` to span a similar range of decode batch sizes (52 batch sizes from 1 → 512).

## TOML configs

One shared `recipe/sglang/foundry.toml` for SAVE and LOAD (the mode comes from
`--cuda-graph-persistence`; a `mode` key in the TOML is ignored on that route):

```toml
base_addr = 0x600000000000
region_size = "256GB"
workspace_root = "foundry_archive"
```

See [`memory-lifecycle.md`](memory-lifecycle.md) for the field semantics.

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
  warmup_state.json                       # shared copy (first rank to write it); MemoryPoolConfig + final_alloc_offset
  rank_0/
    warmup_state.json                     # per-rank (wins on LOAD): MemoryPoolConfig, context overrides, final_alloc_offset, FlashInfer tactics
    decode_graphs.json                    # decode-type scope kinds in capture order + output structures
    prefill_graphs.json                   # prefill graphs on: request slots, DP-gather flag, output structures
    graph_{0..N-1}_FULL_t{bs}_r{bs}_UX_pcN.json       # one per captured graph
    graph_{0..N-1}_FULL_t{bs}_r{bs}_UX_pcN.cugraph    # binary cuGraph blob
    graph_manifest.json                   # topology groups for template + on-demand linking
    region_layout.json                    # per-rank layout: start offset, watermark, live ranges
    fatbin_image_packed.img               # packed kernel fatbins
    fatbin_entrypoint_packed.txt          # fatbin entry-point index
```

Filename scheme (defined in `graph_ops._graph_filename`):

```
graph_{state.capture_index}_FULL_t{bs}_r{bs}_UX_pcN.json
```

`state.capture_index` increments per `save_graph` call so files sort in SAVE-time order. `_GRAPH_FILENAME_RE` in `graph_ops.py` parses them on LOAD. Other decode-type scopes (EAGLE / EAGLE3) and prefill graphs use `graph_{i}_DECODE-{kind}_t{size}_UX_pcN[_v{variant}].json` and `graph_{i}_PREFILL_t{tokens}_r{slots}_UX_pc{chunks}.json`.

**Archive format note (LOAD template build).** A `.cugraph` written by the current SAVE carries the header flag
`FLAG_COMPLETE_KERNEL_ATTRS` when it holds every kernel-node attribute the JSON records (per-node programmatic stream
serialization and device-updatable live in spare node-table space; an access-policy window, or those attributes in the
common set, leave the flag unset). LOAD then builds each template's graph from the binary node table only; without the flag it builds from the JSON copy.

## Expected logs

SAVE (success):

```
[Foundry] SGLang graph persistence active: mode=save workspace=foundry_archive_qwen_1.7b
[Foundry] SGLang rank=0 (tp_rank=0 pp_rank=0 dp_rank=… replica=0 attention_dp=False) workspace_dir=foundry_archive_qwen_1.7b/rank_0
[Foundry] SGLang graph extension setup completed in 0.x s (binary restore 0.000 s, region reserve …, cuBLAS handle …)
[Foundry] Saved SGLang warmup state to foundry_archive_qwen_1.7b/rank_0/warmup_state.json
[Foundry] SGLang layout start offset=…
…
[Foundry] Saved SGLang CUDA graph graph_0_FULL_t512_r512_UX_pcN.json key=512
…
[Foundry] Saved SGLang CUDA graph graph_51_FULL_t1_r1_UX_pcN.json key=1
[foundry] Saved graph_manifest.json (templates=on): 9 topology groups (9 templates, 43 on-demand, 43 stripped)
[Foundry] SAVE packed the recorded binaries in … ms
[Foundry] SGLang region layout: … live ranges (… MB) in the region, …
[Foundry] Saved SGLang warmup state to foundry_archive_qwen_1.7b/warmup_state.json
[Foundry] SGLang final_alloc_offset=22785556480
…
INFO:     Application startup complete.
```

LOAD (success):

```
[Foundry] SGLang graph persistence active: mode=load workspace=foundry_archive_qwen_1.7b
[Foundry] SGLang graph extension setup completed in 0.x s (binary restore …, region reserve …, cuBLAS handle …)
[Foundry] SGLang reused saved memory pool config (0 context overrides replayed)
[Foundry] SGLang alloc_offset[before_preallocate]=… (… MB)
[Foundry] LOAD memory restore: … live ranges (… MB) mapped in … ms
[Foundry] SGLang alloc_offset[after_preallocate]=… (… MB)
[foundry] Using graph_manifest.json (9 topology groups)
[foundry] Phase 1: 52 graphs parsed in 0.x ms (…), 9 topologies, 4 threads, 52 binary + 0 json
[foundry BUILD] Template 0 (...): N nodes, done in X.X ms
…
[foundry] Phase 2: 9 templates + 43 on-demand = 52 graphs built in xx.x ms (…)
[Foundry] SGLang alloc_offset[before_target_decode_restore]=… (… MB)
[Foundry] Loaded 52 SGLang graphs in 0.0xs (builds 0.0xs, handover 0.00xs)
[Foundry] SGLang decode capture loop on LOAD: 0.xs (52 shapes, restore 0.0xs, per-shape prep 0.xs)
[Foundry] SGLang alloc_offset[after_load_target_decode_graphs]=22785556480 (… MB)
[Foundry] SGLang alloc_offset[after_load_all_graphs]=22785556480 (… MB)
…
INFO:     Application startup complete.
```

The `after_load_all_graphs` value **must** equal SAVE's `final_alloc_offset`. If it doesn't, see [`memory-consistency.md`](memory-consistency.md).
The build lines (`[foundry] Phase 1/2`) come from the background build thread and may interleave with the offset lines.
The graphs are finished one per shape inside sglang's capture loop. The two timing lines, logged once per runner
(decode above; the prefill runner logs `Loaded N SGLang prefill graphs in ...` and `SGLang prefill capture loop on
LOAD: ...` the same way when prefill graphs are on), mean:

- `Loaded N ... in X s`: only Foundry's restore work. `builds` is the time from the start of this runner's restore
  (the builds being launched) to the last build / instantiate
  finishing (Phase 1 + Phase 2). `handover` is the part of the per-shape `restore_next_*` calls after the builds
  finished (allocator replay, output reconstruction). A call's wait for the builds is counted once, in `builds`.
  X = builds + handover.
- `capture loop on LOAD: Y s`: the wall time of sglang's own `capture()` on LOAD. `per-shape prep` is Y minus the time
  spent inside the restore calls: sglang's work around the restores (the run-once kernel `warmup()`, the two
  `gc.collect()` of `freeze_gc`, and per shape the dummy batch and the attention metadata). The builds run on the
  background thread while the loop runs, so restore + prep can exceed Y.

On Qwen3-1.7B (FlashInfer, prefill 8-64 + decode 1-8, one H200) Phase 2 took ~11 ms (prefill) and ~19 ms (decode),
so the restore is tens of milliseconds per runner, while the loops take ~1.1 s (prefill, which also runs the run-once
`warmup()`) and ~0.7 s (decode).

An archive saved by older code (no `capture_loop_version` in `region_layout.json`) is refused at setup, before the
weights load: `Foundry archive ... has capture_loop_version=None, this code needs 2: re-SAVE with the current code`.
Re-run SAVE.
