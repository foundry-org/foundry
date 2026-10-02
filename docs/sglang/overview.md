# SGLang Integration Overview

Foundry persists SGLang's decode (and optionally prefill) CUDA graphs to disk on SAVE and restores them on LOAD, skipping graph capture, kernel warmup, and the per-batch-size attention metadata setup costs.

Foundry runs as an **SGLang plugin**: `pip install -e foundry` registers the entry point `foundry = "foundry_sglang_plugin:load"` in group `sglang.srt.plugins`, and `FOUNDRY_GRAPH_EXTENSION_CONFIG=<TOML>` in the launcher's environment switches it on. No SGLang change is needed (validated on upstream main `fa090f7755`, fork branch `foundry-plugin`). The plugin wraps three whitelisted resolution steps to pin the graph settings and installs the runtime patches in [`hooks.md`](hooks.md) in the launcher and in every scheduler process. The earlier in-tree route (fork flag `--foundry-graph-extension-config-path`, [`direct-edits.md`](direct-edits.md)) is superseded.

**Dependency route.** An SGLang that declares Foundry as an optional dependency (`sglang[foundry]`, fork branch
`foundry-dep`) calls `foundry.integration.sglang.api` from its own call sites instead of being patched:
`--cuda-graph-persistence {save,load}` switches it on and `--cuda-graph-persistence-config <TOML>` names the
Foundry TOML (optional; the flag's mode wins over the TOML's `mode`). The pins, rejects and runtime steps are the
same code as the plugin's (`plugin.py`, `hooks.py`); only the messages name the flag. `INTEGRATION_API_VERSION` in
`api.py` is checked by SGLang's adapter (major must match). The scheduler preload is set only around each spawn
(`api.configure_subprocess`), not for the rest of the launcher. On such an SGLang the plugin refuses to activate
(`FOUNDRY_GRAPH_EXTENSION_CONFIG` set -> exit with a pointer to the flag).

## Parallelism

| Mode | Status | Notes |
|---|:---:|---|
| Single GPU | ✅ | Qwen3-1.7B / 4B / 14B |
| Data parallel (DP) | ✅ | One full replica per rank; validated DP=2. Requires the per-rank device binding (below); `NCCL_CUMEM_ENABLE=0` / `NCCL_NVLS_ENABLE=0` are pinned by the plugin. |
| Tensor parallel (TP) | ✅ | torch symmetric-memory allreduce inside the decode graphs (`--enable-torch-symm-mem --disable-custom-all-reduce`, pinned by the plugin); validated TP=2 (Qwen3-1.7B, Qwen3-32B) and TP=4 (Qwen3-32B). Without multicast (no IMEX channels) LOAD needs the optional sglang branch `symm-mem-no-multicast-fallback` (two-shot instead of disabling the communicator). |
| Expert parallel (DeepEP) | ✅ | DP attention + DeepEP low-latency, TP attention + EP, and attention-TP inside DP groups + EP; DeepEP v2. Validated up to EP8 (Qwen3.5-122B/397B, DeepSeek-V4-Flash, GLM, Inkling; see `recipe/sglang/README.md` **Validation**). See **Expert parallel** below. |

**Expert parallel (DeepEP).** The default EP recipe runs DP-attention + DeepEP for the
MoE all-to-all (NVSHMEM, foundry-compatible); TP attention + EP (`serve_qwen3-30ba3b_ep_tpattn.sh`,
the `tpepN` / `tpAepN` topologies of `serve_common.sh`) routes the attention all-reduce
through torch symmetric memory. The serve script is `recipe/sglang/serve_qwen3-30ba3b_ep.sh
<ep_size> [--save|--load]` with: `--enable-dp-attention --moe-a2a-backend deepep
--deepep-mode low_latency --moe-runner-backend deep_gemm --attention-backend fa3`
(custom all-reduce off and torch symm-mem on come from the plugin's pins). The EP kernels (`sgl-deep-ep`, `sgl-deep-gemm`, fa3 inside
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

## What the plugin pins and why

Foundry's rule: state that cannot be made static across processes is not supported; Foundry uses the cuMem-backed
alternatives instead. A graph restored on LOAD replays only what was recorded inside the capture region. A buffer
registration, IPC handle exchange or pool carve-out made outside it (when `graph_capture()` exits, at first use
inside the forward, by a second graph or allocator) does not exist in the LOAD process, and the restored kernels read
addresses nobody set up. Every such feature that a flag or an environment variable can switch off is therefore a
config the plugin sets, not a recipe requirement:

- each pin prints one `[Foundry] pin: <name>=<value> (<sglang default | as given | overrides sglang default X |
  was unset | already set>): <reason>` line;
- a value set explicitly to the contrary (a flag on the command line, a variable in the launcher's environment)
  stops the launch with that pin's reason, so a pin never overrides a user's choice silently;
- a later sglang resolution step (model override, platform fallback) that turns a strict pin back is rejected,
  naming the step, instead of being re-pinned after other fields were derived from it.

Flags are declared by the plugin's resolution hook on `apply_inkling_prefill_cuda_graph_default` (before the CUDA-graph,
platform and memory steps read them) and re-checked on `handle_other_validations`. Environment variables are set in
the launcher's `activate()` (before the server args are built and before any engine process is spawned: spawned
processes inherit `os.environ`, and sglang reads some of them once at import time), re-asserted at the top of every
scheduler process and at each spawn site (`setup_ld_preload_env`). `NCCL_CUMEM_ENABLE` depends on the all-to-all
backend and is pinned by the `handle_other_validations` hook, once `--moe-a2a-backend` is resolved.

| Pinned | Value | Why | Error when |
|---|---|---|---|
| `--disable-custom-all-reduce` (sglang default off; covers custom all-reduce v1 and v2, `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` only picks the class) | on | custom all-reduce registers the graphs' buffers over CUDA IPC when `graph_capture()` exits, outside the recorded region | never (the contrary value is sglang's default) |
| `--enable-torch-symm-mem` | on | the in-graph all-reduce runs on torch symmetric memory, a cuMem buffer Foundry places at the same address on SAVE and LOAD; sglang may turn it off (deterministic inference), then the all-reduce is NCCL, which replays | never |
| `--enable-symm-mem` (NCCL windows) | off | registered at first use inside the forward; needs NCCL's cuMem buffers | set on the command line |
| `--enable-nccl-nvls` | off | NVLS multicast buffers registered at first use inside the forward | set on the command line |
| `--enable-mscclpp` | off | MSCCL++ registers its buffers at first use inside the forward | set on the command line |
| `--enable-two-batch-overlap` | off | not validated with Foundry (children micro-batch graphs and DeepEP capture events); pinned off until a TBO row is validated | set on the command line |
| `--enable-memory-saver` | off | torch_memory_saver owns the pools (and, with `SGLANG_MEMORY_SAVER_CUDA_GRAPH`, the graph memory) outside Foundry's region | set on the command line |
| `--dsv4-attn-backend` | `auto` (resolves to `flashmla`), or the user's `flashmla` | the `trtllm` variant asserts at its first call, which falls inside the capture, and needs an eager bootstrap Foundry does not run | `trtllm` |
| `NCCL_GRAPH_REGISTER`, `NCCL_LOCAL_REGISTER` | `0` | NCCL registers the buffers of graph-captured collectives on the host at capture time; a restored graph replays the kernels without the registration (illegal address in the DP-attention all-gather at bs >= 4 with NCCL 2.30) | set to anything but 0 |
| `NCCL_CUMEM_ENABLE` | `0`, except with `--moe-a2a-backend deepep_v2` (left to sglang, which sets 1: v2's NCCL windows need cuMem) | NCCL's cuMem buffers are mapped with driver flags Foundry's VMM region does not carry; the plain allocator keeps them at deterministic offsets | set to anything but 0 (not with deepep_v2) |
| `NCCL_NVLS_ENABLE` | `0` | NVLS multicast buffers: same mapping and registration problem | set to anything but 0 |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | `0` | the precompile sweep runs on the first rank inside the first capture: a device synchronization, and scratch only SAVE allocates; kernels still JIT per shape | set to true |
| `SGLANG_MEMORY_SAVER_CUDA_GRAPH` | `0` | the memory-saver graph context owns the graph memory outside Foundry's region | set to true |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH` | `0` | the attention-metadata prep is captured into a second graph Foundry does not save | set to true |
| `SGLANG_ENABLE_GRAPH_POOL_PRECARVE` | `0` | only upstream `capture_one` runs the precarve (measured over its two eager warmups, minted at the first capture); Foundry's `capture_one` replaces it on SAVE and LOAD, so the flag would be a no-op and the graph pool would not match a native run's layout | set to true |
| `SGLANG_ENABLE_GRAPH_POOL_BORROW` | `0` | eager allocations would borrow free graph-pool extents whose addresses the restored graphs reference | set to true |

Rejected instead, because Foundry has no logic for them: speculative decoding, LoRA, PD multiplexing, elastic-EP
recapture, attention graph variants, non-full graph backends, `--disable-cuda-graph`, CUDA-graph profiling,
`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`. Profiling and FlashInfer autotune are pinned off with the graph backends.

Still documented requirements (a flag cannot express them): dense no-padding decode capture with an explicit
batch-size list, DeepEP low-latency on EP rows, JIT caches warm before SAVE, pinned hybrid state pools, and identical
flags and environment on SAVE and LOAD (see `recipe/sglang/README.md`).

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
- [`known-issues.md`](known-issues.md) — open issues and their status
- [`graph-state-checklist.md`](graph-state-checklist.md) — what a restored graph must reproduce: the four divergence mechanisms, every SGLang flag / backend / kernel that changes graph attributes or contents, pinned vs. rejected settings, and the open gaps (with the driver-side and SGLang-side reviews it merges)
- [`validated-configs.md`](validated-configs.md) — every model and configuration that passed SAVE + LOAD, with timings; the stage breakdown is `figs/stages_pr.png`
