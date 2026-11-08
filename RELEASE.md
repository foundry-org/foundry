# Foundry 0.1.0rc4

Third release candidate of 0.1.0 (same wheel pairs and index layout as rc2).

## Highlights

- **Native fallback outside the supported configurations** (SGLang integration
  API 2.4). Before any graph is restored, Foundry raises `Unsupported(step,
  reason)` at its detection points (environment and server-args pins, the
  `invalid.json` marker of an archive whose SAVE fell back, the capture state at
  the first capture scope, and on SAVE every captured shape); SGLang's adapter
  logs `[Foundry] unsupported at <step>: <reason>; cuda-graph persistence
  disabled, continuing natively` and captures its graphs natively. The scope and
  capture decisions are voted over the world group, so all ranks fall back
  together; `--disable-cuda-graph` makes persistence a quiet no-op; every
  scheduler ends its init with one summary line (graphs saved / restored /
  disabled at step). After LOAD starts restoring graphs, failures stay errors.
- **EAGLE / EAGLE3 speculative decoding graphs are persisted**: the
  target-verify, draft-decode and draft-extend graph sets, each saved and
  restored on its own (`capture_scope('decode', kind=...)`,
  `decode_graphs.json` records scope order and output structure). Adaptive and
  multi-layer EAGLE and the other algorithms fall back.
- **FlashInfer tuned tactics travel with the archive**: autotune stays pinned
  off, SAVE installs SGLang's per-rank autotune cache file and keeps its content
  in the rank's `warmup_state.json`, LOAD installs that section (12-17 % decode
  throughput on FlashInfer MoE paths; foundry#25).
- **Pre-capture state is built by SGLang, checked by Foundry**: SGLang's
  `BaseRunner._prepare_capture_state` builds the lazy communication state
  (DeepEP / Mooncake buffers, dispatcher expert mappings, the logits gatherer)
  before the first capture scope on both modes through the dispatcher
  `prebuild()` protocol; Foundry's per-backend bootstraps are gone and a
  dispatcher without the protocol is reported as unsupported at `capture state`.
- **Deterministic pre-layout allocations**: `ProcessGroup.barrier()`'s async
  Work is waited at the call and the cyclic collector is held off from
  `before_parallel_init` to the layout start, so two run-dependent frees no
  longer move host tensors that EAGLE3 draft graphs address (DeepEP v2 target:
  draft-decode index out of bounds or a lower accept length on LOAD).
- **Graph fidelity across SAVE/LOAD**: event creation flags (rebuilt event
  nodes no longer carry timing: 2.6 us per record/wait pair on replay), kernel
  node cluster dimensions, the cluster count in the exec-share topology key (a
  grid-only `cuGraphExecUpdate` of a cluster node leaves its device cluster
  constants stale on driver 595: FlashInfer's Lamport all-reduce fusion then
  read a wrong token stride), failed kernel-node attribute queries are reported
  (`FOUNDRY_ALLOW_ATTR_QUERY_FAILURE=1` to continue), `loaded_events` /
  `bind_events` let the host bind its own events to a restored graph, the hook
  resolves `dlsym(RTLD_NEXT)` from the caller.
- **Pins and rejections** live in `pins.py` (the plugin module is gone); new
  since rc3: `SGLANG_ENABLE_DCP_PUSH_COMBINE=0` (the DCP push-combine path
  builds its own CustomAllReduceV2 before capture, independent of
  `--disable-custom-all-reduce`) and `ShapeKey.logits_variant` (sharded verify)
  rejected like `attention_variant`. What rc3 rejected with an error (a
  contrary pin, LoRA, PD multiplexing, elastic-EP recapture, non-EAGLE
  speculative decoding, `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`) now falls back
  natively.
- FlashInfer SM90 GDN CuTe-DSL prefill kernels come from Foundry's object cache
  again by default (`FOUNDRY_FLASHINFER_GDN_CUTE_CACHE=0` turns it off): 10 s
  of JIT per Qwen3.5 start.

## Removed

- The plugin route (`FOUNDRY_GRAPH_EXTENSION_CONFIG`, `plugin.py`): SGLang's
  own call sites (`--cuda-graph-persistence`) are the only route; the API
  2.2-only entry points (`record_memory_pool_overrides`,
  `before/after_alloc_memory_pool`) with it.
- Foundry's per-backend pre-capture bootstraps (`bootstrap_logits_gatherer`,
  `bootstrap_deepep_buffer` and the v2 / Mooncake variants,
  `bootstrap_expert_mappings`); SGLang builds that state (see Highlights).
- The scratch skip: `scratch_space_size` in the TOML is ignored (logged once).
- The post-load `empty_cache`, the dynamo-related SAVE bootstraps (dynamo is
  pinned off), `FOUNDRY_SGLANG_CAPTURE_TRACE`.
- LOAD of archives saved before API 2.1 (no `decode_graphs.json` record, or no
  output structure): refused with a message; re-SAVE.

## Validation

2x H200 (driver 595.71.05, torch 2.14.1 / cu130), 2026-11-07/08, SGLang
`foundry-dep` (9fccf82eda): unit + `tests/sgl` suites and SGLang's adapter
and e2e tests green; Qwen3-30B-A3B-FP8 TP2 (262 graphs per rank), EP2 DeepEP
v2 (256) and EP2 + EAGLE3 (72 across three kinds): SAVE and LOAD allocation
offsets equal on every rank, LOAD greedy outputs and logprobs identical to
native (sequential and batched), EAGLE3 accept length identical; native -> LOAD
to the first `/health` 200: 103 -> 37 s, 116 -> 37 s, 45 -> 38 s. Fallback
rows (a dispatcher without `prebuild`, LOAD of the archive its SAVE marked
invalid, state created inside a capture, a reason found on one rank only, a
contrary pin, `--disable-cuda-graph`) start natively with output identical to
native. 1x H200, 256 decode
graphs per set with SGLang's server warmup skipped: Qwen3.5-2B 53.5 -> 25.3 s,
Qwen3-1.7B (FlashInfer, 4 prefill graphs) 40.5 -> 28.8 s, Qwen3-30B-A3B-FP8 +
EAGLE3 (768 graphs) 109.4 -> 31.4 s; restore 0.35 / 0.65 / 1.3 s; the first
request after LOAD costs the same as after a native start.

## Upgrading

- Re-SAVE every archive: the archive layout (per-kind graph sets,
  `decode_graphs.json`, tactics in `warmup_state.json`) and the pre-layout
  allocation sequence changed; archives from rc3 are refused.
- SAVE and LOAD must run with the same environment and flags: DeepEP v2
  graphs take NCCL's window by an address the driver places (512 MiB steps),
  which Foundry does not control; the recipes' `set_env.sh` is the shared set.
- SGLang `foundry-dep` is rebased on upstream main (FlashInfer 0.7.3,
  sgl-deep-gemm 0.2.6) and requires integration API 2.x with minor >= 4, i.e.
  `foundry-core >= 0.1.0rc4`; the `sglang[foundry]` extra pins it.
- Archives saved with a DeepEP v3 / NIXL / Mori / Pplx dispatcher, LoRA, PD
  multiplexing or a non-EAGLE speculative algorithm were never valid; such
  servers now start natively with the `unsupported at` line instead.

## Previous Releases

## Foundry 0.1.0rc3

Second release candidate of 0.1.0 (same wheel pairs and index layout as rc2).

### Highlights

- **Exec share mode** (`FOUNDRY_ENABLE_EXEC_SHARE=1`): one executable graph per
  template, members switched with `cuGraphExecUpdate`; members are grouped by
  their per-node kernel binaries, not only by function name. The lazy exec path
  is gone; Phase-2 errors propagate. Works without cudagraph-patch (real
  `cuGraphExecUpdate` route) and takes the patched route when the patch is
  preloaded (weak `cgp_*` symbols).
- **Faster SGLang bootstrap:** `NVSHMEM_DISABLE_NCCL=1` is pinned (NVSHMEM's
  per-team NCCL communicators cost 2.4-6 s and 1.5 GB per DeepEP bootstrap and
  are unused by the low-latency kernels: 3.5 -> 0.7 s), and Triton's cache key
  is computed on a background thread during weight load (-0.3 s per LOAD).
- A device-linked library loaded with `cuLibraryLoadData` (NCCL 2.30.7) is
  restored with the same call instead of `cuLinkAddData`, which failed with
  error 209 on drivers older than the library's toolkit.
- **Recipes** (`recipe/sglang/`): plain `--cuda-graph-persistence` serve
  scripts, one shared `set_env.sh` (dynamo off, DeepGEMM precompile off,
  NVSHMEM NCCL off, DeepEP v2 GIN off, single-node IB off, host-registered
  pinned buffers), dense decode graphs 1..256 by default, GLM-5.3-Flash and
  Qwen3-235B-A22B added, `--attn-dp-size` instead of the deprecated
  `--dp-size N --enable-dp-attention`.
- **Tests** restructured into tiers with markers applied by directory
  (`tests/unit`, `core`, `multigpu`, `sgl`, `sgl/e2e`); `pytest tests` runs in
  about 5 minutes on one GPU, the Qwen3.5-35B-A3B 2-GPU e2e (tp2, ep2, tpep2)
  and the 4-GPU cases are opt-in (`tests/README.md`).

### Removed

- `foundry.graph.CUDAGraph.instantiate` / `raw_cuda_graph_exec` (never bound).

### Validation

2x H200 (driver 595.71.05, torch 2.14.1 / cu130, SGLang foundry-dep
e5ba719e30), 2026-10-16: default test run 207 passed; Qwen3.5-35B-A3B tp2 /
ep2 / tpep2, Qwen3-14B tp1 / tp2 / dp2 and Qwen3-30B-A3B-FP8 ep2 (DeepEP v2):
native, SAVE and LOAD greedy outputs identical, LOAD restores 256 graphs per
rank in 0.26-0.44 s against 16-25 s of native capture, TPOT within 0.3% of
native. Exec share LOAD without cudagraph-patch verified (real route, 256
graphs per rank in 0.7 s). The 16 cold-start rows of the paper (4x H200)
remain strict-exact with the NVSHMEM pin.

### Upgrading

- Archives written with rc2 still load; re-SAVE to get the `cuLibraryLoadData`
  restore and the NVSHMEM-pin allocation layout (SAVE and LOAD must run with
  the same `set_env.sh`).
- SGLang's `--dp-size N --enable-dp-attention` is deprecated: use
  `--attn-dp-size N` (the recipes do).

## Foundry 0.1.0rc2

Release candidate of 0.1.0: the first release on PyPI (`foundry-core`, import
name `foundry`) and the first one SGLang can depend on.

### Highlights

- `pip install foundry-core`: manylinux_2_28 wheels for CPython 3.10-3.13,
  built against torch 2.14.1 / cu130 (SGLang's pin).
- One package index per torch/CUDA pair:
  `pip install foundry-core --extra-index-url https://foundry-org.github.io/foundry/whl/<cuda>/torch<A.B>/`
  (0.1.x also builds torch 2.13.0 / cu130).
- Optional SGLang dependency: `pip install "sglang[foundry]"`, then
  `--cuda-graph-persistence {save,load}` (integration API 2.0).
- Prefill CUDA graphs are saved and restored next to the decode graphs.
- torch 2.14 support: the scheduler processes run with `TORCHDYNAMO_DISABLE=1`
  (pinned by Foundry).
- Faster restore: binary templates, build/instantiate pipeline, exec-pool
  prewarm.
- No Boost runtime dependency (header-only, vendored subset).

### Fixes

- Fatbin packing: the archived image is 30-35 MB and mmap'd on LOAD.
- One preallocation mechanism on LOAD.
- Qwen3.5 + FlashInfer LOAD divergence.
- FlashInfer SM90 GDN prefill kernels are loaded from FlashInfer's on-disk
  cache instead of being compiled per process.
- Optional udev-wait shim for bare hosts (`tools/host/no_cdev_wait.c`, TOML
  `verbs_udev_wait_shim_path`).
- The hook preload is scoped to the scheduler spawn.

### Validation

4xH200, torch 2.14.1 / cu130: Qwen3-30B-A3B-FP8 (real weights, 7 layouts),
Qwen3-235B-A22B-FP8, Qwen3.5-122B-A10B-FP8, Qwen3.5-35B-A3B,
DeepSeek-V4-Flash-FP8, GLM-5.3-Flash and gpt-oss-120b in EP, TP and
DP-attention layouts, plus the 1-GPU e2e; native, SAVE and LOAD outputs agree
and LOAD TTFT / TPOT match native (`docs/sglang/validated-configs.md`).

### Packaging

- `FOUNDRY_WHEEL_BUILD=1` builds a relocatable wheel; `FOUNDRY_SDIST=1` builds
  the sdist without a CUDA toolkit.
- `.github/workflows/release.yml` builds, repairs, smoke-tests and publishes
  (PyPI trusted publishing + GitHub Release + the per-pair wheel index on
  `gh-pages`); `tools/release/build_wheel.sh` runs the same steps locally. See
  `docs/release.md`.

### Upgrading

- `pip uninstall foundry` before installing `foundry-core`.
- Archives written before this release (`capture_loop_version` < 2) must be
  re-SAVEd.
- Source builds need Boost headers only (no `libboost-*` libraries).
- Other torch versions keep building from source
  (`pip install -e . --no-build-isolation`).

## Foundry 0.0.3

Restored graphs now run exactly like captured ones. This release closes the
last per-token latency gap between foundry-restored and natively captured CUDA
graphs, validates SGLang tensor parallelism, and adds DeepEP v2 (NCCL symmetric
windows) support.

### Highlights

- **Restored-graph TPOT at parity with native capture.** Two root causes fixed:
  `cuGraphExecUpdate` on a shared template exec left it permanently slower
  (now every on-demand member gets its own `cudaGraphInstantiate` in LOAD
  Phase 2; `FOUNDRY_ENABLE_EXEC_SHARE=1` shares one exec per template), and libcuda 580–610 applying
  `edge_data[0]` to every edge of a bulk `cuGraphAddDependencies_v2` call,
  which silently dropped all programmatic-dependent-launch (PDL) edges of a
  rebuilt graph (now inserted per homogeneous edge record,
  `include/GraphDependencies.h`, see `docs/pdl-edge-batching.md`). Median TPOT
  of restored vs unmodified SGLang is within ±0.7 % at bs 1–128 on single GPU,
  TP2, EP2 and DeepEP v2 EP2 with all 256 decode graphs.
- **SGLang tensor parallel validated.** torch symmetric-memory allreduce inside
  the decode graphs, TP=2/4, including a two-shot fallback for hosts without
  multicast.
- **DeepEP v2 (NCCL 2.30.7 symmetric windows / GIN) under foundry.** The hook
  answers `cuPointerSetAttribute(SYNC_MEMOPS)` on VMM memory and marks region
  allocations GPUDirect-RDMA capable; the SGLang integration bootstraps the v2
  buffer before capture and disables NCCL graph-time buffer registration.
  Recipe: `recipe/sglang/serve_qwen3-30ba3bfp8_ep_v2.sh`.
- **Greedy-output equality checked, not assumed.** 32 prompts × bs 1/8/32
  against unmodified SGLang run twice: dense TP is bit-identical; MoE configs
  match one of the two baseline runs in every cell (baseline itself is not
  run-to-run deterministic at bs ≥ 8).

### Engine integrations

- **SGLang** — pairs with `foundry-org/sglang` branch `foundry` at `f1d688e52`
  (upstream main of 2026-09-01, post-0.5.18, plus the integration; the 0.0.2-era
  fork is kept on `foundry-0.0.2`). Single GPU, DP, TP (symm-mem), EP (DeepEP
  low-latency) and EP with DeepEP v2 all validated end-to-end with the shipped recipes
  (`recipe/sglang/`, see its Validation section). Recipes gained an
  `SGL_EXTRA_ARGS` passthrough and a DeepEP v2 script; `graph_templates` TOML
  knob to disable topology templating.
- **vLLM** — unchanged; the legacy CUDA-IPC DeepEP recipe moved to
  `recipe/vllm/experimental/`.

### Fixes

- Member kernels whose smem variant differs from the template's re-target
  through the per-context `CUfunction` with `MAX_DYNAMIC_SHARED` set
  (>48 KB DeepGEMM kernels).
- `setup.py` forwards `FOUNDRY_DEBUG` to the hook's CMake build; with the flag,
  every LOAD logs a per-graph check that restored edge records survived.
- Driver/compat notes: `cuLinkAddData ... 209` on LOAD means the userspace
  CUDA library is older than the toolkit NCCL was built with (NCCL 2.30.7
  needs a 13.3-capable driver or `cuda-compat-13-3` for every recipe).

### Docs

- `docs/pdl-edge-batching.md` (root cause, fix, driver reproducer), updated
  `docs/exec-update-penalty.md`, README status and performance tables.

## Foundry 0.0.2

SGLang graduates to a fully validated engine. This release brings the SGLang
integration to parity with vLLM across single GPU, data parallel, and expert
parallel — with a self-contained recipe and no vLLM build dependency for EP.

### Highlights

- **SGLang single GPU / DP / EP all validated end-to-end.**
  SAVE → LOAD → query verified on single-GPU Qwen3-1.7B / 4B / 14B,
  data-parallel (DP=2) Qwen3-1.7B, and expert-parallel (EP=2, DeepEP
  low-latency) Qwen3-30B-A3B-FP8. Foundry-restored decode graphs match baseline
  throughput within run-to-run noise; cold start drops from ~30 s of capture to
  a ~0.4 s restore.
- **NVSHMEM auto-detection — no manual path, no vLLM ep_kernels.**
  Foundry now resolves DeepEP's `libnvshmem_host.so` from the `nvidia-nvshmem`
  wheel (a `torch` dependency) the same way it auto-detects `libcuda_hook.so`.
  DeepEP installs via SGLang's own `ci_install_deepep.sh`; the vLLM EP-kernel
  helper is no longer required.
- **Minimal SGLang fork — 47 lines, activation only.**
  The fork carries just a CLI flag, a config field, and three `apply_server_args`
  call sites (`server_args.py`, `foundry_shim.py`, `scheduler.py`,
  `data_parallel_controller.py`). All save/load logic lives in the integration
  layer; the edits are inert unless `--foundry-graph-extension-config-path` is set.

### Engine integrations

- **SGLang** — integration for SGLang v0.5.13. Working configurations: single
  GPU, data parallel (DP), expert parallel (EP, DeepEP low-latency + DP-attention with fa3). Self-contained recipes under `recipe/sglang/` (shared TOML pair +
  per-config serve scripts, mirroring `recipe/vllm/`) for Qwen3-1.7B (single),
  Qwen3-1.7B (DP), and Qwen3-30B-A3B-FP8 (EP). TP attention stays unsupported
  (NCCL all-reduce vs the VMM region) — EP uses DP-attention instead. EP requires
  SGLang's pinned kernels — DeepEP `9af0e0d` (not vLLM's `29d31c0`),
  `sgl-deep-gemm >= 0.1.2`, and `flash-attn-3` (fa3 sidesteps a FlashInfer
  ragged-prefill off-by-one under this config).

- **Manual DeepEP / NVSHMEM bootstrap (SGLang has no `prepare_comm_buffer`).**
  vLLM exposes `prepare_communication_buffer_for_model`, a clean upstream site
  where foundry brings up the NVSHMEM runtime. SGLang has no equivalent — it
  creates the singleton DeepEP `Buffer` lazily on the first MoE dispatch, inside
  the warmup forwards foundry suppresses, which would push `Buffer(...)` into the
  captured stream ("operation not permitted when stream is capturing"). Foundry
  instead walks the model to the `DeepEPDispatcher` and forces buffer creation
  (`bootstrap_deepep_buffer`) before the capture loop on SAVE and before replay
  on LOAD, with `init_nvshmem_for_loaded_modules` run once on LOAD before any
  NVSHMEM-kernel graph replays.

### Fixes

- **Per-rank VMM device binding (DP/TP/EP).** `set_allocation_region` binds to
  the current CUDA device, so the integration now calls `set_device(gpu_id)`
  before reserving the region — rank > 0 previously reserved on `cuda:0` and
  faulted.
- **Set the main CUDA context on graph-build pool workers (C++).** EP graphs
  carry `NODE_EVENT_RECORD`/`WAIT` nodes, so on-demand prep calls `cuEventCreate`
  on `SimpleThreadPool` workers that had no current context. Added
  `cuCtxSetCurrent(main_ctx)` as the first call on those workers (mirroring the
  background thread) — without it LOAD faulted with "invalid device context".
  Dense graphs never hit this path, which is why it surfaced only on SGLang EP.
- **EP capture/load correctness.** A SAVE-only warmup pass (triggers DeepGEMM
  per-shape JIT + lazy init outside the captured graph), `deepep_adapter` mode
  init on LOAD (replay asserts on `_captured_deepep_mode`, which the replaced
  capture loop never sets), and the FlashInfer per-bs pre-pass gated off for fa3
  while still populating `decode_cuda_graph_metadata` post-load for replay.

### Docs

- New `docs/sglang/` set (overview, direct-edits, hooks, memory-lifecycle,
  save-load-workflow, memory-consistency) and a self-contained `recipe/sglang/`
  README with install, run, performance, and troubleshooting.
- Top-level README parallelism status table updated to mark SGLang single GPU,
  DP, and EP as validated.

---

## Foundry 0.0.1

First public release of Foundry — a CUDA-graph persistence library that
captures an entire model's CUDA graphs (plus their device context: modules,
workspaces, VMM layout) once and replays them at startup, eliminating compile,
warmup, and capture from cold-start time.

## Highlights

- **Deterministic memory layout — zero patching on graph load.**
  Foundry indirects memory allocation to the same reserved memory region
  with monotonic cursor advancing for both SAVE and LOAD. Therefore,
  during CUDA graph reconstruction, captured pointers resolve as-is — no
  pointer rewriting, no per-graph fix-ups, no runtime recompilation.
- **CUDA module extraction — library-agnostic execution context saving.**
  A driver-level `LD_PRELOAD` hook (`libcuda_hook.so`) intercepts CUDA module
  loads at the driver boundary, so foundry captures the device-resident code
  irrespective of which framework, kernel library, or codegen path produced
  it (PyTorch, Inductor, cuBLAS NVJET, NVSHMEM, DeepEP, DeepGEMM, …).
- **Fast graph reconstruction via template-based CUDA graph grouping.**
  Captured graphs are grouped by topology into templates with node-param sets;
  On load, the template is rebuilt once per group and instances are reconstructed
  on demand, keeping load fast and asynchronous.

## Engine integrations

- **vLLM** — compatible with vLLM v0.21. Working configurations: single GPU,
  data parallel (DP), expert parallel (EP, DeepEP low-latency). End-to-end
  recipes under `recipe/vllm/` for Qwen3-1.7B, Qwen3-14B (DP),
  Qwen3-30B-A3B (EP), Qwen3-30B-A3B-FP8 (EP).
- **SGLang** — integration layer for SGLang v0.5.13 (adapted fork coming
  soon).
- **Integration architecture.** Engine-specific logic lives in a thin
  integration layer (`foundry/integration/<engine>/`); engine forks contain
  only minimal hook calls.

## Verified kernel & comm support

- **cuBLAS NVJET** kernels (Hopper+).
- **torch.compile** modules.
- **NVSHMEM / DeepEP** validated.
- **DeepGEMM FP8 MoE** validated.

## Dependency

- **PyTorch 2.11.0** (compatible with 2.9 – 2.11).
- **CUDA 12+**, CMake 4.0+, Boost.

## Documentation

- Integration design notes and per-engine recipes under `docs/` and `recipe/`.
- vLLM recipe README covers save/load workflow, archive layout, and required
  env settings.

## Repository hygiene

- Open-source pre-commit hooks (ruff, ruff-format, clang-format,
  markdownlint, actionlint, DCO sign-off).
- Smoke tests covering re-export imports and archive round-trip.

## Roadmap

- Adapted vLLM and SGLang forks published alongside the release.
- Tensor parallel support.
- Host-process checkpoint/restore.
- PD-disaggregated serving.
- Single shared communication-stub capture across replicas.
