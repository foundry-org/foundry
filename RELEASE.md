# Foundry 0.1.0rc2

Release candidate of 0.1.0: the first release on PyPI (`foundry-core`, import
name `foundry`) and the first one SGLang can depend on. It carries everything
since 0.0.3.

## Highlights

- **`pip install foundry-core`.** manylinux_2_28 wheels for CPython 3.10-3.13,
  built against torch 2.14.1 / cu130, SGLang's current pin; the wheel requires
  `torch==2.14.1`.
  Importing under another torch or CUDA major raises a readable `ImportError`.
- **Optional SGLang dependency.** Installing Foundry does not require SGLang;
  an SGLang started with `--cuda-graph-persistence {save,load}` requires
  Foundry (`pip install "sglang[foundry]"`) and calls
  `foundry.integration.sglang.api` from its own call sites. Settings a
  restored graph cannot replay are pinned with a reason; unsupported features
  are rejected at resolution. `python -m foundry.integration.sglang.preflight`
  checks a launch beforehand.
- **Prefill CUDA graphs** are saved and restored next to the decode graphs.
  With power-of-two buckets, prefill TTFT and throughput of a LOADed engine
  match native capture.
- **torch 2.14 support without disabling Dynamo:** SAVE runs SGLang's warm-up
  forwards in a private CUDA memory pool between two preparation passes and
  releases it before capturing, so compiled helpers and autotuning run
  outside the capture; LOAD still runs no forward.
- **LOAD re-runs SGLang's own capture loop** and substitutes only
  `capture_one`; the restore time is logged apart from the loop.
- **Faster restore:** 0.5-1.6 s per rank for 128 decode graphs on 30B-235B
  models (binary templates, build/instantiate pipeline, exec-pool prewarm).
  Non-portable cluster kernels are opted in on restore.
- **No Boost runtime dependency.** Boost is header-only (vendored subset under
  `third_party/boost`, system Boost >= 1.83 as fallback); `std::filesystem`
  replaces `boost::filesystem`.

## Fixes

- SAVE over-read every fatbin (30 s per rank in distributed init, 4-5 GB
  archives); the packed image is now 30-35 MB and mmap'd on LOAD.
- One preallocation mechanism on LOAD (backing segments, fenced release,
  on-demand holes).
- Qwen3.5 + FlashInfer LOAD divergence.
- Optional udev-wait shim for bare hosts with blocked verbs devices
  (`tools/host/no_cdev_wait.c`, TOML `verbs_udev_wait_shim_path`).
- The hook preload is scoped to the scheduler spawn; only schedulers and the
  DP controller carry `LD_PRELOAD`.

## Validation

4xH200, SGLang main with `--cuda-graph-persistence`: Qwen3-30B-A3B-FP8 (real
weights, 7 layouts), Qwen3-235B-A22B-FP8, Qwen3.5-122B-A10B-FP8,
Qwen3.5-35B-A3B, DeepSeek-V4-Flash-FP8, GLM-5.3-Flash and gpt-oss-120b in EP,
TP and DP-attention layouts. Every row restores in 0.5-1.6 s per rank, reaches
`/health` within a few seconds of the eager engine, and greedy output, TTFT
and TPOT match native capture (`docs/sglang/validated-configs.md`).

## Packaging

- `FOUNDRY_WHEEL_BUILD=1` builds a relocatable wheel; `FOUNDRY_SDIST=1` builds
  the sdist without a CUDA toolkit.
- `.github/workflows/release.yml` builds, repairs, smoke-tests and publishes
  (PyPI trusted publishing + GitHub Release); `tools/release/build_wheel.sh`
  runs the same steps locally. See `docs/release.md`.

## Upgrading

- `pip uninstall foundry` before installing `foundry-core`.
- Archives written before this release (`capture_loop_version` < 2) must be
  re-SAVEd.
- Source builds need Boost headers only (no `libboost-*` libraries).
- Other torch versions keep building from source
  (`pip install -e . --no-build-isolation`).

## Previous Releases

## Foundry 0.0.3

Restored graphs now run exactly like captured ones. This release closes the
last per-token latency gap between foundry-restored and natively captured CUDA
graphs, validates SGLang tensor parallelism, and adds DeepEP v2 (NCCL symmetric
windows) support.

### Highlights

- **Restored-graph TPOT at parity with native capture.** Two root causes fixed:
  `cuGraphExecUpdate` on a shared template exec left it permanently slower
  (now every on-demand member gets its own `cudaGraphInstantiate`, eager by
  default, `FOUNDRY_LAZY_GRAPH_EXEC=1` to defer), and libcuda 580–610 applying
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
