# Foundry 0.1.0

Foundry is now installable as prebuilt wheels from PyPI under the
distribution name **`foundry-core`** (the import name stays `foundry`). This
release is about packaging; the graph save/restore code is unchanged from
0.0.3 apart from the Boost and filesystem changes below.

## Highlights

- **`pip install foundry-core`.** manylinux_2_28 x86_64 wheels for CPython
  3.10-3.13, built against **torch 2.13 (cu130)**. One torch/CUDA pairing per
  release line, the sglang-kernel convention: the PyPI version is plain
  (`0.1.0`) and the wheel requires `torch==2.13.*`. Wheels for other
  torch/CUDA pairs, when built, go on the GitHub Release only, with a local
  version such as `0.1.0+cu128.torch2.12` (PyPI rejects local versions).
- **SGLang dependency route.** SGLang can declare
  `foundry-core>=0.1.0,<0.2` as an ordinary PyPI requirement; a git-ref
  dependency would block SGLang's own PyPI upload. The plugin entry point
  (`sglang.srt.plugins` / `foundry`) and `foundry.integration.sglang.api`
  ship in the wheel, and `foundry/libcuda_hook.so` keeps its path next to
  `foundry.ops` so the preload path SGLang derives is unchanged.
- **No Boost runtime dependency.** Boost is used header-only: Boost.JSON
  compiles through `csrc/boost_json_src.cpp` (once per shared object, hidden
  in the preloaded hook), `boost::filesystem` is replaced by
  `std::filesystem`, and neither `libcuda_hook.so` nor `foundry.ops` links a
  `libboost_*` library. The headers are vendored under `third_party/boost`
  (bcp subset of Boost 1.90.0, `tools/release/vendor_boost.sh`); source builds
  without the vendored copy fall back to a system Boost >= 1.83.
- **Torch build guard.** `foundry.__version__` comes from the installed
  distribution. The build records its torch and CUDA versions in
  `foundry/_build_info.py`; importing `foundry` under a torch with a different
  major.minor or CUDA major raises an `ImportError` that names both builds and
  the fix, instead of an undefined-symbol error
  (`FOUNDRY_SKIP_TORCH_CHECK=1` bypasses it).

## Packaging and release

- `pyproject.toml`: name `foundry-core`, version 0.1.0; `install_requires`
  is set by `setup.py` (`torch` for source builds, `torch==A.B.*` for release
  wheels). `MANIFEST.in` ships the native sources and `third_party` in the
  sdist.
- `FOUNDRY_WHEEL_BUILD=1` builds a relocatable wheel: `foundry.ops` keeps
  only the `$ORIGIN` RPATH (it finds `libcuda_hook.so` beside it; torch is
  imported first) and links `-lcuda` against the toolkit stub.
- `.github/workflows/release.yml`: sdist, a wheel matrix (`PYTHONS` x
  `BUILD_PAIRS`) built in `pytorch/manylinux2_28-builder:cuda13.0`,
  `auditwheel repair` with torch, CUDA runtime, NVRTC, the driver and
  `libcuda_hook.so` excluded, layout and DT_NEEDED checks, an import smoke
  test in a clean venv, PyPI trusted publishing and a GitHub Release.
  `tools/release/build_wheel.sh` runs the same steps on a local host.
- How to cut a release: [`docs/release.md`](docs/release.md).

## Upgrading

- `pip uninstall foundry` before installing `foundry-core`: both own the
  `foundry` import package.
- Source builds no longer need the compiled Boost libraries
  (`libboost-filesystem-dev`, `libboost-json-dev`) or a Boost entry on
  `LD_LIBRARY_PATH`; Boost headers >= 1.83 or the vendored copy suffice.
- The wheels pair with torch 2.13 / cu130. Environments on another torch
  (the vLLM recipe uses torch 2.11) keep building from source with
  `pip install -e . --no-build-isolation`.

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
