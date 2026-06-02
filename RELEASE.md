# Foundry 0.0.2

SGLang graduates to a fully validated engine. This release brings the SGLang
integration to parity with vLLM across single GPU, data parallel, and expert
parallel — with a self-contained recipe and no vLLM build dependency for EP.

## Highlights

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

## Engine integrations

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

## Fixes

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

## Docs

- New `docs/sglang/` set (overview, direct-edits, hooks, memory-lifecycle,
  save-load-workflow, memory-consistency) and a self-contained `recipe/sglang/`
  README with install, run, performance, and troubleshooting.
- Top-level README parallelism status table updated to mark SGLang single GPU,
  DP, and EP as validated.

---

## Previous Releases

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
