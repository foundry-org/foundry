# Foundry 0.0.1

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
