# vLLM Direct Edits

Total footprint in the `vllm/` repo: **~97 lines across 5 files** (+ a small flashinfer compatibility adjustment). Every other line of integration logic lives in `foundry.integration.vllm`. Foundry-side config patches (env vars, compilation_config fields) are applied at runtime from `install_hooks` — no additional vLLM-side changes are needed for them.

```
 vllm/compilation/decorators.py    | 11 +++++++++
 vllm/compilation/foundry_shim.py  | 50 ++++++++++++++++++++++++++++++++++++++++
 vllm/config/compilation.py        | 13 +++++++++++
 vllm/utils/flashinfer.py          |  8 +++----
 vllm/v1/engine/core.py            |  6 +++++
 vllm/v1/worker/gpu_worker.py      | 13 +++++++++++
```

## 1. `vllm/compilation/foundry_shim.py` (new, ~50 lines)

The re-export point. Imports foundry's public API with an `ImportError` fallback so vLLM still builds when foundry is not installed.

```python
try:
    from foundry.integration.vllm import (
        CUDAGraphExtensionMode, get_graph_extension_mode, install_hooks,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    import enum

    class CUDAGraphExtensionMode(str, enum.Enum):
        NONE = "none"; SAVE = "save"; LOAD = "load"

    def get_graph_extension_mode():
        return CUDAGraphExtensionMode.NONE

    def install_hooks(compilation_config):
        return None
```

`install_hooks` is idempotent (guarded by `_INSTALLED`) so the multiple activation sites below all converge on a single install per process.

## 2. `vllm/config/compilation.py` (~13 lines)

```python
@dataclass
class CompilationConfig:
    ...
    graph_extension_config_path: str | None = None
    """Path to the foundry CUDA graph-extension TOML config. ..."""

    def __post_init__(self):
        ...
        if self.graph_extension_config_path:
            from vllm.compilation.foundry_shim import install_hooks
            install_hooks(self)
```

This is the earliest activation point in the parent process. As soon as a `CompilationConfig` is fully constructed, foundry's runtime patches are in place.

## 3. `vllm/v1/engine/core.py` (~6 lines)

```python
class EngineCore:
    def __init__(self, vllm_config, ...):
        ...
        # foundry: install_hooks runs here because CompilationConfig
        # __post_init__ does not re-fire on pickle/spawn deserialization.
        if vllm_config.compilation_config.graph_extension_config_path:
            from vllm.compilation.foundry_shim import install_hooks
            install_hooks(vllm_config.compilation_config)
        ...
```

`EngineCore` runs in the engine-core process. `CompilationConfig` was pickled into this process, so `__post_init__` did not re-fire — we re-install explicitly. Idempotent.

## 4. `vllm/v1/worker/gpu_worker.py` (~13 lines)

```python
class Worker(WorkerBase):
    def __init__(self, vllm_config, ...):
        ...
        # foundry: same reason as EngineCore — re-install in worker.
        if vllm_config.compilation_config.graph_extension_config_path:
            from vllm.compilation.foundry_shim import install_hooks
            install_hooks(vllm_config.compilation_config)
        ...
```

Workers run in their own processes (one per GPU under MP). The `Worker.__init__` line guarantees foundry is installed before any worker-side CUDA work.

## 5. `vllm/compilation/decorators.py` (~11 lines)

```python
def _support_torch_compile(...):
    ...
    from vllm.compilation.foundry_shim import (
        CUDAGraphExtensionMode, get_graph_extension_mode,
    )
    self.do_not_compile = (
        ...                                            # upstream conditions
        or get_graph_extension_mode() == CUDAGraphExtensionMode.LOAD
    )
    if self.do_not_compile:
        if get_graph_extension_mode() == CUDAGraphExtensionMode.LOAD:
            logger.warning(
                "Compile is disabled due to graph extension mode is LOAD."
            )
        return
    ...
```

On LOAD, the wrapped class never gets the `_support_torch_compile.__call__` machinery installed (early return). Dispatch goes through `gpu_model_runner.py:4884`'s `CUDAGraphWrapper(self.model, ...)` reassignment instead — `self.model(...)` from `_dummy_run` calls the wrapper, which on LOAD goes straight to `finish_one_graph_load(...)` for replay. The inner runnable is never invoked, so skipping torch.compile setup is safe and saves the ~2-4s Dynamo + AOT-cache-load cost on cold start.

## 6. `vllm/utils/flashinfer.py` (~8 lines, small adjustment)

Minor compatibility fix to keep the foundry-mode flashinfer import path clean. Not foundry-specific in intent; folded in with the integration.

## Runtime config patches (no vLLM source change)

`install_hooks` applies one config patch at runtime — it's not in the vLLM tree, but it's load-bearing for correctness. Lives in `foundry/python/foundry/integration/vllm/hooks.py::install_hooks` and stomps a value that vLLM's `VllmConfig.__post_init__` would otherwise force:

```python
# vLLM hard-overrides this to 1 (vllm/config/vllm.py:1052) whenever cudagraph
# mode is enabled. The warmup _dummy_run(rm=NONE, force_attention=True) goes
# through CUDAGraphWrapper.__call__ which short-circuits to self.runnable(...) —
# an *uncaptured* forward whose allocations break SAVE↔LOAD VMM parity.
compilation_config.cudagraph_num_of_warmups = 0
```

`VLLM_USE_AOT_COMPILE` is not touched — it defaults to `False` in `vllm/envs.py`, which is what we want. If a future vLLM build flips the default to `True`, we'd need to force-set it back to `False` here (an AOT cache-hit forward would otherwise produce non-deterministic allocations between SAVE and LOAD).

See [`overview.md`](overview.md#critical-invariants-read-this-first) for the failure modes these prevent.

## Why none of these can move into the integration layer

| Edit | Why it's a direct edit |
|---|---|
| `foundry_shim.py` | Has to be importable as `vllm.compilation.foundry_shim`. Cannot live elsewhere. |
| `CompilationConfig.graph_extension_config_path` | Needs to be a real dataclass field for pickling across spawn / Ray and for serialization in vLLM's config system. |
| `CompilationConfig.__post_init__` | Earliest activation; parent-process bootstrap. |
| `EngineCore.__init__` | Re-install for the engine-core child process. |
| `Worker.__init__` | Re-install for worker child processes (multiproc / Ray). |
| `decorators._support_torch_compile` | The compile-disable check needs to see foundry's mode at the precise point upstream computes `do_not_compile`. Wrapping the decorator function from outside is fragile. |

## Patches that *did* move into the integration layer

| Edit | Now |
|---|---|
| Pre-fork `LD_PRELOAD` env mutation | `_patch_subprocess_spawn_sites` patches `WorkerProc.make_worker_process`, `CoreEngineProcManager.__init__`, and `CoreEngineActorManager.__init__` |
| `CUDAGraphWrapper.__call__` capture-block swap | `_patch_cuda_graph_wrapper_call` replaces the inner capture stanza with `capture_or_load_graph(...)` |
| `kernel_warmup`, V1 sampler-warmup full-forward skipping | `_patch_kernel_warmup`, `_patch_dummy_runs` (kwarg-signature targeted) |
| `capture_final_alloc_offset` on SAVE | `_patch_capture_model` post-hook (folded in; no separate `_patch_compile_or_warm_up_model` anymore) |
| MoE `DeepEPLLAll2AllManager._make_all2all_kwargs` `use_fabric=True` | `_patch_deepep` |
| `_initialize_kv_caches` profile skip + `WarmupState` write | `_patch_initialize_kv_caches` |
