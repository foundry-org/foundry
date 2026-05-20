# SGLang Direct Edits

Total footprint in the `sglang/` repo: **47 lines across 4 files**. Every other line of integration logic lives in `foundry.integration.sglang`.

## 1. `sglang/srt/foundry_shim.py` (new, ~27 lines)

The activation entry point. Re-exports the foundry public API and forces flags that conflict with SAVE/LOAD determinism.

```python
def apply_server_args(server_args) -> None:
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return

    server_args.disable_piecewise_cuda_graph = True
    server_args.enable_profile_cuda_graph = False
    server_args.disable_flashinfer_autotune = True

    from foundry.integration.sglang.hooks import install_hooks
    install_hooks(server_args)
```

`install_hooks` is idempotent (guarded by a module-level `_INSTALLED` flag) so calling it multiple times across processes is safe.

## 2. `sglang/srt/server_args.py` (~12 lines)

Adds the config-path field, the CLI argument, and the `__post_init__` activation:

```python
class ServerArgs:
    ...
    foundry_graph_extension_config_path: Optional[str] = None
    ...

    def __post_init__(self):
        ...
        if self.foundry_graph_extension_config_path:
            from sglang.srt.foundry_shim import apply_server_args
            apply_server_args(self)
        ...

    @staticmethod
    def add_cli_args(parser):
        ...
        parser.add_argument(
            "--foundry-graph-extension-config-path",
            type=str,
            default=ServerArgs.foundry_graph_extension_config_path,
            help="Path to Foundry CUDA graph extension TOML config.",
        )
```

The field must live on `ServerArgs` (not be monkey-attached) because:

- `argparse` consumes it from the CLI.
- `ServerArgs` is pickled across the `spawn` boundary; only declared fields survive serialization.

## 3. `sglang/srt/managers/scheduler.py` (~4 lines)

```python
def run_scheduler_process(server_args, ...):
    load_plugins()
    if server_args.foundry_graph_extension_config_path:
        from sglang.srt.foundry_shim import apply_server_args
        apply_server_args(server_args)
    ...
```

This is the first thing every spawned scheduler child runs. Without it, foundry's runtime monkey-patches aren't installed in the scheduler process and the model-loading lifecycle takes the upstream path.

## 4. `sglang/srt/managers/data_parallel_controller.py` (~4 lines)

```python
def run_data_parallel_controller_process(server_args, ...):
    parent_process = psutil.Process().parent()
    configure_logger(server_args)
    if server_args.foundry_graph_extension_config_path:
        from sglang.srt.foundry_shim import apply_server_args
        apply_server_args(server_args)
    ...
```

Same purpose as the scheduler call, but for the DP-controller child. This child later spawns its own scheduler subprocesses, so it must have foundry's spawn-site patches installed before doing so — otherwise its nested `proc.start()` calls won't set `LD_PRELOAD` in their environments.

## Why these can't be moved into the integration layer

| Edit | Why direct |
|---|---|
| `foundry_shim.py` | Has to be importable as `sglang.srt.foundry_shim`. It's the only file foundry's hooks can rely on being present in the sglang import path. |
| `ServerArgs.foundry_graph_extension_config_path` | Needs to be a real dataclass field (argparse + pickle). |
| `ServerArgs.__post_init__` call | Earliest activation point in the parent process. |
| `run_scheduler_process` install | `spawn` does not inherit Python state; the child needs an explicit re-install. |
| `run_data_parallel_controller_process` install | Same reason; this child also spawns further children. |
