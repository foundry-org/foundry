# MoE / DeepEP Support

vLLM-side MoE / expert-parallelism support is what makes the integration interesting beyond dense decode. The remaining MoE-specific concern is that DeepEP all-to-all communicator buffers must be set up in fabric mode for foundry to interpose them, and NVSHMEM module init must fire after the runtime is bootstrapped.

The dense-only baseline doesn't exercise any of this; turn it on by running an MoE model with EP.

## MoE weight packing

vLLM's recent versions use modular kernels for both quantized and unquantized MoE — the same `model_loader.load_model()` path produces deterministic packing regardless of foundry mode. Foundry's integration takes the unified preallocate → `start_graph_builds` → weight-load → `prepare_communication_buffer_for_model` path for all MoE variants (and for dense models), so no quant-metadata roundtrip through `WarmupState` is needed. The earlier `collect_moe_quant_metadata` / `inject_moe_quant_metadata` / `reset_moe_quant_config` / `reinit_moe_quant_config` / `split_load_model` ceremony has been removed.

## DeepEP fabric mode

### Problem

`DeepEPLLAll2AllManager` builds the all-to-all communicator with one of several backends:

- NVL (NVLink direct) — uses `cudaIpcGetMemHandle` for buffer sharing.
- Fabric — uses NVSHMEM + symmetric memory.

NVL's IPC handles are process-local and not persistable; the captured graph can't reference them on LOAD. Fabric mode allocates buffers via NVSHMEM, which foundry's hook can interpose (`init_nvshmem_for_loaded_modules` re-binds them on LOAD).

### Solution

`_patch_deepep` wraps `DeepEPLLAll2AllManager._make_all2all_kwargs`:

```python
@functools.wraps(orig)
def patched(self, *args, **kwargs):
    kwargs = orig(self, *args, **kwargs)
    if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
        kwargs["use_fabric"] = True
        kwargs["num_nvl_bytes"] = 0          # RDMA-only path
    return kwargs
```

Forces every foundry-mode DeepEP setup onto the fabric path. The patch lives behind a try/except in case the import path moves.

## NVSHMEM initialization order

NVSHMEM modules must be loaded into device code memory **before** any captured graph that references NVSHMEM kernels is replayed.

1. `runtime.setup_graph_extension(...)` on LOAD calls `fops.set_skip_fatbin_processing(True)` then `fops.load_cuda_modules_and_libraries(workspace_dir)`. The loader queues each NVSHMEM-using module into `pending_nvshmem_init` but does not flush it (the NVSHMEM runtime isn't up yet).

2. `_load_model_with_overlap` runs `preallocate_for_load_mode()` and `start_graph_builds()`, then calls `do_original_load()` which eventually invokes `prepare_communication_buffer_for_model(model)`. The orig call constructs the DeepEP `Buffer`, which bootstraps the NVSHMEM runtime.

3. `_patch_prepare_comm_buffer`'s post-orig hook then calls `fops.init_nvshmem_for_loaded_modules()` to flush the queue — well before any captured graph that uses NVSHMEM symbols is replayed by `finish_one_graph_load` inside `capture_model`.

If NVSHMEM init runs after the first `finish_one_graph_load`, the captured graph that calls into NVSHMEM aborts with `cuLaunchKernel` errors because the kernel handles don't resolve yet. If it runs *before* `prepare_communication_buffer_for_model` (the bug we hit when init was placed inside `start_graph_builds`), `nvshmemx_cumodule_init` silently no-ops because the runtime is not bootstrapped, and replay later sees uninitialized modules — also fatal.

For dense single-GPU models the NVSHMEM count is 0 and the call is a no-op; the post-orig hook is kept on all paths for parity.

## Validation

For a working MoE SAVE→LOAD cycle:

- All ranks produce `rank_{N}/warmup_state.json`-equivalent per-rank artifacts.
- All ranks' `final_alloc_offset` are reproducible across SAVE pass 1 → pass 2.
- On LOAD, the first inference call returns coherent tokens.
