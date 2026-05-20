# MoE / DeepEP Support

vLLM-side MoE / expert-parallelism support is what makes the integration interesting beyond dense decode. The core problem: MoE weight packing depends on a per-expert quant config that's computed during weight load, and DeepEP all-to-all communicator buffers must be set up in fabric mode for foundry to interpose them.

The dense-only baseline doesn't exercise any of this; turn it on by running an MoE model (e.g. DeepSeek-V2 with EP).

## MoE quant metadata

### Problem

`load_model` reads the model checkpoint and, for MoE layers, packs expert weights using a quant config derived from the checkpoint. The quant config can vary slightly across runs because of non-determinism in fp8 calibration paths or numerical thresholds that fall right on a boundary. If SAVE pass 1's packed weights differ from LOAD's, the captured graph's kernel args reference one packing and LOAD has the other — silent wrong output.

### Solution

`foundry.integration.vllm.moe`:

1. **SAVE pass 1**: after `loader.load_model()`, call `collect_moe_quant_metadata(model)` to extract the resolved quant config per MoE layer. Serialize into `WarmupState.moe_quant_metadata`.

2. **SAVE pass 2 / LOAD**: before `loader.load_model()`, call `inject_moe_quant_metadata(model, ws.moe_quant_metadata)` so the loader uses the saved config instead of re-deriving.

3. **After load**: `reinit_moe_quant_config(model)` is called on both SAVE and LOAD to re-bind any per-layer state that depends on the now-known quant config (e.g. compiled expert routing kernels).

This is wrapped in `_load_model_with_cge_overlap` in `foundry/integration/vllm/hooks.py::_patch_load_model`. The dense path skips all of this and falls through to a simpler overlap.

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

`runtime.setup_graph_extension(...)` on LOAD calls `cge.set_skip_fatbin_processing(True)` and `cge.load_cuda_modules_and_libraries(workspace_dir)` early. Subsequently, `preload_all_graphs(...)` calls `cge.init_nvshmem_for_loaded_modules()` before `finish_graph_loads(...)` runs the captured kernels.

If NVSHMEM init runs after `finish_graph_loads`, the first captured graph that calls into NVSHMEM aborts with `cuLaunchKernel` errors because the kernel handles don't resolve yet.

For dense single-GPU models the NVSHMEM count is 0 and these calls are no-ops; they're kept in the LOAD path for EP parity.

## Validation

For a working MoE SAVE→LOAD cycle:

- All ranks produce `rank_{N}/warmup_state.json`-equivalent per-rank artifacts.
- All ranks' `final_alloc_offset` are reproducible across SAVE pass 1 → pass 2.
- On LOAD, the first inference call returns coherent tokens.
