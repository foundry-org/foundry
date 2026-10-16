# Tests

| Directory | Markers (set by `tests/conftest.py`) | Needs |
|---|---|---|
| `unit/` | none | `foundry.ops` importable, no GPU device |
| `core/` | `gpu` | 1 GPU, `libcuda_hook.so`; `nvcc` for the `.cu` cases |
| `multigpu/` | `gpu`, `multigpu` | 2 GPUs; `deep_ep` + `nvidia-nvshmem` wheel for `test_deepep.py` |
| `sgl/` | `sglang` | sglang installed, no GPU |
| `sgl/e2e/` | `sglang`, `gpu`, `slow` (+ `multigpu`, `xlarge` per case) | GPUs, model weights or configs from the HF cache / network |

`plugin_route` marks `sgl/test_plugin_route.py` and `sgl/test_plugin_hook_points.py`.
`slow` and `xlarge` are deselected by default (`pyproject.toml` addopts); a `-m` on the command line replaces that default.
Missing GPUs skip; missing sglang / deep_ep / triton / nvcc skip.

## Run

```bash
pytest tests/                                   # unit, core, multigpu, sglang (no engine launches)
pytest tests/unit                               # host without a GPU
pytest tests/core                               # 1 GPU
pytest tests/multigpu                           # 2 GPUs
pytest tests/sgl                             # sglang, no engines
pytest tests/sgl/e2e/test_save_load_1gpu.py -m slow -v -s
pytest tests/sgl/e2e -m "slow and multigpu and not xlarge" -v -s     # Qwen3.5-35B-A3B on 2 GPUs
pytest tests/sgl/e2e -m slow -k q35_ep2 -v -s                       # one case
pytest tests/sgl/e2e -m "slow and xlarge" -v -s                     # 4 GPUs
pytest tests/ -m "not slow and not xlarge and not plugin_route"         # without the plugin-route files
pytest tests/core/test_save_load.py -k compat -q -s                     # torch-compat check of a wheel
```

Pin the devices with `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=0,1 pytest tests/multigpu`.
The `exec_share` patched-route case runs only with cudagraph-patch preloaded:

```bash
eval "$(cudagraph-patch env)" && pytest tests/core/test_exec_share.py -q -s
```

## Environment overrides

| Variable | Used by | Default |
|---|---|---|
| `FOUNDRY_E2E_MODEL` | `sgl/e2e/test_save_load_1gpu.py` | `Qwen/Qwen3-1.7B` (ungated) |
| `FOUNDRY_E2E_MAXBS` | `sgl/e2e/test_save_load_multigpu.py` | `64` (the recipes use `256`) |
| `FOUNDRY_E2E_WEIGHTS` | `sgl/e2e/test_save_load_multigpu.py` | `real` for 35B, `dummy` for `xlarge` |
| `TEST_BUF_MB` | `multigpu/test_symm_mem.py` | `8` |
| `FOUNDRY_TEST_DEEPEP_FABRIC=1` | `multigpu/test_deepep.py` | unset (no `use_fabric` kwarg) |

## Benchmarks

```bash
python tools/bench/bench_preallocation.py --cases performance,boundary
python tools/bench/bench_exec_update.py --cases core
```
