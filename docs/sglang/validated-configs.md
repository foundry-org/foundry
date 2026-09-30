# Validated SGLang models and configurations

Every model and configuration that passed Foundry SAVE + LOAD with SGLang, with the latest measurement per row. Graphs are counted per rank as prefill + decode. For the 4xH200 and 1-GPU rows, the native -> LOAD column is the log time of the first `/health` 200; the older rows use the driver's curl time. Restore is restore only; "a + b" means prefill + decode. "Prefill graphs: native limitation" means plain SGLang cannot capture FULL prefill graphs for that model, so only decode graphs exist:

- Qwen3.5 hybrid GDN: `IndexError` in `hybrid_linear_attn_backend`;
- DeepSeek-V4: EXTEND `NotImplementedError`;
- GLM-5.3: DSA backend `AttributeError`;
- Inkling: triton rejects EXTEND capture.

"Dummy text" parity means the greedy completion was identical under dummy weights. It shows that the same kernels ran, not model quality.

Code labels:

- `main`: sglang upstream `fa090f7755` with the Foundry plugin (entry point, `FOUNDRY_GRAPH_EXTENSION_CONFIG`).
- `fork`: sglang `foundry-prefill` `4f018fd052` with the in-tree flag.
- `H200-V8`: fork on upstream `03ea13a545`, Foundry `220896e`.
- `H200-C4`: container, fork, Foundry `b4a1f7f`.

| model | config (layout, GPUs) | weights | host / sglang / foundry code | graphs (prefill + decode per rank) | native graph -> LOAD to /health (s) | restore (s) | output check | notes |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-122B-A10B-FP8 | EP4, DP attention, DeepEP LL (4) | dummy | 4xH200 / main / `64b23ac` | 0 + 128 (prefill: native limitation) | 97.2 -> **47.8** | 0.83 | dummy text, native == LOAD | eager 45.6 s |
| Qwen3.5-122B-A10B-FP8 | attn TP2 x DP2 + EP4 (4) | dummy | 4xH200 / main / `64b23ac` | 0 + 128 (native limitation) | 103.1 -> **49.6** | 1.13 | dummy text, native == LOAD | bs off the multiple-of-2 grid runs eagerly in both |
| GLM-5.3-Flash (FP8) | EP4, DP attention (4) | dummy | 4xH200 / main / `64b23ac` | 0 + 128 (native limitation) | 138.3 -> **79.3** | 1.02 | dummy text, native == LOAD | dist init 23-29 s on every engine, eager included |
| gpt-oss-120b | EP4, plain EP (4) | dummy | 4xH200 / main / `64b23ac` | 4 + 128 | 76.8 -> **50.5** | 0.72 + 0.51 | **not testable**: mxfp4 dummy experts are uninitialized memory, so the text differs after token 1; TPOT native = LOAD (9.59 / 9.51 ms at bs 128) | the only non-Qwen3 row with prefill graphs |
| DeepSeek-V4-Flash-FP8 | EP4, DP attention (4) | dummy | 4xH200 / main / `6d5bec5` (cluster fix) | 0 + 128 (native limitation) | 105.2 -> **46.4** | 1.16 | dummy text, native == LOAD | before the fix, LOAD failed with error 912 on the cluster-16 top-k kernel |
| Qwen3-235B-A22B-FP8 | attn TP4 + EP4, DeepEP LL (4) | dummy | 4xH200 / main / `10a2681` | 4 + 128 | 111.8 -> **36.2** | 0.06 + 1.51-1.54 | dummy text, native == LOAD | TTFT / TPOT equal to native (C8 1030 / 1023 ms; bs128 75.9 / 75.8 ms) |
| Qwen3-30B-A3B-FP8 | EP4, DP attention, DeepEP LL (4) | **real** | 4xH200 / main / `10a2681` | 4 + 128 | 80.7 -> **47.8** | 0.025 + 0.70-0.72 | greedy 20/20 x 2 reps, native vs LOAD; eager vs LOAD 20/20 | TTFT C8 439 / 434 ms; TPOT bs128 10.2 / 9.2 ms (native / LOAD) |
| Qwen3-30B-A3B-FP8 | TP2 (2) | **real** | 4xH200 / main / `c5e43c4` | 5 + 128 | 71.5 -> **37.9** | 1.22 + 0.52 | greedy 20/20 per rep, native vs LOAD | native is not run-to-run deterministic at TP2 (19/20); native bar from part-2; rechecked on `056b154`: SAVE vs LOAD 20/20 |
| Qwen3-1.7B | 1 GPU, flashinfer | real | H200 / sglang `foundry-plugin` (main + PR) / `056b154` | 4 + 8 | 23.0 -> **20.0** | 0.012 + 0.024 | e2e test: 8 prompts sequential + batched equal to native; 8/8 prefill replays | e2e test 6/6 OK |
| Qwen3.5-2B | 1 GPU, fa3 | real | H200 / sglang `foundry-plugin` / `056b154` | 0 + 8 (native limitation) | 28.0 -> **27.0** | 0.015 | e2e test greedy equal (3 runs) | 2 of 3 runs 6/6 OK; one run tripped the TPOT check on a slow native bench (greedy output still matched) |
| Qwen3-30B-A3B-FP8 | TP4 + EP4 (4) | real | 4xH200 `P1-real` / fork, flag / `6bd4275` | 0 + 128 | 72.5 -> **39.2** | 0.67 | greedy: native r0 vs r1 9/20, LOAD within native's variants (any-rep 20/20) | native is nondeterministic here |
| Qwen3-30B-A3B-FP8 | attn TP2 + EP4 (4) | real | `P1-real` | 0 + 128 | 82.7 -> **47.7** | 0.75 | greedy 20/20 | |
| Qwen3-30B-A3B-FP8 | DP4 (4) | real | `P1-real` | 0 + 128 | 76.6 -> **48.5** | 0.56 | greedy 20/20 | |
| Qwen3-30B-A3B-FP8 | DP2 (2) | real | `P1-real` | 0 + 128 | 74.3 -> **44.2** | 0.60 | greedy 20/20 | |
| Qwen3-30B-A3B-FP8 | EP2, DP attention, DeepEP (2) | real | `P1-real` | 0 + 128 | 74.3 -> **47.9** | 0.68 | greedy 20/20 | |
| Qwen3-30B-A3B-FP8 | TP2 + EP2 (2) | real | `P1-real` | 0 + 128 | 70.4 -> **36.6** | 0.69 | greedy 20/20 | |
| Qwen3.5-122B-A10B-FP8 | EP8, DP attention (8) | dummy | 8xH200 `H200-V8` (older code) | 0 + 128 (native limitation) | 114.8 -> **60.7** | 2.05 | dummy text, native == LOAD | |
| Qwen3.5-122B-A10B-FP8 | attn TP2 + EP8 (8) | dummy | `H200-V8` | 0 + 128 (native limitation) | 112.8 -> **64.6** | 4.27 | dummy text | |
| Qwen3.5-397B-A17B-FP8 | EP8, DP attention (8) | dummy | `H200-V8` | 0 + 128 (native limitation) | 131.9 -> **64.1** | 2.31 | dummy text | no eager engine |
| DeepSeek-V4-Flash-FP8 | EP8, DP attention (8) | dummy | `H200-V8` (fork: no cluster-16 top-k path) | 0 + 128 (native limitation) | 119.6 -> **56.8** | 2.77 | dummy text | |
| GLM-5.3-Flash (FP8) | EP8, DP attention (8) | dummy | `H200-V8` | 0 + 128 (native limitation) | 167.6 -> **112.4** | 4.00 | dummy text | |
| Inkling-Small | TP8 + EP8 (8) | dummy | `H200-V8` | 0 + 64 (native limitation) | 65.4 -> **51.3** | 1.99 | dummy text | `--moe-a2a-backend none` |
| Inkling-Small | EP8, DP attention (8) | dummy | `H200-V8` | 0 + 102 (native limitation) | 88.1 -> **62.0** | 3.60 | dummy text | |
| Inkling-Small | attn TP4 + EP8 (8) | dummy | `H200-V8` | 0 + 128 (native limitation) | 105.1 -> **63.8** | 5.18 | dummy text | |
| Qwen3.5-27B | TP2 (2) | dummy | `H200-V8` | 0 + 128 (native limitation) | 60.6 -> **38.7** | 1.03 | dummy text | |
| Qwen3.5-27B | TP4 (4) | dummy | `H200-V8` | 0 + 128 (native limitation) | 61.2 -> **40.3** | 1.03 | dummy text | |
| Qwen3-235B-A22B-FP8 | attn TP4 x DP2 + EP8 (8) | dummy | `H200-V8` | 4 + 128 | 128.1 -> **58.8** | 0.91 + 4.64 | dummy text | input throughput / TTFT equal to native with prefill graphs |
| Qwen3-30B-A3B | EP4, DP attention (4) | dummy | `H200-V8` | 4 + 128 | 69.1 -> **51.4** | 0.77 + 1.39 | dummy text | |
| Qwen3-30B-A3B | TP4 + EP4 (4) | dummy | `H200-V8` | 4 + 128 | 63.8 -> **36.9** | 0.80 + 1.83 | dummy text | |
| Qwen3.5-35B-A3B | EP4 (4) | dummy | 8xH200 venv, before the udev shim (older code) | 0 + 128 (native limitation) | 116.1 -> **93.2** | 1.89 | dummy text | every EP engine paid a ~40 s NVSHMEM probe |
| Qwen3.5-35B-A3B | TP4 + EP4 (4) | dummy | same | 0 + 128 (native limitation) | 137.5 -> **85.5** | 3.04 | dummy text | same |
| Inkling-Small-21L (**reduced model**, 21 of 42 layers) | EP4, DP attention (4) | dummy | 4xH200 `H200-C4` | 0 + 128 (native limitation) | 96.1 -> **56.3** | 2.09 | dummy text | a2a `deepep` |
| Inkling-Small-21L (**reduced model**) | TP4 + EP4 (4) | dummy | `H200-C4` | 0 + 128 (native limitation) | 61.2 -> **46.2** | 1.61 | dummy text | |

33 rows. The `P1-real` TP2 and EP4 rows are superseded by the two `main` Qwen3-30B-A3B-FP8 rows above. The `10a2681` rows are the final validation of the plugin route (2026-09-30, 4xH200); the same lease also passed the 1-GPU e2e test (4 runs), the Qwen3.5-35B-A3B tp2 / ep2 large tests (real weights) and the xlarge Qwen3.5-122B-FP8 / DeepSeek-V4-Flash-FP8 EP4 tests. Not included: an older 8xH100 real-weight matrix on much older code. DeepEP v2 (`--moe-a2a-backend deepep_v2`) is validated only on the earlier fork route (H100 EP2 / EP4, see `recipe/sglang/README.md`); on the plugin route it could not be started because the bare 4xH200 hosts have no RDMA NIC and v2's NCCL GIN asserts before any Foundry code runs.

The stage breakdown of the `main` rows is in [figs/stages_pr.png](figs/stages_pr.png).
