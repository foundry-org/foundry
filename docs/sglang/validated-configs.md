# Validated SGLang models and configurations

Every model and configuration that passed Foundry SAVE + LOAD with SGLang, with the latest measurement per row. Graphs are counted per rank as prefill + decode. For the 4xH200 and 1-GPU rows, the native -> LOAD column is the log time of the first `/health` 200; the older rows use the driver's curl time. Restore is restore only; "a + b" means prefill + decode. "Prefill graphs: native limitation" means plain SGLang cannot capture FULL prefill graphs for that model, so only decode graphs exist:

- Qwen3.5 hybrid GDN: `IndexError` in `hybrid_linear_attn_backend`;
- DeepSeek-V4: EXTEND `NotImplementedError`;
- GLM-5.3: DSA backend `AttributeError`;
- Inkling: triton rejects EXTEND capture.

"Dummy text" parity means the greedy completion was identical under dummy weights. It shows that the same kernels ran, not model quality.

## foundry-core 0.1.0rc4, torch 2.14.1, 2x / 1x H200 (2026-11-08)

Driver 595.71.05, SGLang `foundry-dep` (9fccf82eda), real weights unless noted. Times are launch to the first `/health` 200; the 1-GPU rows run with `--skip-server-warmup` on every engine (256 decode graphs per set, no padding). Output check: LOAD greedy tokens, selected and top-3 logprobs identical to native over 20 sequential requests (TP2 also batched); EAGLE3 accept length identical to native.

| model | layout | weights | graphs per rank | native graph -> LOAD (s) | restore (s) |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-FP8 | TP2, fa3, prefill full | real | 6 + 256 | 103.4 -> **37.0** | 0.05 + 1.27 |
| Qwen3-30B-A3B-FP8 | EP2, DP attention, DeepEP v2 | real | 0 + 256 | 116.1 -> **37.0** | 1.3 |
| Qwen3-30B-A3B-FP8 + EAGLE3 | EP2, DP attention, DeepEP v2, decode bs 1..32 | real | 24 x 3 kinds | 45.1 -> **37.6** | 0.1 |
| Qwen3.5-2B | 1 GPU, default attention | real | 0 + 256 | 53.5 -> **25.3** | 0.35 |
| Qwen3-1.7B | 1 GPU, flashinfer, prefill full bs 8/16/32/64 | real | 4 + 256 | 40.5 -> **28.8** | 0.01 + 0.64 |
| Qwen3-30B-A3B-FP8 + EAGLE3 | 1 GPU, flashinfer, decode bs 1..256 | dummy | 256 x 3 kinds | 109.4 -> **31.4** | 1.35 |

Records: `claude-doc/merge_plan/rng_state_fix/RESULTS.md` (2-GPU rows), `claude-doc/report/sgl_tests_timing_256/README.md` (1-GPU rows, medians of 3).

## Dependency route, torch 2.14.1, 4xH200 (2026-10-11)

sglang `--cuda-graph-persistence` (PR branch `d487b80da4` on upstream `234ccba7f6`) + foundry-core 0.1.0rc2 (`a7fe034`), torch 2.14.1 / cu130. Times are launch to the first `/health` 200 in the
engine log; restore is prefill + decode graphs per rank. Output check: native graph, SAVE and LOAD agree on the
selected and top-3 logprobs of 74 greedy requests.

| model | layout (4 GPUs) | weights | graphs (prefill + decode per rank) | native graph -> LOAD (s) | restore (s) |
|---|---|---|---|---|---|
| Qwen3-235B-A22B-FP8 | attn TP4 + EP4, DeepEP LL | dummy | 10 + 128 | 134.3 -> **35.6** | 0.13 + 1.56 |
| Qwen3-235B-A22B-FP8 | attn TP2 x DP2 + EP4 | dummy | 10 + 128 | 143.7 -> **35.2** | 0.13 + 1.60 |
| Qwen3.5-122B-A10B-FP8 | EP4, DP attention | dummy | 0 + 128 | 114.0 -> **42.2** | 0.97 |
| Qwen3.5-122B-A10B-FP8 | attn TP2 x DP2 + EP4 | dummy | 0 + 256 | 180.4 -> **38.3** | 3.62 |
| DeepSeek-V4-Flash-FP8 | EP4, DP attention | dummy | 0 + 256 | 167.9 -> **39.6** | 2.75 |
| GLM-5.3-Flash (FP8) | EP4, DP attention | dummy | 0 + 128 | 135.4 -> **51.0** | 1.04 |
| gpt-oss-120b | EP4, a2a none | dummy | 10 + 128 | 87.7 -> **42.3** | 0.05 + 0.63 |
| Qwen3.5-35B-A3B | EP4, DP attention, DeepEP | dummy | 0 + 256 | 100.1 -> **34.8** | 1.28 |
| Qwen3.5-35B-A3B | attn TP4 + EP4 | dummy | 0 + 128 | 76.5 -> **35.8** | 0.90 |
| Qwen3-30B-A3B-FP8 | EP4, DP attention, DeepEP LL | **real** | 10 + 128 | 77.5 -> **39.2** | 0.05 + 0.71 |
| Qwen3-30B-A3B-FP8 | TP2 | **real** | 12 + 128 | 74.8 -> **36.8** | 0.06 + 0.54 |
| Qwen3-30B-A3B-FP8 | attn TP2 x DP2 + EP4 | **real** | 10 + 128 | 89.7 -> **39.5** | 0.06 + 0.78 |
| Qwen3-30B-A3B-FP8 | DP4 | **real** | 12 + 128 | 103.3 -> **46.2** | 0.06 + 0.58 |
| Qwen3-30B-A3B-FP8 | attn TP4 + EP4 | **real** | 10 + 128 | 81.2 -> **47.1** | 0.06 + 0.73 |
| Qwen3-30B-A3B-FP8 | EP2, DP attention | **real** | 10 + 128 | 72.9 -> **36.7** | 0.05 + 0.73 |
| Qwen3-30B-A3B-FP8 | attn TP2 + EP2 | **real** | 10 + 128 | 76.8 -> **35.9** | 0.06 + 0.75 |
| Qwen3-1.7B | 1 GPU, flashinfer | real | 4 + 8 | e2e | 0.01 + 0.15 |
| Qwen3.5-2B | 1 GPU, fa3 | real | 0 + 8 | e2e | 0.02 |
