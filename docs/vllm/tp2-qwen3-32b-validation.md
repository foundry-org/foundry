# Experimental vLLM TP=2 validation: Qwen3-32B

This document records a scoped SAVE → LOAD check for Qwen3-32B BF16 on a
single machine with two NVIDIA H20 GPUs and NVLink. It does not validate every
vLLM tensor-parallel configuration. The recipe uses the V1 model runner,
`FULL_DECODE_ONLY`, torch symmetric-memory allreduce, and disables custom
allreduce and allreduce/RMSNorm fusion. LOAD enables the companion vLLM fork's
`FOUNDRY_LOAD_ALLOW_COMPILE=1` option for uncaptured forwards; there is no
mixed-forward warmup.

## Source and cache policy

The local Qwen3-32B weights came from Hugging Face revision
`9216db5781bf21249d130ec9da846c4624c16137` (the revision recorded for
all downloaded files). The measured run used Foundry
`3e034b0865320478f33abc8e1028c6df62427d49`
plus the SAVE autotune fix and vLLM fork
`eed7819a258033030a03e0f56d8fc7543434e1dc` plus the LOAD compile option.
Python 3.12.16, PyTorch 2.11.0+cu130, Triton 3.6.0, vLLM
`0.1.dev18535+geed7819a2`, and Transformers 5.22.2 were installed. The
5-pair benchmark started a new server process each time but reused the same
two-pass SAVE archive, vLLM compile cache, and OS page cache. SAVE time is not
included in LOAD startup time.

## SAVE → LOAD and correctness

A separate check after removal of the unused mixed-forward warmup code started
with an empty archive and compile cache. SAVE pass 1 and pass 2 both served a
query successfully. Pass 2 restored eight cached Triton `best_config` choices
during graph capture. The archive had `warmup_state.json`, a manifest for each
rank, and 256 `.cugraph` files per rank. LOAD became healthy and answered
requests. Three single greedy completions and two overlapping concurrent
completions matched ordinary vLLM byte for byte. These requests exercise
prefill and decode; client overlap alone does not prove a particular server
step contained a mixed prefill/decode batch. The cached-selection regression
tests cover missing and incompatible cache entries and passed 7/7 without a GPU.

## Process startup and TPOT

The earlier 5-pair run, in which the optional mixed warmup was disabled,
measured the following medians. Each mode started five times in alternating
order. For each service and client concurrency 1/8/32/128, one warmup round
preceded three measured rounds of 64 output tokens per request.

| Metric | Ordinary vLLM | Foundry LOAD |
| --- | ---: | ---: |
| Init to `/health` | 66.061 s | 41.441 s |
| Init to first completed request | 66.335 s | 47.186 s |
| Decode graph capture / restore phase | 21 s | 1 s |

| Client concurrency | vLLM TPOT | LOAD TPOT | Relative delta |
| ---: | ---: | ---: | ---: |
| 1 | 13.705 ms/token | 13.689 ms/token | -0.1% |
| 8 | 14.186 ms/token | 14.183 ms/token | 0.0% |
| 32 | 16.361 ms/token | 16.352 ms/token | -0.1% |
| 128 | 41.855 ms/token | 41.834 ms/token | -0.1% |

The five ordinary vLLM `/health` values ranged from 65.670 to 66.666 s;
the five LOAD values ranged from 41.244 to 41.654 s. The paired startup
measurements were:

| Pair | vLLM `/health` | LOAD `/health` | vLLM first request | LOAD first request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 66.059 s | 41.654 s | 66.332 s | 47.327 s |
| 2 | 66.666 s | 41.441 s | 66.941 s | 47.186 s |
| 3 | 66.067 s | 41.643 s | 66.342 s | 47.345 s |
| 4 | 65.670 s | 41.244 s | 65.944 s | 46.905 s |
| 5 | 66.061 s | 41.246 s | 66.335 s | 46.897 s |

[Startup observations](data/qwen3-32b-tp2-startup.csv) contain all ten
service starts; [TPOT observations](data/qwen3-32b-tp2-tpot.csv) contain all
120 measured rounds (five starts per mode, four concurrency levels, three
rounds each). The CSVs retain the numeric observations and omit personal
filesystem paths. TPOT in the table is the median of the per-round request
p50 values. A later repeat using a
fresh SAVE archive completed all five pairs but had one ordinary-vLLM graph
capture phase of 72 s versus 22–24 s in the other four pairs. That repeat is
excluded from the acceleration claim; its cause was not established. The
correctness check from its fresh archive remains valid.

Only Qwen3-32B, TP=2, this hardware topology, the stated graph/collective
settings and request distribution were checked. Mixed-batch scheduler steps,
other models and larger TP sizes need separate evidence.
