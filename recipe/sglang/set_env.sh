#!/bin/bash
# Common environment for every recipe in this directory (native, --save and --load alike), so the three
# engines of a comparison run with the same pins. `source` it after SCRIPT_DIR is set; each variable
# keeps a value the caller already exported.
#
# TORCHDYNAMO_DISABLE=1      Foundry pins it for SAVE/LOAD (dynamo frames fall outside the capture); the
#                            native engine must match or its decode TPOT differs for reasons unrelated to Foundry.
# SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
#                            DeepGEMM kernels compile on first use instead of a startup sweep.
# NVSHMEM_DISABLE_NCCL=1     NVSHMEM otherwise builds one NCCL communicator per team (5 on a 4-rank node,
#                            ~0.4 s each) while DeepEP creates its buffer; the low-latency kernels never use
#                            them. Cuts the DeepEP bootstrap from ~3.5 s to ~0.7 s on native and LOAD and
#                            frees the 1.5 GB of region memory their buffers took. It changes the allocation
#                            sequence, so archives are captured with it (SAVE and LOAD share this file).
# EP_DISABLE_GIN=1           DeepEP v2 only (serve_qwen3-30ba3bfp8_ep_v2.sh): the GIN mode hangs in the first
#                            warm-up forward (ElasticBuffer) on every H200 host tried, with or without usable
#                            verbs; the NVLink-only direct mode is the validated path. DeepEP v2's JIT also needs
#                            `cuobjdump` on PATH (absent from the pip CUDA 13 toolkit: install cuda_cuobjdump).
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}"
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-1}"
