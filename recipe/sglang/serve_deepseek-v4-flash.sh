#!/bin/bash
# DeepSeek-V4-Flash (sgl-project FP8 conversion), expert parallel: DP-attention + DeepEP + DeepGEMM.
# Usage: bash serve_deepseek-v4-flash.sh <epN> [--save|--load]     e.g. ep4 (4 GPUs) or ep8 (8 GPUs)
# Validated: ep8 on 8xH100 and ep4 on 4xH200 (256 decode graphs/rank). Any epN is passed through;
# if the checkpoint does not fit, sglang itself reports it. No mode = plain SGLang.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <epN> [--save|--load]}
case "$CFG" in ep[0-9]*) N=${CFG#ep} ;; *) echo "Usage: $0 <epN> [--save|--load]"; exit 1 ;; esac
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <epN> [--save|--load]"; exit 1 ;;
esac
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
# FP8 experts; without a probe-able checkpoint sglang would assume FP4, which Hopper DeepGEMM rejects.
export SGLANG_DSV4_FP4_EXPERTS="${SGLANG_DSV4_FP4_EXPERTS:-0}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-0.7}" \
  --tp-size "$N" --dp-size "$N" --ep-size "$N" --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm \
  --chunked-prefill-size 2048 --max-running-requests $((256 * N)) \
  --cuda-graph-max-bs-decode "${MAXBS:-256}" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
