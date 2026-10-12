#!/bin/bash
# DeepSeek-V4-Flash (sgl-project FP8 conversion), EP over 8 GPUs: DP-attention + DeepEP + DeepGEMM.
# Usage: CUDA_VISIBLE_DEVICES=0-7 bash serve_deepseek-v4-flash.sh ep8 [--save|--load]
# Validated (2026-09-17, 8xH100): ep8 at 0.7, 256 decode graphs/rank. Only ep8 is offered
# (attention-TP hybrids fail in sglang weight loading; stock FP4 checkpoint has no EP path on Hopper).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "${1:-}" == ep8 ]] || { echo "Usage: $0 ep8 [--save|--load]"; exit 1; }
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 ep8 [--save|--load]"; exit 1 ;;
esac
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DSV4_FP4_EXPERTS="${SGLANG_DSV4_FP4_EXPERTS:-0}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec sglang serve \
  --model-path "${SGL_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-0.7}" \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm \
  --chunked-prefill-size 2048 --max-running-requests 2048 \
  --cuda-graph-max-bs-decode "${MAXBS:-256}" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
