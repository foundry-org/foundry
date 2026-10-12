#!/bin/bash
# Qwen3-30B-A3B (MoE), expert parallel with TP attention (symm-mem allreduce, no DP-attention).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3b_ep_tpattn.sh <ep_size> [--save|--load]
# SGL_MODEL=Qwen/Qwen3-30B-A3B-FP8 for FP8. No mode = plain SGLang.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <ep_size> [--save|--load]"; exit 1 ;;
esac
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((EP_SIZE - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-30B-A3B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size "$EP_SIZE" --ep-size "$EP_SIZE" --cuda-graph-backend-prefill disabled \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm \
  --mem-fraction-static 0.8 --disable-radix-cache \
  --chunked-prefill-size 256 --attention-backend fa3 --cuda-graph-max-bs-decode 128 \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
