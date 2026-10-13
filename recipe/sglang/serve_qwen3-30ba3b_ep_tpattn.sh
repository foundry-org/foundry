#!/bin/bash
# Qwen3-30B-A3B (MoE), expert parallel with TP attention (symm-mem allreduce, no DP-attention).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3b_ep_tpattn.sh <ep_size> [--save|--load]
# SGL_MODEL=Qwen/Qwen3-30B-A3B-FP8 for FP8. No mode = plain SGLang.
# Decode graphs for MAXBS (default 256) batch sizes per rank, no padding; prefill graphs off. The batch is
# scattered over the EP_SIZE attention-TP ranks, so the captured sizes are EP_SIZE, 2*EP_SIZE, .., MAXBS*EP_SIZE.
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
MAXBS=${MAXBS:-256}; ATP=$EP_SIZE
GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * ATP)) --cuda-graph-bs-decode $(seq -s' ' "$ATP" "$ATP" $((MAXBS * ATP)))
        --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled)
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((EP_SIZE - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-30B-A3B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size "$EP_SIZE" --ep-size "$EP_SIZE" \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm \
  --mem-fraction-static 0.8 --disable-radix-cache \
  --chunked-prefill-size 256 --attention-backend fa3 "${GRAPHS[@]}" \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
