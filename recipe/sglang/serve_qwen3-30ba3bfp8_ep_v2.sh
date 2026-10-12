#!/bin/bash
# Qwen3-30B-A3B-FP8, expert parallel over DeepEP v2 (NCCL symmetric windows + GIN).
# Needs nvidia-nccl-cu13 >= 2.30.7 and a driver accepting its cu13.3 fatbins (see README).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep_v2.sh <ep_size> [--save|--load]
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
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-128}
export SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$(( CUDA_GRAPH_MAX_BS > 128 ? CUDA_GRAPH_MAX_BS : 128 ))}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((EP_SIZE - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-30B-A3B-FP8}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size "$EP_SIZE" --dp-size "$EP_SIZE" --ep-size "$EP_SIZE" --enable-dp-attention \
  --moe-a2a-backend deepep_v2 --moe-runner-backend deep_gemm \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" --disable-radix-cache \
  --chunked-prefill-size 256 --attention-backend fa3 --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
