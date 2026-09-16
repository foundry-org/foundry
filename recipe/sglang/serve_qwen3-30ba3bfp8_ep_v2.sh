#!/bin/bash
# Qwen3-30B-A3B-FP8, expert parallel over DeepEP v2 (NCCL symmetric-memory
# windows + GIN) with DP-attention. Prototype recipe (Sep 2026).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep_v2.sh <ep_size> [--save|--load]
# SGL_EXTRA_ARGS: extra `sglang serve` flags appended verbatim (e.g. \"--cuda-graph-backend-prefill disabled\").
#
# Requirements beyond the v1 EP recipe (see README "DeepEP v2"):
# - nvidia-nccl-cu13 >= 2.30.7 (sgl-deep-ep's ElasticBuffer is built against it);
#   torch pins 2.29.7, so install it with --no-deps and re-pin after any
#   `pip install -e` of foundry.
# - A driver (or cuda-compat-13-3 forward-compat lib on LD_LIBRARY_PATH) that
#   accepts NCCL 2.30.7's cuda13.3 fatbins.
# - 128x128 blockwise FP8 experts and the deep_gemm MoE runner (v2 hard limits).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
MODEL_NAME="${SGL_MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-128}

if [[ "$2" == "--save" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_save.toml"
    echo "Using foundry SAVE: ${FOUNDRY_TOML}"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_load.toml"
    echo "Using foundry LOAD: ${FOUNDRY_TOML}"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <ep_size> [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline SGLang)"
fi

FOUNDRY_ARGS=()
if [[ -n "${FOUNDRY_TOML:-}" ]]; then
    FOUNDRY_ARGS+=( --foundry-graph-extension-config-path "${FOUNDRY_TOML}" )
fi

# Do NOT set NCCL_CUMEM_ENABLE=0 here (unlike the v1 recipe): v2's NCCL windows
# need cuMem; sglang defaults it to 1 for deepep_v2. No multicast on the test
# hosts, so keep NVLS off. Foundry itself disables NCCL graph/local buffer
# registration under SAVE/LOAD (restored graphs cannot replay a registration).
export NCCL_NVLS_ENABLE=0

# Rank-0-only DeepGEMM pre-compile warmup skews allocation across ranks and
# SAVE vs LOAD; keep it off on both paths (see the v1 recipe).
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# Decode dispatches up to <bs> tokens per rank; v2's default cap is 128.
export SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$(( CUDA_GRAPH_MAX_BS > 128 ? CUDA_GRAPH_MAX_BS : 128 ))}

sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size "$EP_SIZE" \
    --dp-size "$EP_SIZE" \
    --ep-size "$EP_SIZE" \
    --enable-dp-attention \
    --moe-a2a-backend deepep_v2 \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --disable-custom-all-reduce \
    --chunked-prefill-size 256 \
    --attention-backend fa3 \
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
    "${FOUNDRY_ARGS[@]}" \
    ${SGL_EXTRA_ARGS:-}
