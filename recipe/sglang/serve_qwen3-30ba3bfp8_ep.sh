#!/bin/bash
# Qwen3-30B-A3B-FP8 (MoE), expert parallel via DeepEP low-latency + DP-attention.
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep.sh <ep_size> [--save|--load]
#
# Requires (see README §EP): deep_ep @ 9af0e0d, sgl-deep-gemm >= 0.1.2, flash-attn-3,
# and the nvshmem_host_path uncommented in foundry_{save,load}.toml.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
# FP8 routes through the Fp8Config MoE path (deep_gemm fp8 kernels), avoiding the
# bf16 masked-GEMM the pinned deep_gemm may lack. Override MODEL_NAME for bf16.
MODEL_NAME="${SGL_MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.8

if [[ "$2" == "--save" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_save.toml"
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    echo "Using foundry SAVE: ${FOUNDRY_TOML}"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_load.toml"
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
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

# LD_PRELOAD of libcuda_hook.so AND DeepEP's libnvshmem_host.so are set by
# foundry's setup_ld_preload_env at worker spawn time — both paths auto-detected
# (the hook from the foundry install, NVSHMEM from the nvidia-nvshmem wheel via
# config._detect_nvshmem_host_path), so nothing is preloaded by the shell. Set
# nvshmem_host_path in the TOML only to override the NVSHMEM auto-detection.
# Assumes foundry + the sglang fork are pip-installed (see README).

# DeepEP low-latency caps tokens/rank at SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
# (default 128; (n+1)*2 must be <= NVSHMEM_QP_DEPTH). Raise to 256 ((256+1)*2=514
# <= 1024 default QP) and chunk prefill to 256 so prefill chunks and decode batches
# fit — applied identically to SAVE and LOAD so the captured graphs match.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size "$EP_SIZE" \
    --dp-size "$EP_SIZE" \
    --ep-size "$EP_SIZE" \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --disable-custom-all-reduce \
    --chunked-prefill-size 256 \
    --attention-backend fa3 \
    --cuda-graph-max-bs 128 \
    "${FOUNDRY_ARGS[@]}"
