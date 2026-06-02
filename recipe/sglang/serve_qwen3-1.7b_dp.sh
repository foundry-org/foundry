#!/bin/bash
# Qwen3-1.7B, data parallel (one full replica per DP rank).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_dp.sh <dp_size> [--save|--load]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DP_SIZE=${1:?Usage: $0 <dp_size> [--save|--load]}
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.6

if [[ "$2" == "--save" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_save.toml"
    # CUMEM P2P / NVLS multicast cuMemMap with driver-capability flags the
    # foundry VMM region doesn't carry — pin NCCL to IPC/ring. Foundry runs only.
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    echo "Using foundry SAVE: ${FOUNDRY_TOML}"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_load.toml"
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    echo "Using foundry LOAD: ${FOUNDRY_TOML}"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <dp_size> [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline SGLang)"
fi

FOUNDRY_ARGS=()
if [[ -n "${FOUNDRY_TOML:-}" ]]; then
    FOUNDRY_ARGS+=( --foundry-graph-extension-config-path "${FOUNDRY_TOML}" )
fi

# LD_PRELOAD of libcuda_hook.so is set by foundry's setup_ld_preload_env at
# worker spawn time (path auto-detected; propagated to the DP controller and
# every rank's scheduler child). Assumes foundry + the sglang fork are
# pip-installed (see README) so both import without PYTHONPATH.

sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --dp-size "$DP_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs 512 \
    "${FOUNDRY_ARGS[@]}"
