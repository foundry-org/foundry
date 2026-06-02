#!/bin/bash
# Qwen3-1.7B, single GPU. SAVE / LOAD CUDA graphs via the foundry SGLang integration.
# Usage: bash serve_qwen3-mini.sh [--save|--load]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.6

if [[ "$1" == "--save" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_save.toml"
    echo "Using foundry SAVE: ${FOUNDRY_TOML}"
elif [[ "$1" == "--load" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_load.toml"
    echo "Using foundry LOAD: ${FOUNDRY_TOML}"
elif [[ -n "$1" ]]; then
    echo "Usage: $0 [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline SGLang)"
fi

FOUNDRY_ARGS=()
if [[ -n "${FOUNDRY_TOML:-}" ]]; then
    FOUNDRY_ARGS+=( --foundry-graph-extension-config-path "${FOUNDRY_TOML}" )
fi

# LD_PRELOAD of libcuda_hook.so is set by foundry's setup_ld_preload_env at
# worker spawn time (path auto-detected from the foundry install). Baseline runs
# don't need it preloaded by the shell. Assumes foundry + the sglang fork are
# pip-installed (see README) so both import without PYTHONPATH.

sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs 512 \
    "${FOUNDRY_ARGS[@]}"
