#!/bin/bash
# Qwen3-1.7B, single GPU. SAVE / LOAD CUDA graphs via the foundry SGLang integration.
# Usage: bash serve_qwen3-mini.sh [--save|--load]
# SGL_EXTRA_ARGS: extra `sglang serve` flags appended verbatim (e.g. \"--cuda-graph-backend-prefill disabled\").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/serve_common.sh"   # foundry_select / foundry_serve (plugin route + preflight)

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

# --save/--load: export FOUNDRY_GRAPH_EXTENSION_CONFIG="$FOUNDRY_TOML" after the preflight; baseline: unset it.
foundry_select "$1" "${FOUNDRY_TOML:-}"

# LD_PRELOAD of libcuda_hook.so is set by foundry's setup_ld_preload_env at
# worker spawn time (path auto-detected from the foundry install). Baseline runs
# don't need it preloaded by the shell. Assumes foundry + sglang are
# pip-installed (see README) so both import without PYTHONPATH.

foundry_serve "$1" sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs-decode 512 \
    ${SGL_EXTRA_ARGS:-}
