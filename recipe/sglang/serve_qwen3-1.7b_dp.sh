#!/bin/bash
# Qwen3-1.7B, data parallel (one full replica per DP rank).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_dp.sh <dp_size> [--save|--load]
# SGL_EXTRA_ARGS: extra `sglang serve` flags appended verbatim (e.g. \"--cuda-graph-backend-prefill disabled\").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/serve_common.sh"   # foundry_select / foundry_serve (plugin route + preflight)

DP_SIZE=${1:?Usage: $0 <dp_size> [--save|--load]}
MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
MEM_FRACTION_STATIC=0.6

if [[ "$2" == "--save" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_save.toml"
    echo "Using foundry SAVE: ${FOUNDRY_TOML}"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_TOML="${SCRIPT_DIR}/foundry_load.toml"
    echo "Using foundry LOAD: ${FOUNDRY_TOML}"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <dp_size> [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline SGLang)"
fi

# --save/--load: export FOUNDRY_GRAPH_EXTENSION_CONFIG="$FOUNDRY_TOML" after the preflight; baseline: unset it.
foundry_select "$2" "${FOUNDRY_TOML:-}"
foundry_baseline_pins   # baseline only: the plugin's collective pins (custom AR off, torch symm-mem on), see serve_common.sh

# LD_PRELOAD of libcuda_hook.so is set by foundry's setup_ld_preload_env at
# worker spawn time (path auto-detected; propagated to the DP controller and
# every rank's scheduler child). Assumes foundry + sglang are
# pip-installed (see README) so both import without PYTHONPATH.

foundry_serve "$2" sglang serve \
    --model-path "$MODEL_NAME" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 \
    --dp-size "$DP_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --attention-backend flashinfer \
    --cuda-graph-max-bs-decode 512 \
    $FOUNDRY_BASELINE_ARGS \
    ${SGL_EXTRA_ARGS:-}
