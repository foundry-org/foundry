#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="Qwen/Qwen3-1.7B"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.8

FOUNDRY_ARGS=()
if [[ "$1" == "--save" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_save.toml" )
    echo "Using foundry SAVE: ${SCRIPT_DIR}/foundry_save.toml"
elif [[ "$1" == "--load" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_load.toml" )
    echo "Using foundry LOAD: ${SCRIPT_DIR}/foundry_load.toml"
elif [[ -n "$1" ]]; then
    echo "Usage: $0 [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline vLLM)"
fi

# LD_PRELOAD of libcuda_hook.so is set by foundry's setup_ld_preload_env at
# worker spawn time (uses the path in the TOML config). Baseline runs don't
# need it preloaded by the shell.

# Foundry only patches the V1 model runner. vLLM defaults certain
# architectures (e.g. Qwen3ForCausalLM) to the V2 runner, which our
# patches don't touch — pin V1 here.
export VLLM_USE_V2_MODEL_RUNNER=0

CUDAGRAPH_CAPTURE_SIZES=($(seq 1 256))

ARGS=(
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --no-enable-prefix-caching
    --max-num-batched-tokens 256
    --max-num-seqs 256
    --attention-config.backend FLASH_ATTN
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY
    --compilation-config.cudagraph_num_of_warmups 0
    --chat-template-content-format string
    --cudagraph-capture-sizes ${CUDAGRAPH_CAPTURE_SIZES[@]}
)

vllm serve "$MODEL_NAME" "${ARGS[@]}" "${FOUNDRY_ARGS[@]}"
