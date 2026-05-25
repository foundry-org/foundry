#!/bin/bash
# Usage: bash serve_qwen3-30ba3b_ep.sh <ep_size> [--save|--load]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
MODEL_NAME="Qwen/Qwen3-30B-A3B"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.8

FOUNDRY_ARGS=()
if [[ "$2" == "--save" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_save.toml" )
    # Foundry pins NCCL to the IPC/ring transports — the CUMEM P2P and NVLS
    # multicast fast paths cuMemMap with driver-capability flags the foundry
    # VMM region doesn't carry.
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    # Foundry only patches the V1 model runner; pin V1 explicitly so vLLM
    # doesn't quietly route Qwen3ForCausalLM-class architectures to V2.
    export VLLM_USE_V2_MODEL_RUNNER=0
    echo "Using foundry SAVE: ${SCRIPT_DIR}/foundry_save.toml"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_load.toml" )
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V2_MODEL_RUNNER=0
    echo "Using foundry LOAD: ${SCRIPT_DIR}/foundry_load.toml"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <ep_size> [--save|--load]"
    exit 1
else
    echo "Running without foundry (baseline vLLM)"
fi

# LD_PRELOAD of libcuda_hook.so / libnvshmem_host.so is set by foundry's
# setup_ld_preload_env at worker spawn time (uses the paths in the TOML
# config). Baseline runs don't need either preloaded by the shell.

# Foundry only patches the V1 model runner. vLLM defaults certain
# architectures (e.g. Qwen3ForCausalLM) to the V2 runner, which our
# patches don't touch — pin V1 here.
export VLLM_USE_V2_MODEL_RUNNER=0

export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1

CUDAGRAPH_CAPTURE_SIZES=($(seq 1 256))

ARGS=(
    --trust-remote-code
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --data-parallel-size "$EP_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --distributed-executor-backend uni
    --enable-expert-parallel
    --all2all-backend deepep_low_latency
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
