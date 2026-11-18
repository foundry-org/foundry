#!/bin/bash
# Usage: bash serve_qwen3-32b_tp.sh <tp_size> [--save|--load]
#
# Tensor parallelism with torch symmetric-memory allreduce. The decode
# graphs capture vLLM's SymmMemCommunicator ops (two_shot_all_reduce_ at
# TP=2 on Hopper) directly on the persistent symmetric buffer; foundry's
# VMM region makes the buffer, its peer-pointer device arrays, and the
# multicast VA land at identical addresses across SAVE and LOAD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TP_SIZE=${1:?Usage: $0 <tp_size> [--save|--load]}
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
HOST="0.0.0.0"
PORT=12000
GPU_MEMORY_UTILIZATION=0.8

FOUNDRY_ARGS=()
if [[ "$2" == "--save" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_save.toml" )
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V2_MODEL_RUNNER=0
    echo "Using foundry SAVE: ${SCRIPT_DIR}/foundry_save.toml"
elif [[ "$2" == "--load" ]]; then
    FOUNDRY_ARGS+=( --compilation-config.graph_extension_config_path "${SCRIPT_DIR}/foundry_load.toml" )
    export NCCL_CUMEM_ENABLE=0
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V2_MODEL_RUNNER=0
    echo "Using foundry LOAD: ${SCRIPT_DIR}/foundry_load.toml"
elif [[ -n "$2" ]]; then
    echo "Usage: $0 <tp_size> [--save|--load]"
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

# Torch symmetric memory is vLLM main's default TP allreduce backend;
# pinned here for clarity.
export VLLM_ALLREDUCE_USE_SYMM_MEM=1

CUDAGRAPH_CAPTURE_SIZES=($(seq 1 256))

ARGS=(
    --trust-remote-code
    --dtype bfloat16
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    # Custom all-reduce registers IPC buffers per captured graph — a
    # replay path foundry does not support yet. With it disabled every
    # decode-graph allreduce dispatches to symm-mem (validated).
    --disable-custom-all-reduce
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --no-enable-prefix-caching
    --max-num-batched-tokens 256
    --max-num-seqs 256
    --attention-config.backend FLASH_ATTN
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY
    --compilation-config.cudagraph_num_of_warmups 0
    # Keep allreduce as an explicit symm-mem op in the decode graphs:
    # vLLM main's FlashInfer allreduce+rmsnorm fusion pass would otherwise
    # rewrite it into trtllm fused kernels — a workspace foundry does not
    # replay yet.
    --compilation-config.pass_config.fuse_allreduce_rms false
    --chat-template-content-format string
    --cudagraph-capture-sizes ${CUDAGRAPH_CAPTURE_SIZES[@]}
)

vllm serve "$MODEL_NAME" "${ARGS[@]}" "${FOUNDRY_ARGS[@]}"
