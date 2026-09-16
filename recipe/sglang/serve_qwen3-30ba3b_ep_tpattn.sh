#!/bin/bash
# Qwen3-30B-A3B (MoE), expert parallel via DeepEP low-latency with TP ATTENTION:
# the attention allreduce runs through torch symmetric memory instead of using
# DP-attention. This mirrors the vLLM EP topology (recipe/vllm/*_ep.sh).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3b_ep_tpattn.sh <ep_size> [--save|--load]
# SGL_EXTRA_ARGS: extra `sglang serve` flags appended verbatim (e.g. \"--cuda-graph-backend-prefill disabled\").
#
# vs serve_qwen3-30ba3b_ep.sh: --dp-size/--enable-dp-attention are replaced by
# --disable-custom-all-reduce --enable-torch-symm-mem (decode-graph allreduce =
# symm_mem two_shot on the persistent symmetric buffer foundry places
# deterministically) and --cuda-graph-backend-prefill disabled — without
# DP-attention every rank dispatches the full prefill chunk, and prefill-graph
# capture trips DeepEP's num_max_dispatch_tokens_per_rank assert even on
# baseline sglang. Requires the fork's two-shot-without-multicast fix
# (`foundry` branch) on hosts without IMEX channels.
#
# The EP kernel stack (sgl-deep-ep, sgl-deep-gemm, fa3 inside sglang-kernel) is
# wheel-provided by the sglang install; NVSHMEM auto-detects from the
# nvidia-nvshmem wheel (leave nvshmem_host_path unset in the TOMLs — README §EP).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
# bf16 is the validated default on the `foundry` branch (sgl-deep-gemm's masked bf16
# GEMM is present in the wheel); override for FP8: SGL_MODEL=Qwen/Qwen3-30B-A3B-FP8.
MODEL_NAME="${SGL_MODEL:-Qwen/Qwen3-30B-A3B}"
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
    --ep-size "$EP_SIZE" \
    --enable-torch-symm-mem \
    --cuda-graph-backend-prefill disabled \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --disable-radix-cache \
    --disable-custom-all-reduce \
    --chunked-prefill-size 256 \
    --attention-backend fa3 \
    --cuda-graph-max-bs-decode 128 \
    "${FOUNDRY_ARGS[@]}" \
    ${SGL_EXTRA_ARGS:-}
