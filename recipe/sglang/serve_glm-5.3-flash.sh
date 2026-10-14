#!/bin/bash
# GLM-5.3-Flash (FP8), expert parallel: DP-attention + DeepEP low-latency + DeepGEMM, DSA attention.
# Usage: bash serve_glm-5.3-flash.sh <epN> [--save|--load]     e.g. ep4 (4 GPUs) or ep8 (8 GPUs)
# Validated: ep4 on 4xH200 and ep8 on 8xH200 (128 decode graphs/rank). Any epN is passed through;
# if the checkpoint does not fit, sglang itself reports it. No mode = plain SGLang.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <epN> [--save|--load]}
case "$CFG" in ep[0-9]*) N=${CFG#ep} ;; *) echo "Usage: $0 <epN> [--save|--load]"; exit 1 ;; esac
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <epN> [--save|--load]"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
# Validated at 128 decode graphs/rank on 4 GPUs (the KDA state pool for 256 slots/rank does not fit there); MAXBS=256 works on 8.
MAXBS=${MAXBS:-128}
# The KDA (mamba-style) state pool is global across DP ranks: MAXBS slots per rank (MAMBA overrides).
MAMBA=${MAMBA:-$((MAXBS * N))}
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-zai-org/GLM-5.3-Flash}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-0.8}" \
  --max-mamba-cache-size "$MAMBA" \
  --tp-size "$N" --dp-size "$N" --ep-size "$N" --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm \
  --chunked-prefill-size 512 --max-running-requests $((256 * N)) \
  --cuda-graph-max-bs-decode "$MAXBS" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
