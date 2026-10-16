#!/bin/bash
# Qwen3.5-35B-A3B (MoE hybrid). Usage: bash serve_qwen3.5-35ba3b.sh <tp2|tp4|ep2|ep4|ep8|tpep2|tpep4> [--save|--load]
# Validated (2026-09-16, 8xH100), 256 decode graphs: tp/ep/tpep at 0.8; ep8 at 0.6. Any size passes through.
# No mode = plain SGLang. MAXBS / MEMFRAC / PORT / SGL_EXTRA_ARGS override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <tpN|epN|tpepN> [--save|--load]}
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <tpN|epN|tpepN> [--save|--load]"; exit 1 ;;
esac
MAXBS=${MAXBS:-256}
GRAPHS=(--cuda-graph-max-bs-decode "$MAXBS" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled)
case "$CFG" in
  tp[0-9]*) N=${CFG#tp}; TOPO=(--tp-size "$N"); MEMFRAC_DEFAULT=0.8 ;;
  ep[0-9]*) N=${CFG#ep}
    TOPO=(--tp-size "$N" --attn-dp-size "$N" --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size 256 --attention-backend fa3 --max-running-requests $((256 * N)))
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
    if [[ "$CFG" == ep8 ]]; then MEMFRAC_DEFAULT=0.6; else MEMFRAC_DEFAULT=0.8; fi ;;
  tpep[0-9]*) N=${CFG#tpep}
    TOPO=(--tp-size "$N" --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size 256 --attention-backend fa3 --max-running-requests $((256 * N)))
    GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * N)) --cuda-graph-bs-decode $(seq -s' ' "$N" "$N" $((MAXBS * N)))
            --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled)
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
    MEMFRAC_DEFAULT=0.8 ;;
  *) echo "unknown topology '$CFG' (tpN|epN|tpepN)"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3.5-35B-A3B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-$MEMFRAC_DEFAULT}" \
  "${TOPO[@]}" "${GRAPHS[@]}" \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
