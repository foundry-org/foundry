#!/bin/bash
# Qwen3.5-122B-A10B (MoE hybrid), FP8 by default; SGL_MODEL=Qwen/Qwen3.5-122B-A10B for bf16.
# Usage: bash serve_qwen3.5-122ba10b.sh <epN|tpepN|tpAepN> [--save|--load]
# Validated (2026-09-17, 8xH100): FP8 ep4/tpep4/tp2ep4 at 0.6; ep8 at 0.79 with MAXBS=64; tp4ep8 at 0.65.
# --max-mamba-cache-size caps the GDN state pool (and thus graphs/rank); see README "Qwen3.5 and DeepSeek".
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <epN|tpepN|tpAepN> [--save|--load]}
MODEL_PATH="${SGL_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <epN|tpepN|tpAepN> [--save|--load]"; exit 1 ;;
esac
if [[ "$CFG" == ep8 && "$MODEL_PATH" == *FP8* ]]; then MAXBS=${MAXBS:-64}; MAMBA=64; else MAMBA=256; fi
MAXBS=${MAXBS:-256}
GRAPHS=(--cuda-graph-max-bs-decode "$MAXBS" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled)
case "$CFG" in
  ep[0-9]*) N=${CFG#ep}
    TOPO=(--tp-size "$N" --dp-size "$N" --ep-size "$N" --enable-dp-attention
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size 256 --attention-backend fa3 --max-running-requests $((256 * N))) ;;
  tpep[0-9]*) N=${CFG#tpep}
    TOPO=(--tp-size "$N" --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size 256 --attention-backend fa3 --max-running-requests $((256 * N)))
    GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * N)) --cuda-graph-bs-decode $(seq -s' ' "$N" "$N" $((MAXBS * N)))
            --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled) ;;
  tp[0-9]*ep[0-9]*) ATP=${CFG%%ep*}; ATP=${ATP#tp}; N=${CFG##*ep}
    TOPO=(--tp-size "$N" --dp-size $((N / ATP)) --ep-size "$N" --enable-dp-attention
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size 256 --attention-backend fa3 --max-running-requests $((256 * N)))
    GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * ATP)) --cuda-graph-bs-decode $(seq -s' ' "$ATP" "$ATP" $((MAXBS * ATP)))
            --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled) ;;
  *) echo "unknown topology '$CFG' (epN|tpepN|tpAepN)"; exit 1 ;;
esac
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
case "$CFG" in
  ep8) if [[ "$MODEL_PATH" == *FP8* ]]; then MEMFRAC_DEFAULT=0.79; else MEMFRAC_DEFAULT=0.6; fi ;;
  tp4ep8) MEMFRAC_DEFAULT=0.65 ;;
  *) MEMFRAC_DEFAULT=0.6 ;;
esac
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "$MODEL_PATH" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-$MEMFRAC_DEFAULT}" \
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-$MAMBA}" \
  "${TOPO[@]}" "${GRAPHS[@]}" \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
