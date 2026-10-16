#!/bin/bash
# Qwen3-235B-A22B (MoE, 94 layers, 128 experts), FP8 by default; SGL_MODEL=Qwen/Qwen3-235B-A22B for bf16 (8 GPUs).
# Usage: bash serve_qwen3-235ba22b.sh <epN|tpepN|tpAepN> [--save|--load]
# Validated (2026-10-14, 4xH200, dummy weights, foundry d67f154): FP8 tpep4 and tp2ep4 at 0.7 (LOAD /health 31.5 s /
# 33.8 s vs 105-127 s native capture). The FP8 checkpoint is ~235 GB: 4 GPUs minimum. epN is untested on this model.
# Same topology vocabulary as serve_qwen3.5-122ba10b.sh (README "Qwen3.5 and DeepSeek"); no Mamba/GDN state pool here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <epN|tpepN|tpAepN> [--save|--load]}
MODEL_PATH="${SGL_MODEL:-Qwen/Qwen3-235B-A22B-FP8}"
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <epN|tpepN|tpAepN> [--save|--load]"; exit 1 ;;
esac
MAXBS=${MAXBS:-256}
# Prefill chunk 2048 (the validated rows; 256 as in the 30B recipes also works but prefills more slowly on 94 layers).
CHUNK=${CHUNK:-2048}
GRAPHS=(--cuda-graph-max-bs-decode "$MAXBS" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled)
case "$CFG" in
  ep[0-9]*) N=${CFG#ep}
    TOPO=(--tp-size "$N" --attn-dp-size "$N" --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size "$CHUNK" --attention-backend fa3 --max-running-requests $((256 * N))) ;;
  tpep[0-9]*) N=${CFG#tpep}
    TOPO=(--tp-size "$N" --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size "$CHUNK" --attention-backend fa3 --max-running-requests $((256 * N)))
    GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * N)) --cuda-graph-bs-decode $(seq -s' ' "$N" "$N" $((MAXBS * N)))
            --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled) ;;
  tp[0-9]*ep[0-9]*) ATP=${CFG%%ep*}; ATP=${ATP#tp}; N=${CFG##*ep}
    TOPO=(--tp-size "$N" --attn-dp-size $((N / ATP)) --ep-size "$N"
          --moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm
          --chunked-prefill-size "$CHUNK" --attention-backend fa3 --max-running-requests $((256 * N)))
    GRAPHS=(--cuda-graph-max-bs-decode $((MAXBS * ATP)) --cuda-graph-bs-decode $(seq -s' ' "$ATP" "$ATP" $((MAXBS * ATP)))
            --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled) ;;
  *) echo "unknown topology '$CFG' (epN|tpepN|tpAepN)"; exit 1 ;;
esac
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048
source "$SCRIPT_DIR/set_env.sh"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "$MODEL_PATH" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-0.7}" \
  "${TOPO[@]}" "${GRAPHS[@]}" \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
