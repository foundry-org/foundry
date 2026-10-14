#!/bin/bash
# Qwen3.5-27B (dense hybrid). Usage: bash serve_qwen3.5-27b.sh <single|dp2|dp4|dp8|tp2|tp4|tp8> [--save|--load]
# Validated (2026-09-16, 8xH100): every topology listed, mem-fraction 0.8, 256 decode graphs.
# No mode = plain SGLang. MAXBS / MEMFRAC / PORT / SGL_EXTRA_ARGS override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=${1:?Usage: $0 <single|dpN|tpN> [--save|--load]}
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <single|dpN|tpN> [--save|--load]"; exit 1 ;;
esac
case "$CFG" in
  single)    N=1; TOPO=(--tp-size 1) ;;
  dp[0-9]*) N=${CFG#dp}; TOPO=(--tp-size 1 --dp-size "$N") ;;
  tp[0-9]*) N=${CFG#tp}; TOPO=(--tp-size "$N") ;;
  *) echo "unknown topology '$CFG' (single|dpN|tpN)"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3.5-27B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --disable-radix-cache --mem-fraction-static "${MEMFRAC:-0.8}" \
  "${TOPO[@]}" \
  --cuda-graph-max-bs-decode "${MAXBS:-256}" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
