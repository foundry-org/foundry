#!/bin/bash
# Qwen3.5-122B-A10B (MoE hybrid). FP8 checkpoint (127 GB) by default; SGL_MODEL=Qwen/Qwen3.5-122B-A10B for bf16
# (250 GB: 8-GPU topologies only).
# Usage: bash serve_qwen3.5-122ba10b.sh <ep4|tpep4|tp2ep4|ep8|tp4ep8> [--save|--load]
# Validated (2026-09-17, 8xH100):
#   FP8   ep4 / tpep4 / tp2ep4 at 0.6 (256 graphs); ep8 at 0.79 with MAXBS=64; tp4ep8 at 0.65
#   bf16  ep8 at 0.6; tp4ep8 at 0.65
# Two knobs the 30B-class recipes do not need:
#   --max-mamba-cache-size caps the GDN state pool; without it sglang sizes it from the free memory left after the
#   weights and the pool crowds out the graphs (8 ranks) or fails to size at all (4 ranks at 0.6). The cap also
#   bounds the running requests per DP worker (cache / state slots per request / dp) and sglang captures decode
#   graphs only up to that bound: 256 on ep4 -> 64 graphs, 64 on ep8 -> 8 graphs, 256 on tp4ep8 -> bs 4..128.
#   0.8 is too tight for the 8-rank rows: SAVE captures with ~2.5 GB left and the first request's lazy Triton loads
#   fail; 0.79 with a cache of 64 is the measured ceiling for FP8 EP8.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_common.sh"
MODEL_PATH="${SGL_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
if [[ "$1" == ep8 && "$MODEL_PATH" == *FP8* ]]; then
  MAXBS=${MAXBS:-64}; MODEL_EXTRA="${MODEL_EXTRA:---max-mamba-cache-size 64}"
else
  MODEL_EXTRA="${MODEL_EXTRA:---max-mamba-cache-size 256}"
fi
memfrac_default() {
  case $1 in
    ep8)    [[ "$MODEL_PATH" == *FP8* ]] && echo 0.79 || echo 0.6 ;;
    tp4ep8) echo 0.65 ;;
    *)      echo 0.6 ;;
  esac
}
serve_main "$@"
