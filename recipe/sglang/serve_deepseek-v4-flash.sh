#!/bin/bash
# DeepSeek-V4-Flash, expert parallel over 8 GPUs (DP attention + DeepEP low-latency + DeepGEMM).
# Usage: CUDA_VISIBLE_DEVICES=0-7 bash serve_deepseek-v4-flash.sh ep8 [--save|--load]
# Validated (2026-09-17, 8xH100): ep8 at 0.7, 256 decode graphs per rank; capture 107 s -> restore 5.7 s, identical
# greedy output and TPOT.
#
# Checkpoint: sgl-project/DeepSeek-V4-Flash-FP8 (all-FP8 conversion). The stock deepseek-ai checkpoint ships FP4
# experts, which sglang serves TP-only on Hopper (no DeepEP/EP path), so it cannot use this topology on H100.
# Attention: sglang's own DeepSeek-V4 backend (dsv4, KV page 256); under DP attention the prefill chunk is divided by
# dp and must stay a multiple of the page, hence --chunked-prefill-size 2048 on 8 ranks.
# The attention-TP hybrids (tp2ep4, tp4ep8) fail in sglang's weight loading for this checkpoint (expert shard
# mismatch) and are not offered.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_common.sh"
MODEL_PATH="${SGL_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}"
EP_ATTN=""        # sglang picks the dsv4 backend
EP_CHUNK=2048
memfrac_default() { echo 0.7; }
[[ "${1:-}" == ep8 ]] || { echo "Usage: $0 ep8 [--save|--load] (validated topology: ep8)"; exit 1; }
serve_main "$@"
