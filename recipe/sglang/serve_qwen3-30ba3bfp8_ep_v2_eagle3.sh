#!/bin/bash
# Qwen3-30B-A3B-FP8 with the EAGLE3 draft Tengyunw/qwen3_30b_moe_eagle3, expert parallel over DeepEP v2
# (same layout and requirements as serve_qwen3-30ba3bfp8_ep_v2.sh). Foundry saves and restores the
# target-verify, draft-decode and draft-extend graph sets; the draft's MoE runs with --speculative-moe-a2a-backend none.
# Validated with decode graphs 1..32 on 2x H200 (set CUDA_GRAPH_MAX_BS to change the set, same value on SAVE and LOAD).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-30ba3bfp8_ep_v2_eagle3.sh <ep_size> [--save|--load]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_SIZE=${1:?Usage: $0 <ep_size> [--save|--load]}
export CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
export SGL_EXTRA_ARGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path ${SGL_DRAFT_MODEL:-Tengyunw/qwen3_30b_moe_eagle3} \
  --speculative-num-steps ${SPEC_NUM_STEPS:-3} --speculative-eagle-topk ${SPEC_TOPK:-1} --speculative-num-draft-tokens ${SPEC_DRAFT_TOKENS:-4} \
  --speculative-moe-a2a-backend none ${SGL_EXTRA_ARGS:-}"
exec bash "$SCRIPT_DIR/serve_qwen3-30ba3bfp8_ep_v2.sh" "$EP_SIZE" "${2:-}"
