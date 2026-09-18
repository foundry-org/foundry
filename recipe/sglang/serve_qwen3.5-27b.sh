#!/bin/bash
# Qwen3.5-27B (dense hybrid: gated DeltaNet + full attention; the text path of a VL architecture).
# Usage: bash serve_qwen3.5-27b.sh <single|dp2|dp4|dp8|tp2|tp4|tp8> [--save|--load]
# Validated (2026-09-16, 8xH100): every topology listed, --mem-fraction-static 0.8, 256 decode graphs.
# sglang's default attention backend; the hybrid state pool needs no cap at this size.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_common.sh"
MODEL_PATH="${SGL_MODEL:-Qwen/Qwen3.5-27B}"
serve_main "$@"
