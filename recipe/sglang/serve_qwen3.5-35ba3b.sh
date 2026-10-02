#!/bin/bash
# Qwen3.5-35B-A3B (MoE hybrid: gated DeltaNet + full attention, routed experts).
# Usage: bash serve_qwen3.5-35ba3b.sh <tp2|tp4|ep2|ep4|ep8|tpep2|tpep4> [--save|--load]
# Validated (2026-09-16, 8xH100), 256 decode graphs: tp2 tp4 ep2 ep4 tpep2 tpep4 at 0.8; ep8 at 0.6 (LOAD needs a
# few GB more headroom than SAVE on 8 ranks: resident kernel images + the preallocated graph range). tpep8 LOAD ran
# out of memory instantiating 256 execs of bs up to 2048 and is not offered here.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_common.sh"
MODEL_PATH="${SGL_MODEL:-Qwen/Qwen3.5-35B-A3B}"
memfrac_default() { case $1 in ep8) echo 0.6 ;; *) echo 0.8 ;; esac; }
serve_main "$@"
