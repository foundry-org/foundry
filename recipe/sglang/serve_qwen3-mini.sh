#!/bin/bash
# Qwen3-1.7B, single GPU. Usage: bash serve_qwen3-mini.sh [--save|--load]
# No mode = plain SGLang (also warms the JIT caches before the first --save).
# Archive: ./foundry_archive (foundry.toml). SGL_MODEL / SGL_EXTRA_ARGS override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSIST=()
case "${1:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 [--save|--load]"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-1.7B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size 1 --mem-fraction-static 0.6 --disable-radix-cache \
  --attention-backend flashinfer --cuda-graph-max-bs-decode 512 \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
