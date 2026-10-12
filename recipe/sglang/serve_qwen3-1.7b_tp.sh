#!/bin/bash
# Qwen3-1.7B, tensor parallel (torch symm-mem allreduce in the decode graphs).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_tp.sh <tp_size> [--save|--load]
# No mode = plain SGLang. Archive: ./foundry_archive (foundry.toml).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP_SIZE=${1:?Usage: $0 <tp_size> [--save|--load]}
PERSIST=()
case "${2:-}" in
  --save) PERSIST=(--cuda-graph-persistence save --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml") ;;
  --load) PERSIST=(--cuda-graph-persistence load --cuda-graph-persistence-config "$SCRIPT_DIR/foundry.toml")
          export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
  "") ;;
  *) echo "Usage: $0 <tp_size> [--save|--load]"; exit 1 ;;
esac
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-1.7B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size "$TP_SIZE" --mem-fraction-static 0.6 --disable-radix-cache \
  --attention-backend flashinfer --cuda-graph-max-bs-decode 128 \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
