#!/bin/bash
# Shared driver for the multi-topology recipes (serve_qwen3.5-*.sh, serve_deepseek-v4-flash.sh).
#
# A model script sets MODEL_PATH (and optionally the per-model knobs below), then calls
#     serve_main <cfg> [--save|--load|--warm]
# where <cfg> names the parallel topology with the same vocabulary as the cold-start matrix
# (experimental/matrix3/coldstart.sh), so every row here was run exactly like this:
#   single            1 GPU
#   dp2 dp4 dp8       data parallel, one replica per rank
#   tp2 tp4 tp8       tensor parallel, torch symm-mem allreduce inside the decode graphs
#   ep2 ep4 ep8       expert parallel: DP attention + DeepEP low-latency + DeepGEMM, capture bs per rank
#   tpep2 tpep4       TP attention (symm-mem) + EP over the same ranks; a global batch is reduce-scattered over
#                     N ranks, so the capture list is bs = N, 2N, ... (bs < N runs eagerly)
#   tp2ep4 tp2ep8 tp4ep8   attention TP a inside DP-attention groups (dp = n/a) + EP over all n ranks
#
# Environment overrides: MAXBS (256: decode graphs for every bs 1..MAXBS, no padding), MEMFRAC (per-model default),
# PORT (12000), SGL_EXTRA_ARGS (appended verbatim). Per-model knobs a script may set before serve_main:
#   EP_ATTN   attention backend flag on the EP rows (default "--attention-backend fa3"; "" = sglang's own default)
#   EP_CHUNK  --chunked-prefill-size on the EP rows (256; under DP attention it is divided by dp and must stay a
#             multiple of the KV page size)
#   EP_A2A    all-to-all + MoE runner flags on the EP rows (DeepEP low-latency + deep_gemm)
#   MODEL_EXTRA  flags for every topology (e.g. a Mamba/GDN state-pool cap for hybrid models)
#   memfrac_default <cfg>  optional function returning the validated --mem-fraction-static for a topology
#
# Both modes share the TOMLs next to this file (workspace_root = "foundry_archive"): rm -rf foundry_archive before a
# SAVE for a different model or topology. Baseline runs (no flag) capture the same decode-graph set natively.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_ATTN="${EP_ATTN-"--attention-backend fa3"}"
EP_CHUNK="${EP_CHUNK:-256}"
EP_A2A="${EP_A2A-"--moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm"}"
MODEL_EXTRA="${MODEL_EXTRA:-}"

topology_args() {  # $1 cfg -> sets N, ARGS, GRAPHS
  local cfg=$1 maxbs=${MAXBS:-256}
  GRAPHS="--cuda-graph-max-bs-decode $maxbs --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled"
  case "$cfg" in
    single)        N=1; ARGS="--tp-size 1" ;;
    dp2|dp4|dp8)   N=${cfg#dp}; ARGS="--tp-size 1 --dp-size $N" ;;
    tp2|tp4|tp8)   N=${cfg#tp}; ARGS="--tp-size $N --disable-custom-all-reduce --enable-torch-symm-mem" ;;
    ep2|ep4|ep8)   N=${cfg#ep}
      # The DP-attention gather is an all-reduce over the TP group: route it through torch symm-mem like the tp
      # rows (uniform collective path across topologies).
      ARGS="--tp-size $N --dp-size $N --ep-size $N --enable-dp-attention $EP_A2A --enable-torch-symm-mem
            --disable-custom-all-reduce --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
      export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048 ;;
    tpep2|tpep4|tpep8) N=${cfg#tpep}
      ARGS="--tp-size $N --ep-size $N --enable-torch-symm-mem $EP_A2A --disable-custom-all-reduce
            --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
      GRAPHS="--cuda-graph-max-bs-decode $(( maxbs * N )) --cuda-graph-bs-decode $(seq -s' ' $N $N $(( maxbs * N )))
              --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled"
      export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048 ;;
    tp2ep4|tp2ep8|tp4ep8) local atp=${cfg%%ep*}; atp=${atp#tp}; N=${cfg##*ep}
      ARGS="--tp-size $N --dp-size $(( N / atp )) --ep-size $N --enable-dp-attention $EP_A2A --enable-torch-symm-mem
            --disable-custom-all-reduce --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
      GRAPHS="--cuda-graph-max-bs-decode $(( maxbs * atp )) --cuda-graph-bs-decode $(seq -s' ' $atp $atp $(( maxbs * atp )))
              --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled"
      export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048 ;;
    *) echo "unknown topology '$cfg' (single | dpN | tpN | epN | tpepN | tp2ep4 | tp2ep8 | tp4ep8)" >&2; return 1 ;;
  esac
}

serve_main() {
  local cfg=${1:?Usage: $0 <cfg> [--save|--load|--warm]} mode=${2:-}
  : "${MODEL_PATH:?MODEL_PATH must be set by the model script}"
  topology_args "$cfg" || exit 1
  local memfrac=${MEMFRAC:-}
  if [[ -z "$memfrac" ]] && declare -F memfrac_default >/dev/null; then memfrac=$(memfrac_default "$cfg"); fi
  memfrac=${memfrac:-0.8}

  local foundry_args=()
  case "$mode" in
    --save) foundry_args=( --foundry-graph-extension-config-path "${SCRIPT_DIR}/foundry_save.toml" ); echo "Using foundry SAVE" ;;
    --load) foundry_args=( --foundry-graph-extension-config-path "${SCRIPT_DIR}/foundry_load.toml" ); echo "Using foundry LOAD" ;;
    --warm)
      # Cache warm-up, once per machine and model: plain SGLang with the SAME decode-graph set (identical to
      # running without a mode). SAVE compiles inside the capture window on purpose (no eager forward); a
      # freshly compiled inductor kernel is autotuned on first run and the benchmark synchronizes the device,
      # which capture forbids. SGLang's own pre-capture warmup forwards compile and autotune every kernel with
      # capture-mode inputs (views of the static batch buffers), so after this run the inductor / Triton /
      # DeepGEMM caches under SGLANG_CACHE_DIR hold exactly the artifacts SAVE needs. An eager run
      # (--disable-cuda-graph) compiles different artifacts and does not help.
      echo "Cache warm-up: plain SGLang with the recipe's decode-graph set" ;;
    "")     echo "Running without foundry (baseline SGLang, same decode-graph set)" ;;
    *)      echo "Usage: $0 <cfg> [--save|--load|--warm]"; exit 1 ;;
  esac

  # Identical on baseline, SAVE and LOAD so the captured graphs match:
  # - NCCL_CUMEM_ENABLE=0 / NCCL_NVLS_ENABLE=0: NCCL buffers through the plain allocator (deterministic VMM offsets)
  # - SGLANG_JIT_DEEPGEMM_PRECOMPILE=0: the rank-0-only DeepGEMM precompile warmup would put ~14 GB of scratch into
  #   rank 0's deterministic range; kernels still JIT lazily per shape.
  export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( N - 1 )))

  echo "model=$MODEL_PATH cfg=$cfg gpus=$CUDA_VISIBLE_DEVICES mem_fraction_static=$memfrac"
  # shellcheck disable=SC2086
  sglang serve \
      --model-path "$MODEL_PATH" \
      --trust-remote-code \
      --host 0.0.0.0 --port "${PORT:-12000}" \
      --disable-radix-cache \
      --mem-fraction-static "$memfrac" \
      $ARGS $GRAPHS $MODEL_EXTRA \
      "${foundry_args[@]}" \
      ${SGL_EXTRA_ARGS:-}
}
