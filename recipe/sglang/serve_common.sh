#!/bin/bash
# Shared driver for the multi-topology recipes (serve_qwen3.5-*.sh, serve_deepseek-v4-flash.sh), and the Foundry
# plugin helpers every serve_*.sh script uses (foundry_select, foundry_serve; see "Foundry plugin route" below).
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
# SAVE for a different model or topology. Baseline runs (no mode) capture the same decode-graph set natively.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Foundry plugin route -------------------------------------------------------------------------------------------
# Foundry runs as an SGLang plugin (entry point `foundry` in group sglang.srt.plugins, registered by installing foundry)
# and is switched on by FOUNDRY_GRAPH_EXTENSION_CONFIG=<TOML> in the launcher's environment; there is no CLI flag.
# SGLang silently runs natively when the entry point is not registered in the serving venv or when SGLANG_PLUGINS is
# set without `foundry`, so --save/--load run a preflight first (python -m foundry.integration.sglang.preflight, in
# the interpreter of the `sglang` launcher on PATH; SGL_PYTHON overrides it) and, once /health answers, check the
# engine log for the plugin's activation line (and, on LOAD, the restored-graph count).

# Interpreter of the `sglang` launcher on PATH, i.e. the venv that will serve.
foundry_sglang_python() {
  if [[ -n "${SGL_PYTHON:-}" ]]; then echo "$SGL_PYTHON"; return 0; fi
  local bin py
  bin=$(command -v sglang) || { echo "[Foundry preflight] FAILED: no 'sglang' on PATH (activate the SGLang venv)" >&2; return 1; }
  py=$(head -n 1 "$bin"); py=${py#\#!}; py=${py%% *}
  if [[ "$py" == /*python* && -x "$py" ]]; then echo "$py"; return 0; fi
  py="$(dirname "$bin")/python"                       # '#!/usr/bin/env python' or pip's '#!/bin/sh' long-path wrapper
  if [[ -x "$py" ]]; then echo "$py"; return 0; fi
  echo "[Foundry preflight] FAILED: cannot tell which python runs $bin; set SGL_PYTHON" >&2; return 1
}

# foundry_preflight <toml> <--save|--load>: exits the script with a one-line reason when Foundry would not run.
foundry_preflight() {
  local toml=$1 mode=$2 py
  py=$(foundry_sglang_python) || exit 1
  # -c / -m put the cwd first on sys.path, and a workspace root holding the sglang/ (or foundry/) checkout would
  # shadow the installed package that `sglang serve` imports. PYTHONSAFEPATH=1 (python >= 3.11) drops that entry;
  # the neutral cwd covers older interpreters, and --cwd keeps relative TOML paths resolving against this directory.
  local here=$PWD
  if ! (cd / && PYTHONSAFEPATH=1 "$py" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("foundry") is None)') 2>/dev/null; then
    echo "[Foundry preflight] FAILED: foundry is not importable by $py: pip install foundry-core (or pip install -e foundry) in this venv" >&2
    exit 1
  fi
  (cd / && PYTHONSAFEPATH=1 "$py" -m foundry.integration.sglang.preflight --toml "$toml" --cwd "$here" "$mode") || exit 1
}

# foundry_select <mode> [toml]: --save / --load export FOUNDRY_GRAPH_EXTENSION_CONFIG (default: the recipe TOML for
# the mode) and run the preflight. Any other mode (baseline, --warm) unsets it, so a stray export in the calling shell
# cannot turn a baseline run into a SAVE.
foundry_select() {
  local mode=$1 toml=${2:-}
  case "$mode" in
    --save) toml=${toml:-${SCRIPT_DIR}/foundry_save.toml} ;;
    --load) toml=${toml:-${SCRIPT_DIR}/foundry_load.toml} ;;
    *)      unset FOUNDRY_GRAPH_EXTENSION_CONFIG; return 0 ;;
  esac
  foundry_preflight "$toml" "$mode"
  export FOUNDRY_GRAPH_EXTENSION_CONFIG="$toml"
  echo "FOUNDRY_GRAPH_EXTENSION_CONFIG=$FOUNDRY_GRAPH_EXTENSION_CONFIG"
}

# foundry_verify <mode> <log>: after /health, report whether the plugin ran (it cannot report its own absence).
foundry_verify() {
  local mode=$1 log=$2 url="http://127.0.0.1:${PORT:-12000}/health" waited=0 limit=${FOUNDRY_HEALTH_TIMEOUT:-3600}
  until curl -sf -o /dev/null "$url"; do
    kill -0 $$ 2>/dev/null || return 0                  # the script (and its server) is gone
    sleep 2; waited=$(( waited + 2 ))
    if (( waited >= limit )); then echo "[Foundry verify] no /health after ${limit} s; log not checked" >&2; return 1; fi
  done
  local active loaded errors
  active=$(( $(grep -a -o '\[Foundry\] sglang plugin active: pid=[0-9]*' "$log" | sort -u | wc -l) ))
  if (( active == 0 )); then
    echo "[Foundry verify] WARNING: /health is up but '$log' has no '[Foundry] sglang plugin active' line:" \
         "SGLang ran NATIVELY, nothing was saved or restored (entry point not registered in this venv, or" \
         "SGLANG_PLUGINS without foundry)" >&2
    return 1
  fi
  echo "[Foundry verify] plugin active in $active process(es)"
  if [[ "$mode" == --load ]]; then
    loaded=$(grep -a -o '\[Foundry\] Loaded [0-9]* SGLang graphs in [0-9.]*s' "$log")
    if [[ -z "$loaded" ]]; then
      echo "[Foundry verify] WARNING: LOAD reached /health without a '[Foundry] Loaded N SGLang graphs' line in $log" >&2
    else
      echo "$loaded" | sed 's/^/[Foundry verify] /'
    fi
  fi
  errors=$(grep -a -c '\[HOOK\] ERROR' "$log")
  if (( errors > 0 )); then echo "[Foundry verify] WARNING: $errors '[HOOK] ERROR' line(s) in $log" >&2; fi
  return 0
}

# foundry_serve <mode> <command...>: run the server in the foreground. With Foundry selected, its output is also
# written to FOUNDRY_SERVE_LOG (default logs/sglang_<mode>_<time>.log) and checked by foundry_verify in the background.
foundry_serve() {
  local mode=$1; shift
  if [[ -z "${FOUNDRY_GRAPH_EXTENSION_CONFIG:-}" ]]; then "$@"; return; fi
  local log=${FOUNDRY_SERVE_LOG:-logs/sglang_${mode#--}_$(date +%Y%m%d_%H%M%S).log} verifier rc
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT:-12000}/health"; then
    echo "port ${PORT:-12000} already answers /health: stop that server first (the log check would read the wrong engine)" >&2
    return 1
  fi
  mkdir -p "$(dirname "$log")"
  echo "engine log: $log"
  foundry_verify "$mode" "$log" &
  verifier=$!
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  kill "$verifier" 2>/dev/null
  wait "$verifier" 2>/dev/null
  return "$rc"
}

# ---- Baseline parity with the plugin's pins ------------------------------------------------------------------------
# On --save / --load the Foundry plugin pins every config that selects state it cannot replay (custom all-reduce off,
# torch symm-mem all-reduce on, NCCL buffer registration / cuMem / NVLS off, the DeepGEMM precompile sweep off, ...;
# table in docs/sglang/overview.md "What the plugin pins and why"), so no script passes them. A baseline or --warm run
# has no plugin: foundry_baseline_pins (after foundry_select) gives it the pinned values that change what it runs, so
# it captures the same graph set and warms the same kernels as SAVE. The NCCL cuMem / NVLS pins are sglang's own
# defaults when the variables are unset, and buffer registration does not change what a baseline computes.
foundry_baseline_pins() {  # sets FOUNDRY_BASELINE_ARGS (empty with Foundry selected)
  FOUNDRY_BASELINE_ARGS=""
  [[ -n "${FOUNDRY_GRAPH_EXTENSION_CONFIG:-}" ]] && return 0
  FOUNDRY_BASELINE_ARGS="--disable-custom-all-reduce --enable-torch-symm-mem"
  export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
}

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
    tp2|tp4|tp8)   N=${cfg#tp}; ARGS="--tp-size $N" ;;
    ep2|ep4|ep8)   N=${cfg#ep}
      # The DP-attention gather is an all-reduce over the TP group; like the tp rows it runs on torch symm-mem
      # (pinned by the plugin, see foundry_baseline_pins).
      ARGS="--tp-size $N --dp-size $N --ep-size $N --enable-dp-attention $EP_A2A
            --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
      export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048 ;;
    tpep2|tpep4|tpep8) N=${cfg#tpep}
      ARGS="--tp-size $N --ep-size $N $EP_A2A
            --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
      GRAPHS="--cuda-graph-max-bs-decode $(( maxbs * N )) --cuda-graph-bs-decode $(seq -s' ' $N $N $(( maxbs * N )))
              --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled"
      export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 NVSHMEM_QP_DEPTH=2048 ;;
    tp2ep4|tp2ep8|tp4ep8) local atp=${cfg%%ep*}; atp=${atp#tp}; N=${cfg##*ep}
      ARGS="--tp-size $N --dp-size $(( N / atp )) --ep-size $N --enable-dp-attention $EP_A2A
            --chunked-prefill-size $EP_CHUNK $EP_ATTN --max-running-requests $(( 256 * N ))"
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

  case "$mode" in
    --save) echo "Using foundry SAVE" ;;
    --load) echo "Using foundry LOAD"
            # Map the packed kernel image instead of reading it: the eager read of the 4.7-5.5 GB image was 1.6-1.8 s
            # of every rank's LOAD (Qwen3.5-122B-FP8 EP8: 1.67 s -> 0.10 s with mmap, identical output). Measured with
            # the archive in the page cache; set FOUNDRY_MMAP_ARCHIVE=0 to read eagerly.
            export FOUNDRY_MMAP_ARCHIVE="${FOUNDRY_MMAP_ARCHIVE:-1}" ;;
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
  foundry_select "$mode"
  foundry_baseline_pins
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( N - 1 )))

  echo "model=$MODEL_PATH cfg=$cfg gpus=$CUDA_VISIBLE_DEVICES mem_fraction_static=$memfrac"
  # shellcheck disable=SC2086
  foundry_serve "$mode" sglang serve \
      --model-path "$MODEL_PATH" \
      --trust-remote-code \
      --host 0.0.0.0 --port "${PORT:-12000}" \
      --disable-radix-cache \
      --mem-fraction-static "$memfrac" \
      $ARGS $GRAPHS $MODEL_EXTRA $FOUNDRY_BASELINE_ARGS \
      ${SGL_EXTRA_ARGS:-}
}
