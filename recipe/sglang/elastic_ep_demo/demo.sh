#!/bin/bash
# Elastic-EP fault-recovery demo, one engine flavour per run.
#
#   bash demo.sh orig      SGLang without CUDA graphs (--disable-cuda-graph)
#   bash demo.sh graph     SGLang with natively captured decode graphs (bs 1..MAXBS) on every (re)boot
#   bash demo.sh foundry   the same graphs restored by foundry LOAD on every (re)boot
#
# Topology: Qwen3-30B-A3B-FP8, EP8 as 4 "nodes" x 2 GPUs on one host (mooncake all-to-all + process groups, elastic
# EP with DP attention, 384 redundant experts). Timeline, relative to the launch command:
#   +0        launch all four nodes; probe load starts as soon as the engine is healthy
#   +T_KILL   kill nodes 1-3 (6 of 8 GPUs); the two survivors detect the fault and rebalance the experts onto themselves
#   +T_KILL+T_REBOOT   relaunch nodes 1-3 in recover mode; they rejoin and receive their experts back
#   +T_KILL+T_END      stop
# Load: every PROBE_PERIOD s, PROBE_BATCH requests with identical fixed prompts (PROBE_IN tokens in, PROBE_OUT out,
# greedy) are submitted and their TTFT/TPOT recorded, so every stage of every run measures the same work.
# The foundry mode needs two archives (built once by this script, see README): a fresh-launch SAVE of all four
# nodes and a recover-mode SAVE of nodes 1-3.
#
# Layout: WORK (default: current directory) holds the archives (foundry_archive_eep_fresh, foundry_archive_eep_recover)
# and logs (eep_demo_logs/<mode>/: node logs, events.txt, probe_*.csv). Requires 8 GPUs, the sglang fork with elastic
# EP (branch foundry-elastic-ep), foundry installed, the model in the local HF cache (HF_HUB_OFFLINE=1).
set -u; ulimit -c 0
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-${MODE:?orig|graph|foundry}}; case $MODE in orig|graph|foundry) ;; *) echo "usage: demo.sh orig|graph|foundry"; exit 1;; esac
WORK=${WORK:-$PWD}; cd "$WORK"
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-FP8}
IB=${IB_DEVICES:-mlx5_ib0,mlx5_ib1}      # RDMA devices for the mooncake process groups
MAXBS=${MAXBS:-256}                       # decode graphs bs 1..MAXBS per rank (graph / foundry modes)
REDUNDANT=${REDUNDANT:-384}               # redundant expert slots (128 + 384 = 512 physical, 64 per GPU)
DIST=${DIST_INIT_ADDR:-127.0.0.1:25555}
T_KILL=${T_KILL:-330}; T_REBOOT=${T_REBOOT:-120}; T_END=${T_END:-300}
NP=${LOAD_PROCS:-4}; PROBE_BATCH=${PROBE_BATCH:-256}; PROBE_IN=${PROBE_IN:-128}; PROBE_OUT=${PROBE_OUT:-128}; PROBE_PERIOD=${PROBE_PERIOD:-10}
PROBE_SRC=${PROBE_SRC:-sharegpt}          # sharegpt (ShareGPT_V3 from the HF cache) | random
JOIN_DIR=/tmp/eep_join
OUT=$WORK/eep_demo_logs/$MODE; rm -rf "$OUT"; mkdir -p "$OUT"; EV=$OUT/events.txt; : > "$EV"

CU12=$(python3 -c "import glob,os,site;print(os.path.dirname(glob.glob(site.getsitepackages()[0]+'/nvidia/**/libcudart.so.12',recursive=True)[0]))" 2>/dev/null)
export LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:+$EXTRA_LD_LIBRARY_PATH:}${CU12:+$CU12:}${LD_LIBRARY_PATH:-}
export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 HF_HUB_OFFLINE=1 PYTORCH_NVML_BASED_CUDA_CHECK=1
export SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512          # dispatch buffer per rank = 256 requests x 2 (max-running-requests / dp)
export SGLANG_MOONCAKE_EP_TIMEOUT_US=${EP_TIMEOUT_US:-2000000}          # a2a peer timeout = fault detection latency; baked into the captured graphs
export SGLANG_ELASTIC_PROACTIVE_DEACTIVATE=world                        # survivors deactivate the dead ranks as soon as one a2a times out
export SGLANG_ELASTIC_JOIN_MARKER_DIR=$JOIN_DIR; rm -rf "$JOIN_DIR"
export SGLANG_ELASTIC_REBALANCE_AFTER_RECOVER=1                         # rejoined ranks get their experts back right after the join
export SGLANG_ELASTIC_POST_RECOVERY_LAYOUT=${POST_LAYOUT:-trivial}      # trivial (reinstall the initial layout) | eplb (recompute with --eplb-algorithm)
export SGLANG_EXPERT_LOCATION_UPDATER_TIMING=${EPLB_TIMING:-0}
OPT="SGLANG_PRESPAWN_WORKERS=1 SGLANG_EARLY_FORKSERVER=1 SGLANG_FORKSERVER_ALLOW_FOUNDRY=1"   # fast process start (forkserver daemon + pre-spawn)
case $MODE in
  orig)    GRAPH_ARGS="--disable-cuda-graph"; HOOK=0 ;;
  graph)   GRAPH_ARGS="--cuda-graph-max-bs-decode $MAXBS --disable-cuda-graph-padding"; HOOK=0 ;;
  foundry) GRAPH_ARGS="--cuda-graph-max-bs-decode $MAXBS --disable-cuda-graph-padding"; HOOK=1 ;;
esac

ev() { echo "$(date +%s.%N) $1" >> "$EV"; echo "=== $1 $(date -u +%T)"; }
args() { # $1 node rank, $2 port, [$3 recover]
  echo --model-path $MODEL --trust-remote-code --tp 8 --dp 8 --nnodes 4 --node-rank $1 --dist-init-addr $DIST --random-seed 42 \
       --enable-dp-attention --enable-dp-lm-head --elastic-ep-backend mooncake --mooncake-ib-device $IB --moe-a2a-backend mooncake --deepep-mode low_latency \
       --moe-dense-tp-size 1 --disable-custom-all-reduce --enable-eplb ${EPLB_ALGO:+--eplb-algorithm $EPLB_ALGO} --eplb-rebalance-num-iterations 100000000 \
       --expert-distribution-recorder-buffer-size 1000 --ep-num-redundant-experts $REDUNDANT --chunked-prefill-size ${CHUNK:-4096} --max-running-requests ${MAX_RUNNING:-2048} \
       --decode-log-interval ${DECODE_LOG_INTERVAL:-1} $GRAPH_ARGS --mem-fraction-static ${MEMFRAC:-0.5} --moe-runner-backend deep_gemm --disable-radix-cache --attention-backend fa3 \
       --host 127.0.0.1 --port $2 ${3:+--elastic-ep-join-mode recover} ${EXTRA_ARGS:-}
}
gen() { curl -s -m 120 http://127.0.0.1:21000/generate -H "Content-Type: application/json" -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}' | cut -c1-70; }
launch() { # $1 node rank, $2 "" | recover, $3 extra args, $4 log tag, $5 extra env
  local k=$1 rec=$2 extra=$3 tag=$4 xenv=${5:-}; local g=$((2*k)),$((2*k+1)); local log=$OUT/node${k}${tag}.log
  local envs="$OPT $xenv"; [ "$HOOK" = 1 ] && [ -n "$rec$extra" ] && envs="$envs SGLANG_FORKSERVER_FILE=$FS_HOOK"   # foundry engines fork from the hook daemon
  env $envs CUDA_VISIBLE_DEVICES=$g sglang serve $(args $k $((21000+1000*k)) $rec) $extra > "$log" 2>&1 &
  disown; }  # engines are stopped by pkill; disown keeps bash from reporting them as Killed
wait_healthy() { # $1 log glob tag, $2 timeout s
  local t0=$(date +%s)
  for _ in $(seq 1 $2); do curl -fsS -m 3 http://127.0.0.1:21000/health >/dev/null 2>&1 && { echo $(( $(date +%s) - t0 )); return 0; }
    grep -qa "Scheduler hit\|scheduler died\|AssertionError" $OUT/node?$1.log 2>/dev/null && { echo LAUNCH_FAILED; grep -a -m3 "Error\|assert" $OUT/node?$1.log | cut -c1-200 >&2; return 1; }; sleep 1; done
  echo TIMEOUT; return 1; }
n_rec() { grep -ac "recover ranks .* done" $OUT/node0.log; }
wait_recover() { # $1 count before, $2 label
  local n0=$1 label=$2 t=$(date +%s.%N) deadline=$(( $(date +%s) + ${RECOVER_TIMEOUT:-300} ))
  while [ $(date +%s) -lt $deadline ]; do [ "$(n_rec)" -gt "$n0" ] && break
    grep -qa "Scheduler hit\|scheduler died" $OUT/node*_r*.log $OUT/node*_save.log 2>/dev/null && { echo "RECOVER_FAILED ($label)"; grep -a -m2 "Error\|assert" $OUT/node*_r*.log $OUT/node*_save.log 2>/dev/null | cut -c1-200; break; }
    curl -s -m 2 http://127.0.0.1:21000/generate -H "Content-Type: application/json" -d '{"text":"hi","sampling_params":{"max_new_tokens":1}}' >/dev/null 2>&1; sleep 1; done
  echo "$label: $(python3 -c "import time;print(f'{time.time()-$t:.1f}')") s; $(grep -a "recover ranks" $OUT/node0.log | tail -1 | cut -c1-90)"; }
kill_nodes() { rm -rf "$JOIN_DIR"; for k in "$@"; do pkill -9 -f "port $((21000+1000*k))"; done; pkill -9 -f "sglang::scheduler_DP[2-7]"; sleep 2; }
wait_fault() { local n0=$(grep -ac "EPLB due to rank faults" $OUT/node0.log); for _ in $(seq 1 120); do [ "$(grep -ac 'EPLB due to rank faults' $OUT/node0.log)" -gt "$n0" ] && break; sleep 1; done; sleep 3; }
stop_engines() { bash "$RECIPE/cleanup_engines.sh"; }
sleep_until() { local now=$(date +%s); [ $1 -gt $now ] && sleep $(( $1 - now )); }

echo "##### elastic-EP demo MODE=$MODE MAXBS=$MAXBS WORK=$WORK $(date -u +%T)"
stop_engines
HOOK=0 LOGDIR=$OUT bash "$RECIPE/start_daemon.sh"
FS_HOOK=""; if [ "$HOOK" = 1 ]; then FS_HOOK=/tmp/sglang_forkserver_hook.json; HOOK=1 FS_FILE=$FS_HOOK LOGDIR=$OUT bash "$RECIPE/start_daemon.sh"; fi

# ---- one-time archives for the foundry mode -------------------------------------------------------------------
if [ $MODE = foundry ] && { [ ! -d foundry_archive_eep_recover/rank_7 ] || [ ! -d foundry_archive_eep_fresh/rank_0 ] || [ "${FORCE_SETUP:-0}" = 1 ]; }; then
  # The a2a timeout is a kernel argument inside the captured graphs, so SAVE runs with the serving timeout; a file
  # barrier (FOUNDRY_WARMUP_BARRIER_*) lines the ranks up before the warmup forwards so the short timeout does not trip.
  rm -rf foundry_archive_eep_fresh foundry_archive_eep_recover
  echo "##### setup 1/2: fresh-launch SAVE of all four nodes -> foundry_archive_eep_fresh"
  rm -f $JOIN_DIR/warmup_ready_*; ev "setup: launch EP8 (SAVE fresh)"
  for k in 0 1 2 3; do launch $k "" "--foundry-graph-extension-config-path $RECIPE/foundry_save_fresh.toml" "_savefresh" "FOUNDRY_WARMUP_BARRIER_DIR=$JOIN_DIR FOUNDRY_WARMUP_BARRIER_COUNT=8"; done
  T=$(wait_healthy _savefresh 300) || { echo SETUP_SAVE_FRESH_FAILED; exit 1; }; echo "fresh SAVE healthy in $T s; gen: $(gen)"; sleep 3; stop_engines; sleep 3
  echo "##### setup 2/2: recover-mode SAVE of nodes 1-3 -> foundry_archive_eep_recover"
  ev "setup: launch EP8"; for k in 0 1 2 3; do launch $k "" "" ""; done
  T=$(wait_healthy "" 240) || { echo SETUP_LAUNCH_FAILED; exit 1; }; echo "healthy in $T s"
  ev "setup: kill nodes 1-3"; kill_nodes 1 2 3; wait_fault
  rm -f $JOIN_DIR/warmup_ready_*; ev "setup: SAVE reboot nodes 1-3"; n0=$(n_rec)
  for k in 1 2 3; do launch $k recover "--foundry-graph-extension-config-path $RECIPE/foundry_save_recover.toml" "_save" "FOUNDRY_WARMUP_BARRIER_DIR=$JOIN_DIR FOUNDRY_WARMUP_BARRIER_COUNT=6"; done
  RECOVER_TIMEOUT=420 wait_recover $n0 "setup: nodes 1-3 save-reboot"; sleep 3
  echo "archives: fresh $(du -sh foundry_archive_eep_fresh | cut -f1), recover $(du -sh foundry_archive_eep_recover | cut -f1)"
  ev "setup: tear down"; stop_engines; sleep 3
  mkdir -p $OUT/setup; mv $OUT/node*.log $OUT/setup/ 2>/dev/null || true; : > "$EV"
fi
LOAD_FRESH=""; LOAD_RECOVER=""
if [ $MODE = foundry ]; then LOAD_FRESH="--foundry-graph-extension-config-path $RECIPE/foundry_load_fresh.toml"; LOAD_RECOVER="--foundry-graph-extension-config-path $RECIPE/foundry_load_recover.toml"; fi

# ---- fixed probe prompts ----------------------------------------------------------------------------------------
python3 "$RECIPE/make_probe_pool.py" $OUT/probe_pool --n $NP --batch $(( PROBE_BATCH / NP )) --in-len $PROBE_IN --source $PROBE_SRC --seed ${PROBE_SEED:-0} 2>&1 | tail -n 1
echo "##### timed run: kill at +$T_KILL s, reboot at kill+$T_REBOOT s, end at kill+$T_END s"
LOADPIDS=()
for i in $(seq 0 $((NP-1))); do
  python3 "$RECIPE/probe_gen.py" $OUT/probe_$i.csv $(( T_KILL + T_END + 10 )) --period $PROBE_PERIOD --batch $(( PROBE_BATCH / NP )) --in-len $PROBE_IN --out-len $PROBE_OUT --seed $i --stall-timeout ${STALL_TIMEOUT:-45} --pool-file $OUT/probe_pool_$i.json --temperature 0 > $OUT/probe_gen_$i.log 2>&1 &
  LOADPIDS+=($!)
done
ev "launch EP8"; TL=$(date +%s)
for k in 0 1 2 3; do launch $k "" "$LOAD_FRESH" ""; done
T=$(wait_healthy "" 300) || { echo LAUNCH_FAILED; stop_engines; exit 1; }
ev "healthy"; echo "healthy in $T s; gen: $(gen)"; grep -a "Engine startup timings" $OUT/node0.log | head -1 | cut -c1-160
sleep_until $(( TL + T_KILL )); ev "kill"; TK=$(date +%s); kill_nodes 1 2 3
( wait_fault; ev "fault EPLB done (2 survivors)" ) &
sleep_until $(( TK + T_REBOOT )); ev "reboot nodes 1-3"; n0=$(n_rec); for k in 1 2 3; do launch $k recover "$LOAD_RECOVER" "_r$k"; done
wait_recover $n0 "nodes 1-3 (-> EP8)"; ev "EP8 recovered"
for _ in $(seq 1 120); do [ "$(grep -ac "post-recovery EPLB rebalance done" $OUT/node0.log)" -ge 1 ] && break; sleep 0.5; done; ev "post-recovery EPLB done"
sleep_until $(( TK + T_END )); ev "load end"; wait ${LOADPIDS[@]} 2>/dev/null
echo "probe batches: $(cat $OUT/probe_*.csv 2>/dev/null | grep -vc epoch)"
stop_engines
python3 "$RECIPE/tpot_summary.py" "$OUT"
echo "##### demo $MODE done $(date -u +%T); results in $OUT"
