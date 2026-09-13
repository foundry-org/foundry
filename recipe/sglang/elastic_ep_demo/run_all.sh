#!/bin/bash
# Run the three engines back to back (foundry first so its archives get built), keep each result directory under
# WORK/eep_demo_results/<tag>_<mode>, and plot the decode speed of the three runs on one figure.
set -u
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); WORK=${WORK:-$PWD}; cd "$WORK"
MODES=${MODES:-"foundry graph orig"}; TAG=${TAG:-$(date -u +%m%d_%H%M)}; RES=$WORK/eep_demo_results; mkdir -p "$RES"
for m in $MODES; do
  echo "##### $m $(date -u +%T)"; WORK=$WORK bash "$RECIPE/demo.sh" $m > "$RES/${TAG}_$m.out" 2>&1
  rm -rf "$RES/${TAG}_$m"; cp -r "$WORK/eep_demo_logs/$m" "$RES/${TAG}_$m"; tail -n 12 "$RES/${TAG}_$m.out"
done
python3 "$RECIPE/plot_decode_speed.py" "$RES/${TAG}_decode_speed.png" "$RES/${TAG}_orig:SGLang w/o CUDA graph" "$RES/${TAG}_graph:SGLang w/ CUDA graph (native capture)" "$RES/${TAG}_foundry:Foundry (restored graphs)" 2>&1 | tail -n 1
python3 "$RECIPE/plot_decode_speed.py" "$RES/${TAG}_tpot.png" --metric tpot "$RES/${TAG}_orig:SGLang w/o CUDA graph" "$RES/${TAG}_graph:SGLang w/ CUDA graph (native capture)" "$RES/${TAG}_foundry:Foundry (restored graphs)" 2>&1 | tail -n 1
echo "##### done $(date -u +%T): $RES/${TAG}_*"
