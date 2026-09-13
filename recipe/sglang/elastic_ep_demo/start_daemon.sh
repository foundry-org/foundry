#!/bin/bash
# (Re)start the SGLang forkserver daemon that the demo's engines fork from (SGLANG_EARLY_FORKSERVER=1 +
# SGLANG_PRESPAWN_WORKERS=1 give the fast process start). HOOK=1 preloads foundry's libcuda_hook.so into the daemon
# so foundry SAVE/LOAD engines can fork from it; HOOK=0 (default) starts a hook-less daemon for plain SGLang engines.
# FS_FILE selects the daemon's json handle (a non-default path starts an additional daemon next to the default one).
# EXTRA_LD_LIBRARY_PATH: prepended for the daemon and every forked engine (e.g. a CUDA compat driver directory).
set -u
FS_FILE=${FS_FILE:-/tmp/sglang_forkserver.json}
LOGDIR=${LOGDIR:-/tmp/eep_demo}; mkdir -p "$LOGDIR"
DLOG=$LOGDIR/forkserver_daemon$( [ "$FS_FILE" = /tmp/sglang_forkserver.json ] || echo _$(basename "$FS_FILE" .json) ).log
fs=$(python3 -c "import json;print(json.load(open('$FS_FILE'))['pid'])" 2>/dev/null)
dp=$(python3 -c "import json;print(json.load(open('$FS_FILE'))['daemon_pid'])" 2>/dev/null)
for c in $(pgrep -P "${fs:-0}" 2>/dev/null); do kill -9 "$c" 2>/dev/null; done
[ -n "$dp" ] && kill "$dp" 2>/dev/null; [ -n "$fs" ] && kill -9 "$fs" 2>/dev/null
rm -f "$FS_FILE"
# mooncake dlopens libcudart.so.12 from the pip wheel; forked engines inherit the daemon's loader search path.
CU12=$(python3 -c "import glob,os,site;print(os.path.dirname(glob.glob(site.getsitepackages()[0]+'/nvidia/**/libcudart.so.12',recursive=True)[0]))" 2>/dev/null)
# libcuda_hook.so and DeepEP's NVSHMEM host lib, auto-detected the same way a foundry engine detects them.
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -r HOOK_SO NVSHMEM_SO <<<"$(python3 -c "
from foundry.integration.sglang.config import CUDAGraphExtensionConfig as C
c = C.from_toml('$RECIPE/foundry_load_fresh.toml'); print(c.hook_library_path or '', c.nvshmem_host_path or '')")"
[ "${HOOK:-0}" = 1 ] && [ -z "$HOOK_SO" ] && { echo "libcuda_hook.so not found (is foundry installed?)"; exit 1; }
PRE=$NVSHMEM_SO
[ "${HOOK:-0}" = 1 ] && PRE=$HOOK_SO${NVSHMEM_SO:+:$NVSHMEM_SO}
LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:+$EXTRA_LD_LIBRARY_PATH:}${CU12:+$CU12:}${LD_LIBRARY_PATH:-} \
LD_PRELOAD=$PRE PYTORCH_NVML_BASED_CUDA_CHECK=1 SGLANG_FORKSERVER_FILE=$FS_FILE \
  setsid nohup python3 -m sglang.srt.utils.early_forkserver > "$DLOG" 2>&1 &
for _ in $(seq 1 60); do [ -f "$FS_FILE" ] && break; sleep 1; done
grep -a ready "$DLOG" | tail -1 | cut -c1-80
