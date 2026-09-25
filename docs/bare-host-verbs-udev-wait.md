# Bare-host DeepEP engines start 40 s late: rdma-core's udev wait, and the shim that removes it

Applies to: SGLang (and any other engine) with DeepEP low-latency all-to-all, run directly on a host
whose InfiniBand verbs devices exist but cannot be opened by the user. Seen on the radix H200 hosts in
venv mode (no container). Foundry SAVE, LOAD, the native graph engine and the eager engine are all
affected in the same way; Foundry's graph restore is not.

## Symptom

Every DeepEP low-latency engine pays about 40 s once, at the first forward that creates the DeepEP
buffer:

| engine | where the 40 s shows up |
|---|---|
| native SGLang with CUDA graphs | the first captured decode shape (43-46 s instead of 3-6 s) |
| eager SGLang | the first forward |
| Foundry SAVE and LOAD | the pre-capture DeepEP bootstrap, so `Init torch distributed` / capture look slow |

The per-graph capture cost of every later shape is unchanged, and Foundry's restore time is unchanged,
so the effect looks like "capture is 2-3x slower" on a 128-graph set and disappears in the per-graph
numbers once the first shape is excluded. In the podman container image on the same host the same
engine does not pay it.

The engine log carries no error for it. The closest hint is NVSHMEM's
`nvshmemi_transport_init:282: init failed for transport: IBGDA`, which is printed in both the slow and
the fast case and is harmless on a single node (traffic goes over NVLink).

## Cause

1. DeepEP's low-latency mode always sets `NVSHMEM_IB_ENABLE_IBGDA=1` (`deep_ep/buffers/legacy.py`), so
   NVSHMEM loads its IBGDA transport and asks libibverbs for every HCA.
2. On the bare host `/dev/infiniband/uverbsN` exists but `open()` returns `EPERM` for the unprivileged
   user. rdma-core's `open_cdev()` then falls back to `open_cdev_robust()`, which sets an inotify watch
   on `/dev/char/` and waits **up to 5 s per device** for udev to create the node. Eight HCAs are
   40 s per rank. The wait ends early whenever some other process creates a node in `/dev/char`, which is
   why the cost varies between 22 and 46 s.

   ```
   openat("/dev/infiniband/uverbs0", O_RDWR) = -1 EPERM
   inotify_add_watch(3, "/dev/char/", IN_CREATE) = 1
   openat("/dev/char/231:192", O_RDWR)      = -1 EPERM
   poll([{fd=3..},{fd=4..}], 2, -1) = 1 ... <4.999966>      # fd 4 = the 5 s timer
   ```
3. In the container neither `/dev/infiniband` nor `/dev/char` exists: libibverbs finds zero devices and
   IBGDA fails instantly, which is the "fast" behaviour.

What does not help: `NVSHMEM_REMOTE_TRANSPORT=none` (does not cover IBGDA), `NVSHMEM_HCA_LIST` (NVSHMEM
opens every device before filtering), `IBV_CONFIG_DIR` / `SYSFS_PATH`, and a stub libibverbs (NVSHMEM
init then fails and the engine crashes). `NCCL_IB_DISABLE=1` fixes the same 5 s/HCA probe on the NCCL
side (another ~40 s in `init torch distributed`) but not this one.

## The shim

`tools/host/no_cdev_wait.c` is a 15-line `LD_PRELOAD` library that makes the single call
`inotify_add_watch(fd, "/dev/char/...", ...)` fail with `ENOENT`. That is exactly the condition the
container produces, so rdma-core returns immediately and everything else is untouched: every other
inotify watch passes through to libc, no verbs, NVSHMEM or NCCL behaviour changes, and the engines
still fall back to NVLink/P2P as before.

Verified on 8xH200 (Qwen3-30B-A3B, EP4 with DP attention, DeepEP low latency, 128 decode graphs, warm
JIT caches, runs alternated on the same GPUs):

| engine | capture | first shape | time to `/health` |
|---|---|---|---|
| venv, no shim | 71.4 s | 44 s | 108.9 s |
| venv + shim | 31.5 / 32.5 s | 3-4 s | 67.4 / 68.1 s |
| container | 33.9 s | 6 s | 68.7 s |

TPOT was identical in every row (5.14 / 6.25 ms at bs 1 / 8).

## How to use it

Build once per host (any C compiler, no CUDA needed):

```bash
make -C foundry/tools/host                      # -> foundry/tools/host/libno_cdev_wait.so
# or: gcc -O2 -shared -fPIC -o libno_cdev_wait.so foundry/tools/host/no_cdev_wait.c -ldl
```

For Foundry SAVE and LOAD engines, name it in the graph-extension TOML, next to the NVSHMEM knob:

```toml
verbs_udev_wait_shim_path = "/abs/path/to/libno_cdev_wait.so"
```

The integration layer (`integration/sglang/runtime.py`, `setup_ld_preload_env`) then adds it to
`LD_PRELOAD` together with `libcuda_hook.so` and the NVSHMEM host library, before the engine processes
are spawned; a missing file is rejected when the TOML is read. Native SGLang engines (no Foundry) do not
read that TOML, so for a fair native-vs-Foundry comparison export it in the shell as well, appending to
whatever `LD_PRELOAD` already holds:

```bash
export LD_PRELOAD=/path/to/libno_cdev_wait.so${LD_PRELOAD:+:$LD_PRELOAD}
```

Foundry itself adds `libcuda_hook.so` from Python (`integration/sglang/runtime.py`,
`setup_ld_preload_env`) by prepending to whatever `LD_PRELOAD` already holds, so an exported shim
survives into the SAVE and LOAD engines; the recipe scripts and the `experimental/matrix3` harness set
nothing else. A launcher that overwrites the variable drops one of the two libraries, and the SAVE/LOAD
engines then silently lose the shim while the native engines keep it, which makes their start times
incomparable.

When to enable it: only when the verbs devices are present but blocked, i.e. `ls /dev/infiniband`
lists `uverbs*` and opening one fails with `EPERM`. With usable HCAs the shim is unnecessary (there is
no wait), and in a container without `/dev/char` it is a no-op. The bare-host bootstrap
(`experimental/host_setup_venv.sh`) builds the same source and exports it under that condition, together
with `NCCL_IB_DISABLE=1` for the NCCL-side probe.

## Evidence

`claude-doc/report_coldstart/h200_8gpu_venv/capture_speed_env_diff.md`: the container-vs-venv
environment diff (everything else identical: compat libcuda 610.57.04, torch 2.13.0+cu130, NCCL 2.29.7,
NVSHMEM, triton, flashinfer, sgl-kernel, glibc 2.39, ulimits, CPU settings), the per-shape capture
timings, the strace, and the bisection that leaves the udev wait as the only difference.
