# Elastic-EP fault-recovery demo

Shows what foundry buys during an elastic expert-parallel (EP) fault and recovery: the rebooted ranks
come back with their decode CUDA graphs restored from an archive instead of recapturing them, so the
engine is back at full decode speed sooner, while the steady-state decode speed matches natively
captured graphs. Three engine flavours run the same scripted fault on the same fixed workload:

| mode      | decode graphs on every (re)boot                            |
|-----------|------------------------------------------------------------|
| `orig`    | none (`--disable-cuda-graph`)                              |
| `graph`   | captured natively, batch sizes 1..`MAXBS` (default 256)    |
| `foundry` | the same graphs restored by foundry LOAD                   |

## Setup

- Qwen3-30B-A3B-FP8 as **EP8 on one 8-GPU host, run as 4 "nodes" x 2 GPUs** (`--nnodes 4`, one
  `sglang serve` per node pair), DP attention, mooncake all-to-all and process groups (`--elastic-ep-backend mooncake`),
  fault-aware EPLB with 384 redundant experts, DeepGEMM MoE runner, FA3 attention, radix cache off.
- **Timeline** (relative to the launch command): `+0` launch all four nodes, probe load starts once
  `/health` passes; `+T_KILL` (330 s) kill nodes 1-3, the two survivors detect the fault through the
  all-to-all timeout and rebalance all experts onto themselves; `+T_KILL+T_REBOOT` (120 s) relaunch
  nodes 1-3 with `--elastic-ep-join-mode recover`, they rejoin and the initial expert layout is
  reinstalled; `+T_KILL+T_END` (300 s) stop.
- **Workload**: every `PROBE_PERIOD` (10 s), `LOAD_PROCS` (4) client processes each submit
  `PROBE_BATCH/LOAD_PROCS` (64) requests from a fixed prompt pool (ShareGPT prompts cut to `PROBE_IN`
  128 tokens, `PROBE_OUT` 128 output tokens, greedy). The same prompts are replayed in every period and
  every stage of every mode, so the per-period TPOT medians are directly comparable; decode speed is
  reported as `1000 / TPOT` tokens/s per request.
- Every (re)boot uses the fast process start (forkserver daemon + pre-spawned workers, see
  `start_daemon.sh`). Plain engines fork from a hook-less daemon; foundry SAVE/LOAD engines fork from a
  second daemon that has `libcuda_hook.so` preloaded.

## Prerequisites

- 8 GPUs (H100/H200 class) with RDMA devices for mooncake (`IB_DEVICES`, default `mlx5_ib0,mlx5_ib1`).
- The SGLang fork with the elastic-EP work (branch `foundry-elastic-ep`), foundry installed
  (`pip install -e foundry --no-build-isolation`), mooncake + DeepGEMM + FA3 kernels importable.
- `Qwen/Qwen3-30B-A3B-FP8` and (for `PROBE_SRC=sharegpt`) `ShareGPT_V3_unfiltered_cleaned_split.json`
  in the local Hugging Face cache; the scripts run with `HF_HUB_OFFLINE=1`. `PROBE_SRC=random`
  needs no dataset.
- `curl`, `python3` with `matplotlib` (plots only).

## Quick start

```bash
cd /some/work/dir                      # archives and logs land here (WORK, default: cwd)
R=/path/to/foundry/recipe/sglang/elastic_ep_demo
bash $R/demo.sh orig                   # ~12 min
bash $R/demo.sh graph                  # ~13 min
bash $R/demo.sh foundry                # ~35 min the first time (builds the two archives), ~12 min after
```

Or all three back to back with the comparison plots:

```bash
bash $R/run_all.sh                     # MODES="foundry graph orig" TAG=<utc stamp>
```

Each `demo.sh` run ends with a per-phase table (probe batches, failed requests, TPOT/TTFT medians)
from `tpot_summary.py`. `run_all.sh` keeps every run under `WORK/eep_demo_results/<TAG>_<mode>/` and
writes `<TAG>_decode_speed.png` (tokens/s per request vs time, all three modes, fault/recovery events
marked) and `<TAG>_tpot.png`.

## What the foundry mode does the first time

Foundry needs two archives, which `demo.sh foundry` builds when they are missing (or with
`FORCE_SETUP=1`), then tears everything down and runs the timed demo:

1. **Fresh-launch SAVE** of all four nodes into `foundry_archive_eep_fresh` (used by the timed cold
   start of the demo).
2. A plain EP8 launch, kill nodes 1-3, then a **recover-mode SAVE** reboot of nodes 1-3 into
   `foundry_archive_eep_recover` (used by the timed reboot). Recover-mode joins allocate differently
   from a fresh launch, so the reboot needs its own archive.

The mooncake all-to-all timeout is a kernel argument inside the captured graphs, so SAVE runs with
the serving timeout (`SGLANG_MOONCAKE_EP_TIMEOUT_US`); `FOUNDRY_WARMUP_BARRIER_*` lines the ranks up
before the warmup forward so the short timeout does not trip during capture. Archives must be rebuilt
when anything that changes the captured graphs changes (`MAXBS`, `REDUNDANT`, `CHUNK`, `MAX_RUNNING`,
`MEMFRAC`, the timeout, the model, the kernels).

## Knobs (environment variables)

| variable | default | meaning |
|----------|---------|---------|
| `WORK` | `$PWD` | directory for archives (`foundry_archive_eep_*`) and logs (`eep_demo_logs/<mode>/`) |
| `MAXBS` | 256 | largest decode graph batch size per rank (`graph`/`foundry`) |
| `T_KILL`, `T_REBOOT`, `T_END` | 330, 120, 300 | schedule in seconds (see timeline) |
| `PROBE_BATCH`, `PROBE_IN`, `PROBE_OUT`, `PROBE_PERIOD`, `LOAD_PROCS` | 256, 128, 128, 10, 4 | workload |
| `PROBE_SRC`, `PROBE_SEED` | `sharegpt`, 0 | prompt pool source |
| `IB_DEVICES` | `mlx5_ib0,mlx5_ib1` | RDMA devices for mooncake |
| `MODEL` | `Qwen/Qwen3-30B-A3B-FP8` | model path |
| `REDUNDANT` | 384 | redundant expert slots |
| `MEMFRAC` | 0.5 | `--mem-fraction-static` |
| `EP_TIMEOUT_US` | 2000000 | mooncake all-to-all peer timeout (fault detection latency) |
| `POST_LAYOUT` | `trivial` | expert layout after recovery: `trivial` (initial layout) or `eplb` (recompute; set `EPLB_ALGO`, e.g. `elasticity_aware_node_cover`) |
| `EPLB_TIMING` | 0 | 1 logs per-layer expert-relocation timing |
| `FORCE_SETUP` | 0 | 1 rebuilds the foundry archives |
| `EXTRA_ARGS` | | extra `sglang serve` flags appended to every node |
| `EXTRA_LD_LIBRARY_PATH` | | prepended to `LD_LIBRARY_PATH` of the daemons and engines (e.g. a CUDA compat driver directory) |

## Outputs (`WORK/eep_demo_logs/<mode>/`)

- `node<k>.log`, `node<k>_r<k>.log`: engine logs of the initial launch and of the reboot; `setup/`
  holds the SAVE-phase logs of the foundry mode.
- `events.txt`: `<epoch> <event>` lines (`launch EP8`, `healthy`, `kill`, `fault EPLB done (2 survivors)`,
  `reboot nodes 1-3`, `EP8 recovered`, `post-recovery EPLB done`, `load end`).
- `probe_<i>.csv`: one row per probe batch per client (epoch, n_ok, n_fail, TTFT/TPOT percentiles);
  `probe_pool_<i>.json`: the fixed prompts.
- `forkserver_daemon*.log`: daemon logs.

Reading the result: in `graph` mode the reboot's "EP8 recovered" comes ~25 s after the relaunch
(graph capture); in `foundry` mode within a few seconds (graph restore); in `orig` mode it is quick
too but the decode speed stays lower throughout. Steady-state decode speed of `foundry` and `graph`
should agree within noise.

## Reference run

One host with 8x H100 80GB, driver 580 with the CUDA 13.3 compat library (`EXTRA_LD_LIBRARY_PATH`),
defaults everywhere (`run_all.sh`, September 2026):

| mode      | launch to serving | reboot command to EP8 recovered | TPOT p50 before fault / on 2 survivors / after recovery |
|-----------|------------------:|--------------------------------:|:--------------------------------------------------------|
| `orig`    | 32 s              | 23 s                            | 99.4 / 95.0 / 97.4 ms                                   |
| `graph`   | 99 s              | 91 s                            | 19.4 / 28.5 / 19.4 ms                                   |
| `foundry` | 37 s              | 30 s                            | 19.4 / 28.7 / 19.3 ms                                   |

The foundry archives took about 9 minutes to build (fresh SAVE healthy in 136 s, plus a plain launch,
fault and recover-mode SAVE); `foundry_archive_eep_fresh` is 46 GB, `foundry_archive_eep_recover` 35 GB.

## Files

```
elastic_ep_demo/
├── demo.sh                    # one mode: (setup SAVEs,) launch, probe load, kill, reboot, summary
├── run_all.sh                 # all three modes back to back + comparison plots
├── start_daemon.sh            # (re)start a forkserver daemon (HOOK=1: with libcuda_hook.so preloaded)
├── cleanup_engines.sh         # stop every demo engine / probe client
├── make_probe_pool.py         # fixed prompt pools (ShareGPT or random token ids)
├── probe_gen.py               # periodic probe client: submits a pool, records TTFT/TPOT percentiles
├── tpot_summary.py            # per-phase table from probe_*.csv + events.txt
├── plot_decode_speed.py       # decode speed (or --metric tpot) vs time for several runs
├── foundry_save_fresh.toml    # SAVE  -> foundry_archive_eep_fresh   (fresh launch, all nodes)
├── foundry_load_fresh.toml    # LOAD  <- foundry_archive_eep_fresh
├── foundry_save_recover.toml  # SAVE  -> foundry_archive_eep_recover (recover-mode reboot, nodes 1-3)
└── foundry_load_recover.toml  # LOAD  <- foundry_archive_eep_recover
```

## Notes

- The demo pins the two per-node engines to GPUs `2k, 2k+1` via `CUDA_VISIBLE_DEVICES` and uses ports
  `21000 + 1000*k`; node 0 (port 21000) serves the probes and never dies.
- `cleanup_engines.sh` kills processes by command line (`sglang serve`, `sglang::`, the probe client
  and the daemons' children); run the demo on a host you do not share with other SGLang jobs.
- `SGLANG_ELASTIC_PROACTIVE_DEACTIVATE=world` makes the survivors deactivate the dead ranks as soon as
  one all-to-all times out, so the fault is handled in a few seconds instead of after the mooncake
  heartbeat timeout.
