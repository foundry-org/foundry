# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Sampled decode speed vs time for the fault-recovery demo.

Default metric: 1000/TPOT = decode tokens/s per request (higher is better); --metric tpot plots the
TPOT in ms instead (lower is better).
Usage: plot_decode_speed.py out.png [--period 10] [--metric tokps|tpot] <dir>:<label> [...]
<dir> holds probe_*.csv (one row per probe batch per client process) and events.txt. Per period slot
the TPOT p50 of all processes is combined (n_ok-weighted mean of per-process medians). Slots whose
batch produced no completed request are drawn as crosses at the top (engine was unavailable).
"""

import csv
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
out = args[0]
args = args[1:]
PERIOD = 10
METRIC = "tokps"
if args and args[0] == "--period":
    PERIOD = float(args[1])
    args = args[2:]
if args and args[0] == "--metric":
    METRIC = args[1]
    args = args[2:]


def load(d):
    slots = {}
    for f in glob.glob(f"{d}/probe_*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                s = int(int(r["epoch"]) // PERIOD)
                # n_ok, sum(n_ok*tpot50), n_fail, sum(n_ok*ttft50)
                e = slots.setdefault(s, [0, 0.0, 0, 0.0])
                n = int(r["n_ok"])
                e[2] += int(r["n_fail"])
                if n > 0 and r["tpot_p50_ms"]:
                    e[0] += n
                    e[1] += n * float(r["tpot_p50_ms"])
                    e[3] += n * float(r["ttft_p50_ms"])
    ev = {}
    with open(f"{d}/events.txt") as fh:
        for line in fh:
            if line.strip():
                t, lab = line.split(None, 1)
                ev.setdefault(lab.strip(), float(t))
    return slots, ev


fig, ax = plt.subplots(figsize=(12.5, 5.2))
colors = ["tab:blue", "tab:orange", "tab:green"]
shared = None
ymax = 0
for ci, spec in enumerate(args):
    d, label = spec.rsplit(":", 1)
    slots, ev = load(d)
    c = colors[ci % len(colors)]
    t0 = ev["launch EP8"]
    xs = []
    ys = []
    fx = []
    for s in sorted(slots):
        n, acc, nf, _ = slots[s]
        t = s * PERIOD + PERIOD / 2 - t0
        if t < 0:
            continue
        if n > 0:
            xs.append(t)
            ys.append(1000.0 / (acc / n) if METRIC == "tokps" else acc / n)
        elif nf > 0:
            fx.append(t)
    ymax = max(ymax, max(ys) if ys else 0)
    healthy = ev.get("healthy", 0) - t0
    rec = ev.get("EP8 recovered")
    rec = None if rec is None or rec >= ev.get("load end", 1e18) else rec - t0
    rec_s = "never" if rec is None else f"{rec:.0f} s"
    lbl = f"{label}  (serving at {healthy:.0f} s; recovered {rec_s})"
    ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=c, label=lbl)
    if fx:
        ax.plot(fx, [None] * len(fx), color=c)  # placeholder; failures drawn after ymax is known
    slots_fail = fx
    ax.plot([healthy], [0], marker="^", color=c, ms=11, clip_on=False, zorder=5)
    if rec is not None:
        ax.plot([rec], [0], marker="^", color=c, ms=11, clip_on=False, zorder=5)
    if shared is None:
        shared = (ev["kill"] - t0, ev["reboot nodes 1-3"] - t0)
    ax._fail = getattr(ax, "_fail", []) + [(c, slots_fail)]
top = ymax * 1.15
for c, fx in ax._fail:
    if fx:
        ax.plot(fx, [top] * len(fx), marker="x", ls="none", ms=7, color=c)
ax.set_ylim(0, ymax * 1.3)
for x in shared:
    ax.axvline(x, color="black", ls="--", lw=1.0)
ax.text(
    shared[0] - 2,
    ymax * 1.28,
    f"6 of 8 GPUs killed at {shared[0]:.0f} s\nreboot command at {shared[1]:.0f} s",
    fontsize=9,
    va="top",
    ha="right",
)
ax.set_xlabel(
    f"time since engine launch (s)   —   one probe batch of identical-length requests every "
    f"{PERIOD:.0f} s; x = batch got no output; triangles: first served token / rebooted ranks back"
)
ax.set_ylabel(
    "decode speed per request = 1000 / median TPOT of the probe batch (tok/s)"
    if METRIC == "tokps"
    else "TPOT of the probe batch, median (ms) — lower is better"
)
ax.set_title(
    "EP8 Qwen3-30B-A3B-FP8, 4 nodes x 2 GPUs: cold start, 3 nodes killed, all three rebooted"
)
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
