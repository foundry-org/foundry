#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Per-phase summary of the sampled TPOT probes.

Usage: python3 tpot_summary.py <demo_dir>   (reads probe_*.csv + events.txt)
"""

import csv
import glob
import statistics as st
import sys

d = sys.argv[1]
rows = []
for f in glob.glob(f"{d}/probe_*.csv"):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                (
                    int(r["epoch"]),
                    int(r["n_ok"]),
                    int(r["n_fail"]),
                    float(r["tpot_p50_ms"]) if r["tpot_p50_ms"] else None,
                    float(r["ttft_p50_ms"]) if r["ttft_p50_ms"] else None,
                )
            )
with open(f"{d}/events.txt") as fh:
    ev = [
        (float(line.split(None, 1)[0]), line.split(None, 1)[1].strip())
        for line in fh
        if line.strip()
    ]
print(f"{'phase':62} {'batches':>7} {'failed reqs':>11} {'TPOT p50 ms':>11} {'TTFT p50 ms':>11}")
for (ta, la), (tb, lb) in zip(ev, ev[1:]):
    seg = [r for r in rows if ta <= r[0] < tb]
    ok = [r for r in seg if r[3] is not None]
    tp = st.median(r[3] for r in ok) if ok else float("nan")
    tt = st.median(r[4] for r in ok) if ok else float("nan")
    n_fail = sum(r[2] for r in seg)
    print(f"  {la:28} -> {lb:28} {len(seg):7d} {n_fail:11d} {tp:11.1f} {tt:11.0f}")
