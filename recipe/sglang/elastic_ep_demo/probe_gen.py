#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Sampled latency probe for the fault-recovery demo.

Every --period seconds submit a batch of --batch identical-length requests (fixed --in-len input
tokens, fixed --out-len output tokens, EOS ignored) and record the batch's TTFT and TPOT
percentiles. The probe batches ARE the load; between batches the engine idles. TPOT is measured per
request from the first token to the last, so it is independent of client throughput limits.

Usage: probe_gen.py <out.csv> <duration_s> [--period 10] [--batch 64] [--in-len 128] [--out-len 128]
CSV row per batch: epoch,n_ok,n_fail,ttft_p50_ms,tpot_p50_ms,tpot_p90_ms,tpot_max_ms,batch_wall_s
Run several processes for large batches (each process's batch is a share); the plot merges them.
"""

import argparse
import asyncio
import json
import random
import time
import uuid

import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("duration", type=float)
ap.add_argument("--period", type=float, default=10.0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--in-len", type=int, default=128)
ap.add_argument("--out-len", type=int, default=128)
ap.add_argument("--port", type=int, default=21000)
ap.add_argument("--vocab", type=int, default=150000)
ap.add_argument(
    "--stall-timeout",
    type=float,
    default=45.0,
    help="abort a request that produced no token for this long",
)
ap.add_argument("--req-timeout", type=float, default=120.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument(
    "--pool-file",
    default=None,
    help="JSON list of token-id prompts (from make_probe_pool.py); the same prompts are replayed "
    "every period so all demo stages measure identical requests",
)
ap.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="0 = greedy, so the decode path is also identical across periods",
)
ap.add_argument(
    "--max-outstanding",
    type=int,
    default=2,
    help="skip a period when this many probe batches are still in flight (bounds the load during "
    "stalls: with 2 survivors an unbounded backlog exceeded the per-rank dispatch buffer and "
    "crashed the engine)",
)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/generate"
ABORT_URL = f"http://127.0.0.1:{a.port}/abort_request"
rng = random.Random(a.seed)
POOL = None
if a.pool_file:
    with open(a.pool_file) as pf:
        POOL = json.load(pf)
if POOL is not None:
    assert len(POOL) >= a.batch and all(len(x) == a.in_len for x in POOL[: a.batch]), (
        "pool does not match --batch/--in-len"
    )
    POOL = POOL[: a.batch]


async def one(session, ids):
    rid = uuid.uuid4().hex
    body = {
        "input_ids": ids,
        "rid": rid,
        "sampling_params": {
            "max_new_tokens": a.out_len,
            "temperature": a.temperature,
            "ignore_eos": True,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    n = 0
    try:
        async with session.post(
            URL,
            json=body,
            timeout=aiohttp.ClientTimeout(total=a.req_timeout, sock_read=a.stall_timeout),
        ) as r:
            if r.status != 200:
                return None
            try:
                async for raw in r.content:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        m = json.loads(payload)["meta_info"]["completion_tokens"]
                    except Exception:
                        continue
                    if m > n:
                        if t_first is None:
                            t_first = time.perf_counter()
                        n = m
            except Exception:
                try:
                    async with session.post(
                        ABORT_URL, json={"rid": rid}, timeout=aiohttp.ClientTimeout(total=5)
                    ) as ar:
                        await ar.read()
                except Exception:
                    pass
                return None
    except Exception:
        return None
    if t_first is None or n < 2:
        return None
    t_end = time.perf_counter()
    return (1000 * (t_first - t0), 1000 * (t_end - t_first) / (n - 1))  # ttft_ms, tpot_ms


async def batch(session, f):
    t0 = time.time()
    ids = (
        POOL
        if POOL is not None
        else [[rng.randrange(1000, a.vocab) for _ in range(a.in_len)] for _ in range(a.batch)]
    )  # fixed pool, or identical length with fresh content
    res = await asyncio.gather(*[one(session, x) for x in ids])
    ok = [r for r in res if r]
    if ok:
        ttft = sorted(r[0] for r in ok)
        tpot = sorted(r[1] for r in ok)

        def q(xs, p):
            return xs[min(len(xs) - 1, int(p * len(xs)))]

        row = (
            f"{int(t0)},{len(ok)},{len(res) - len(ok)},{q(ttft, 0.5):.1f},{q(tpot, 0.5):.2f},"
            f"{q(tpot, 0.9):.2f},{tpot[-1]:.2f},{time.time() - t0:.1f}"
        )
    else:
        row = f"{int(t0)},0,{len(res)},,,,,{time.time() - t0:.1f}"
    f.write(row + "\n")
    f.flush()


async def main():
    stop = time.time() + a.duration
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        with open(a.out, "w") as f:
            f.write(
                "epoch,n_ok,n_fail,ttft_p50_ms,tpot_p50_ms,tpot_p90_ms,tpot_max_ms,batch_wall_s\n"
            )
            f.flush()
            tasks = set()
            next_t = time.time()
            while time.time() < stop:
                now = time.time()
                if now >= next_t:
                    tasks = {t for t in tasks if not t.done()}
                    if len(tasks) < a.max_outstanding:
                        tasks.add(asyncio.create_task(batch(session, f)))
                    else:
                        # skipped: engine still busy with earlier probe batches
                        f.write(f"{int(now)},0,0,,,,,0.0\n")
                        f.flush()
                    next_t += a.period
                await asyncio.sleep(0.05)
            if tasks:
                await asyncio.wait(tasks, timeout=a.req_timeout + 5)


asyncio.run(main())
