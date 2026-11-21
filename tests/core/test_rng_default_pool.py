# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""torch's default memory pools after SAVE's captures == after LOAD (1 GPU).

    pytest tests/core/test_rng_default_pool.py -q -s

LOAD does not capture, so every caching-allocator block a SAVE capture leaves in
torch's default pools (pool (0,0) of any stream) is one LOAD does not have, and every
later default-pool tensor that a graph bakes in lands at another address on LOAD
(foundry PR #36: torch's capture RNG seed/offset tensors). SAVE routes them into a
Foundry pool, so a capture must leave the default pools untouched.

Each case: graph 0 (with or without RNG) is captured, then a small CUDA tensor is
allocated on the default stream, then graph 1 bakes it. SAVE records the default-pool
block layout (default stream and capture stream) after both captures; LOAD loads graph 0,
allocates the tensor at the same program point, loads graph 1 and must reproduce the
layout, the tensor's address and SAVE's replays bit for bit.
Variants: rng (dropout + rand in graph 0) or not; full = before the first capture the
small default-pool segments of the default and the capture stream are filled to 0 free
bytes (the #36 trigger: a pair allocated there would grow the pool inside the record window).
"""

import os
import sys

import pytest
import torch
from _helpers import assert_bit_exact, prefix_region, run_phase

BASE_ADDR = 0x3F0000000000
SEED = 4321
VARIANTS = {
    "norng": (False, False),
    "rng": (True, False),
    "norng_full": (False, True),
    "rng_full": (True, True),
}


def _body0(x, rng):
    y = x * 2.0 + 1.0
    if rng:
        return [torch.nn.functional.dropout(y, p=0.5, training=True), torch.rand_like(x)]
    return [y]


def _small_free(stream_handle):
    free = 0
    for seg in torch.cuda.memory_snapshot():
        if tuple(seg.get("segment_pool_id", (0, 0))) != (0, 0) or seg["stream"] != stream_handle:
            continue
        if seg["segment_type"] != "small":
            continue
        free += sum(b["size"] for b in seg["blocks"] if b["state"] == "inactive")
    return free


def _fill_small_pool(stream):
    """Take every free byte of the small default-pool segments of the default and the
    capture stream with 512-byte tensors (identical in SAVE and LOAD)."""
    hold = []
    for s in (torch.cuda.default_stream(), stream):
        with torch.cuda.stream(s):
            n = _small_free(s.cuda_stream) // 512
            hold += [torch.empty(64, dtype=torch.int64, device="cuda") for _ in range(n)]
            assert _small_free(s.cuda_stream) == 0, "could not fill the small pool"
    torch.cuda.synchronize()
    return hold


def _default_pool_layout(stream):
    """Every pool-(0,0) segment of the default and the capture stream with its blocks."""
    roles = {torch.cuda.default_stream().cuda_stream: "default", stream.cuda_stream: "capture"}
    out = []
    for seg in torch.cuda.memory_snapshot():
        if tuple(seg.get("segment_pool_id", (0, 0))) != (0, 0):
            continue
        role = roles.get(seg["stream"], f"other:{seg['stream']:#x}")
        blocks = [(b["address"], b["size"], b["state"]) for b in seg["blocks"]]
        out.append((role, seg["address"], seg["total_size"], blocks))
    return sorted(out)


def _prefix(variant):
    rng, full = VARIANTS[variant]
    g, stream = prefix_region(BASE_ADDR, "1GB")
    x = torch.randn(64, 128, generator=g).cuda()
    with torch.cuda.stream(stream):
        _body0(x, rng)  # warm-up on the capture stream
    torch.cuda.synchronize()
    hold = _fill_small_pool(stream) if full else []
    torch.cuda.manual_seed(SEED)
    return x, stream, hold, rng


def _scalar():
    return torch.full((1,), 3.0, device="cuda")  # default stream, small default pool


def _replays(graphs, outs, n=2):
    got = []
    for _ in range(n):
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        got.append([t.cpu().clone() for o in outs for t in o])
    return got


def _run_save(archive, variant):
    import foundry as fdry

    x, stream, hold, rng = _prefix(variant)
    g0 = fdry.CUDAGraph()
    with fdry.graph(g0):
        o0 = _body0(x, rng)
    s = _scalar()
    g1 = fdry.CUDAGraph()
    with fdry.graph(g1):
        o1 = [x * s + 1.0]
    layout = _default_pool_layout(stream)
    replays = _replays([g0, g1], [o0, o1])
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    g1.save(os.path.join(archive, "g1.json"), output_tensors=o1)
    torch.save(
        {"layout": layout, "scalar": s.data_ptr(), "replays": replays},
        os.path.join(archive, "ref.pt"),
    )
    fdry.stop_allocation_region()


def _load_one(fdry, path, api):
    if api == "load":
        g, outs = fdry.CUDAGraph.load(path)
    else:
        pending = fdry.CUDAGraph.start_graph_builds([path])
        ((g, outs),) = fdry.CUDAGraph.finish_graph_loads(pending)
    return g, [outs] if isinstance(outs, torch.Tensor) else list(outs)


def _run_load(archive, variant, api):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    x, stream, hold, rng = _prefix(variant)
    ref = torch.load(os.path.join(archive, "ref.pt"))
    g0, o0 = _load_one(fdry, os.path.join(archive, "g0.json"), api)
    s = _scalar()
    g1, o1 = _load_one(fdry, os.path.join(archive, "g1.json"), api)
    layout = _default_pool_layout(stream)
    if layout != ref["layout"]:
        want, got = set(map(repr, ref["layout"])), set(map(repr, layout))
        raise AssertionError(
            "torch default-pool layout after LOAD differs from SAVE's after capture:\n"
            f"  SAVE only: {sorted(want - got)}\n  LOAD only: {sorted(got - want)}"
        )
    assert s.data_ptr() == ref["scalar"], (
        f"default-pool tensor baked by graph 1 at {s.data_ptr():#x} on LOAD, {ref['scalar']:#x} on SAVE"
    )
    s.fill_(3.0)
    assert_bit_exact(_replays([g0, g1], [o0, o1]), ref["replays"], f"{variant} replays")
    fdry.stop_allocation_region()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    archives = {}

    def get(variant):
        if variant not in archives:
            work = tmp_path_factory.mktemp(f"rng_default_pool_{variant}")
            run_phase(__file__, "save", work, variant)
            archives[variant] = work
        return archives[variant]

    return get


@pytest.mark.parametrize("api", ["load", "builds"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_default_pool_unchanged_by_capture(saved, variant, api):
    run_phase(__file__, "load", saved(variant), variant, api)


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
