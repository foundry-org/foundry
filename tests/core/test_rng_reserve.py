# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""foundry.reserve_rng_pool() and the capture-context edge cases around it (1 GPU).

pytest tests/core/test_rng_reserve.py -q -s
"""

import sys

import pytest
import torch
from _helpers import run_phase

BASE_ADDR = 0x3F0000000000


def _region():
    import foundry as fdry

    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size("1GB"))
    return fdry


def _generator_unchanged(archive):
    """The generator-contract check in reserve_rng_pool leaves the draws unchanged."""
    fdry = _region()
    torch.cuda.manual_seed(7)
    torch.rand(3, device="cuda")
    want = torch.rand(16, device="cuda").cpu()
    torch.cuda.manual_seed(7)
    torch.rand(3, device="cuda")
    fdry.reserve_rng_pool()
    got = torch.rand(16, device="cuda").cpu()
    assert torch.equal(got, want), "reserve_rng_pool changed the default generator's draws"
    fdry.reserve_rng_pool()  # second call: no-op
    print("OUTCOME ok", flush=True)


def _reserve_after_capture(archive):
    fdry = _region()
    x = torch.zeros(8, device="cuda")
    g = fdry.CUDAGraph()
    with fdry.graph(g):
        x.add_(1.0)
    try:
        fdry.reserve_rng_pool()
    except fdry.CaptureUnsupported as e:
        print(f"OUTCOME refused: {e}", flush=True)
        return
    print("OUTCOME reserved", flush=True)


def _stream_after_refused_begin(archive):
    """A capture_begin refused by Foundry leaves the caller's current stream in place."""
    fdry = _region()
    fdry.reserve_rng_pool()
    before = torch.cuda.current_stream()
    torch.cuda.get_allocator_backend = lambda: "cudaMallocAsync"
    g = fdry.CUDAGraph()
    try:
        with fdry.graph(g):
            pass
    except fdry.CaptureUnsupported as e:
        print(f"refused: {e}", flush=True)
    after = torch.cuda.current_stream()
    assert after == before, f"current stream {after} after the refusal, was {before}"
    print("OUTCOME ok", flush=True)


def _save_out_of_order(archive):
    """Two captures whose windows allocate nothing, saved in reverse capture order."""
    import os

    fdry = _region()
    fdry.reserve_rng_pool()
    x = torch.zeros(8, device="cuda")
    g0 = fdry.CUDAGraph()
    with fdry.graph(g0):
        y = x + 1.0
    g0.save(os.path.join(archive, "g0.json"), output_tensors=y)
    ga, gb = fdry.CUDAGraph(), fdry.CUDAGraph()
    with fdry.graph(ga, pool=g0.pool()):
        x.add_(1.0)
    with fdry.graph(gb, pool=g0.pool()):
        x.mul_(1.0)
    gb.save(os.path.join(archive, "gb.json"))
    try:
        ga.save(os.path.join(archive, "ga.json"))
    except fdry.CaptureUnsupported as e:
        print(f"OUTCOME refused: {e}", flush=True)
        return
    print("OUTCOME saved", flush=True)


CASES = {
    "generator_unchanged": _generator_unchanged,
    "reserve_after_capture": _reserve_after_capture,
    "stream_after_refused_begin": _stream_after_refused_begin,
    "save_out_of_order": _save_out_of_order,
}


@pytest.mark.parametrize(
    "case, want",
    [
        ("generator_unchanged", "OUTCOME ok"),
        (
            "reserve_after_capture",
            "OUTCOME refused: foundry capture: unsupported: reserve_rng_pool()",
        ),
        ("stream_after_refused_begin", "OUTCOME ok"),
        (
            "save_out_of_order",
            "OUTCOME refused: foundry save: unsupported: capture #1 saved after capture #2",
        ),
    ],
)
def test_reserve_and_edges(tmp_path, case, want):
    out = run_phase(__file__, "run", tmp_path, case)
    assert want in out, out[-2000:]


if __name__ == "__main__":
    mode, archive, case = sys.argv[1:]
    CASES[case](archive)
