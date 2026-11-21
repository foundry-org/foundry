# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SAVE routes the capturing thread's allocations on non-capturing streams into Foundry's
RNG pool (that is where torch's capture RNG state goes). Anything there that is not RNG
state is refused at SAVE; RNG state created on another thread is either refused at SAVE
or loads bit for bit (1 GPU).

    pytest tests/core/test_rng_pool_strays.py -q -s
"""

import os
import sys
import threading

import pytest
import torch
from _helpers import assert_bit_exact, prefix_region, run_phase

BASE_ADDR = 0x3F0000000000


def _prefix():
    g, stream = prefix_region(BASE_ADDR, "1GB")
    x = torch.randn(64, 128, generator=g).cuda()
    with torch.cuda.stream(stream):
        torch.nn.functional.dropout(x * 2.0, p=0.5, training=True)
    torch.cuda.synchronize()
    torch.cuda.manual_seed(77)
    return x, stream


def _body(case, x, stream):
    y = x * 2.0
    if case == "stray":  # a tensor on the default stream inside the captured body
        with torch.cuda.stream(torch.cuda.default_stream()):
            _body.keep = torch.empty(1000, device="cuda")
    elif case == "stray_int64":
        with torch.cuda.stream(torch.cuda.default_stream()):
            _body.keep = torch.zeros(1, dtype=torch.int64, device="cuda")
    elif case == "graph_pool_int64":  # an int64[1] body tensor in the graph pool, read by a kernel
        idx = torch.ones(1, dtype=torch.int64, device="cuda")
        return y.index_select(0, idx) + 1.0
    elif case == "rng_thread":  # RNG op launched onto the capturing stream from another thread
        out = {}

        def worker():
            torch.cuda.set_device(0)
            with torch.cuda.stream(stream):
                out["d"] = torch.nn.functional.dropout(y, p=0.5, training=True)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        if "d" not in out:
            raise RuntimeError("worker thread failed")
        return out["d"] + 1.0
    return y + 1.0


def _run_save(archive, case):
    import foundry as fdry

    x, stream = _prefix()
    g = fdry.CUDAGraph()
    try:
        with fdry.graph(g, capture_error_mode="relaxed"):
            out = _body(case, x, stream)
    except RuntimeError as e:
        print(f"OUTCOME refused: {e}", flush=True)
        fdry.stop_allocation_region()
        return
    reps = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        reps.append(out.cpu().clone())
    g.save(os.path.join(archive, "g.json"), output_tensors=out)
    torch.save(reps, os.path.join(archive, "ref.pt"))
    print(f"OUTCOME saved rng={g._native_rng}", flush=True)
    fdry.stop_allocation_region()


def _run_load(archive, case):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    _prefix()
    want = torch.load(os.path.join(archive, "ref.pt"))
    g, out = fdry.CUDAGraph.load(os.path.join(archive, "g.json"))
    out = out if isinstance(out, torch.Tensor) else out[0]
    got = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        got.append(out.cpu().clone())
    assert_bit_exact(got, want, f"{case} replays")
    print("OUTCOME load bitwise", flush=True)
    fdry.stop_allocation_region()


@pytest.mark.parametrize("case", ["stray", "stray_int64"])
def test_stray_allocation_refused(tmp_path, case):
    out = run_phase(__file__, "save", tmp_path, case)
    assert "OUTCOME refused: foundry capture: unsupported:" in out, out[-2000:]
    assert "non-capturing stream" in out


def test_graph_pool_int64_not_rng(tmp_path):
    """An int64[1] tensor of the body (graph pool) that a kernel reads is not RNG state."""
    out = run_phase(__file__, "save", tmp_path, "graph_pool_int64")
    assert "OUTCOME saved" in out, out[-2000:]
    assert "OUTCOME load bitwise" in run_phase(__file__, "load", tmp_path, "graph_pool_int64")


def test_rng_state_from_other_thread(tmp_path):
    """Refused at SAVE (2.14/2.15: the state is created on the worker thread, outside the
    routing scope) or, if saved (2.16: the worker claims capture_begin's pair, which is in
    the RNG pool), loads bit for bit."""
    out = run_phase(__file__, "save", tmp_path, "rng_thread")
    if "OUTCOME refused:" in out:
        assert "foundry capture: unsupported:" in out
        return
    assert "OUTCOME saved" in out, out[-2000:]
    assert "OUTCOME load bitwise" in run_phase(__file__, "load", tmp_path, "rng_thread")


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
