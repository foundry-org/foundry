# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Every captured graph that draws torch RNG is either archived with its RNG state (LOAD
replays bit for bit what torch's own replays drew at SAVE from the same seed) or refused
loudly at SAVE ("foundry capture: unsupported: ..."); never saved as RNG-free (1 GPU).

    pytest tests/core/test_rng_detect.py -q -s

SAVE seeds the default generator, captures, re-seeds and replays the graphs with torch
(the reference), then draws eagerly, saves. LOAD loads every graph, re-seeds, replays in
the same order and draws eagerly.
"""

import os
import sys
import threading

import pytest
import torch
from _helpers import assert_bit_exact, prefix_region, run_phase

BASE_ADDR = 0x3F0000000000
SEED = 1234
REPLAYS = 3


def _prefix():
    g, stream = prefix_region(BASE_ADDR, "1GB")
    x = torch.randn(64, 128, generator=g).cuda()
    with torch.cuda.stream(stream):  # warm-up on the capture stream (identical on LOAD)
        torch.nn.functional.dropout(x * 2.0, p=0.5, training=True)
        torch.rand_like(x)
    torch.cuda.synchronize()
    torch.cuda.manual_seed(SEED)
    return x, stream


def _capture(fdry, body, mode="global", stream=None):
    g = fdry.CUDAGraph()
    with fdry.graph(g, capture_error_mode=mode, stream=stream):
        out = body()
    return g, out


def _rng_body(x):
    return lambda: [torch.nn.functional.dropout(x, p=0.5, training=True), torch.rand_like(x)]


# Each case captures on SAVE and returns (graphs to save, names to replay); a graph to save
# is (name, graph, outputs). It may save graphs itself (h1_reuse saves g0 before g1).
def case_multi(fdry, x, stream, archive):
    """Two RNG graphs sharing the default generator, replayed alternately."""
    g0, o0 = _capture(fdry, lambda: [torch.rand_like(x)])
    g1, o1 = _capture(fdry, _rng_body(x))
    return [("g0", g0, o0), ("g1", g1, o1)], ["g0", "g1"]


def case_h1_reuse(fdry, x, stream, archive):
    """An earlier RNG graph is destroyed inside the capture body; the new capture's RNG
    state reuses its freed seed/offset blocks (same addresses as before the capture)."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    old = dict(g0._native_rng)

    def body():
        g0.reset()
        return [torch.rand_like(x)]

    g1, o1 = _capture(fdry, body, mode="relaxed")
    print(f"PAIRS old={old} new={g1._native_rng}", flush=True)
    return [("g1", g1, o1)], ["g1"]


def case_h1_unaccounted(fdry, x, stream, archive):
    """The state of a refused capture (not held by Foundry) is freed inside a later capture,
    whose RNG state can then take its address."""
    gen = torch.Generator(device="cuda")
    bad = fdry.CUDAGraph()
    try:
        with fdry.graph(bad):
            torch.rand(x.shape, device="cuda", generator=gen)
    except RuntimeError as e:
        print(f"first capture refused: {str(e)[:80]}", flush=True)

    def body():
        bad._native_graph.reset()
        return [torch.rand_like(x)]

    g, o = _capture(fdry, body, mode="relaxed")
    return [("g", g, o)], ["g"]


def case_c3_pair_reuse(fdry, x, stream, archive):
    """A saved RNG graph is reset on SAVE before the next RNG capture: the archived pairs
    of the two graphs must not alias (both stay live on LOAD)."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    old = (g0._native_rng["seed_ptr"], g0._native_rng["offset_ptr"])
    g0.reset()
    g1, o1 = _capture(fdry, _rng_body(x))
    new = (g1._native_rng["seed_ptr"], g1._native_rng["offset_ptr"])
    print(f"PAIRS old={old} new={new}", flush=True)
    if set(old) & set(new):
        print("OUTCOME pair_reused", flush=True)
    return [("g1", g1, o1)], ["g1"]


def _strip_requested_size():
    orig = torch.cuda.memory_snapshot

    def snap(*a, **k):
        segs = orig(*a, **k)
        for s in segs:
            for b in s["blocks"]:
                b.pop("requested_size", None)
        return segs

    torch.cuda.memory_snapshot = snap


def case_h3_no_requested_size(fdry, x, stream, archive):
    """memory_snapshot without requested_size (or a torch that changes the state's shape)."""
    _strip_requested_size()
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


def case_h4_thread_memcpy(fdry, x, stream, archive):
    """Another thread draws from the default generator on the capturing stream and the
    graph reads the state through a copy (no kernel holds the address by value)."""

    def body():
        buf = torch.zeros(1, dtype=torch.int64, device="cuda")  # graph pool (this thread)
        done = []

        def worker():
            torch.cuda.set_device(0)
            with torch.cuda.stream(stream):
                _, off, _ = torch.cuda.default_generators[0].philox_state(4)
                buf.copy_(off)  # a copy node, no allocation on the worker
                done.append(True)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        if not done:
            raise RuntimeError("worker thread failed")
        return [buf + 0]

    g, o = _capture(fdry, body, mode="relaxed")
    return [("g", g, o)], ["g"]


def case_raise_then_rng(fdry, x, stream, archive):
    """A capture that raises midway (after an RNG op, the first RNG state of the process),
    then a normal RNG capture."""
    bad = fdry.CUDAGraph()
    try:
        with fdry.graph(bad):
            torch.rand_like(x)
            raise ValueError("body failed")
    except ValueError:
        pass
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


def case_default_stream(fdry, x, stream, archive):
    """Capture on the default stream (torch refuses it)."""
    g, o = _capture(fdry, _rng_body(x), stream=torch.cuda.default_stream())
    return [("g", g, o)], ["g"]


def case_custom_generator(fdry, x, stream, archive):
    """RNG from a non-default torch.Generator(device='cuda')."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(5)
    g, o = _capture(fdry, lambda: [torch.rand(x.shape, device="cuda", generator=gen)])
    return [("g", g, o)], ["g"]


def case_eager_rng(fdry, x, stream, archive):
    """Eager RNG between seeding and the replays (shares torch's generator counter)."""
    g, o = _capture(fdry, lambda: [torch.rand_like(x)])
    return [("g", g, o)], ["g"]


EAGER_BEFORE = {"eager_rng"}
CASES = {n[len("case_") :]: f for n, f in list(globals().items()) if n.startswith("case_")}


def _replays(graphs, names, case):
    torch.cuda.manual_seed(SEED)
    if case in EAGER_BEFORE:
        torch.rand(16, device="cuda")
    got = []
    for _ in range(REPLAYS):
        for n in names:
            graphs[n][0].replay()
        torch.cuda.synchronize()
        got.append([t.cpu().clone() for n in names for t in graphs[n][1]])
    # Eager RNG after the replays: the replays advanced the default generator as torch's do.
    got.append([torch.rand(16, device="cuda").cpu()])
    return got


def _run_save(archive, case):
    import foundry as fdry

    x, stream = _prefix()
    try:
        saved, names = CASES[case](fdry, x, stream, archive)
    except RuntimeError as e:
        print(f"OUTCOME refused: {e}", flush=True)
        fdry.stop_allocation_region()
        return
    graphs = {n: (g, o) for n, g, o in saved}
    for n, g, o in saved:
        g.save(os.path.join(archive, f"{n}.json"), output_tensors=o)
    ref = _replays(graphs, names, case)
    torch.save({"names": names, "replays": ref}, os.path.join(archive, "ref.pt"))
    gens = {n: g._native_rng for n, g, _ in saved}
    print(f"OUTCOME saved rng={gens}", flush=True)
    fdry.stop_allocation_region()


def _run_load(archive, case):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    _prefix()
    ref = torch.load(os.path.join(archive, "ref.pt"))
    graphs = {}
    for n in ("g0", "g1", "g"):  # save order
        path = os.path.join(archive, f"{n}.json")
        if os.path.exists(path):
            g, out = fdry.CUDAGraph.load(path)
            graphs[n] = (g, [out] if isinstance(out, torch.Tensor) else list(out))
    assert_bit_exact(_replays(graphs, ref["names"], case), ref["replays"], f"{case} replays")
    print("OUTCOME load bitwise", flush=True)
    fdry.stop_allocation_region()


def _save_then_load(tmp_path, case):
    out = run_phase(__file__, "save", tmp_path, case)
    if "OUTCOME refused:" in out:
        assert "foundry capture: unsupported:" in out or "non-default stream" in out, out[-2000:]
        return "refused"
    assert "OUTCOME saved" in out, out[-2000:]
    assert "OUTCOME pair_reused" not in out, out[-2000:]
    assert "OUTCOME load bitwise" in run_phase(__file__, "load", tmp_path, case)
    return "saved"


@pytest.mark.parametrize(
    "case",
    ["multi", "h1_reuse", "h1_unaccounted", "c3_pair_reuse", "raise_then_rng", "h4_thread_memcpy"],
)
def test_rng_handled_or_refused(tmp_path, case):
    _save_then_load(tmp_path, case)


@pytest.mark.parametrize("case", ["h3_no_requested_size", "custom_generator", "default_stream"])
def test_rng_refused(tmp_path, case):
    assert _save_then_load(tmp_path, case) == "refused"


def test_rng_follows_eager_counter(tmp_path):
    """Loaded graphs take (seed, offset) from torch's default generator per replay, like
    torch's own replay: eager RNG before the replays shifts both alike."""
    _save_then_load(tmp_path, "eager_rng")


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
