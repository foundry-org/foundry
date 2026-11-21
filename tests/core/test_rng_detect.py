# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Every captured graph that draws torch RNG is either archived with its RNG state (LOAD
replays bit for bit what torch's own replays drew at SAVE from the same seed) or refused
loudly at SAVE ("foundry capture: unsupported: ..."); never saved as RNG-free (1 GPU).
Each case expects one outcome (per torch version where torch's behaviour differs).

    pytest tests/core/test_rng_detect.py -q -s

SAVE seeds the default generator, captures, re-seeds and replays the graphs with torch
(the reference), then draws eagerly, saves. LOAD loads every graph, re-seeds, replays in
the same order and draws eagerly.
"""

import importlib
import os
import sys
import threading

import pytest
import torch
from _helpers import assert_bit_exact, prefix_region, run_phase

BASE_ADDR = 0x3F0000000000
SEED = 1234
REPLAYS = 3


# torch >= 2.16 pre-allocates a capture's seed/offset pairs in capture_begin (in Foundry's
# RNG pool), so another thread drawing from the default generator claims one of those.
_TORCH = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
PREALLOCATED_PAIRS = _TORCH >= (2, 16)
_KEEP = []  # default-pool int64[1] blocks a case frees inside its capture body


def _prefix():
    g, stream = prefix_region(BASE_ADDR, "1GB")
    # Two 8 B default-pool blocks (on SAVE and LOAD alike; s1_thread_freed frees them).
    _KEEP[:] = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(2)]
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
    """An earlier RNG graph is destroyed inside the capture body (its archived pair is held,
    so the new capture's state cannot take its address)."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    old = dict(g0._native_rng)

    def body():
        g0.reset()
        return [torch.rand_like(x)]

    g1, o1 = _capture(fdry, body, mode="relaxed")
    print(f"PAIRS old={old} new={g1._native_rng}", flush=True)
    return [("g1", g1, o1)], ["g1"]


def case_h1_live_reuse(fdry, x, stream, archive):
    """An earlier RNG-free graph (its pair not archived, alive with torch's graph) is
    destroyed inside the capture body; the new capture's state may take the freed address.
    Handled by the positive default-generator probe alone (the finaliser plays no part)."""
    g0, o0 = _capture(fdry, lambda: [x * 2.0])
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    old = g0._native_rng

    def body():
        g0.reset()
        return [torch.rand_like(x)]

    g1, o1 = _capture(fdry, body, mode="relaxed")
    print(f"PAIRS old={old} new={g1._native_rng}", flush=True)
    return [("g1", g1, o1)], ["g1"]


def case_freed_live_custom(fdry, x, stream, archive):
    """An earlier RNG-free graph is destroyed inside the capture body, then a non-default
    generator draws: its state can take the freed pair's addresses (live before the
    capture, so only the finaliser-tracked candidates find it)."""
    g0, o0 = _capture(fdry, lambda: [x * 2.0])
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    gen = torch.Generator(device="cuda")
    live = sorted(importlib.import_module("foundry.graph")._RNG_LIVE.get(0, ()))
    print(f"PAIRS live before the capture: {[hex(a) for a in live]}", flush=True)

    def body():
        g0.reset()
        return [torch.rand(x.shape, device="cuda", generator=gen)]

    g, o = _capture(fdry, body, mode="relaxed")
    return [("g", g, o)], ["g"]


def case_save_order(fdry, x, stream, archive):
    """An RNG-free graph (its window allocates) captured first, an RNG graph second, the
    second saved before the first."""
    g0, o0 = _capture(fdry, lambda: [x * 2.0])
    g1, o1 = _capture(fdry, _rng_body(x))
    g1.save(os.path.join(archive, "g1.json"), output_tensors=o1)
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    return [], ["g1"]


def case_unsaved_window(fdry, x, stream, archive):
    """A captured graph whose window allocated is never saved; a later graph is saved."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    _capture(fdry, lambda: [x * 3.0])  # new output: its window allocates; not saved
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


def case_unsaved_empty_window(fdry, x, stream, archive):
    """A captured graph whose window allocated nothing (in-place, into a pool with room) is
    never saved; a later graph is saved."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    ga = fdry.CUDAGraph()
    with fdry.graph(ga, pool=g0.pool()):
        x.mul_(1.0)
    g, o = _capture(fdry, _rng_body(x))
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


def _refused_capture(fdry, x):
    """A capture refused at SAVE (RNG from a non-default generator); its window allocated."""
    gen = torch.Generator(device="cuda")
    bad = fdry.CUDAGraph()
    try:
        with fdry.graph(bad):
            torch.rand(x.shape, device="cuda", generator=gen)
    except RuntimeError as e:
        print(f"first capture refused: {str(e)[:80]}", flush=True)
    return bad


def _raised_capture(fdry, x, draw=True):
    """A capture whose body raises (after an RNG op, which allocates, if draw)."""
    bad = fdry.CUDAGraph()
    try:
        with fdry.graph(bad):
            if draw:
                torch.rand_like(x)
            raise ValueError("body failed")
    except ValueError:
        pass
    return bad


def case_h1_unaccounted(fdry, x, stream, archive):
    """A refused capture, then a capture that frees its state inside the body."""
    bad = _refused_capture(fdry, x)

    def body():
        bad._native_graph.reset()
        return [torch.rand_like(x)]

    g, o = _capture(fdry, body, mode="relaxed")
    return [("g", g, o)], ["g"]


def case_rng_then_refused_then_rng(fdry, x, stream, archive):
    """A saved RNG graph, a refused capture, another RNG capture."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    _refused_capture(fdry, x)
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g0", "g"]


def case_raise_then_rng(fdry, x, stream, archive):
    """A capture that raises midway (after an RNG op, the first RNG state of the process),
    then a normal RNG capture."""
    _raised_capture(fdry, x)
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


def case_rng_then_raise_then_rng(fdry, x, stream, archive):
    """A saved RNG graph, a capture that raises after an RNG op, another RNG capture."""
    g0, o0 = _capture(fdry, _rng_body(x))
    g0.save(os.path.join(archive, "g0.json"), output_tensors=o0)
    _raised_capture(fdry, x)
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


def case_raise_before_alloc(fdry, x, stream, archive):
    """A capture whose body raises before anything is allocated, then an RNG capture."""
    _raised_capture(fdry, x, draw=False)
    g, o = _capture(fdry, _rng_body(x))
    return [("g", g, o)], ["g"]


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


def _on_worker(fn):
    done = []

    def worker():
        torch.cuda.set_device(0)
        fn()
        done.append(True)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if not done:
        raise RuntimeError("worker thread failed")


def case_h4_thread_memcpy(fdry, x, stream, archive):
    """Another thread draws from the default generator on the capturing stream and the
    graph reads the state through a copy (no kernel holds the address by value)."""

    def body():
        buf = torch.zeros(1, dtype=torch.int64, device="cuda")  # graph pool (this thread)

        def draw():
            with torch.cuda.stream(stream):
                _, off, _ = torch.cuda.default_generators[0].philox_state(4)
                buf.copy_(off)  # a copy node, no allocation on the worker

        _on_worker(draw)
        return [buf + 0]

    g, o = _capture(fdry, body, mode="relaxed")
    return [("g", g, o)], ["g"]


def case_s1_thread_freed(fdry, x, stream, archive):
    """The body frees two 8 B default-pool blocks, then another thread draws from the
    default generator on the capturing stream: its state can take the freed addresses."""

    def body():
        buf = torch.empty_like(x)  # graph pool (this thread)
        _KEEP.clear()

        def draw():
            with torch.cuda.stream(stream):
                buf.uniform_()

        _on_worker(draw)
        return [buf + 0]

    g, o = _capture(fdry, body, mode="relaxed")
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
        for n, g, o in saved:
            g.save(os.path.join(archive, f"{n}.json"), output_tensors=o)
    except RuntimeError as e:
        kind = "CaptureUnsupported" if isinstance(e, fdry.CaptureUnsupported) else type(e).__name__
        print(f"OUTCOME refused: {e} [{kind}]", flush=True)
        fdry.stop_allocation_region()
        return
    graphs = {n: (g, o) for n, g, o in saved}
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
        line = out[out.index("OUTCOME refused:") :].splitlines()[0]
        if "non-default stream" not in line:  # torch's own refusal
            assert "[CaptureUnsupported]" in line, line
            assert "foundry capture: unsupported:" in line or "foundry save: unsupported:" in line
        return "refused", line
    assert "OUTCOME saved" in out, out[-2000:]
    assert "OUTCOME pair_reused" not in out, out[-2000:]
    assert "OUTCOME load bitwise" in run_phase(__file__, "load", tmp_path, case)
    return "saved", ""


POISONED = "ended without being saved after its record window allocated"
THREAD = "not in Foundry's RNG pool" if not PREALLOCATED_PAIRS else None
EXPECTED = {
    "multi": ("saved", None),
    "h1_reuse": ("saved", None),
    "h1_live_reuse": ("saved", None),
    "c3_pair_reuse": ("saved", None),
    # 2.16 pre-allocates the pairs in capture_begin on the capturing stream: a segment of
    # the RNG pool for that stream (the reserved one is the default stream's), so even this
    # window allocated.
    "raise_before_alloc": ("refused", POISONED) if PREALLOCATED_PAIRS else ("saved", None),
    "eager_rng": ("saved", None),
    # A capture that failed after its window allocated: every later capture is refused.
    "h1_unaccounted": ("refused", POISONED),
    "rng_then_refused_then_rng": ("refused", POISONED),
    "raise_then_rng": ("refused", POISONED),
    "rng_then_raise_then_rng": ("refused", POISONED),
    "freed_live_custom": ("refused", "not the default CUDA generator's seed/offset pair"),
    "save_order": ("refused", "was not saved before this graph"),
    "unsaved_window": ("refused", "was not saved before this graph"),
    "unsaved_empty_window": ("saved", None),
    "h3_no_requested_size": ("refused", "has no 'requested_size' field"),
    "custom_generator": ("refused", "not the default CUDA generator's seed/offset pair"),
    "default_stream": ("refused", "non-default stream"),
    # 2.14/2.15: the other thread's state is created in its default pool, refused;
    # 2.16: it claims a pair capture_begin pre-allocated in the RNG pool, archived.
    "h4_thread_memcpy": ("refused", THREAD) if THREAD else ("saved", None),
    "s1_thread_freed": ("refused", THREAD) if THREAD else ("saved", None),
}


@pytest.mark.parametrize("case", list(EXPECTED))
def test_rng_case(tmp_path, case):
    want, message = EXPECTED[case]
    got, line = _save_then_load(tmp_path, case)
    assert got == want, f"{case}: {got} ({line}), expected {want}"
    if message is not None:
        assert message in line, line


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
