# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Eager vs exec-share LOAD (1 GPU, no vLLM/SGLang).

Usage: pytest tests/core/test_exec_share.py -q -s
With cudagraph-patch preloaded (the patched-route case; the real-route case then skips):
    eval "$(cudagraph-patch env)" && pytest tests/core/test_exec_share.py -q -s

SAVE captures one elementwise chain (>128 kernel nodes) at several batch sizes:
same kernels and DAG, different grids, so one template plus members. Two LOAD
processes, eager (default) and share (FOUNDRY_ENABLE_EXEC_SHARE=1), replay the
sizes in alternation, which in share mode switches the template's exec on
every replay, and must match the SAVE outputs bit for bit. Share mode must log
its mode line, instantiate no member exec and start no exec-pool prewarm.
Per-replay times of a switch vs a repeat are printed, not asserted. The route
line must say cudagraph-patch when cgp_exec_update resolves in this process
(LD_PRELOAD, inherited by the LOAD processes), the real update otherwise.

A second workload (needs Triton) launches one Triton kernel with two BLOCK
specializations: same function name and DAG, different binary. Share mode
must split the saved group into one template per binary.
"""

import ctypes
import json
import os
import re
import statistics
import sys
import time

import pytest
import torch
from _helpers import load_reference, prefix_region, run_phase, save_reference

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Weak cgp_* symbols bind in the LOAD processes only when the patch is preloaded,
# and they inherit this process's LD_PRELOAD.
_HAS_CGP = hasattr(ctypes.CDLL(None), "cgp_exec_update")
ROUTE_LINE = {
    True: "[foundry] exec share: patched route (cgp)",
    False: "[foundry] exec share: real cuGraphExecUpdate",
}

BASE_ADDR = 0x4F0000000000
REGION_SIZE_STR = "1GB"
HIDDEN = 1024
SIZES = [64, 128, 256, 512, 1024]  # rows; one graph each
DEPTH = 48  # 3 kernels per step: 144 kernel nodes per graph
# Every consecutive pair differs: each replay is a member switch in share mode.
ORDER = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4, 3, 2, 1, 0, 4, 0]
TIMING_ITERS = 50
SHARE_ENV = "FOUNDRY_ENABLE_EXEC_SHARE"
WORKLOAD_ENV = "FOUNDRY_TEST_EXEC_SHARE_WORKLOAD"
TRITON_DEPTH = 8
PHASE2_RE = re.compile(
    r"Phase 2: (\d+) templates \+ (\d+) on-demand = (\d+) graphs built in [\d.]+ ms "
    r"\(members: (\d+) execs instantiated"
)
REFINE_RE = re.compile(r"\[foundry\] exec share: (\d+) saved groups -> (\d+) templates")


def _body(x, n):
    y = x[:n]
    for _ in range(DEPTH):
        y = torch.sin(y)
        y = y * 0.5
        y = y + 0.25
    return y


if _HAS_TRITON:

    @triton.jit
    def _affine_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 0.5 + 0.25, mask=mask)


def _triton_block(n):
    """Rows 64 and 128 use one specialization, 256..1024 the other."""
    return 512 if n <= 128 else 1024


def _body_triton(x, n):
    y = x[:n]
    block = _triton_block(n)
    for _ in range(TRITON_DEPTH):
        out = torch.empty_like(y)
        _affine_kernel[(triton.cdiv(y.numel(), block),)](y, out, y.numel(), BLOCK=block)
        y = out
    return y


def _workload_body():
    return _body_triton if os.environ.get(WORKLOAD_ENV) == "triton" else _body


def _prefix():
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator."""
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.randn(max(SIZES), HIDDEN, generator=g).cuda()
    body = _workload_body()
    with torch.cuda.stream(stream):
        for n in SIZES if body is _body_triton else SIZES[:1]:
            body(x, n)  # warm-up: lazy module loads, Triton compiles
    torch.cuda.synchronize()
    return x


def _graph_paths(archive):
    # save_graph_manifest expects graph_<index>_<suffix>.json
    return [os.path.join(archive, f"graph_{i}_bs{n}.json") for i, n in enumerate(SIZES)]


def _run_save(archive):
    import foundry as fdry

    x = _prefix()
    body = _workload_body()
    graphs, outs = [], []
    for path, n in zip(_graph_paths(archive), SIZES):
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = body(x, n)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, body(x, n))
        graph.save(path, output_tensors=out)
        graphs.append(graph)
        outs.append(out.cpu().clone())
    fdry.save_graph_manifest(archive)
    save_reference(archive, outs, [x])
    fdry.stop_allocation_region()


def _replay_ms(graph, iters):
    """Per-replay GPU (event) and CPU (replay call) medians in us."""
    gpu, cpu = [], []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t = time.perf_counter()
        graph.replay()
        cpu.append((time.perf_counter() - t) * 1e6)
        end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end) * 1e3)
    return statistics.median(gpu), statistics.median(cpu)


def _run_load(archive):
    import foundry as fdry

    mode = "share" if os.environ.get(SHARE_ENV) == "1" else "eager"
    fdry.load_cuda_modules_and_libraries("hook_archive")
    x = _prefix()
    ref_outs = load_reference(archive, [x])

    paths = _graph_paths(archive)
    fdry.CUDAGraph.start_exec_pool_prewarm(paths)  # as SGLang LOAD does; no-op in share mode
    pending = fdry.CUDAGraph.start_graph_builds(paths)
    loaded = [fdry.CUDAGraph.finish_one_graph_load(pending, i) for i in range(len(paths))]
    graphs = [g for g, _ in loaded]
    outs = [o if isinstance(o, torch.Tensor) else o[0] for _, o in loaded]

    replays = []
    for i in ORDER:
        graphs[i].replay()
        torch.cuda.synchronize()
        got = outs[i].cpu().clone()
        assert torch.equal(got, ref_outs[i]), f"{mode}: graph {i} (rows {SIZES[i]}) != SAVE"
        replays.append(got)
    torch.save(replays, os.path.join(archive, f"load_{mode}.pt"))

    # Timing (printed only): a switch alternates two members, a repeat replays one.
    a, b = len(SIZES) - 2, len(SIZES) - 1
    sw = [_replay_ms(graphs[i], 1) for _ in range(TIMING_ITERS) for i in (a, b)]
    sw_gpu = statistics.median(t[0] for t in sw)
    sw_cpu = statistics.median(t[1] for t in sw)
    graphs[a].replay()
    rep_gpu, rep_cpu = _replay_ms(graphs[a], 2 * TIMING_ITERS)
    print(
        f"[exec-share-test] {mode}: switch replay gpu {sw_gpu:.1f} us, cpu {sw_cpu:.1f} us; "
        f"repeat replay gpu {rep_gpu:.1f} us, cpu {rep_cpu:.1f} us"
    )

    stats = {}
    for g in graphs:
        s = json.loads(g.exec_share_stats())
        stats.setdefault(str(s.get("exec")), s)  # one entry per shared exec
    print("EXEC_SHARE_STATS " + json.dumps({"mode": mode, "execs": list(stats.values())}))
    fdry.stop_allocation_region()


def _spawn(cmd_mode, workdir, share=False, workload="elementwise"):
    """SAVE/LOAD child; returns its combined stdout/stderr."""
    env = {WORKLOAD_ENV: workload}
    if share:
        env[SHARE_ENV] = "1"
    return run_phase(__file__, cmd_mode, workdir, env=env)


def _phase2(log):
    m = PHASE2_RE.search(log)
    assert m, "no Phase 2 summary line"
    templates, members, total, member_execs = map(int, m.groups())
    return templates, members, total, member_execs


def _stats(log):
    line = next(ln for ln in log.splitlines() if ln.startswith("EXEC_SHARE_STATS "))
    return json.loads(line.split(" ", 1)[1])


@pytest.fixture(scope="module", params=["elementwise", "triton"])
def runs(request, tmp_path_factory):
    workload = request.param
    if workload == "triton" and not _HAS_TRITON:
        pytest.skip("triton not installed")
    work = tmp_path_factory.mktemp(f"exec_share_{workload}")
    _spawn("save", work, workload=workload)
    assert (work / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
    return (
        work,
        _spawn("load", work, workload=workload),
        _spawn("load", work, share=True, workload=workload),
    )


def test_eager_instantiates_every_member(runs):
    _, eager, _ = runs
    templates, members, total, member_execs = _phase2(eager)
    assert total == len(SIZES)
    assert members > 0, "SAVE produced no template members; nothing to share"
    assert member_execs == members
    assert "[foundry] exec share:" not in eager
    assert all(not s["enabled"] for s in _stats(eager)["execs"])


def test_share_instantiates_templates_only(runs):
    _, _, share = runs
    templates, members, total, member_execs = _phase2(share)
    assert total == len(SIZES)
    assert members > 0, "share mode refined every group to one graph; no switch tested"
    assert member_execs == 0
    assert "[foundry] exec share:" in share
    assert not [ln for ln in share.splitlines() if ln.startswith("[foundry] exec pool prewarm")]
    execs = _stats(share)["execs"]
    assert all(s["enabled"] and s["valid"] for s in execs)
    assert sum(s["updates"] for s in execs) > 0, "no member switch happened"


def test_share_splits_kernel_binaries(runs):
    work, eager, share = runs
    if work.name.startswith("exec_share_elementwise"):
        pytest.skip("Triton workload only")
    assert _phase2(eager)[0] == 1, "eager keeps the saved group"
    m = REFINE_RE.search(share)
    assert m, "no share-mode refinement line"
    saved, templates = map(int, m.groups())
    # Same names and DAG, two binaries of _affine_kernel: one template per binary.
    assert (saved, templates) == (1, 2)
    assert _phase2(share)[:2] == (2, len(SIZES) - 2)


@pytest.mark.skipif(_HAS_CGP, reason="cudagraph-patch preloaded")
def test_share_route_real(runs):
    _, _, share = runs
    assert ROUTE_LINE[False] in share
    assert all(s["route"] == "cuGraphExecUpdate" for s in _stats(share)["execs"])


@pytest.mark.skipif(
    not _HAS_CGP, reason="cudagraph-patch not preloaded (cgp_exec_update unresolved)"
)
def test_share_route_patched(runs):
    _, _, share = runs
    assert ROUTE_LINE[True] in share
    assert all(s["route"] == "cgp" for s in _stats(share)["execs"])


def test_outputs_identical_between_modes(runs):
    work, _, _ = runs
    eager = torch.load(work / "archive" / "load_eager.pt")
    share = torch.load(work / "archive" / "load_share.pt")
    assert len(eager) == len(share) == len(ORDER)
    for k, (a, b) in enumerate(zip(eager, share)):
        assert torch.equal(a, b), f"replay {k} (graph {ORDER[k]}): eager != share"


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
