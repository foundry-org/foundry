# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Eager vs exec-share LOAD (1 GPU, no vLLM/SGLang).

Usage: pytest tests/test_exec_share.py -q -s

SAVE captures one elementwise chain (>128 kernel nodes) at several batch sizes:
same kernels and DAG, different grids, so one template plus members. Two LOAD
processes, eager (default) and share (FOUNDRY_ENABLE_EXEC_SHARE=1), replay the
sizes in alternation, which in share mode switches the template's exec on
every replay, and must match the SAVE outputs bit for bit. Share mode must log
its mode line, instantiate no member exec and start no exec-pool prewarm.
Per-replay times of a switch vs a repeat are printed, not asserted.
"""

import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

_spec = importlib.util.find_spec("foundry")
_HAS_OPS = _spec is not None and any(
    next(Path(p).glob("ops*.so"), None) for p in (_spec.submodule_search_locations or [])
)
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs 1 GPU"),
    pytest.mark.skipif(not _HAS_OPS, reason="foundry.ops not built/installed"),
    pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set"),
]

BASE_ADDR = 0x4F0000000000
REGION_SIZE_STR = "1GB"
HIDDEN = 1024
SIZES = [64, 128, 256, 512, 1024]  # rows; one graph each
DEPTH = 48  # 3 kernels per step: 144 kernel nodes per graph
# Every consecutive pair differs: each replay is a member switch in share mode.
ORDER = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4, 3, 2, 1, 0, 4, 0]
TIMING_ITERS = 50
SHARE_ENV = "FOUNDRY_ENABLE_EXEC_SHARE"
PHASE2_RE = re.compile(
    r"Phase 2: (\d+) templates \+ (\d+) on-demand = (\d+) graphs built in [\d.]+ ms "
    r"\(members: (\d+) execs instantiated"
)


def _body(x, n):
    y = x[:n]
    for _ in range(DEPTH):
        y = torch.sin(y)
        y = y * 0.5
        y = y + 0.25
    return y


def _prefix(fdry):
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator."""
    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(max(SIZES), HIDDEN, generator=g).cuda()
    stream = torch.cuda.Stream()
    fdry.graph.default_capture_stream = stream
    with torch.cuda.stream(stream):
        _body(x, SIZES[0])  # warm-up: lazy module loads
    torch.cuda.synchronize()
    return x


def _graph_paths(archive):
    # save_graph_manifest expects graph_<index>_<suffix>.json
    return [os.path.join(archive, f"graph_{i}_bs{n}.json") for i, n in enumerate(SIZES)]


def _run_save(archive):
    import foundry as fdry

    x = _prefix(fdry)
    graphs, outs = [], []
    for path, n in zip(_graph_paths(archive), SIZES):
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = _body(x, n)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, _body(x, n))
        graph.save(path, output_tensors=out)
        graphs.append(graph)
        outs.append(out.cpu().clone())
    fdry.save_graph_manifest(archive)
    torch.save({"outs": outs, "input_ptr": x.data_ptr()}, os.path.join(archive, "save.pt"))
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
    x = _prefix(fdry)
    ref = torch.load(os.path.join(archive, "save.pt"))
    assert x.data_ptr() == ref["input_ptr"], "VMM layout diverged from SAVE"

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
        assert torch.equal(got, ref["outs"][i]), f"{mode}: graph {i} (rows {SIZES[i]}) != SAVE"
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


def _spawn(cmd_mode, workdir, share=False):
    """SAVE/LOAD subprocess with the hook preloaded (cwd: ./hook_archive lives there).
    Returns the combined stdout/stderr."""
    from test_load_cublas_ws import _get_hook_so_path

    env = os.environ.copy()
    env.pop(SHARE_ENV, None)
    if share:
        env[SHARE_ENV] = "1"
    preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = f"{_get_hook_so_path()}:{preload}" if preload else _get_hook_so_path()
    env["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = "1"
    archive = workdir / "archive"
    archive.mkdir(exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{cmd_mode}", str(archive)]
    proc = subprocess.run(
        cmd, env=env, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(proc.stdout)
    assert proc.returncode == 0, f"{cmd_mode} (share={share}) failed"
    return proc.stdout


def _phase2(log):
    m = PHASE2_RE.search(log)
    assert m, "no Phase 2 summary line"
    templates, members, total, member_execs = map(int, m.groups())
    return templates, members, total, member_execs


def _stats(log):
    line = next(ln for ln in log.splitlines() if ln.startswith("EXEC_SHARE_STATS "))
    return json.loads(line.split(" ", 1)[1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    work = tmp_path_factory.mktemp("exec_share")
    _spawn("save", work)
    assert (work / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
    return work, _spawn("load", work), _spawn("load", work, share=True)


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


def test_outputs_identical_between_modes(runs):
    work, _, _ = runs
    eager = torch.load(work / "archive" / "load_eager.pt")
    share = torch.load(work / "archive" / "load_share.pt")
    assert len(eager) == len(share) == len(ORDER)
    for k, (a, b) in enumerate(zip(eager, share)):
        assert torch.equal(a, b), f"replay {k} (graph {ORDER[k]}): eager != share"


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
