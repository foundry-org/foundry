# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Failed kernel-node attribute queries at SAVE are reported, not silently dropped.

tests/core/csrc/attr_query_shim.c is preloaded into the SAVE / LOAD children (after
libcuda_hook.so). It makes cuGraphKernelNodeGetAttribute return CUDA_ERROR_INVALID_VALUE
for the attribute ids in FOUNDRY_TEST_FAIL_ATTRS. The workload is the cluster probe of
test_cluster_restore.py: two graphs (4 and 8 blocks) of the same topology, each with a
launch without cluster attribute and one with clusterDim 2x1x1.

* attr 11 (PREFERRED_CLUSTER_DIMENSION), failing at SAVE and LOAD, as on driver
  580.105.08: SAVE succeeds, its summary line counts one failure per kernel node, the
  archive has no preferredClusterDim, and LOAD (eager and FOUNDRY_ENABLE_EXEC_SHARE=1)
  replays SAVE's values.
* attr 4 (CLUSTER_DIMENSION): SAVE raises. With FOUNDRY_ALLOW_ATTR_QUERY_FAILURE=1 it
  writes the nodes with attrQueryAvailable=false, LOAD warns, and the 2x1x1 launch is
  restored without a cluster (what the warning means).
* control (no failing ids): no summary line beyond attributes the driver itself rejects,
  no LOAD warning, everything restored.

    pytest tests/core/test_attr_query.py -q -s
"""

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _helpers import map_saved_layout, prefix_region, run_phase, save_layout

ROOT = Path(__file__).resolve().parents[2]
BASE_ADDR = 0x3F0000000000
SIZES = [4, 8]  # blocks per launch, one graph each
LAUNCHES = [0, 2]  # clusterDim.x per launch (0: no cluster attribute)
# clusterDim.x -> (is_explicit_cluster, cluster_nctarank) per block
SEEN = {0: (0, 1), 2: (1, 2)}
NO_CLUSTER = (0, 1)
ORDER = [0, 1, 0, 1]  # LOAD replay order
ATTR_CLUSTER_DIM = 4
ATTR_PREFERRED_CLUSTER_DIM = 11
# 580.126.20 rejects the PROGRAMMATIC_STREAM_SERIALIZATION (6) query on every kernel node.
DRIVER_REJECTED = {6}

SUMMARY = re.compile(
    r"\[foundry\] (\S+): attribute queries on (\d+) kernel node\(s\): (\d+) saved without "
    r"attribute data \(attrQueryAvailable=false\); failed queries, not recorded:(.*)"
)
WARNING = re.compile(
    r"\[foundry\] (\S+): (\d+) kernel node\(s\) saved without attribute data "
    r"\(attrQueryAvailable=false\): cluster dims and launch attributes were not restored"
)


def _paths(archive):
    return [os.path.join(archive, f"graph_{i}_bs{n}.json") for i, n in enumerate(SIZES)]


def _expected(blocks, restored_cluster=True):
    out = []
    for c in LAUNCHES:
        pair = SEEN[c] if restored_cluster or c == 0 else NO_CLUSTER
        out += list(pair) * blocks
    return out


def _alloc():
    return [
        torch.full((len(LAUNCHES) * n * 2,), -1, dtype=torch.int32, device="cuda") for n in SIZES
    ]


def _launch_all(lib, out, blocks):
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for i, cluster in enumerate(LAUNCHES):
        ptr = ctypes.c_void_p(out.data_ptr() + i * blocks * 2 * out.element_size())
        assert lib.launch_cluster(ptr, blocks, cluster, stream) == 0


def _run_save(archive, lib_path):
    import foundry as fdry

    lib = ctypes.CDLL(lib_path)
    _, stream = prefix_region(BASE_ADDR)
    outs = _alloc()
    with torch.cuda.stream(stream):
        for out, n in zip(outs, SIZES):
            _launch_all(lib, out, n)  # warm-up: module load
    torch.cuda.synchronize()
    for out, n, path in zip(outs, SIZES, _paths(archive)):
        out.fill_(-1)
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            _launch_all(lib, out, n)
        graph.replay()
        torch.cuda.synchronize()
        assert out.tolist() == _expected(n), f"native replay bs{n}: {out.tolist()}"
        graph.save(path, output_tensors=[out])
    fdry.save_graph_manifest(archive)
    save_layout(archive)
    fdry.stop_allocation_region()


def _run_load(archive, load_api, expect):
    import foundry as fdry

    restored_cluster = expect == "save"
    # The library is not loaded here: the kernel's module must come from hook_archive.
    fdry.load_cuda_modules_and_libraries("hook_archive")
    prefix_region(BASE_ADDR)
    outs = _alloc()  # SAVE's addresses
    paths = _paths(archive)
    if load_api == "load":
        # save_graph_manifest strips the member's dependencies: only the template loads alone.
        paths = paths[:1]
        loaded = [fdry.CUDAGraph.load(paths[0])]
    else:
        map_saved_layout(archive)
        loaded = fdry.CUDAGraph.finish_graph_loads(fdry.CUDAGraph.start_graph_builds(paths))
    for (_, got), out in zip(loaded, outs):
        got = [got] if isinstance(got, torch.Tensor) else list(got)
        assert got[0].data_ptr() == out.data_ptr(), "output not restored at SAVE's address"
    for i in (i for i in ORDER if i < len(loaded)):
        outs[i].fill_(-1)
        loaded[i][0].replay()
        torch.cuda.synchronize()
        want = _expected(SIZES[i], restored_cluster)
        assert outs[i].tolist() == want, f"graph {i} replay: {outs[i].tolist()} (want {want})"
        print(f"LOAD graph {i} (bs{SIZES[i]}): {expect} values ok")
    fdry.stop_allocation_region()


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    """(cluster probe library, attr query shim), built once."""
    nvcc, gcc = shutil.which("nvcc"), shutil.which("gcc")
    if nvcc is None or gcc is None or not torch.cuda.is_available():
        pytest.skip("CUDA toolkit, gcc and GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("thread-block clusters require Hopper or newer")
    out = tmp_path_factory.mktemp("attr_query_libs")
    probe, shim = out / "libcluster_probe.so", out / "libattr_query_shim.so"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch=sm_{major}{minor}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-cudart",
            "shared",
            str(ROOT / "tests/core/csrc/cluster_probe.cu"),
            "-o",
            str(probe),
        ],
        check=True,
    )
    subprocess.run(
        [
            gcc,
            "-O2",
            "-shared",
            "-fPIC",
            str(ROOT / "tests/core/csrc/attr_query_shim.c"),
            "-o",
            str(shim),
            "-ldl",
            "-lpthread",
        ],
        check=True,
    )
    return probe, shim


def _env(shim, fail="", **extra):
    return {"LD_PRELOAD": str(shim), "FOUNDRY_TEST_FAIL_ATTRS": fail, **extra}


def _summaries(log):
    return {Path(m.group(1)).name: m for m in SUMMARY.finditer(log)}


def _failures(match):
    """attr id -> count from a summary line's failed-query list."""
    return {int(a): int(n) for a, n in re.findall(r"attr (\d+) x(\d+)", match.group(4))}


def _kernel_attrs(path):
    root = json.loads(Path(path).read_text())
    common = root.get("common_kernel_node_attrs", {})
    return [
        {**common, **node["params"].get("kernel_node_attrs", {})}
        for node in root["nodes"]
        if node["type"] == "KernelNode"
    ]


def _save(tmp_path, libs, **env):
    probe, shim = libs
    log = run_phase(__file__, "save", tmp_path, probe, env=_env(shim, **env))
    assert "[attr_query_shim] active" in log, "shim not interposed"
    return log


def _load(tmp_path, libs, load_api, expect, fail="", share=False):
    _, shim = libs
    extra = {"FOUNDRY_ENABLE_EXEC_SHARE": "1"} if share else {}
    return run_phase(__file__, "load", tmp_path, load_api, expect, env=_env(shim, fail, **extra))


def test_control(tmp_path, libs):
    log = _save(tmp_path, libs)
    for name, m in _summaries(log).items():
        # Only attributes the driver itself rejects may show up; none drops a node's data.
        assert m.group(3) == "0", m.group(0)
        assert set(_failures(m)) <= DRIVER_REJECTED, m.group(0)
    for path in _paths(tmp_path / "archive"):
        attrs = _kernel_attrs(path)
        assert [a.get("clusterDimX") for a in attrs] == [None, 2], attrs
        assert all(a.get("attrQueryAvailable") is True for a in attrs), attrs
    for api, share in (("load", False), ("builds", False), ("builds", True)):
        log = _load(tmp_path, libs, api, "save", share=share)
        assert not WARNING.search(log), WARNING.search(log).group(0)


def test_preferred_cluster_dim_query_fails(tmp_path, libs):
    fail = str(ATTR_PREFERRED_CLUSTER_DIM)
    log = _save(tmp_path, libs, fail=fail)
    summaries = _summaries(log)
    assert sorted(summaries) == sorted(Path(p).name for p in _paths(tmp_path)), log
    for m in summaries.values():
        assert m.group(2) == str(len(LAUNCHES)) and m.group(3) == "0", m.group(0)
        assert _failures(m).get(ATTR_PREFERRED_CLUSTER_DIM) == len(LAUNCHES), m.group(0)
    for path in _paths(tmp_path / "archive"):
        attrs = _kernel_attrs(path)
        assert not any("preferredClusterDimX" in a for a in attrs), attrs
        assert [a.get("clusterDimX") for a in attrs] == [None, 2], attrs
    for share in (False, True):
        log = _load(tmp_path, libs, "builds", "save", fail=fail, share=share)
        assert not WARNING.search(log)
        if share:
            # link_on_demand_shared_exec queries attr 11 on the template's nodes.
            failed = re.search(r"\[attr_query_shim\] failed (\d+) queries", log)
            assert failed and int(failed.group(1)) > 0, log


def test_cluster_dim_query_fails(tmp_path, libs):
    fail = str(ATTR_CLUSTER_DIM)
    with pytest.raises(AssertionError) as exc:
        _save(tmp_path / "strict", libs, fail=fail)
    msg = str(exc.value)
    assert "CLUSTER_DIMENSION" in msg and "cluster dims" in msg, msg
    assert "FOUNDRY_ALLOW_ATTR_QUERY_FAILURE" in msg, msg

    work = tmp_path / "allowed"
    log = _save(work, libs, fail=fail, FOUNDRY_ALLOW_ATTR_QUERY_FAILURE="1")
    summaries = _summaries(log)
    assert sorted(summaries) == sorted(Path(p).name for p in _paths(work)), log
    for m in summaries.values():
        assert m.group(3) == str(len(LAUNCHES)), m.group(0)
        assert _failures(m) == {ATTR_CLUSTER_DIM: len(LAUNCHES)}, m.group(0)
    for path in _paths(work / "archive"):
        attrs = _kernel_attrs(path)
        assert [a.get("attrQueryAvailable") for a in attrs] == [False, False], attrs
        assert not any("clusterDimX" in a for a in attrs), attrs
    # LOAD on a driver that answers the query: the archive itself lacks the cluster dims.
    for api, share in (("load", False), ("builds", False), ("builds", True)):
        log = _load(work, libs, api, "nocluster", share=share)
        warned = {Path(m.group(1)).name: m.group(2) for m in WARNING.finditer(log)}
        loaded = _paths(work)[:1] if api == "load" else _paths(work)
        assert warned == {Path(p).name: str(len(LAUNCHES)) for p in loaded}, log


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
