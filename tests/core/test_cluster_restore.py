# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Kernel nodes launched with a cluster-dimension attribute must be restored with it.

tests/core/csrc/cluster_probe.cu is built into a shared library. Its kernel writes
%is_explicit_cluster and %cluster_nctarank per block, and launch_cluster() launches it
with clusterDim (n,1,1), or without the attribute for n = 0. SAVE captures one graph per
workload (WORKLOADS: the n of each launch, in order). When every kernel node has the
same attrs, save() stores them once at the graph level (common_kernel_node_attrs).
The archive must record each node's cluster dims, and LOAD (``load`` and ``builds``
APIs, fresh process) must replay the same values.

    pytest tests/core/test_cluster_restore.py -q -s
"""

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _helpers import map_saved_layout, prefix_region, run_phase, save_layout

ROOT = Path(__file__).resolve().parents[2]
BASE_ADDR = 0x3F0000000000
GRAPH_JSON = "graph.json"
BLOCKS = 4
# clusterDim.x passed to launch_cluster -> (is_explicit_cluster, cluster_nctarank) per block
SEEN = {0: (0, 1), 2: (1, 2)}
WORKLOADS = {
    "mixed": [0, 2],
    "uniform": [2, 2],  # one attr set for every node: common_kernel_node_attrs
}
# Archived clusterDimX per kernel node (None: no cluster attr recorded)
ARCHIVED = {"mixed": [None, 2], "uniform": [2, 2]}
COMMON = {"uniform"}


def _expected(workload):
    return [v for c in WORKLOADS[workload] for _ in range(BLOCKS) for v in SEEN[c]]


def _launch_all(lib, out, workload):
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for i, cluster in enumerate(WORKLOADS[workload]):
        ptr = ctypes.c_void_p(out.data_ptr() + i * BLOCKS * 2 * out.element_size())
        assert lib.launch_cluster(ptr, BLOCKS, cluster, stream) == 0


def _run_save(archive, lib_path, workload):
    import foundry as fdry

    lib = ctypes.CDLL(lib_path)
    want = _expected(workload)
    _, stream = prefix_region(BASE_ADDR)
    out = torch.full((len(want),), -1, dtype=torch.int32, device="cuda")
    with torch.cuda.stream(stream):
        _launch_all(lib, out, workload)  # warm-up: module load
    torch.cuda.synchronize()
    assert out.tolist() == want, f"eager launches: {out.tolist()}"
    out.fill_(-1)
    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        _launch_all(lib, out, workload)
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == want, f"native replay: {out.tolist()}"
    graph.save(os.path.join(archive, GRAPH_JSON), output_tensors=[out])
    save_layout(archive)
    fdry.stop_allocation_region()


def _run_load(archive, workload, load_api):
    import foundry as fdry

    # The library is not loaded here: the kernel's module must come from hook_archive.
    fdry.load_cuda_modules_and_libraries("hook_archive")
    prefix_region(BASE_ADDR)
    want = _expected(workload)
    out = torch.full((len(want),), -1, dtype=torch.int32, device="cuda")  # SAVE's address
    path = os.path.join(archive, GRAPH_JSON)
    if load_api == "load":
        graph, outs = fdry.CUDAGraph.load(path)
    else:
        map_saved_layout(archive)
        ((graph, outs),) = fdry.CUDAGraph.finish_graph_loads(
            fdry.CUDAGraph.start_graph_builds([path])
        )
    outs = [outs] if isinstance(outs, torch.Tensor) else list(outs)
    assert outs[0].data_ptr() == out.data_ptr(), "output not restored at SAVE's address"
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == want, f"restored replay: {out.tolist()} (want {want})"
    fdry.stop_allocation_region()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Build the kernel library once and SAVE each workload once."""
    nvcc = shutil.which("nvcc")
    if nvcc is None or not torch.cuda.is_available():
        pytest.skip("CUDA toolkit and GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("thread-block clusters require Hopper or newer")
    lib = tmp_path_factory.mktemp("cluster_probe_lib") / "libcluster_probe.so"
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
            str(lib),
        ],
        check=True,
    )
    works = {}

    def get(workload):
        if workload not in works:
            work = tmp_path_factory.mktemp(f"cluster_restore_{workload}")
            run_phase(__file__, "save", work, lib, workload)
            works[workload] = work
        return works[workload]

    return get


@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_cluster_dims_archived(saved, workload):
    root = json.loads((saved(workload) / "archive" / GRAPH_JSON).read_text())
    common = root.get("common_kernel_node_attrs", {})
    assert ("clusterDimX" in common) == (workload in COMMON), sorted(common)
    # Kernel nodes are saved in the driver's node order, which is capture order here.
    dims = [
        {**common, **node["params"].get("kernel_node_attrs", {})}.get("clusterDimX")
        for node in root["nodes"]
        if node["type"] == "KernelNode"
    ]
    assert dims == ARCHIVED[workload], dims


@pytest.mark.parametrize("load_api", ["load", "builds"])
@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_cluster_dims_restored(saved, workload, load_api):
    run_phase(__file__, "load", saved(workload), workload, load_api)


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
