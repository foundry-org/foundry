# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Single-graph SAVE / LOAD round trip (1 GPU, no vLLM/SGLang).

    pytest tests/core/test_save_load.py -q -s
    pytest tests/core/test_save_load.py -k compat -q -s     # torch-compat check of a wheel

One SAVE process per workload captures a graph, replays it twice and saves it;
each LOAD process (fresh, same archive) recreates the inputs at the SAVE
addresses, restores the graph and must replay it bit for bit like SAVE.

Workloads:
  add          elementwise add; the graph is captured in (and restored into) a
               second allocation region, separate from the inputs' region
  sgemm        fp32 128x64x32 matmul into a preallocated output, cuBLAS handle on
               the capture stream; saved without output tensors (LOAD returns a bare graph)
  hgemm_nvjet  as sgemm, fp16 192x4096x192 (the shape selects an nvjet kernel on Hopper)
  compat       what foundry.ops takes from libtorch: Philox capture state (rand,
               dropout advance per replay), cuBLAS / cuBLASLt handle and workspace
               (preallocate_cublas_workspaces), linear with bias (cuBLASLt epilogue)
Load APIs: ``load`` (CUDAGraph.load) and ``builds`` (start_graph_builds +
finish_graph_loads). compat also runs LOAD with FOUNDRY_MMAP_ARCHIVE=0.
add, sgemm and hgemm_nvjet then change an input in place and replay once
more: the graph reads its inputs, it does not bake them.
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from _helpers import assert_bit_exact, load_reference, prefix_region, run_phase, save_reference

BASE_ADDR = 0x3F0000000000
GRAPH_BASE_ADDR = 0x3F8000000000  # add: the graph's own region
REGION_SIZE_STR = "1GB"
SEED = 1234
GRAPH_JSON = "graph.json"


@dataclass(frozen=True)
class Workload:
    names: tuple  # one per observed output
    make_inputs: Callable  # (cpu generator) -> tuple of cuda tensors
    body: Callable  # (*inputs) -> list of observed outputs
    reference: Callable  # (*inputs) -> eager value per output (None: not checked)
    save_outputs: bool = True  # False: graph.save without outputs, LOAD observes the inputs
    cublas: str | None = None  # "handle" | "workspaces"
    graph_region: bool = False
    rng: tuple = ()  # names of outputs that advance on every replay
    mutate: bool = False  # LOAD: change the first input and replay once more


def _add_inputs(g):
    return torch.full((100, 100), 2.0, device="cuda"), torch.full((100, 100), 3.0, device="cuda")


def _add(a, b):
    return [a + b]


def _gemm_inputs(m, k, n, dtype):
    def make(g):
        a = torch.full((m, k), 2.0, device="cuda", dtype=dtype)
        b = torch.full((k, n), 3.0, device="cuda", dtype=dtype)
        return a, b, torch.zeros(m, n, device="cuda", dtype=dtype)

    return make


def _gemm(a, b, out):
    out.copy_(a @ b)
    return [out]


def _gemm_ref(a, b, out):
    return [a @ b]


def _compat_inputs(g):
    x = torch.randn(64, 128, generator=g).cuda()
    w1 = torch.randn(128, 256, generator=g).cuda()
    x16 = x.to(torch.bfloat16)
    w2 = torch.randn(256, 128, generator=g).to(torch.bfloat16).cuda()
    b2 = torch.randn(256, generator=g).to(torch.bfloat16).cuda()
    return x, w1, x16, w2, b2


def _compat(x, w1, x16, w2, b2):
    h = torch.relu(x @ w1) * 2.0 + 1.0  # cuBLAS gemm + elementwise
    y = torch.nn.functional.linear(x16, w2, b2)  # addmm with 1-D bias: cuBLASLt epilogue
    r = torch.rand_like(x)  # Philox, offset taken from the capture state
    d = torch.nn.functional.dropout(y, p=0.5, training=True)  # fused Philox dropout
    return [h, y, r, d]


def _compat_ref(x, w1, *_):
    return [torch.relu(x @ w1) * 2.0 + 1.0, None, None, None]


WORKLOADS = {
    "add": Workload(("sum",), _add_inputs, _add, _add, graph_region=True, mutate=True),
    "sgemm": Workload(
        ("out",),
        _gemm_inputs(128, 64, 32, torch.float32),
        _gemm,
        _gemm_ref,
        save_outputs=False,
        cublas="handle",
        mutate=True,
    ),
    "hgemm_nvjet": Workload(
        ("out",),
        _gemm_inputs(192, 4096, 192, torch.float16),
        _gemm,
        _gemm_ref,
        save_outputs=False,
        cublas="handle",
        mutate=True,
    ),
    "compat": Workload(
        ("gemm_relu", "lt_linear", "rand", "dropout"),
        _compat_inputs,
        _compat,
        _compat_ref,
        cublas="workspaces",
        rng=("rand", "dropout"),
    ),
}


def _prefix(w: Workload):
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator's
    small pool (where the per-capture seed/offset tensors land) must match."""
    import foundry as fdry

    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    inputs = w.make_inputs(g)
    with torch.cuda.stream(stream):
        if w.cublas == "handle":
            torch._C._cuda_getCurrentBlasHandle()
        elif w.cublas == "workspaces":
            fdry.preallocate_cublas_workspaces()  # cuBLAS + cuBLASLt handle and workspace
        w.body(*inputs)  # warm-up: lazy module loads, cuBLASLt heuristics
    torch.cuda.synchronize()
    if w.graph_region:
        fdry.set_allocation_region(GRAPH_BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    return inputs


def _replay_twice(graph, outs):
    replays = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        replays.append([t.cpu().clone() for t in outs])
    return replays


def _check_eager(w: Workload, outs, inputs):
    for name, got, want in zip(w.names, outs, w.reference(*inputs)):
        if want is not None:
            torch.testing.assert_close(got.cpu(), want.cpu(), msg=f"{name}: replay != eager")


def _run_save(archive, workload):
    import foundry as fdry

    w = WORKLOADS[workload]
    inputs = _prefix(w)
    torch.cuda.manual_seed(SEED)  # seed + philox offset 0, as a fresh LOAD state
    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        outs = w.body(*inputs)
    replays = _replay_twice(graph, outs)
    graph.save(os.path.join(archive, GRAPH_JSON), output_tensors=outs if w.save_outputs else None)

    for name, a, b in zip(w.names, *replays):
        if name in w.rng:
            assert not torch.equal(a, b), f"{name}: RNG did not advance between replays"
        else:
            assert torch.equal(a, b), f"{name}: deterministic output changed between replays"
    _check_eager(w, replays[0], inputs)
    save_reference(archive, replays, inputs)
    fdry.stop_allocation_region()


def _run_load(archive, workload, load_api):
    import foundry as fdry

    w = WORKLOADS[workload]
    fdry.load_cuda_modules_and_libraries("hook_archive")
    inputs = _prefix(w)
    want = load_reference(archive, inputs)
    path = os.path.join(archive, GRAPH_JSON)
    if load_api == "load":
        result = fdry.CUDAGraph.load(path)
        if w.save_outputs:
            graph, outs = result
        else:
            assert isinstance(result, fdry.ops.CUDAGraph), f"expected a bare graph: {result!r}"
            graph, outs = result, None
    else:
        pending = fdry.CUDAGraph.start_graph_builds([path])
        ((graph, outs),) = fdry.CUDAGraph.finish_graph_loads(pending)
    if w.save_outputs:
        outs = [outs] if isinstance(outs, torch.Tensor) else list(outs)
    else:
        assert not outs, f"saved without outputs, LOAD reconstructed {outs!r}"
        outs = [inputs[-1]]  # the preallocated output buffer
    assert_bit_exact(_replay_twice(graph, outs), want, f"{workload} replays")

    if w.mutate:
        inputs[0].fill_(1.0)
        graph.replay()
        torch.cuda.synchronize()
        _check_eager(w, outs, inputs)
    fdry.stop_allocation_region()


CASES = [
    pytest.param(w, api, mmap, id=f"{w}-{api}" + ("" if mmap else "-no_mmap"))
    for w in WORKLOADS
    for api in ("load", "builds")
    for mmap in ((True, False) if w == "compat" else (True,))
]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One SAVE per workload, shared by its LOAD cases."""
    archives = {}

    def get(workload):
        if workload not in archives:
            work = tmp_path_factory.mktemp(f"save_load_{workload}")
            run_phase(__file__, "save", work, workload)
            assert (work / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
            archives[workload] = work
        return archives[workload]

    return get


@pytest.mark.parametrize("workload, load_api, mmap", CASES)
def test_load_matches_save(saved, workload, load_api, mmap):
    env = {} if mmap else {"FOUNDRY_MMAP_ARCHIVE": "0"}
    run_phase(__file__, "load", saved(workload), workload, load_api, env=env)


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
