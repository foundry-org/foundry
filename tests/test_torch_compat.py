# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Torch-compat check for an installed foundry-core wheel (1 GPU, no vLLM/SGLang).

Usage, in a venv with the wheel and one torch pair: pytest tests/test_torch_compat.py -q -s

Covers what foundry.ops takes from libtorch: the build-time torch check, the
generator capture state (per-capture Philox offsets of RNG ops inside a graph),
MemPool::graph_pool_handle + allocate-to-pool, and the cuBLAS / cuBLASLt handle
and workspace. A SAVE process captures matmul + linear-with-bias + elementwise +
rand + dropout and replays twice (RNG advances); two fresh LOAD processes
replay the archive twice and must match the SAVE replays bit for bit.
"""

import importlib.util
import os
import subprocess
import sys
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

BASE_ADDR = 0x3F0000000000
REGION_SIZE_STR = "1GB"
SEED = 1234
NAMES = ("gemm_relu", "lt_linear", "rand", "dropout")
RNG_NAMES = ("rand", "dropout")


def _body(x, w1, x16, w2, b2):
    h = torch.relu(x @ w1) * 2.0 + 1.0  # cuBLAS gemm + elementwise
    y = torch.nn.functional.linear(x16, w2, b2)  # addmm with 1-D bias: cuBLASLt epilogue
    r = torch.rand_like(x)  # Philox, offset taken from the capture state
    d = torch.nn.functional.dropout(y, p=0.5, training=True)  # fused Philox dropout
    return [h, y, r, d]


def _prefix(fdry):
    """Identical in SAVE and LOAD: the VMM cursor and the caching allocator's
    small pool (where the per-capture seed/offset tensors land) must match."""
    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 128, generator=g).cuda()
    w1 = torch.randn(128, 256, generator=g).cuda()
    x16 = x.to(torch.bfloat16)
    w2 = torch.randn(256, 128, generator=g).to(torch.bfloat16).cuda()
    b2 = torch.randn(256, generator=g).to(torch.bfloat16).cuda()
    inputs = (x, w1, x16, w2, b2)
    stream = torch.cuda.Stream()
    fdry.graph.default_capture_stream = stream
    with torch.cuda.stream(stream):
        fdry.preallocate_cublas_workspaces()  # cuBLAS + cuBLASLt handle and workspace
        _body(*inputs)  # warm-up: lazy module loads, cuBLASLt heuristics
    torch.cuda.synchronize()
    return inputs


def _replay_twice(graph, outs):
    replays = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        replays.append([t.cpu().clone() for t in outs])
    return replays


def _run_save(archive):
    import foundry as fdry

    inputs = _prefix(fdry)
    torch.cuda.manual_seed(SEED)  # seed + philox offset 0, as a fresh LOAD state
    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        outs = _body(*inputs)
    replays = _replay_twice(graph, outs)
    graph.save(os.path.join(archive, "graph.json"), output_tensors=outs)

    for name, a, b in zip(NAMES, *replays):
        if name in RNG_NAMES:
            assert not torch.equal(a, b), f"{name}: RNG did not advance between replays"
        else:
            assert torch.equal(a, b), f"{name}: deterministic output changed between replays"
    expected = torch.relu(inputs[0] @ inputs[1]) * 2.0 + 1.0
    torch.testing.assert_close(replays[0][0], expected.cpu())
    torch.save(
        {"replays": replays, "input_ptrs": [t.data_ptr() for t in inputs]},
        os.path.join(archive, "save_outputs.pt"),
    )
    fdry.stop_allocation_region()


def _run_load(archive):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    inputs = _prefix(fdry)
    ref = torch.load(os.path.join(archive, "save_outputs.pt"))
    assert [t.data_ptr() for t in inputs] == ref["input_ptrs"], "VMM layout diverged from SAVE"
    graph, outs = fdry.CUDAGraph.load(os.path.join(archive, "graph.json"))
    outs = [outs] if isinstance(outs, torch.Tensor) else list(outs)
    replays = _replay_twice(graph, outs)
    for i, (got, want) in enumerate(zip(replays, ref["replays"])):
        for name, a, b in zip(NAMES, got, want):
            assert torch.equal(a, b), f"replay {i} {name}: LOAD differs from SAVE"
    fdry.stop_allocation_region()


def _spawn(mode, workdir):
    """SAVE/LOAD subprocess with the hook preloaded, env as test_load_cublas_ws.
    cwd is the workdir: the hook packs/reads ./hook_archive."""
    from _helpers import hook_so_path

    env = os.environ.copy()
    preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = f"{hook_so_path()}:{preload}" if preload else hook_so_path()
    env["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = "1"
    archive = workdir / "archive"
    archive.mkdir(exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{mode}", str(archive)]
    subprocess.check_call(cmd, env=env, cwd=workdir)


def test_versions():
    import foundry
    from foundry import _loader

    info = _loader._build_info()
    built = f"{info.TORCH_VERSION} (CUDA {info.TORCH_CUDA_VERSION})" if info else "unknown"
    print(
        f"\nfoundry {foundry.__version__} built against torch {built}; installed torch "
        f"{torch.__version__} (CUDA {torch.version.cuda}), GPU {torch.cuda.get_device_name(0)}"
    )
    assert foundry.__version__
    if info is not None:
        assert _loader._major_minor(info.TORCH_VERSION) == _loader._major_minor(torch.__version__)


def test_save_load_rng_cublas(tmp_path):
    _spawn("save", tmp_path)
    assert (tmp_path / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
    _spawn("load", tmp_path)
    _spawn("load", tmp_path)  # the archive loads repeatedly


if __name__ == "__main__":
    {"--save": _run_save, "--load": _run_load}[sys.argv[1]](sys.argv[2])
