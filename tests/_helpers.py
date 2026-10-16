# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Shared helpers for the GPU tests (tests/core, tests/multigpu).

Each test file there is also the script its SAVE / LOAD child processes run
(``python <test file> --<mode> <archive>``); the children import this module
too (``run_phase`` puts tests/ on their PYTHONPATH). Keep it import-light:
no pytest, torch or foundry import at module top.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Foundry switches a child must not inherit from the calling shell; pass them
# through run_phase(env=...) to set them.
_STRIPPED_ENV = ("FOUNDRY_ENABLE_EXEC_SHARE", "FOUNDRY_MMAP_ARCHIVE", "FOUNDRY_MODE")
_TAIL_LINES = 80


def _skip(reason: str):
    import pytest

    pytest.skip(reason)


def hook_so_path() -> str:
    """libcuda_hook.so next to foundry.ops (the lookup the sglang integration
    uses); skips the test when the native build is missing."""
    try:
        from foundry.integration.sglang.config import CUDAGraphExtensionConfig
    except ImportError as exc:
        _skip(f"foundry not importable: {exc}")
    path = CUDAGraphExtensionConfig._detect_hook_so_path()
    if path is None:
        _skip("libcuda_hook.so not found next to foundry.ops")
    return path


def nvshmem_host_path() -> str | None:
    """libnvshmem_host.so of the installed nvidia-nvshmem wheel (the path the
    sglang integration preloads for DeepEP), or None."""
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig

    return CUDAGraphExtensionConfig._detect_nvshmem_host_path()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_phase(
    script,
    mode: str,
    workdir,
    *extra,
    env: dict | None = None,
    preload_nvshmem: bool = False,
    timeout: float = 900,
) -> str:
    """Run ``python script --<mode> <workdir>/archive *extra`` with the hook
    preloaded and ``cwd=workdir`` (the hook packs / reads ./hook_archive
    there). Returns the combined stdout/stderr; fails the test with its tail
    on a non-zero exit."""
    workdir = Path(workdir)
    archive = workdir / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    child_env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    child_env["TORCH_CUBLASLT_UNIFIED_WORKSPACE"] = "1"
    child_env.update(env or {})
    preload = [hook_so_path()]
    if preload_nvshmem:
        nvshmem = nvshmem_host_path()
        if nvshmem is None:
            _skip("libnvshmem_host.so not found (nvidia-nvshmem wheel not installed)")
        preload.insert(0, nvshmem)
    if child_env.get("LD_PRELOAD"):
        preload.append(child_env["LD_PRELOAD"])
    child_env["LD_PRELOAD"] = ":".join(preload)
    paths = [str(TESTS_DIR)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
    child_env["PYTHONPATH"] = os.pathsep.join(paths)

    cmd = [sys.executable, str(script), f"--{mode}", str(archive), *map(str, extra)]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    print(proc.stdout)
    tail = "\n".join(proc.stdout.splitlines()[-_TAIL_LINES:])
    assert proc.returncode == 0, f"{Path(script).name} --{mode} exited {proc.returncode}:\n{tail}"
    return proc.stdout


# ---------------------------------------------------------------------------
# Child-process side (SAVE / LOAD scripts)
# ---------------------------------------------------------------------------


def prefix_region(base: int, size: str = "1GB", seed: int = 0):
    """Identical in SAVE and LOAD: device 0, the allocation region, a seeded
    CPU generator for the inputs and a capture stream set as foundry's
    default. Returns (generator, stream)."""
    import foundry as fdry
    import torch

    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(base, fdry.parse_size(size))
    stream = torch.cuda.Stream()
    fdry.graph.default_capture_stream = stream
    return torch.Generator().manual_seed(seed), stream


def _cpu(x):
    if isinstance(x, (list, tuple)):
        return [_cpu(v) for v in x]
    return x.detach().cpu().clone()


def save_reference(archive, outputs, inputs, name: str = "reference.pt") -> None:
    """SAVE: store ``outputs`` (tensors or nested lists of them) and the input
    pointers for check_against_reference / load_reference."""
    import torch

    torch.save(
        {"outputs": _cpu(outputs), "input_ptrs": [t.data_ptr() for t in inputs]},
        os.path.join(archive, name),
    )


def load_reference(archive, inputs, name: str = "reference.pt"):
    """LOAD: assert the inputs sit where SAVE put them; return SAVE's outputs."""
    import torch

    ref = torch.load(os.path.join(archive, name))
    got = [t.data_ptr() for t in inputs]
    assert got == ref["input_ptrs"], f"VMM layout diverged from SAVE: {got} != {ref['input_ptrs']}"
    return ref["outputs"]


def assert_bit_exact(got, want, what: str = "output") -> None:
    import torch

    if isinstance(want, (list, tuple)):
        assert len(got) == len(want), f"{what}: {len(got)} values, SAVE had {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_bit_exact(g, w, f"{what}[{i}]")
        return
    assert torch.equal(_cpu(got), want), f"{what}: LOAD differs from SAVE"


def check_against_reference(archive, outputs, inputs, name: str = "reference.pt") -> None:
    assert_bit_exact(outputs, load_reference(archive, inputs, name))


LAYOUT_JSON = "layout.json"


def save_layout(archive: str) -> None:
    """SAVE, after capture: record the region cursor so a LOAD can back the span."""
    import foundry as fdry

    with open(os.path.join(archive, LAYOUT_JSON), "w") as f:
        json.dump({"final_alloc_offset": fdry.get_current_alloc_offset()}, f)


def map_saved_layout(archive: str) -> None:
    """LOAD, before start_graph_builds: map the span up to SAVE's final offset
    (what the sglang route's preallocate_for_load_mode does). The background
    template builds add memcpy nodes before finish_graph_loads replays the
    allocator events, and the driver rejects a not-yet-mapped source pointer
    (cuGraphAddMemcpyNode: invalid argument). CUDAGraph.load needs none of this."""
    import foundry as fdry

    with open(os.path.join(archive, LAYOUT_JSON)) as f:
        final = int(json.load(f)["final_alloc_offset"])
    current = fdry.get_current_alloc_offset()
    if final > current:
        assert fdry.ops.preallocate_ranges([(current, final - current)], final), (
            f"could not map the layout span {current:#x}..{final:#x}"
        )
