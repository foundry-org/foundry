# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Restored graphs must keep mixed default/programmatic (PDL) edge records.

Compiles tests/core/csrc/graph_dependencies.cu against include/GraphDependencies.h and
runs it twice: grouped insertion (what foundry does) must preserve every edge
record; the raw single-call variant only reports whether the installed driver
broadcasts edge_data[0] (libcuda 580-610 do), which is why the grouping exists.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reproducer(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if nvcc is None or not torch.cuda.is_available():
        pytest.skip("CUDA toolkit and GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("PDL edge data requires Hopper or newer")
    exe = tmp_path_factory.mktemp("graph_dependencies") / "graph_dependencies"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch=sm_{major}{minor}",
            "-I",
            str(ROOT / "include"),
            str(ROOT / "tests/core/csrc/graph_dependencies.cu"),
            "-lcuda",
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe


def test_grouped_insertion_preserves_edge_records(reproducer):
    result = subprocess.run([str(reproducer)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_report_driver_bulk_insertion_behavior(reproducer):
    """Informational: does this driver need the grouping? Never fails."""
    result = subprocess.run([str(reproducer), "--raw"], capture_output=True, text=True)
    verdict = "broadcasts edge_data[0]" if result.returncode else "honours per-edge data"
    print(
        f"\ndriver {torch.version.cuda}: raw cuGraphAddDependencies_v2 {verdict}\n{result.stdout}"
    )
