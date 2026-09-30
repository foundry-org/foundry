# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Restored kernel nodes of non-portable (16-block) cluster kernels must be addable.

Compiles tests/cluster_optin.cu against include/ClusterOptIn.h (the helper every LOAD
path uses to add kernel nodes) and runs it: compiled and launch-attribute 16-block
clusters are added, instantiated and launched from functions that never got the
capture process's NonPortableClusterSizeAllowed opt-in; portable clusters are untouched.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_non_portable_cluster_nodes(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None or not torch.cuda.is_available():
        pytest.skip("CUDA toolkit and GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("thread-block clusters require Hopper or newer")
    exe = tmp_path / "cluster_optin"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch=sm_{major}{minor}",
            "-I",
            str(ROOT / "include"),
            str(ROOT / "tests/cluster_optin.cu"),
            "-lcuda",
            "-o",
            str(exe),
        ],
        check=True,
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    print(result.stdout + result.stderr)
    if result.returncode == 3:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
