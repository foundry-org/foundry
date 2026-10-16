# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Tier markers by directory, device-count skips, and the shared fixtures.

    tests/unit         no marker      no GPU device; foundry.ops importable
    tests/core         gpu            1 GPU
    tests/multigpu     gpu multigpu   2 GPUs
    tests/sgl          sglang         sglang installed, no GPU (dir named sgl: tests/ is on sys.path,
                                      a tests/sglang/ would shadow the package)
    tests/sgl/e2e      + gpu slow     engine launches (multi-GPU cases add multigpu / xlarge)

test_plugin_route.py and test_plugin_hook_points.py also get ``plugin_route``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_DIR_MARKS = {
    "core": ("gpu",),
    "multigpu": ("gpu", "multigpu"),
    "sgl": ("sglang",),
}
_E2E_MARKS = ("gpu", "slow")
_PLUGIN_ROUTE_FILES = {"test_plugin_route.py", "test_plugin_hook_points.py"}
_MIN_DEVICES = {"gpu": 1, "multigpu": 2, "xlarge": 4}


def _tier_marks(path: Path) -> list[str]:
    try:
        parts = path.resolve().relative_to(TESTS_DIR).parts
    except ValueError:
        return []
    marks = list(_DIR_MARKS.get(parts[0], ())) if len(parts) > 1 else []
    if parts[:2] == ("sgl", "e2e"):
        marks += _E2E_MARKS
    if parts[-1] in _PLUGIN_ROUTE_FILES:
        marks.append("plugin_route")
    return marks


def pytest_collection_modifyitems(config, items):
    for item in items:
        for name in _tier_marks(Path(str(item.fspath))):
            item.add_marker(name)

    needs = [i for i in items if any(i.get_closest_marker(m) for m in _MIN_DEVICES)]
    if not needs:
        return
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    for item in needs:
        for mark, n in _MIN_DEVICES.items():
            if item.get_closest_marker(mark) and count < n:
                item.add_marker(
                    pytest.mark.skip(reason=f"needs {n} CUDA device(s), {count} visible")
                )
                break


@pytest.fixture
def workdir(tmp_path):
    """Per-test working directory of the SAVE / LOAD children (./hook_archive lands here)."""
    return tmp_path
