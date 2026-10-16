# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Tier markers by directory, device-count skips, and the shared fixtures.

    tests/unit         no marker      no GPU device; foundry.ops importable
    tests/core         gpu            1 GPU
    tests/multigpu     gpu multigpu   2 GPUs
    tests/sglang       sglang         sglang installed, no GPU
    tests/sglang/e2e   + gpu slow     engine launches (multi-GPU cases add multigpu / xlarge)

test_plugin_route.py and test_plugin_hook_points.py also get ``plugin_route``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_DIR_MARKS = {
    "core": ("gpu",),
    "multigpu": ("gpu", "multigpu"),
    "sglang": ("sglang",),
}
_E2E_MARKS = ("gpu", "slow")
_PLUGIN_ROUTE_FILES = {"test_plugin_route.py", "test_plugin_hook_points.py"}
_MIN_DEVICES = {"gpu": 1, "multigpu": 2, "xlarge": 4}


def _guard_sglang_namespace() -> None:
    """tests/ is on sys.path (pyproject ``pythonpath``; pytest also prepends
    this conftest's directory), so ``import sglang`` can resolve to tests/sglang/
    as a namespace package: when sglang is not installed, or when an editable
    sglang is served by a meta-path finder that runs after the path scan. Import
    the real package with tests/ off the path, or block the name so that
    ``pytest.importorskip("sglang")`` skips."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is not None:
        return
    saved = sys.path[:]
    sys.path[:] = [p for p in saved if Path(p or ".").resolve() != TESTS_DIR]
    try:
        importlib.invalidate_caches()
        try:
            importlib.import_module("sglang")
        except Exception:
            sys.modules.pop("sglang", None)
            sys.modules["sglang"] = None
    finally:
        sys.path[:] = saved


_guard_sglang_namespace()


def _tier_marks(path: Path) -> list[str]:
    try:
        parts = path.resolve().relative_to(TESTS_DIR).parts
    except ValueError:
        return []
    marks = list(_DIR_MARKS.get(parts[0], ())) if len(parts) > 1 else []
    if parts[:2] == ("sglang", "e2e"):
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
