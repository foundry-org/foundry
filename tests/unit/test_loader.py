# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""foundry._loader: the build-time torch check that guards the foundry.ops load.

pytest tests/unit/test_loader.py -q
"""

from types import SimpleNamespace

import foundry
import pytest
import torch
from foundry import _loader


def test_installed_torch_matches_the_build():
    assert foundry.__version__
    info = _loader._build_info()
    if info is None:
        pytest.skip("source tree without _build_info.py (never built)")
    print(
        f"\nfoundry {foundry.__version__} built against torch {info.TORCH_VERSION} "
        f"(CUDA {info.TORCH_CUDA_VERSION}); installed torch {torch.__version__} "
        f"(CUDA {torch.version.cuda})"
    )
    assert _loader._major_minor(info.TORCH_VERSION) == _loader._major_minor(torch.__version__)
    assert _loader._cuda_major(info.TORCH_CUDA_VERSION) == _loader._cuda_major(torch.version.cuda)


@pytest.mark.parametrize(
    "version, expected",
    [("2.13.0+cu130", (2, 13)), ("2.14.1", (2, 14)), ("", None), (None, None), ("dev", None)],
)
def test_major_minor(version, expected):
    assert _loader._major_minor(version) == expected


def _fake_build(monkeypatch, torch_version, cuda_version):
    info = SimpleNamespace(TORCH_VERSION=torch_version, TORCH_CUDA_VERSION=cuda_version)
    monkeypatch.setattr(_loader, "_build_info", lambda: info)
    monkeypatch.delenv(_loader.SKIP_ENV, raising=False)


def test_matching_build_passes(monkeypatch):
    _fake_build(monkeypatch, torch.__version__, torch.version.cuda or "")
    _loader.check_torch_build()


def test_other_torch_minor_is_refused_with_the_remedy(monkeypatch):
    _fake_build(monkeypatch, "1.0.0", "11.8")
    with pytest.raises(ImportError, match=r"built against torch 1\.0\.0") as exc:
        _loader.check_torch_build()
    assert f"{_loader.SKIP_ENV}=1" in str(exc.value)
    assert "torch==1.0.0" in str(exc.value)


def test_other_cuda_major_is_refused(monkeypatch):
    if torch.version.cuda is None:
        pytest.skip("CPU torch")
    other = str(_loader._cuda_major(torch.version.cuda) + 1) + ".0"
    _fake_build(monkeypatch, torch.__version__, other)
    with pytest.raises(ImportError, match=f"CUDA {other}"):
        _loader.check_torch_build()


def test_skip_env_disables_the_check(monkeypatch):
    _fake_build(monkeypatch, "1.0.0", "11.8")
    monkeypatch.setenv(_loader.SKIP_ENV, "1")
    _loader.check_torch_build()


def test_unbuilt_tree_is_not_checked(monkeypatch):
    monkeypatch.setattr(_loader, "_build_info", lambda: None)
    monkeypatch.delenv(_loader.SKIP_ENV, raising=False)
    _loader.check_torch_build()
