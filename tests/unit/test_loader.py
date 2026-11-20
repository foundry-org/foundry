# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""foundry._loader: the torch check that guards the foundry.ops load (stable ABI:
any torch >= 2.14 with the build's CUDA major).

pytest tests/unit/test_loader.py -q
"""

from types import SimpleNamespace

import foundry
import pytest
import torch
from foundry import _loader


def test_installed_torch_is_supported_by_the_build():
    assert foundry.__version__
    info = _loader._build_info()
    if info is None:
        pytest.skip("source tree without _build_info.py (never built)")
    print(
        f"\nfoundry {foundry.__version__} built against torch {info.TORCH_VERSION} "
        f"(CUDA {info.TORCH_CUDA_VERSION}); installed torch {torch.__version__} "
        f"(CUDA {torch.version.cuda})"
    )
    assert _loader._major_minor(torch.__version__) >= _loader.MIN_TORCH
    assert _loader._cuda_major(info.TORCH_CUDA_VERSION) == _loader._cuda_major(torch.version.cuda)
    # The build records its stable-ABI target, which is the loader's lower bound.
    assert _loader._major_minor(info.STABLE_ABI_TARGET) == _loader.MIN_TORCH


def test_installed_metadata_does_not_pin_torch():
    """An installed foundry-core requires torch >= the stable-ABI target, never ==."""
    from importlib.metadata import PackageNotFoundError, requires

    try:
        reqs = requires("foundry-core") or []
    except PackageNotFoundError:
        pytest.skip("foundry-core is not installed (source tree on sys.path)")
    torch_reqs = [r for r in reqs if r.replace(" ", "").startswith("torch")]
    print("\nRequires-Dist:", reqs)
    assert torch_reqs == [f"torch>={_loader.MIN_TORCH[0]}.{_loader.MIN_TORCH[1]}"]


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


def test_other_build_torch_minor_passes(monkeypatch):
    _fake_build(monkeypatch, "2.14.1+cu130", torch.version.cuda or "")
    _loader.check_torch_build()


def test_torch_older_than_the_stable_target_is_refused(monkeypatch):
    _fake_build(monkeypatch, torch.__version__, torch.version.cuda or "")
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    with pytest.raises(ImportError, match=r"needs torch >= 2\.14") as exc:
        _loader.check_torch_build()
    assert f"{_loader.SKIP_ENV}=1" in str(exc.value)


def test_other_cuda_major_is_refused(monkeypatch):
    if torch.version.cuda is None:
        pytest.skip("CPU torch")
    other = str(_loader._cuda_major(torch.version.cuda) + 1) + ".0"
    _fake_build(monkeypatch, torch.__version__, other)
    with pytest.raises(ImportError, match=f"CUDA {other}"):
        _loader.check_torch_build()


def test_skip_env_disables_the_check(monkeypatch):
    _fake_build(monkeypatch, "1.0.0", "11.8")
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    monkeypatch.setenv(_loader.SKIP_ENV, "1")
    _loader.check_torch_build()


def test_unbuilt_tree_is_not_checked(monkeypatch):
    monkeypatch.setattr(_loader, "_build_info", lambda: None)
    monkeypatch.delenv(_loader.SKIP_ENV, raising=False)
    _loader.check_torch_build()
