# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Load ``foundry.ops``, refusing a torch it was not built for.

Imported first by ``foundry/__init__.py``: sets ``__version__``, checks the
installed torch against the build-time one (``_build_info.py``, written by
setup.py), imports torch (so the extension's libc10/libtorch dependencies
resolve without an RPATH into the torch install) and loads ``foundry.ops``,
re-raising a load failure with the build-time torch/CUDA.

``foundry.ops`` links ATen/c10 internals, so like other torch C++ extensions
it only works with the torch (major.minor and CUDA major) it was compiled
against. Loading it under another torch fails with an undefined-symbol error
at best and undefined behavior at worst; this check turns that into a
readable message. ``FOUNDRY_SKIP_TORCH_CHECK=1`` skips it.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("foundry-core")
except PackageNotFoundError:  # source tree on sys.path, not installed
    __version__ = "0.0.0+unknown"

SKIP_ENV = "FOUNDRY_SKIP_TORCH_CHECK"


def _major_minor(version: str | None) -> tuple[int, ...] | None:
    if not version:
        return None
    m = re.match(r"(\d+)\.(\d+)", version)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _cuda_major(version: str | None) -> int | None:
    mm = _major_minor(version)
    return mm[0] if mm else None


def _build_info():
    try:
        from . import _build_info
    except ImportError:  # source tree that was never built
        return None
    if _build_info.TORCH_VERSION.startswith("@"):
        return None
    return _build_info


def _remedy(built_torch: str, built_cuda: str) -> str:
    cuda_tag = "cu" + built_cuda.replace(".", "") if built_cuda else "cpu"
    return (
        f"Install the matching torch (pip install 'torch=={built_torch.split('+')[0]}' "
        f"--index-url https://download.pytorch.org/whl/{cuda_tag}), install a foundry-core "
        "wheel built for your torch from the GitHub Release, or build from source against "
        "the installed torch: pip install --no-build-isolation --no-binary foundry-core "
        f"foundry-core. Set {SKIP_ENV}=1 to skip this check."
    )


def check_torch_build() -> None:
    """Raise ImportError if the installed torch differs from the build torch."""
    if os.environ.get(SKIP_ENV):
        return
    info = _build_info()
    if info is None:
        return
    import torch

    built_mm = _major_minor(info.TORCH_VERSION)
    have_mm = _major_minor(torch.__version__)
    built_cuda = _cuda_major(info.TORCH_CUDA_VERSION)
    have_cuda = _cuda_major(torch.version.cuda)
    if built_mm == have_mm and built_cuda == have_cuda:
        return
    raise ImportError(
        f"foundry.ops was built against torch {info.TORCH_VERSION} "
        f"(CUDA {info.TORCH_CUDA_VERSION or 'none'}) but torch {torch.__version__} "
        f"(CUDA {torch.version.cuda or 'none'}) is installed; the extension only loads "
        f"with the torch major.minor and CUDA major it was built for. "
        + _remedy(info.TORCH_VERSION, info.TORCH_CUDA_VERSION)
    )


def ops_import_error(exc: ImportError) -> ImportError:
    """Wrap a failed ``foundry.ops`` load with the build-time torch/CUDA."""
    info = _build_info()
    built = (
        f"torch {info.TORCH_VERSION}, CUDA {info.TORCH_CUDA_VERSION or 'none'}"
        if info is not None
        else "unknown build"
    )
    return ImportError(
        f"failed to load foundry.ops ({built}): {exc}. libcuda.so.1 (the NVIDIA driver) "
        "and the same torch build are required. "
        + (_remedy(info.TORCH_VERSION, info.TORCH_CUDA_VERSION) if info else "")
    )


def _load_ops() -> None:
    check_torch_build()
    import torch  # noqa: F401

    try:
        from . import ops  # noqa: F401
    except ImportError as exc:
        raise ops_import_error(exc) from exc


_load_ops()
