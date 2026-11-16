# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Load ``foundry.ops``, refusing a torch it cannot run with.

Imported first by ``foundry/__init__.py``: sets ``__version__``, checks the
installed torch (``_build_info.py``, written by setup.py), imports torch (so the
extension's libtorch_cpu/libtorch_cuda dependencies resolve to the installed
torch without an RPATH into it) and loads ``foundry.ops``, re-raising a load
failure with the build-time torch/CUDA.

``foundry.ops`` uses only the LibTorch Stable ABI (C shims, target torch 2.14),
so one build loads on any torch >= 2.14 with the same CUDA major (its CUDA
runtime). ``FOUNDRY_SKIP_TORCH_CHECK=1`` skips the check.
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
# Stable-ABI target of foundry.ops (setup.py TORCH_TARGET_VERSION).
MIN_TORCH = (2, 14)


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


def check_torch_build() -> None:
    """Raise ImportError if the installed torch is older than the stable-ABI
    target or uses another CUDA major than the build."""
    if os.environ.get(SKIP_ENV):
        return
    import torch

    have_mm = _major_minor(torch.__version__)
    if have_mm is not None and have_mm < MIN_TORCH:
        raise ImportError(
            f"foundry.ops needs torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} (LibTorch Stable ABI "
            f"target), but torch {torch.__version__} is installed. Upgrade torch. "
            f"Set {SKIP_ENV}=1 to skip this check."
        )
    info = _build_info()
    if info is None:
        return
    built_cuda = _cuda_major(info.TORCH_CUDA_VERSION)
    have_cuda = _cuda_major(torch.version.cuda)
    if built_cuda == have_cuda:
        return
    raise ImportError(
        f"foundry.ops was built for CUDA {info.TORCH_CUDA_VERSION or 'none'} but torch "
        f"{torch.__version__} uses CUDA {torch.version.cuda or 'none'}; the extension needs "
        f"the same CUDA major. Rebuild foundry against this torch, or set {SKIP_ENV}=1."
    )


def ops_import_error(exc: ImportError) -> ImportError:
    """Wrap a failed ``foundry.ops`` load with the build-time torch/CUDA."""
    info = _build_info()
    built = (
        f"built with torch {info.TORCH_VERSION}, CUDA {info.TORCH_CUDA_VERSION or 'none'}"
        if info is not None
        else "unknown build"
    )
    return ImportError(
        f"failed to load foundry.ops ({built}): {exc}. libcuda.so.1 (the NVIDIA driver) "
        f"and torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} are required."
    )


def _load_ops() -> None:
    check_torch_build()
    import torch  # noqa: F401

    try:
        from . import ops  # noqa: F401
    except ImportError as exc:
        raise ops_import_error(exc) from exc


_load_ops()
