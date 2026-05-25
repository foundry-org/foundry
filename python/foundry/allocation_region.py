# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import re
from contextlib import contextmanager

from .ops import (
    free_preallocated_region,
    get_current_alloc_offset,
    preallocate_region,
    resume_allocation_region,
    set_allocation_region,
    set_current_alloc_offset,
    stop_allocation_region,
)

# Re-exported through foundry/__init__.py. Listed here so ruff's --fix doesn't
# strip the imports as unused.
__all__ = [
    "allocation_region",
    "free_preallocated_region",
    "get_current_alloc_offset",
    "parse_size",
    "preallocate_region",
    "resume_allocation_region",
    "set_allocation_region",
    "set_current_alloc_offset",
    "stop_allocation_region",
]


def parse_size(size: int | str) -> int:
    if isinstance(size, int):
        return size

    size_str = size.strip().upper()

    pattern = r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?$"
    match = re.match(pattern, size_str)

    if not match:
        raise ValueError(
            f"Invalid size format: {size}. "
            f"Expected format like '1GB', '16MB', '24KB', or integer bytes"
        )

    value = float(match.group(1))
    unit = match.group(2) or "B"

    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
        "TB": 1024 * 1024 * 1024 * 1024,
    }

    result = int(value * multipliers[unit])
    return result


# THIS CONTEXT MANAGER IS NOT RECOMMENDED
# NOTE: If you are using torch or other libraries that has internal memory pool,
# this context manager will not work as expected.
# See tests/test_stop_resume.py for how to guarantee start/stop of the allocation region.
@contextmanager
def allocation_region(base: int | str, size: int | str, prealloc_size: int | str | None = None):
    """
    Context manager to set up a VMM allocation region.

    Args:
        base: Base address for the allocation region (int or hex string).
        size: Total size of the allocation region.
        prealloc_size: Optional size to preallocate upfront for fast allocations.
                       If specified, memory is physically allocated upfront and
                       subsequent allocations use a fast path (pointer bump only).

    Example:
        >>> with allocation_region(0x500000000000, "16GB"):
        ...     tensor = torch.empty(1024, 1024, device="cuda")

        >>> with allocation_region(0x500000000000, "16GB", prealloc_size="8GB"):
        ...     # Allocations within 8GB use fast path (no VMM calls)
        ...     tensor = torch.empty(1024, 1024, device="cuda")
    """
    if isinstance(base, str):
        base_addr = int(base, 16) if base.lower().startswith("0x") else int(base)
    else:
        base_addr = base

    size_bytes = parse_size(size)
    has_prealloc = prealloc_size is not None

    set_allocation_region(base_addr, size_bytes)

    try:
        if has_prealloc:
            prealloc_bytes = parse_size(prealloc_size)
            success = preallocate_region(prealloc_bytes)
            if not success:
                raise RuntimeError(f"Failed to preallocate {prealloc_bytes} bytes")
        yield
    finally:
        if has_prealloc:
            free_preallocated_region()
        stop_allocation_region()
