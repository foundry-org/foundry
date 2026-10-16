# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""foundry.parse_size: the size strings of allocation_region and the TOML."""

import pytest
from foundry import parse_size

MB = 1 << 20


@pytest.mark.parametrize(
    "size, expected",
    [
        (1024, 1024),
        (50 * MB, 50 * MB),
        ("1024", 1024),
        ("512B", 512),
        ("1KB", 1024),
        ("1MB", MB),
        ("100MB", 100 * MB),
        ("512mb", 512 * MB),
        ("  256 MB  ", 256 * MB),
        ("1GB", 1 << 30),
        ("1.5GB", int(1.5 * (1 << 30))),
        ("2TB", 2 << 40),
    ],
)
def test_valid_sizes(size, expected):
    assert parse_size(size) == expected


@pytest.mark.parametrize("size", ["", "GB", "1 G", "1PB", "-1GB", "0x100", "1,5GB", "one MB"])
def test_invalid_sizes_raise(size):
    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size(size)
