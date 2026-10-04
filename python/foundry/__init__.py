# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
# The ops wildcard must come FIRST: .graph's CUDAGraph (and the
# allocation_region helpers) intentionally override the raw pybind names.
# `# isort: off` PINS this order — ruff's isort (lint rule "I") otherwise
# auto-sorts `.ops` last (alphabetical), which re-shadows the wrapper
# CUDAGraph with the raw pybind class and breaks `foundry.graph(...)`
# (raw capture_begin requires a positional `pool`; the wrapper defaults it).
# isort: off
# _loader sets __version__, checks the installed torch against the one
# foundry.ops was built with, and loads foundry.ops; it must stay first.
from ._loader import __version__
from .allocation_region import (
    allocation_region,
    free_preallocated_region,
    get_current_alloc_offset,
    parse_size,
    preallocate_region,
    resume_allocation_region,
    set_current_alloc_offset,
)

from .ops import *  # isort: skip
from .graph import (
    CUDAGraph,
    graph,
    save_graph_manifest,
)

# isort: on

# Re-exports. Listed here so ruff's --fix doesn't strip them as unused.
# (We also configure per-file-ignores for F401 in pyproject.toml as a backstop.)
__all__ = [
    "CUDAGraph",
    "allocation_region",
    "free_preallocated_region",
    "get_current_alloc_offset",
    "graph",
    "parse_size",
    "preallocate_region",
    "resume_allocation_region",
    "save_graph_manifest",
    "set_current_alloc_offset",
]
