# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The Python surface over foundry.ops: every method of the foundry.graph.CUDAGraph
wrapper resolves on ops.CUDAGraph, every public ops name is backed by a registered
torch op, and the public names stay the ones callers use.

    pytest tests/unit/test_graph_wrapper.py -q
"""

import importlib
import types

import pytest
import torch
from foundry import ops

# `from foundry import graph` yields the capture helper class foundry.graph.graph, which
# foundry/__init__.py re-exports over the submodule; import the module explicitly.
graph = importlib.import_module("foundry.graph")

# Implemented in Python only: SAVE captures with torch.cuda.CUDAGraph and hands the
# raw graph to ops.CUDAGraph.adopt_native_capture (no foundry.ops capture binding).
PYTHON_ONLY = {"capture_begin", "capture_end"}

# The names the pybind11 module bound (minus KernelNodeMetadata, which nothing returned).
PUBLIC_FUNCTIONS = {
    "event_elapsed_time",
    "event_query",
    "event_record",
    "event_synchronize",
    "free_preallocated_region",
    "get_current_alloc_offset",
    "get_graph_ptr",
    "get_live_region_ranges",
    "init_nvshmem_for_loaded_modules",
    "load_cuda_modules_and_libraries",
    "native_record_abort",
    "native_record_begin",
    "pack_fatbins_to_folder",
    "preallocate_cublas_workspaces",
    "preallocate_ranges",
    "preallocate_region",
    "recorded_allocation_contains",
    "resume_allocation_region",
    "set_allocation_region",
    "set_current_alloc_offset",
    "set_nvshmem_auto_init",
    "set_pack_fatbins_on_exit",
    "set_skip_fatbin_processing",
    "stop_allocation_region",
    "stream_wait_event",
}
CUDAGRAPH_METHODS = {
    "adopt_native_capture",
    "bind_events",
    "debug_dump",
    "enable_debug_mode",
    "exec_share_stats",
    "finish_graph_loads",
    "finish_one_graph_load",
    "load",
    "loaded_events",
    "pool",
    "raw_cuda_graph",
    "replay",
    "reset",
    "save",
    "start_exec_pool_prewarm",
    "start_graph_builds",
}


def _wrapper_methods():
    return sorted(
        name
        for name, value in vars(graph.CUDAGraph).items()
        if not name.startswith("__")
        and isinstance(value, (staticmethod, types.FunctionType))
        and name not in PYTHON_ONLY
    )


@pytest.mark.parametrize("name", _wrapper_methods())
def test_wrapper_method_is_bound(name):
    assert hasattr(ops.CUDAGraph, name), (
        f"foundry.graph.CUDAGraph.{name} has no foundry.ops binding"
    )


def test_public_surface():
    assert set(ops.__all__) == PUBLIC_FUNCTIONS | {"CUDAGraph", "PendingGraphLoads"}
    for name in ops.__all__:
        assert hasattr(ops, name), name
    public = {n for n in dir(ops.CUDAGraph) if not n.startswith("_")}
    assert public == CUDAGRAPH_METHODS


def test_every_op_is_registered():
    """Each op the Python surface calls exists in the torch.ops namespace."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(ops.__file__).read_text())
    called = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "_C"
    }
    assert len(called) > 40
    ns = getattr(torch.ops, ops.NAMESPACE)
    for name in sorted(called):
        assert hasattr(ns, name), f"torch.ops.{ops.NAMESPACE}.{name} is not registered"
