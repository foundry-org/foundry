# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The Python surface over foundry.ops: every method of the foundry.graph.CUDAGraph
wrapper resolves on the bound class, and ops.pyi declares nothing the
extension does not bind (names the stub lacks are reported as a warning).

    pytest tests/unit/test_graph_wrapper.py -q
"""

import ast
import importlib
import types
import warnings
from pathlib import Path

import pytest
from foundry import ops

# `from foundry import graph` yields the capture helper class foundry.graph.graph, which
# foundry/__init__.py re-exports over the submodule; import the module explicitly.
graph = importlib.import_module("foundry.graph")

PYI = Path(ops.__file__).with_name("ops.pyi")

# Declared in ops.pyi but bound nowhere in csrc/binding.cpp (to be removed from
# the stub). Listed so that a new stale name fails while these are tracked;
# remove an entry once the stub drops it.
KNOWN_UNBOUND_IN_PYI = {
    "KernelUpdateInfo",
    "KernelUpdateInfoList",
    "MemcpyUpdateInfo",
    "MemcpyUpdateInfoList",
    "MemsetUpdateInfo",
    "MemsetUpdateInfoList",
}


def _wrapper_methods():
    return sorted(
        name
        for name, value in vars(graph.CUDAGraph).items()
        if not name.startswith("__") and isinstance(value, (staticmethod, types.FunctionType))
    )


@pytest.mark.parametrize("name", _wrapper_methods())
def test_wrapper_method_is_bound(name):
    assert hasattr(ops.CUDAGraph, name), (
        f"foundry.graph.CUDAGraph.{name} has no foundry.ops binding"
    )


def _stub():
    if not PYI.is_file():
        pytest.skip(f"{PYI} not installed")
    return ast.parse(PYI.read_text())


def _defs(body):
    return {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _public(obj):
    return {n for n in dir(obj) if not n.startswith("_")}


def test_pyi_declares_only_bound_names():
    declared = _defs(_stub().body)
    unbound = declared - _public(ops)
    assert unbound - KNOWN_UNBOUND_IN_PYI == set(), f"ops.pyi declares unbound names: {unbound}"
    fixed = KNOWN_UNBOUND_IN_PYI - unbound
    assert not fixed, f"bound or gone from ops.pyi now, drop from KNOWN_UNBOUND_IN_PYI: {fixed}"
    missing = sorted(_public(ops) - declared)
    if missing:
        warnings.warn(f"foundry.ops names missing from ops.pyi: {missing}", stacklevel=1)


def test_pyi_cudagraph_methods_are_bound():
    cls = next(n for n in _stub().body if isinstance(n, ast.ClassDef) and n.name == "CUDAGraph")
    declared = {n for n in _defs(cls.body) if not n.startswith("__")}
    assert declared <= _public(ops.CUDAGraph), declared - _public(ops.CUDAGraph)
