# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Smoke-test that every public Foundry import survives a fresh interpreter.

This catches a class of regression that's easy to introduce with auto-format /
auto-fix tooling: ``ruff --fix`` strips imports it thinks are unused from
re-export modules like ``foundry/allocation_region.py``, breaking
``foundry/__init__.py`` which depends on those names being re-exported.

If you see a failure here after a pre-commit run, the formatter probably
removed a re-export. Add the missing name back into the source module and
list it in that module's ``__all__`` so the next run doesn't strip it again.
"""

import importlib

import pytest

PUBLIC_MODULES = [
    # Top-level
    "foundry",
    "foundry.ops",
    "foundry.graph",
    "foundry.allocation_region",
    # SGLang integration (no sglang import at module top)
    "foundry.integration.sglang",
    "foundry.integration.sglang.api",
    "foundry.integration.sglang.archive_check",
    "foundry.integration.sglang.config",
    "foundry.integration.sglang.flashinfer_cute_cache",
    "foundry.integration.sglang.graph_ops",
    "foundry.integration.sglang.hooks",
    "foundry.integration.sglang.nvtx",
    "foundry.integration.sglang.plugin",
    "foundry.integration.sglang.preflight",
    "foundry.integration.sglang.runtime",
    # sglang plugin entry point (pyproject [project.entry-points."sglang.srt.plugins"])
    "foundry_sglang_plugin",
]

# vLLM integration: imports vllm at module import, so tested only when vllm is installed.
VLLM_MODULES = [
    "foundry.integration.vllm",
    "foundry.integration.vllm.config",
    "foundry.integration.vllm.hooks",
    "foundry.integration.vllm.runtime",
    "foundry.integration.vllm.graph_ops",
]

# Names that have historically been ruff --fix victims because they live in
# re-export modules. If any of these stop importing, every integration that
# reaches them through `foundry` fails at import or at its first use.
PUBLIC_NAMES_FROM_FOUNDRY = [
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
    "set_allocation_region",
    "stop_allocation_region",
]


def test_public_modules_importable():
    for name in PUBLIC_MODULES:
        importlib.import_module(name)


def test_vllm_modules_importable():
    pytest.importorskip("vllm")
    for name in VLLM_MODULES:
        importlib.import_module(name)


def test_re_exports_present():
    foundry = importlib.import_module("foundry")
    missing = [n for n in PUBLIC_NAMES_FROM_FOUNDRY if not hasattr(foundry, n)]
    assert not missing, (
        f"Missing re-exports on `foundry`: {missing}. "
        "Did `ruff --fix` strip an unused import from a re-export module? "
        "Check `foundry/__init__.py` and the corresponding source module's `__all__`."
    )


def test_cudagraph_reexport_uses_python_wrapper():
    foundry = importlib.import_module("foundry")
    graph_module = importlib.import_module("foundry.graph")

    assert foundry.CUDAGraph is graph_module.CUDAGraph


def test_vllm_integration_public_api():
    pytest.importorskip("vllm")
    mod = importlib.import_module("foundry.integration.vllm")
    for name in ("install_hooks", "CUDAGraphExtensionMode", "get_graph_extension_mode"):
        assert hasattr(mod, name), f"foundry.integration.vllm.{name} not exported"
