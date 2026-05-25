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

PUBLIC_MODULES = [
    # Top-level
    "foundry",
    "foundry.ops",
    "foundry.graph",
    "foundry.allocation_region",
    # vLLM integration
    "foundry.integration.vllm",
    "foundry.integration.vllm.config",
    "foundry.integration.vllm.hooks",
    "foundry.integration.vllm.runtime",
    "foundry.integration.vllm.graph_ops",
    # SGLang integration
    "foundry.integration.sglang",
    "foundry.integration.sglang.config",
    "foundry.integration.sglang.hooks",
    "foundry.integration.sglang.runtime",
    "foundry.integration.sglang.graph_ops",
]

# Names that have historically been ruff --fix victims because they live in
# re-export modules. If any of these stop importing, the integration breaks
# silently — the serve script "looks fine", logs no [foundry] / [HOOK] lines,
# and vLLM runs without graph-extension hooks. See tools/pre_commit/ notes.
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


def test_re_exports_present():
    foundry = importlib.import_module("foundry")
    missing = [n for n in PUBLIC_NAMES_FROM_FOUNDRY if not hasattr(foundry, n)]
    assert not missing, (
        f"Missing re-exports on `foundry`: {missing}. "
        "Did `ruff --fix` strip an unused import from a re-export module? "
        "Check `foundry/__init__.py` and the corresponding source module's `__all__`."
    )


def test_vllm_integration_public_api():
    mod = importlib.import_module("foundry.integration.vllm")
    for name in ("install_hooks", "CUDAGraphExtensionMode", "get_graph_extension_mode"):
        assert hasattr(mod, name), f"foundry.integration.vllm.{name} not exported"


def test_sglang_integration_submodules_importable():
    # SGLang side currently exposes its API via submodules rather than
    # re-exports on `foundry.integration.sglang`. Just verify each submodule
    # imports cleanly — a ruff --fix that stripped a critical import would
    # break this.
    for sub in ("config", "hooks", "runtime", "graph_ops"):
        importlib.import_module(f"foundry.integration.sglang.{sub}")
