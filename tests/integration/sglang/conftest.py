# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Route-aware helpers for the SGLang integration tests.

Foundry reaches SGLang by one of two routes, decided by the installed sglang:

- dependency route: sglang has ``sglang.srt.utils.foundry_adapter`` and calls
  Foundry from its own call sites; enabled by ``--cuda-graph-persistence
  {save,load}`` and ``--cuda-graph-persistence-config <toml>``; the plugin
  refuses to activate there;
- plugin route: the ``sglang.srt.plugins`` entry point, enabled by
  ``FOUNDRY_GRAPH_EXTENSION_CONFIG=<toml>``.

The end-to-end tests launch through :func:`foundry_launch` so one file
validates whichever route the installed sglang has.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

CONFIG_ENV = "FOUNDRY_GRAPH_EXTENSION_CONFIG"
PLUGIN_ONLY_REASON = (
    "plugin route only: this sglang calls Foundry directly (sglang.srt.utils.foundry_adapter) "
    "and the plugin refuses to activate on it"
)
# Activation line of either route, one per process that activated Foundry.
ACTIVE = re.compile(
    r"\[Foundry\] (?:sglang plugin active|cuda graph persistence active): pid=(\d+)"
)


def sglang_has_adapter() -> bool:
    """True when the installed sglang declares Foundry as a dependency
    (dependency route)."""
    try:
        return importlib.util.find_spec("sglang.srt.utils.foundry_adapter") is not None
    except (ImportError, ValueError):
        return False


plugin_route_only = pytest.mark.skipif(sglang_has_adapter(), reason=PLUGIN_ONLY_REASON)


def foundry_launch(base_env, mode=None, toml=None):
    """(env, extra server args) for a native (``mode=None``) or a Foundry SAVE /
    LOAD launch on the installed sglang's route. A Foundry launch first passes
    the preflight the recipe scripts run (route-aware, ``--save`` / ``--load``);
    a failure fails the test rather than letting the engine run natively."""
    env = {k: v for k, v in base_env.items() if k != CONFIG_ENV}
    if mode is None:
        return env, []
    from foundry.integration.sglang import preflight

    try:
        preflight.run(toml, expect_mode=mode, env=env)
    except preflight.PreflightError as exc:
        pytest.fail(f"Foundry preflight: {exc}")
    if sglang_has_adapter():
        return env, ["--cuda-graph-persistence", mode, "--cuda-graph-persistence-config", toml]
    env[CONFIG_ENV] = toml
    return env, []
