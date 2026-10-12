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

The end-to-end tests launch through :func:`foundry_launch` (via
:func:`serve`) so one file validates whichever route the installed sglang has.
Also shared: the mini model config for CPU-only resolution tests
(:func:`model_dir`) and :func:`run_python` for checks that need a fresh
interpreter.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap

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
    preflight (route-aware, ``--save`` / ``--load``); a failure fails the test
    rather than letting the engine run natively."""
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


# A config.json sglang's server-args resolution accepts without weights.
MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(MINI_CONFIG))
    return str(tmp_path)


def run_python(code, cwd=None, *, env_drop=(CONFIG_ENV,), env_extra=None):
    """Run ``code`` (dedented) in a fresh interpreter with this environment
    minus ``env_drop`` plus ``env_extra``."""
    env = {k: v for k, v in os.environ.items() if k not in env_drop}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def foundry_toml(path, mode, workspace, scratch_space_size="1024MB"):
    """Write a Foundry TOML for an end-to-end engine. The region must hold
    everything sglang allocates after setup, the KV pool included."""
    path.write_text(
        f'mode = "{mode}"\nbase_addr = 0x600000000000\nregion_size = "256GB"\n'
        f'workspace_root = "{workspace}"\nscratch_space_size = "{scratch_space_size}"\n'
    )
    return str(path)


def serve(
    tmp_path,
    name,
    model,
    server_args,
    prompts,
    sampling,
    *,
    mode=None,
    toml=None,
    env=None,
    launch_timeout,
    request_timeout,
    while_up=None,
):
    """Launch a native (``mode=None``) or Foundry engine, POST each prompt to
    /generate, call ``while_up()`` while it still runs, stop it. Returns (the
    /generate responses, ``while_up``'s result, the engine log)."""
    import requests
    from sglang.test.test_utils import (
        DEFAULT_URL_FOR_TEST,
        popen_launch_server,
        terminate_and_kill_process_tree,
    )

    launch_env, foundry_args = foundry_launch({**os.environ, **(env or {})}, mode, toml)
    extra = None
    with open(tmp_path / f"{name}.out", "w") as out, open(tmp_path / f"{name}.err", "w") as err:
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=launch_timeout,
            other_args=server_args + foundry_args,
            env=launch_env,
            return_stdout_stderr=(out, err),
        )
        try:
            responses = []
            for prompt in prompts:
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={"text": prompt, "sampling_params": sampling},
                    timeout=request_timeout,
                )
                r.raise_for_status()
                responses.append(r.json())
            if while_up is not None:
                extra = while_up()
        finally:
            terminate_and_kill_process_tree(process)
    log = (tmp_path / f"{name}.err").read_text() + (tmp_path / f"{name}.out").read_text()
    return responses, extra, log
