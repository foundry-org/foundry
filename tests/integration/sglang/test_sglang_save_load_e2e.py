# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""End to end on 1 GPU: native graph run, Foundry SAVE, Foundry LOAD of a small
model with dummy weights and a small decode set. Greedy output must be
identical across the three, LOAD must restore every decode graph once, and the
plugin must report itself active in the launcher and the scheduler. SAVE and LOAD
first pass the preflight the recipe scripts run (foundry.integration.sglang.preflight).

Needs sglang, a CUDA GPU and the model's config (HF cache or network).
Model override: FOUNDRY_E2E_MODEL (default: sglang's
DEFAULT_SMALL_MODEL_NAME_FOR_TEST, meta-llama/Llama-3.2-1B-Instruct).

    pytest tests/integration/sglang/test_sglang_save_load_e2e.py -v -s
"""

import os
import re

import pytest

pytest.importorskip("sglang")
torch = pytest.importorskip("torch")

import requests  # noqa: E402
from foundry.integration.sglang import preflight  # noqa: E402
from foundry.integration.sglang.plugin import CONFIG_ENV  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

_MODEL = os.environ.get("FOUNDRY_E2E_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
_DECODE_BS = ["1", "2", "4", "8"]
_ARGS = [
    "--load-format",
    "dummy",
    "--cuda-graph-bs-decode",
    *_DECODE_BS,
    "--cuda-graph-backend-prefill",
    "disabled",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "8",
    "--random-seed",
    "0",
]
_PROMPTS = ["The capital of France is", "1, 2, 3, 4,", "def fibonacci(n):"]
_LOADED = re.compile(r"\[Foundry\] Loaded (\d+) SGLang graphs")
_ACTIVE = re.compile(r"\[Foundry\] sglang plugin active: pid=(\d+)")


def _toml(path, mode, workspace):
    # The region must hold everything sglang allocates after setup, the KV pool included
    # (mem_fraction_static 0.6 of an 80-141 GB GPU): same size as the experimental harness.
    path.write_text(
        f'mode = "{mode}"\nbase_addr = 0x600000000000\nregion_size = "256GB"\n'
        f'workspace_root = "{workspace}"\nscratch_space_size = "1024MB"\n'
    )
    return str(path)


def _serve(tmp_path, name, env):
    launch_env = {k: v for k, v in os.environ.items() if k != CONFIG_ENV}
    launch_env.update(env)
    if CONFIG_ENV in launch_env:
        # Same check as the recipe scripts: without it a missing entry point or an
        # SGLANG_PLUGINS allowlist would make the "Foundry" engine run natively.
        try:
            preflight.run(launch_env[CONFIG_ENV], env=launch_env)
        except preflight.PreflightError as exc:
            pytest.fail(f"Foundry preflight: {exc}")
    with open(tmp_path / f"{name}.out", "w") as out, open(tmp_path / f"{name}.err", "w") as err:
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_ARGS,
            env=launch_env,
            return_stdout_stderr=(out, err),
        )
        try:
            texts = []
            for prompt in _PROMPTS:
                r = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                    timeout=120,
                )
                r.raise_for_status()
                texts.append(r.json()["text"])
        finally:
            terminate_and_kill_process_tree(process)
    log = (tmp_path / f"{name}.err").read_text() + (tmp_path / f"{name}.out").read_text()
    return texts, log


def test_save_then_load_matches_native(tmp_path):
    workspace = tmp_path / "archive"

    # Native first: also warms the JIT caches before SAVE.
    native, native_log = _serve(tmp_path, "native", {})
    assert not _ACTIVE.search(native_log), "plugin must be inert without the env var"

    save_cfg = _toml(tmp_path / "save.toml", "save", workspace)
    saved, save_log = _serve(tmp_path, "save", {CONFIG_ENV: save_cfg})

    load_cfg = _toml(tmp_path / "load.toml", "load", workspace)
    loaded, load_log = _serve(tmp_path, "load", {CONFIG_ENV: load_cfg})

    # Launcher + one scheduler (TP1): two distinct pids report activation.
    for log in (save_log, load_log):
        # An allocation outside the region is only logged by the hook; SAVE would still "pass".
        assert "[HOOK] ERROR" not in log, "hook reported an error"
        assert len(set(_ACTIVE.findall(log))) >= 2, "plugin did not load in every process"

    assert saved == native
    assert loaded == native
    counts = [int(n) for n in _LOADED.findall(load_log)]
    assert counts == [len(_DECODE_BS)], f"LOAD must restore every decode graph once: {counts}"
