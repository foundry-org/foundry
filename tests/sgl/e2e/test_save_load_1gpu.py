# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""End to end on 1 GPU: native graph run, Foundry SAVE, Foundry LOAD of a small
model with dummy weights and a small decode set. Greedy output must be
identical across the three, LOAD must restore every decode graph once, and
Foundry must report itself active in the launcher and the scheduler. SAVE and LOAD
first pass the preflight CLI's checks (foundry.integration.sglang.preflight).
Runs on whichever route the installed sglang has (_sglang_helpers.foundry_launch): the
--cuda-graph-persistence flags when sglang has the Foundry adapter, the plugin's
environment variable otherwise.

Needs sglang, a CUDA GPU and the model's config (HF cache or network).
Model override: FOUNDRY_E2E_MODEL (default: sglang's
DEFAULT_SMALL_MODEL_NAME_FOR_TEST, meta-llama/Llama-3.2-1B-Instruct).

    pytest tests/sgl/e2e/test_save_load_1gpu.py -m slow -v -s
"""

import os
import re

import pytest

pytest.importorskip("sglang")
torch = pytest.importorskip("torch")

from _sglang_helpers import ACTIVE as _ACTIVE  # noqa: E402
from _sglang_helpers import foundry_toml, serve  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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
_SAMPLING = {"temperature": 0, "max_new_tokens": 32}
_LOADED = re.compile(r"\[Foundry\] Loaded (\d+) SGLang graphs")


def _serve(tmp_path, name, mode=None, toml=None):
    responses, _, log = serve(
        tmp_path,
        name,
        _MODEL,
        _ARGS,
        _PROMPTS,
        _SAMPLING,
        mode=mode,
        toml=toml,
        launch_timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        request_timeout=120,
    )
    return [r["text"] for r in responses], log


def test_save_then_load_matches_native(tmp_path):
    workspace = tmp_path / "archive"

    # Native first: also warms the JIT caches before SAVE.
    native, native_log = _serve(tmp_path, "native")
    assert not _ACTIVE.search(native_log), "Foundry must be inert without being enabled"

    save_cfg = foundry_toml(tmp_path / "save.toml", "save", workspace)
    saved, save_log = _serve(tmp_path, "save", "save", save_cfg)

    load_cfg = foundry_toml(tmp_path / "load.toml", "load", workspace)
    loaded, load_log = _serve(tmp_path, "load", "load", load_cfg)

    # Launcher + one scheduler (TP1): two distinct pids report activation.
    for log in (save_log, load_log):
        # An allocation outside the region is only logged by the hook; SAVE would still "pass".
        assert "[HOOK] ERROR" not in log, "hook reported an error"
        assert len(set(_ACTIVE.findall(log))) >= 2, "Foundry did not load in every process"

    assert saved == native
    assert loaded == native
    counts = [int(n) for n in _LOADED.findall(load_log)]
    assert counts == [len(_DECODE_BS)], f"LOAD must restore every decode graph once: {counts}"
