# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Recipe scripts must not hard-code a GPU count: any epN topology is passed
through to `sglang serve` (which itself reports OOM / unsupported layouts).
Runs the scripts against a stub `sglang` on PATH; no GPU needed."""

import os
import stat
import subprocess
from pathlib import Path

RECIPE = Path(__file__).resolve().parent.parent / "recipe" / "sglang"


def _run(tmp_path, script, *args):
    stub = tmp_path / "sglang"
    stub.write_text('#!/bin/bash\necho "ARGS: $@"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.run(
        ["bash", str(RECIPE / script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_deepseek_accepts_ep4(tmp_path):
    r = _run(tmp_path, "serve_deepseek-v4-flash.sh", "ep4", "--save")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--tp-size 4 --dp-size 4 --ep-size 4" in r.stdout
    assert "--cuda-graph-persistence save" in r.stdout


def test_glm_accepts_ep4(tmp_path):
    r = _run(tmp_path, "serve_glm-5.3-flash.sh", "ep4", "--load")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--tp-size 4 --dp-size 4 --ep-size 4" in r.stdout
    assert "--cuda-graph-persistence load" in r.stdout
