# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The preflight CLI (python -P -m foundry.integration.sglang.preflight) on the
recipe TOML: SAVE and LOAD checks, the FAILED line and the exit codes
(0 ok, 1 a failed check, 2 bad arguments). No GPU.

    pytest tests/sgl/test_preflight.py -v

recipe/sglang/foundry.toml has no ``mode`` key (the mode comes from
--cuda-graph-persistence), so it passes only on the dependency route; on the
plugin route the CLI must refuse it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import CONFIG_ENV, plugin_route_only, sglang_has_adapter  # noqa: E402
from foundry.integration.sglang import preflight  # noqa: E402

RECIPE_TOML = Path(__file__).resolve().parents[2] / "recipe" / "sglang" / "foundry.toml"
ARCHIVE = "foundry_archive"  # workspace_root of the recipe TOML

dependency_route_only = pytest.mark.skipif(
    not sglang_has_adapter(), reason="dependency route only: this sglang has no foundry_adapter"
)


def _preflight(cwd, *args):
    """Run the CLI from ``cwd`` (no sglang checkout to shadow the package)."""
    env = {k: v for k, v in os.environ.items() if k not in (CONFIG_ENV, "SGLANG_PLUGINS")}
    proc = subprocess.run(
        [sys.executable, "-P", "-m", "foundry.integration.sglang.preflight", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(proc.stdout + proc.stderr)
    return proc


def _recipe(tmp_path, *mode):
    return _preflight(tmp_path, "--toml", str(RECIPE_TOML), "--cwd", str(tmp_path), *mode)


@dependency_route_only
def test_recipe_toml_save_passes(tmp_path):
    proc = _recipe(tmp_path, "--save")
    assert proc.returncode == 0, proc.stderr
    assert f"{preflight.PREFIX} ok: TOML {RECIPE_TOML} mode=save" in proc.stdout
    assert "FAILED" not in proc.stderr


@dependency_route_only
def test_recipe_toml_load_without_archive_fails(tmp_path):
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 1
    assert f"{preflight.PREFIX} FAILED: LOAD archive {tmp_path / ARCHIVE}" in proc.stderr
    assert "run --save first" in proc.stderr


@dependency_route_only
def test_recipe_toml_load_with_empty_archive_fails(tmp_path):
    (tmp_path / ARCHIVE).mkdir()
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 1
    assert "has no rank_* dirs" in proc.stderr


@dependency_route_only
def test_recipe_toml_load_with_rank_dirs_passes(tmp_path):
    for rank in (0, 1):
        (tmp_path / ARCHIVE / f"rank_{rank}").mkdir(parents=True)
    proc = _recipe(tmp_path, "--load")
    assert proc.returncode == 0, proc.stderr
    assert "mode=load" in proc.stdout and "(2 rank dirs)" in proc.stdout


@dependency_route_only
def test_recipe_toml_needs_a_mode_flag(tmp_path):
    proc = _recipe(tmp_path)
    assert proc.returncode == 1
    assert "pass --save or --load" in proc.stderr


@plugin_route_only
def test_recipe_toml_is_refused_on_the_plugin_route(tmp_path):
    proc = _recipe(tmp_path, "--save")
    assert proc.returncode == 1
    assert "has mode = 'none', expected 'save'" in proc.stderr


def test_missing_toml_fails(tmp_path):
    proc = _preflight(tmp_path, "--toml", str(tmp_path / "missing.toml"), "--save")
    assert proc.returncode == 1
    assert (
        f"{preflight.PREFIX} FAILED: Foundry TOML {tmp_path / 'missing.toml'} does not exist"
        in (proc.stderr)
    )


@pytest.mark.parametrize(
    "args", [["--save"], ["--toml", str(RECIPE_TOML), "--save", "--load"]], ids=["no_toml", "both"]
)
def test_bad_arguments_exit_2(tmp_path, args):
    assert _preflight(tmp_path, *args).returncode == 2


@pytest.mark.parametrize("value", ["", "foundry", "other, foundry"])
def test_plugin_allowlist_accepts(value):
    preflight.check_plugin_allowlist({"SGLANG_PLUGINS": value} if value else {})


def test_plugin_allowlist_without_foundry_fails():
    with pytest.raises(preflight.PreflightError, match="does not include 'foundry'"):
        preflight.check_plugin_allowlist({"SGLANG_PLUGINS": "other"})
