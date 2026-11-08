# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Dependency route: an sglang that declares Foundry as a dependency and calls
``foundry.integration.sglang.api`` from its own call sites
(``sglang.srt.utils.foundry_adapter``, ``--cuda-graph-persistence``).

Activation is process-global (the config, the environment pins), so every
check that activates runs in a fresh interpreter. No GPU needed.

    pytest tests/sgl/test_dependency_route.py -v
"""

import functools

import pytest

pytest.importorskip("sglang")
adapter = pytest.importorskip("sglang.srt.utils.foundry_adapter")

from _sglang_helpers import run_python  # noqa: E402
from foundry.integration.sglang import api, pins  # noqa: E402

_PIN_ENV_NAMES = [pin.name for pin in pins.ENV_PINS] + [
    pins.NCCL_CUMEM_PIN.name,
    pins._ENV_PINNED_MARK,
    "LD_PRELOAD",
    "FOUNDRY_MODE",
]


# A fresh interpreter without any pinned variable inherited from this shell.
_run_python = functools.partial(run_python, env_drop=_PIN_ENV_NAMES)


def _resolve_code(model_dir, fields):
    return f"""
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.arg_groups.overrides import resolution_result
        sa = ServerArgs(model_path={model_dir!r}, device="cuda", random_seed=42, **{fields!r})
        sa.resolve_once()
        cfg = resolution_result(sa, "cuda_graph_config")
    """


def test_adapter_speaks_this_api_major():
    assert api.INTEGRATION_API_VERSION[0] == adapter.FOUNDRY_INTEGRATION_API_MAJOR


def test_resolution_without_the_flag_imports_nothing_from_foundry(model_dir, tmp_path):
    result = _run_python(
        _resolve_code(model_dir, {})
        + """
        import sys
        leaked = sorted(m for m in sys.modules if m == "foundry" or m.startswith("foundry."))
        assert not leaked, leaked
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]


def test_flag_pins_graph_fields_and_environment(model_dir, tmp_path):
    result = _run_python(
        _resolve_code(model_dir, {"cuda_graph_persistence": "save"})
        + """
        import os
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from foundry.integration.sglang import pins
        from foundry.integration.sglang.config import get_config
        assert cfg.decode.backend == Backend.FULL, cfg.decode.backend
        assert cfg.prefill.backend == Backend.DISABLED, cfg.prefill.backend
        for pin in pins.FIELD_PINS:
            if pin.strict:
                assert resolution_result(sa, pin.field) == pin.value, pin.field
        for pin in pins.ENV_PINS:
            assert os.environ.get(pin.name) == pin.value, pin.name
        assert os.environ[pins.NCCL_CUMEM_PIN.name] == "0"
        assert get_config().mode.value == "save"
        assert get_config().workspace_root == "foundry_archive"
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]
    assert "[Foundry] cuda graph persistence active:" in result.stderr


def test_contrary_flag_falls_back_naming_the_sglang_flag(model_dir, tmp_path):
    """Resolution succeeds without Foundry's graph pins, and a scheduler
    spawned through the adapter falls back at its activation too."""
    result = _run_python(
        _resolve_code(
            model_dir,
            {"cuda_graph_persistence": "save", "enable_two_batch_overlap": True},
        )
        + """
        import subprocess, sys
        from sglang.srt.utils.foundry_adapter import get_foundry_adapter
        assert not get_foundry_adapter().enabled
        # Nothing of Foundry's was declared (sglang's own TBO rule may still
        # disable the prefill graphs).
        from foundry.integration.sglang import pins
        assert pins.RESOLUTION_SOURCE not in dict(sa._resolved_overrides)
        # Inside the generated source: escaped newlines, so dedent() still sees
        # one indentation.
        child = (
            "from types import SimpleNamespace\\n"
            "from sglang.srt.utils.foundry_adapter import activate_foundry\\n"
            "a = activate_foundry(SimpleNamespace(cuda_graph_persistence='save', "
            "cuda_graph_persistence_config=None))\\n"
            "assert not a.enabled\\n"
            "print('CHILD OK')\\n"
        )
        with get_foundry_adapter().configure_subprocess(sa):
            out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
        assert "CHILD OK" in out.stdout, out.stderr[-4000:]
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]
    assert (
        "[Foundry] unsupported at server args: --enable-two-batch-overlap=True: not validated "
        "with Foundry"
    ) in result.stderr, result.stderr[-4000:]
    assert "cuda-graph persistence disabled, continuing natively" in result.stderr


def test_config_without_mode_is_rejected(model_dir, tmp_path):
    toml = tmp_path / "foundry.toml"
    toml.write_text('mode = "save"\n')
    result = _run_python(
        _resolve_code(model_dir, {"cuda_graph_persistence_config": str(toml)}),
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "--cuda-graph-persistence-config requires --cuda-graph-persistence" in (result.stderr)


def test_flag_mode_wins_over_the_toml(model_dir, tmp_path):
    toml = tmp_path / "foundry.toml"
    toml.write_text('mode = "save"\nworkspace_root = "archive_x"\n')
    result = _run_python(
        _resolve_code(
            model_dir,
            {"cuda_graph_persistence": "load", "cuda_graph_persistence_config": str(toml)},
        )
        + """
        from foundry.integration.sglang.config import get_config
        assert get_config().mode.value == "load", get_config().mode
        assert get_config().workspace_root == "archive_x"
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]


def test_configure_subprocess_is_scoped(tmp_path):
    result = _run_python(
        """
        import os
        from foundry.integration.sglang import api
        from foundry.integration.sglang.config import get_hook_library_path
        os.environ["LD_PRELOAD"] = "/launcher/own.so"
        api.activate(mode="save")
        with api.configure_subprocess():
            inside = os.environ["LD_PRELOAD"].split(":")
            mode = os.environ.get("FOUNDRY_MODE")
        hook = get_hook_library_path()
        # Foundry prepends its libraries (hook, NVSHMEM host lib, optional udev shim) in front of the
        # launcher's own entries; the hook need not be first (the NVSHMEM host lib is prepended after it).
        assert hook and hook in inside and inside[-1] == "/launcher/own.so", inside
        assert mode == "save", mode
        assert os.environ["LD_PRELOAD"] == "/launcher/own.so"
        assert "FOUNDRY_MODE" not in os.environ
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]


def test_resource_tracker_starts_before_the_preload_window(tmp_path):
    """Python starts the launcher's resource tracker lazily in the first
    Process.start(); it must already run when the preload env is set, or it
    inherits LD_PRELOAD."""
    result = _run_python(
        """
        import os
        from unittest import mock
        from multiprocessing import resource_tracker
        from foundry.integration.sglang import api
        api.activate(mode="save")
        seen = []
        real = resource_tracker.ensure_running

        def ensure_running():
            seen.append(("tracker", os.environ.get("FOUNDRY_MODE")))
            real()

        with mock.patch.object(resource_tracker, "ensure_running", ensure_running):
            with api.configure_subprocess():
                seen.append(("window", os.environ.get("FOUNDRY_MODE")))
        assert seen == [("tracker", None), ("window", "save")], seen
        assert resource_tracker._resource_tracker._pid is not None
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]


def test_reactivation_with_other_arguments_is_refused(tmp_path):
    result = _run_python(
        """
        from foundry.integration.sglang import api
        api.activate(mode="save")
        api.activate(mode="save")  # idempotent
        try:
            api.activate(mode="load")
        except RuntimeError as exc:
            print("REFUSED", exc)
        """,
        cwd=tmp_path,
    )
    assert "REFUSED" in result.stdout, result.stderr[-4000:]
