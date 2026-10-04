# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Dependency route: an sglang that declares Foundry as a dependency and calls
``foundry.integration.sglang.api`` from its own call sites
(``sglang.srt.utils.foundry_adapter``, ``--cuda-graph-persistence``).

Activation is process-global (the config, the message labels), so every
check that activates runs in a fresh interpreter. No GPU needed.

    pytest tests/integration/sglang/test_sglang_dependency_route.py -v
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sglang")
adapter = pytest.importorskip("sglang.srt.utils.foundry_adapter")

from foundry.integration.sglang import api, plugin  # noqa: E402

_MINI_CONFIG = {
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
_PIN_ENV_NAMES = [pin.name for pin in plugin.ENV_PINS] + [
    plugin.NCCL_CUMEM_PIN.name,
    plugin._ENV_PINNED_MARK,
    plugin.CONFIG_ENV,
    "LD_PRELOAD",
    "FOUNDRY_MODE",
]


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_MINI_CONFIG))
    return str(tmp_path)


def _run_python(code, cwd, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in _PIN_ENV_NAMES}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


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
        from foundry.integration.sglang import plugin
        from foundry.integration.sglang.config import get_config
        assert cfg.decode.backend == Backend.FULL, cfg.decode.backend
        assert cfg.prefill.backend == Backend.DISABLED, cfg.prefill.backend
        for pin in plugin.FIELD_PINS:
            if pin.strict:
                assert resolution_result(sa, pin.field) == pin.value, pin.field
        for pin in plugin.ENV_PINS:
            assert os.environ.get(pin.name) == pin.value, pin.name
        assert os.environ[plugin.NCCL_CUMEM_PIN.name] == "0"
        assert get_config().mode.value == "save"
        assert get_config().workspace_root == "foundry_archive"
        print("OK")
        """,
        cwd=tmp_path,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]
    assert "[Foundry] cuda graph persistence active:" in result.stderr


def test_contrary_flag_is_rejected_naming_the_sglang_flag(model_dir, tmp_path):
    result = _run_python(
        _resolve_code(
            model_dir,
            {"cuda_graph_persistence": "save", "enable_two_batch_overlap": True},
        ),
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "not supported with Foundry graph persistence (--cuda-graph-persistence)" in (
        result.stderr
    ), result.stderr[-4000:]
    assert "--enable-two-batch-overlap=True: not validated with Foundry" in result.stderr
    assert plugin.CONFIG_ENV not in result.stderr


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


def test_plugin_refuses_to_patch_an_sglang_with_the_adapter(tmp_path):
    toml = tmp_path / "foundry.toml"
    toml.write_text('mode = "save"\n')
    result = _run_python(
        """
        from sglang.srt.plugins import load_plugins
        load_plugins()
        print("SHOULD NOT GET HERE")
        """,
        cwd=tmp_path,
        env_extra={plugin.CONFIG_ENV: str(toml)},
    )
    assert result.returncode != 0
    assert "SHOULD NOT GET HERE" not in result.stdout
    assert "use --cuda-graph-persistence" in result.stderr, result.stderr[-4000:]
