# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Plugin route only (delete with the plugin): the sglang.srt.plugins entry
point is a strict no-op without FOUNDRY_GRAPH_EXTENSION_CONFIG; with it, the
plugin's resolution hooks pin decode to full, prefill to full/disabled,
profiling/autotune off and every FIELD_PIN, and its launcher hook sets the
ENV_PINS.

CPU only (no GPU, no server, no weights): resolution runs on a mini
config.json with device="cuda", like sglang's own
test/registered/unit/server_args/test_resolution_declarations.py.

    pytest tests/sgl/test_plugin_route.py -v
"""

import os
import sys
from importlib.metadata import entry_points
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import plugin_route_only, run_python  # noqa: E402
from foundry.integration.sglang import plugin  # noqa: E402
from sglang.srt.arg_groups import resolution_hooks  # noqa: E402
from sglang.srt.arg_groups.overrides import resolution_result  # noqa: E402
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402


@pytest.fixture
def plugin_hooks(tmp_path, clean_pin_env):
    """The plugin's resolution hooks registered on a private registry (the
    registry is process-global; the runtime patches are not installed)."""
    cfg = tmp_path / "foundry.toml"
    cfg.write_text('mode = "save"\n')
    with mock.patch.dict(resolution_hooks._HOOKS, clear=True):
        os.environ[plugin.CONFIG_ENV] = str(cfg)
        plugin.register_resolution_hooks()
        yield


def _resolve(model_dir, **fields):
    sa = ServerArgs(model_path=model_dir, device="cuda", random_seed=42, **fields)
    sa.resolve_once()
    return sa


# ---------------------------------------------------------------------------
# No-op without the env var
# ---------------------------------------------------------------------------


def test_entry_point_is_registered():
    """Red if pyproject.toml loses the entry point (or the install is stale)."""
    eps = {ep.name: ep.value for ep in entry_points(group="sglang.srt.plugins")}
    assert eps.get("foundry") == "foundry_sglang_plugin:load", eps


def test_load_plugins_without_env_imports_nothing_from_foundry():
    """sglang's load_plugins() imports the entry module in every process; with
    the env var unset nothing from the foundry package (native extension
    included) may be imported and no resolution hook may be registered."""
    result = run_python(
        """
        import sys
        from sglang.srt.plugins import load_plugins
        from sglang.srt.arg_groups import resolution_hooks
        load_plugins()
        assert "foundry_sglang_plugin" in sys.modules, "entry point was not loaded"
        leaked = sorted(m for m in sys.modules if m == "foundry" or m.startswith("foundry."))
        assert not leaked, leaked
        assert not resolution_hooks._HOOKS, resolution_hooks._HOOKS
        print("OK")
        """
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]


def test_resolution_without_plugin_hooks_is_untouched(model_dir, restore_environ):
    with mock.patch.dict(resolution_hooks._HOOKS, clear=True):
        sa = _resolve(model_dir)
    sources = [src for src, _ in getattr(sa, "_resolved_overrides", None) or ()]
    assert plugin.RESOLUTION_SOURCE not in sources


@plugin_route_only
def test_activation_failure_exits_instead_of_being_swallowed(tmp_path):
    """load_plugins() swallows Exceptions from plugins; a bad config must stop
    the process (SystemExit) rather than leave sglang running natively."""
    result = run_python(
        """
        from sglang.srt.plugins import load_plugins
        load_plugins()
        print("SHOULD NOT GET HERE")
        """,
        env_extra={plugin.CONFIG_ENV: str(tmp_path / "missing.toml")},
    )
    assert result.returncode != 0
    assert "SHOULD NOT GET HERE" not in result.stdout
    assert "[Foundry] sglang plugin failed to activate" in result.stderr


# ---------------------------------------------------------------------------
# With the plugin's resolution hooks
# ---------------------------------------------------------------------------


def test_pins_decode_full_and_defaults_prefill_to_disabled(model_dir, plugin_hooks):
    sa = _resolve(model_dir)
    cfg = resolution_result(sa, "cuda_graph_config")
    assert cfg.decode.backend == Backend.FULL
    assert cfg.prefill.backend == Backend.DISABLED  # not breakable, the CUDA default
    assert resolution_result(sa, "enable_profile_cuda_graph") is False
    assert resolution_result(sa, "disable_flashinfer_autotune") is True


def test_explicit_full_prefill_is_honoured(model_dir, plugin_hooks):
    sa = _resolve(model_dir, cuda_graph_backend_prefill=Backend.FULL)
    assert resolution_result(sa, "cuda_graph_config").prefill.backend == Backend.FULL
    declared = dict(sa._resolved_overrides)[plugin.RESOLUTION_SOURCE]
    assert "cuda_graph_backend_prefill" not in declared


def test_non_full_prefill_flag_is_rejected(model_dir, plugin_hooks):
    with pytest.raises(ValueError, match="Foundry persists full prefill graphs only"):
        _resolve(model_dir, cuda_graph_backend_prefill=Backend.BREAKABLE)


def test_non_full_decode_flag_is_rejected(model_dir, plugin_hooks):
    with pytest.raises(ValueError, match="Foundry persists full decode graphs only"):
        _resolve(model_dir, cuda_graph_backend_decode=Backend.BREAKABLE)


@pytest.mark.parametrize(
    "json_cfg, match",
    [
        ({"prefill": {"backend": "breakable"}}, "prefill graphs only"),
        ({"decode": {"backend": "disabled"}}, "decode graphs only"),
    ],
)
def test_cuda_graph_config_json_is_validated(model_dir, plugin_hooks, json_cfg, match):
    """JSON outranks the per-phase flags, so the flag check alone misses it.
    (Passed as the parsed dict: parse_cuda_graph_config reads ``.items()``.)"""
    with pytest.raises(ValueError, match=match):
        _resolve(model_dir, cuda_graph_config=json_cfg)


@pytest.mark.parametrize("field", ["disable_cuda_graph", "disable_decode_cuda_graph"])
def test_disable_cuda_graph_is_rejected_not_overridden(model_dir, plugin_hooks, field):
    with pytest.raises(ValueError, match="cannot be combined with Foundry"):
        _resolve(model_dir, **{field: True})


def test_profile_cuda_graph_is_rejected(model_dir, plugin_hooks):
    with pytest.raises(ValueError, match="enable-profile-cuda-graph"):
        _resolve(model_dir, enable_profile_cuda_graph=True)


def test_every_field_pin_is_applied_on_sglang_defaults(model_dir, plugin_hooks, capsys):
    sa = _resolve(model_dir)
    for pin in plugin.FIELD_PINS:
        assert resolution_result(sa, pin.field) == pin.value, pin.field
    assert resolution_result(sa, plugin.DSV4_ATTN_FIELD) == "auto"
    declared = dict(sa._resolved_overrides)[plugin.PIN_RESOLUTION_SOURCE]
    assert {pin.field for pin in plugin.FIELD_PINS} <= set(declared)
    err = capsys.readouterr().err
    for pin in plugin.FIELD_PINS:
        assert f"[Foundry] pin: {pin.field}={pin.value!r}" in err, pin.field


def test_pinned_value_given_explicitly_is_accepted(model_dir, plugin_hooks):
    sa = _resolve(model_dir, disable_custom_all_reduce=True, enable_torch_symm_mem=True)
    assert resolution_result(sa, "disable_custom_all_reduce") is True
    assert resolution_result(sa, "enable_torch_symm_mem") is True


@pytest.mark.parametrize("field", ["enable_symm_mem", "enable_mscclpp"])
def test_explicit_contrary_flag_is_rejected_by_resolution(model_dir, plugin_hooks, field):
    with pytest.raises(ValueError, match=f"--{field.replace('_', '-')}=True"):
        _resolve(model_dir, **{field: True})


@plugin_route_only
def test_launcher_hook_sets_every_env_pin(tmp_path, clean_pin_env, capsys):
    """activate() is the launcher hook (load_plugins, before the server args
    are built and before any engine process is spawned)."""
    cfg = tmp_path / "foundry.toml"
    cfg.write_text('mode = "save"\n')
    fake_hooks = SimpleNamespace(install=lambda path: None)
    with (
        mock.patch.object(plugin, "_ACTIVATED", False),
        mock.patch.object(plugin, "register_resolution_hooks"),
        mock.patch.dict(sys.modules, {"foundry.integration.sglang.hooks": fake_hooks}),
    ):
        plugin.activate(str(cfg))
    for pin in plugin.ENV_PINS:
        assert os.environ.get(pin.name) == pin.value, pin.name
    # NCCL_CUMEM_ENABLE waits for the resolved all-to-all backend.
    assert plugin.NCCL_CUMEM_PIN.name not in os.environ
    err = capsys.readouterr().err
    for pin in plugin.ENV_PINS:
        assert f"[Foundry] pin: {pin.name}={pin.value} (was unset)" in err, pin.name


def test_nccl_cumem_is_pinned_by_resolution(model_dir, plugin_hooks):
    _resolve(model_dir)
    assert os.environ[plugin.NCCL_CUMEM_PIN.name] == "0"
