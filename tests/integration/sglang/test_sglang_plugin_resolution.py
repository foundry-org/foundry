# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry's sglang plugin: strict no-op without FOUNDRY_GRAPH_EXTENSION_CONFIG;
with it, server-args resolution pins decode to full, prefill to full/disabled,
profiling/autotune off, and rejects what Foundry cannot persist.

CPU only (no GPU, no server, no weights): resolution runs on a mini
config.json with device="cuda", like sglang's own
test/registered/unit/server_args/test_resolution_declarations.py.

    pytest tests/integration/sglang/test_sglang_plugin_resolution.py -v
"""

import json
import os
import subprocess
import sys
import textwrap
from importlib.metadata import entry_points
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from foundry.integration.sglang import plugin  # noqa: E402
from sglang.srt.arg_groups import resolution_hooks  # noqa: E402
from sglang.srt.arg_groups.overrides import resolution_result  # noqa: E402
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

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


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_MINI_CONFIG))
    return str(tmp_path)


@pytest.fixture
def restore_environ():
    # Resolution writes environment variables that outlive the record.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def plugin_hooks(tmp_path, restore_environ):
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


def _run_python(code, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != plugin.CONFIG_ENV}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


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
    result = _run_python(
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


def test_activation_failure_exits_instead_of_being_swallowed(tmp_path):
    """load_plugins() swallows Exceptions from plugins; a bad config must stop
    the process (SystemExit) rather than leave sglang running natively."""
    result = _run_python(
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


@pytest.mark.parametrize("backend", [Backend.BREAKABLE, Backend.TC_PIECEWISE])
def test_non_full_prefill_flag_is_rejected(model_dir, plugin_hooks, backend):
    with pytest.raises(ValueError, match="Foundry persists full prefill graphs only"):
        _resolve(model_dir, cuda_graph_backend_prefill=backend)


def test_non_full_decode_flag_is_rejected(model_dir, plugin_hooks):
    with pytest.raises(ValueError, match="Foundry persists full decode graphs only"):
        _resolve(model_dir, cuda_graph_backend_decode=Backend.BREAKABLE)


@pytest.mark.parametrize(
    "json_cfg, match",
    [
        ({"prefill": {"backend": "breakable"}}, "prefill graphs only"),
        ({"prefill": {"backend": "tc_piecewise"}}, "prefill graphs only"),
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


# ---------------------------------------------------------------------------
# Unsupported features (the check itself, on a plain namespace: building a
# valid speculative / LoRA / elastic-EP record needs weights or a cluster)
# ---------------------------------------------------------------------------


def _ns(**overrides):
    base = dict(
        speculative_algorithm=None,
        enable_lora=False,
        lora_paths=None,
        enable_pdmux=False,
        elastic_ep_backend=None,
        max_ep_size=None,
        tp_size=4,
        pp_size=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_supported_record_passes(restore_environ):
    os.environ.pop("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING", None)
    plugin.reject_unsupported_features(_ns())
    plugin.reject_unsupported_features(_ns(pp_size=2))  # PP is allowed
    # Elastic EP without recapture (max_ep_size == tp_size) is allowed.
    plugin.reject_unsupported_features(_ns(elastic_ep_backend="mooncake", max_ep_size=4))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(speculative_algorithm="EAGLE"), "speculative decoding"),
        (dict(enable_lora=True), "LoRA"),
        (dict(lora_paths=["a=/x"]), "LoRA"),
        (dict(enable_pdmux=True), "pdmux"),
    ],
)
def test_unsupported_features_are_rejected(restore_environ, overrides, match):
    with pytest.raises(ValueError, match=match):
        plugin.reject_unsupported_features(_ns(**overrides))


def test_elastic_recapture_is_rejected_where_the_path_exists(restore_environ):
    ns = _ns(elastic_ep_backend="mooncake", max_ep_size=8)
    with (
        mock.patch.object(plugin, "_elastic_recapture_path_exists", return_value=True),
        pytest.raises(ValueError, match="elastic-EP CUDA-graph recapture"),
    ):
        plugin.reject_unsupported_features(ns)
    with mock.patch.object(plugin, "_elastic_recapture_path_exists", return_value=False):
        plugin.reject_unsupported_features(ns)


def test_post_capture_kv_sizing_is_rejected(restore_environ):
    os.environ["SGLANG_ENABLE_POST_CAPTURE_KV_SIZING"] = "1"
    with pytest.raises(ValueError, match="POST_CAPTURE_KV_SIZING"):
        plugin.reject_unsupported_features(_ns())
