# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry's sglang plugin: strict no-op without FOUNDRY_GRAPH_EXTENSION_CONFIG;
with it, server-args resolution pins decode to full, prefill to full/disabled,
profiling/autotune off, pins every flag and environment variable that selects
state Foundry cannot replay (plugin.FIELD_PINS / ENV_PINS), and rejects what
Foundry cannot persist.

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


_PIN_ENV_NAMES = [pin.name for pin in plugin.ENV_PINS] + [
    plugin.NCCL_CUMEM_PIN.name,
    plugin._ENV_PINNED_MARK,
]


@pytest.fixture
def clean_pin_env(restore_environ):
    """No pinned variable inherited from the calling shell."""
    for name in _PIN_ENV_NAMES:
        os.environ.pop(name, None)
    yield


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
# Pinned flags: configs that select state Foundry cannot replay
# ---------------------------------------------------------------------------


def _unresolved(model_dir, **fields):
    """A ServerArgs record as the CLI builds it, before resolution: the pin
    step is called on it directly so that no other resolution step can reject
    the contrary value first."""
    return ServerArgs(model_path=model_dir, device="cuda", random_seed=42, **fields)


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


def test_field_pin_defaults_match_the_installed_sglang(model_dir):
    """FieldPin.default is what tells a user's contrary value from sglang's
    default; red if sglang changes one of them. ServerArgs is a msgspec
    struct, not a dataclass: read the defaults off an unresolved record."""
    sa = _unresolved(model_dir)
    for pin in plugin.FIELD_PINS:
        assert getattr(sa, pin.field) == pin.default, pin.field
    assert getattr(sa, plugin.DSV4_ATTN_FIELD) == "auto"


def test_pinned_value_given_explicitly_is_accepted(model_dir, plugin_hooks):
    sa = _resolve(model_dir, disable_custom_all_reduce=True, enable_torch_symm_mem=True)
    assert resolution_result(sa, "disable_custom_all_reduce") is True
    assert resolution_result(sa, "enable_torch_symm_mem") is True


def test_non_trtllm_dsv4_backend_is_kept(model_dir):
    sa = _unresolved(model_dir, dsv4_attn_backend="flashmla")
    plugin.pin_unreplayable_fields(sa)
    assert resolution_result(sa, plugin.DSV4_ATTN_FIELD) == "flashmla"


@pytest.mark.parametrize(
    "fields, match",
    [
        (dict(enable_symm_mem=True), r"--enable-symm-mem=True: NCCL symmetric-memory windows"),
        (dict(enable_nccl_nvls=True), r"--enable-nccl-nvls=True: NVLS multicast"),
        (dict(enable_mscclpp=True), r"--enable-mscclpp=True: MSCCL\+\+"),
        (
            dict(enable_two_batch_overlap=True),
            r"--enable-two-batch-overlap=True: not validated with Foundry",
        ),
        (dict(enable_memory_saver=True), r"--enable-memory-saver=True: torch_memory_saver"),
        (dict(dsv4_attn_backend="trtllm"), r"--dsv4-attn-backend=trtllm: the trtllm"),
    ],
)
def test_explicit_contrary_flag_is_rejected(model_dir, fields, match):
    with pytest.raises(ValueError, match=match):
        plugin.pin_unreplayable_fields(_unresolved(model_dir, **fields))


@pytest.mark.parametrize("field", ["enable_symm_mem", "enable_mscclpp"])
def test_explicit_contrary_flag_is_rejected_by_resolution(model_dir, plugin_hooks, field):
    with pytest.raises(ValueError, match=f"--{field.replace('_', '-')}=True"):
        _resolve(model_dir, **{field: True})


def test_later_step_that_undoes_a_strict_pin_is_rejected(model_dir):
    from sglang.srt.arg_groups.overrides import declare_resolution

    sa = _unresolved(model_dir)
    plugin.pin_unreplayable_fields(sa)
    declare_resolution(sa, "some_model_override", enable_two_batch_overlap=True)
    with pytest.raises(ValueError, match="set by some_model_override"):
        plugin.validate_pinned_fields(sa)


def test_torch_symm_mem_may_be_turned_off_by_sglang(model_dir, capsys):
    """Deterministic inference turns torch symm-mem off (NCCL fallback, still
    replayable): logged, not rejected."""
    from sglang.srt.arg_groups.overrides import declare_resolution

    sa = _unresolved(model_dir)
    plugin.pin_unreplayable_fields(sa)
    declare_resolution(sa, "_handle_deterministic_inference", enable_torch_symm_mem=False)
    plugin.validate_pinned_fields(sa)
    assert "enable_torch_symm_mem=False (set by _handle_deterministic_inference)" in (
        capsys.readouterr().err
    )


# ---------------------------------------------------------------------------
# Pinned environment variables
# ---------------------------------------------------------------------------


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


def test_env_pins_are_inherited_by_spawned_processes(clean_pin_env):
    plugin.apply_env_pins()
    names = [pin.name for pin in plugin.ENV_PINS]
    result = _run_python(
        f"""
        import os
        print(",".join(os.environ.get(n, "UNSET") for n in {names!r}))
        """
    )
    assert result.stdout.strip().split(",") == [pin.value for pin in plugin.ENV_PINS]


def test_env_pins_are_idempotent_and_quiet_in_descendants(clean_pin_env, capsys):
    plugin.apply_env_pins()
    capsys.readouterr()
    plugin.apply_env_pins()  # a scheduler re-asserting the inherited values
    assert "[Foundry] pin:" not in capsys.readouterr().err


def test_env_pin_already_set_to_the_pinned_value_is_kept(clean_pin_env):
    os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "false"
    plugin.apply_env_pins()
    assert os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"


@pytest.mark.parametrize("pin", plugin.ENV_PINS, ids=lambda pin: pin.name)
def test_explicit_contrary_env_is_rejected(clean_pin_env, pin):
    os.environ[pin.name] = "1"
    with pytest.raises(ValueError, match=f"{pin.name}='1' \\(Foundry needs 0\\)"):
        plugin.apply_env_pins()


def test_nccl_cumem_is_pinned_once_the_a2a_backend_is_known(clean_pin_env):
    plugin.apply_env_pins(SimpleNamespace(moe_a2a_backend="deepep"))
    assert os.environ[plugin.NCCL_CUMEM_PIN.name] == "0"


def test_nccl_cumem_is_pinned_by_resolution(model_dir, plugin_hooks):
    _resolve(model_dir)
    assert os.environ[plugin.NCCL_CUMEM_PIN.name] == "0"


def test_explicit_nccl_cumem_is_rejected(clean_pin_env):
    os.environ[plugin.NCCL_CUMEM_PIN.name] = "1"
    with pytest.raises(ValueError, match="NCCL_CUMEM_ENABLE='1'"):
        plugin.apply_env_pins(SimpleNamespace(moe_a2a_backend="none"))


def test_nccl_cumem_is_left_to_sglang_for_deepep_v2(clean_pin_env):
    """DeepEP v2's NCCL windows need cuMem; sglang sets it to 1 for deepep_v2."""
    os.environ[plugin.NCCL_CUMEM_PIN.name] = "1"
    plugin.apply_env_pins(SimpleNamespace(moe_a2a_backend="deepep_v2"))
    assert os.environ[plugin.NCCL_CUMEM_PIN.name] == "1"


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
