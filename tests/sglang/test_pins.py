# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The pins and rejections both sglang routes apply (plugin.FIELD_PINS /
ENV_PINS, reject_unsupported_features; the dependency route reaches them
through api.pin_server_args): flags and environment variables that select
state Foundry cannot replay, and features it cannot persist.

CPU only (no GPU, no server, no weights): records are built on a mini
config.json with device="cuda".

    pytest tests/sglang/test_pins.py -v
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import run_python  # noqa: E402
from foundry.integration.sglang import plugin  # noqa: E402
from sglang.srt.arg_groups.overrides import resolution_result  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402


def _unresolved(model_dir, **fields):
    """A ServerArgs record as the CLI builds it, before resolution: the pin
    step is called on it directly so that no other resolution step can reject
    the contrary value first."""
    return ServerArgs(model_path=model_dir, device="cuda", random_seed=42, **fields)


# ---------------------------------------------------------------------------
# Pinned flags: configs that select state Foundry cannot replay
# ---------------------------------------------------------------------------


def test_field_pin_defaults_match_the_installed_sglang(model_dir):
    """FieldPin.default is what tells a user's contrary value from sglang's
    default; red if sglang changes one of them. ServerArgs is a msgspec
    struct, not a dataclass: read the defaults off an unresolved record."""
    sa = _unresolved(model_dir)
    for pin in plugin.FIELD_PINS:
        assert getattr(sa, pin.field) == pin.default, pin.field
    assert getattr(sa, plugin.DSV4_ATTN_FIELD) == "auto"


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


@pytest.mark.parametrize("name", ["NVSHMEM_DISABLE_NCCL", "TORCHDYNAMO_DISABLE"])
def test_env_pin_is_present(name):
    """Dropping one of these costs seconds per engine start or breaks replay;
    removing it must fail a test."""
    assert {pin.name: pin.value for pin in plugin.ENV_PINS}.get(name) == "1"


def test_env_pins_are_inherited_by_spawned_processes(clean_pin_env):
    plugin.apply_env_pins()
    names = [pin.name for pin in plugin.ENV_PINS]
    result = run_python(
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
    contrary = "0" if pin.value == "1" else "1"
    os.environ[pin.name] = contrary
    with pytest.raises(
        ValueError, match=f"{pin.name}='{contrary}' \\(Foundry needs {pin.value}\\)"
    ):
        plugin.apply_env_pins()


def test_nccl_cumem_is_pinned_once_the_a2a_backend_is_known(clean_pin_env):
    plugin.apply_env_pins(SimpleNamespace(moe_a2a_backend="deepep"))
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
