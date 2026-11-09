# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The pins and rejections (pins.FIELD_PINS / ENV_PINS,
reject_unsupported_features; sglang reaches them through api.pin_server_args
and api.validate_resolved_server_args): flags and environment variables that
select state Foundry cannot replay, and features it cannot persist. Every
reject raises api.Unsupported, on which sglang's adapter falls back natively.

CPU only (no GPU, no server, no weights): records are built on a mini
config.json with device="cuda", like sglang's own
test/registered/unit/server_args/test_resolution_declarations.py.

    pytest tests/sgl/test_pins.py -v
"""

import os
import re
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from _sglang_helpers import run_python  # noqa: E402
from foundry.integration.sglang import api, pins  # noqa: E402
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
    for pin in pins.FIELD_PINS:
        assert getattr(sa, pin.field) == pin.default, pin.field
    assert getattr(sa, pins.DSV4_ATTN_FIELD) == "auto"


def test_non_trtllm_dsv4_backend_is_kept(model_dir):
    sa = _unresolved(model_dir, dsv4_attn_backend="flashmla")
    pins.pin_unreplayable_fields(sa)
    assert resolution_result(sa, pins.DSV4_ATTN_FIELD) == "flashmla"


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
    with pytest.raises(api.Unsupported, match=match):
        pins.pin_unreplayable_fields(_unresolved(model_dir, **fields))


def test_later_step_that_undoes_a_strict_pin_is_rejected(model_dir):
    from sglang.srt.arg_groups.overrides import declare_resolution

    sa = _unresolved(model_dir)
    pins.pin_unreplayable_fields(sa)
    declare_resolution(sa, "some_model_override", enable_two_batch_overlap=True)
    with pytest.raises(api.Unsupported, match="set by some_model_override"):
        pins.validate_pinned_fields(sa)


def test_torch_symm_mem_may_be_turned_off_by_sglang(model_dir, capsys):
    """Deterministic inference turns torch symm-mem off (NCCL fallback, still
    replayable): logged, not rejected."""
    from sglang.srt.arg_groups.overrides import declare_resolution

    sa = _unresolved(model_dir)
    pins.pin_unreplayable_fields(sa)
    declare_resolution(sa, "_handle_deterministic_inference", enable_torch_symm_mem=False)
    pins.validate_pinned_fields(sa)
    assert "enable_torch_symm_mem=False (set by _handle_deterministic_inference)" in (
        capsys.readouterr().err
    )


# ---------------------------------------------------------------------------
# Pinned environment variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, value",
    [
        ("NVSHMEM_DISABLE_NCCL", "1"),
        ("TORCHDYNAMO_DISABLE", "1"),
        ("SGLANG_ENABLE_DCP_PUSH_COMBINE", "0"),
    ],
)
def test_env_pin_is_present(name, value):
    """Dropping one of these costs seconds per engine start or breaks replay;
    removing it must fail a test."""
    assert {pin.name: pin.value for pin in pins.ENV_PINS}.get(name) == value


def test_env_pins_are_inherited_by_spawned_processes(clean_pin_env):
    pins.apply_env_pins()
    names = [pin.name for pin in pins.ENV_PINS]
    result = run_python(
        f"""
        import os
        print(",".join(os.environ.get(n, "UNSET") for n in {names!r}))
        """
    )
    assert result.stdout.strip().split(",") == [pin.value for pin in pins.ENV_PINS]


def test_env_pins_are_idempotent_and_quiet_in_descendants(clean_pin_env, capsys):
    pins.apply_env_pins()
    capsys.readouterr()
    pins.apply_env_pins()  # a scheduler re-asserting the inherited values
    assert "[Foundry] pin:" not in capsys.readouterr().err


def test_env_pin_already_set_to_the_pinned_value_is_kept(clean_pin_env):
    os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "false"
    pins.apply_env_pins()
    assert os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"


@pytest.mark.parametrize("pin", pins.ENV_PINS, ids=lambda pin: pin.name)
def test_explicit_contrary_env_is_rejected(clean_pin_env, pin):
    """Rejected before any variable is set: the native fallback inherits the
    environment as the user gave it."""
    contrary = "0" if pin.value == "1" else "1"
    os.environ[pin.name] = contrary
    with pytest.raises(
        api.Unsupported, match=f"{pin.name}='{contrary}' \\(Foundry needs {pin.value}\\)"
    ) as info:
        pins.apply_env_pins()
    assert info.value.step == "environment"
    assert all(other.name not in os.environ for other in pins.ENV_PINS if other is not pin)


def test_nccl_cumem_is_pinned_once_the_a2a_backend_is_known(clean_pin_env):
    pins.apply_env_pins(SimpleNamespace(moe_a2a_backend="deepep"))
    assert os.environ[pins.NCCL_CUMEM_PIN.name] == "0"


def test_explicit_nccl_cumem_is_rejected(clean_pin_env):
    os.environ[pins.NCCL_CUMEM_PIN.name] = "1"
    with pytest.raises(api.Unsupported, match="NCCL_CUMEM_ENABLE='1'"):
        pins.apply_env_pins(SimpleNamespace(moe_a2a_backend="none"))


def test_nccl_cumem_is_left_to_sglang_for_deepep_v2(clean_pin_env):
    """DeepEP v2's NCCL windows need cuMem; sglang sets it to 1 for deepep_v2."""
    os.environ[pins.NCCL_CUMEM_PIN.name] = "1"
    pins.apply_env_pins(SimpleNamespace(moe_a2a_backend="deepep_v2"))
    assert os.environ[pins.NCCL_CUMEM_PIN.name] == "1"


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
    pins.reject_unsupported_features(_ns())
    pins.reject_unsupported_features(_ns(pp_size=2))  # PP is allowed
    # Elastic EP without recapture (max_ep_size == tp_size) is allowed.
    pins.reject_unsupported_features(_ns(elastic_ep_backend="mooncake", max_ep_size=4))
    # EAGLE-family speculative decoding is persisted.
    for algo in ("EAGLE", "EAGLE3", "eagle3", "NEXTN", "DFLASH", "NGRAM", "DSPARK"):
        pins.reject_unsupported_features(_ns(speculative_algorithm=algo))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(speculative_algorithm="UNO"), "speculative decoding"),
        (dict(speculative_algorithm="STANDALONE"), "speculative decoding"),
        (dict(speculative_algorithm="EAGLE", speculative_adaptive=True), "adaptive"),
        (dict(speculative_algorithm="EAGLE3", enable_multi_layer_eagle=True), "multi-layer"),
        (dict(enable_lora=True), "LoRA"),
        (dict(lora_paths=["a=/x"]), "LoRA"),
        (dict(enable_pdmux=True), "pdmux"),
    ],
)
def test_unsupported_features_are_rejected(restore_environ, overrides, match):
    with pytest.raises(api.Unsupported, match=match):
        pins.reject_unsupported_features(_ns(**overrides))


def test_elastic_recapture_is_rejected_where_the_path_exists(restore_environ):
    ns = _ns(elastic_ep_backend="mooncake", max_ep_size=8)
    with (
        mock.patch.object(pins, "_elastic_recapture_path_exists", return_value=True),
        pytest.raises(api.Unsupported, match="elastic-EP CUDA-graph recapture"),
    ):
        pins.reject_unsupported_features(ns)
    with mock.patch.object(pins, "_elastic_recapture_path_exists", return_value=False):
        pins.reject_unsupported_features(ns)


def test_post_capture_kv_sizing_is_rejected(restore_environ):
    os.environ["SGLANG_ENABLE_POST_CAPTURE_KV_SIZING"] = "1"
    with pytest.raises(api.Unsupported, match="POST_CAPTURE_KV_SIZING"):
        pins.reject_unsupported_features(_ns())


# ---------------------------------------------------------------------------
# Through sglang's resolution steps (--cuda-graph-persistence save)
# ---------------------------------------------------------------------------


@pytest.fixture
def persistence(clean_pin_env):
    """Resolution with ``cuda_graph_persistence="save"`` in this process:
    sglang's steps call api.pin_server_args and
    api.validate_resolved_server_args. api.activate (process-global config;
    test_dependency_route runs it in fresh interpreters) is marked done."""
    adapter = pytest.importorskip("sglang.srt.utils.foundry_adapter")

    with (
        mock.patch.object(api, "_active", ("save", None)),
        mock.patch.object(api, "_unsupported", None),
        mock.patch.object(adapter, "_active", adapter._NOOP),
    ):
        yield adapter


def _resolve(model_dir, **fields):
    sa = _unresolved(model_dir, cuda_graph_persistence="save", **fields)
    sa.resolve_once()
    return sa


def _resolve_unsupported(model_dir, persistence, match, **fields):
    """Resolution falls back instead of failing: the adapter is off and the
    reason names the flag. Returns the record."""
    sa = _resolve(model_dir, **fields)
    assert not persistence.get_foundry_adapter().enabled
    step, reason, _quiet = api._unsupported
    assert step == "server args"
    assert re.search(match, reason), reason
    return sa


def test_pins_decode_full_and_defaults_prefill_to_disabled(model_dir, persistence):
    from sglang.srt.model_executor.cuda_graph_config import Backend

    sa = _resolve(model_dir)
    cfg = resolution_result(sa, "cuda_graph_config")
    assert cfg.decode.backend == Backend.FULL
    assert cfg.prefill.backend == Backend.DISABLED  # not breakable, the CUDA default
    assert resolution_result(sa, "enable_profile_cuda_graph") is False
    assert resolution_result(sa, "disable_flashinfer_autotune") is True


def test_explicit_full_prefill_is_honoured(model_dir, persistence):
    from sglang.srt.model_executor.cuda_graph_config import Backend

    sa = _resolve(model_dir, cuda_graph_backend_prefill=Backend.FULL)
    assert resolution_result(sa, "cuda_graph_config").prefill.backend == Backend.FULL
    declared = dict(sa._resolved_overrides)[pins.RESOLUTION_SOURCE]
    assert "cuda_graph_backend_prefill" not in declared


@pytest.mark.parametrize(
    "field, match",
    [
        ("cuda_graph_backend_prefill", "Foundry persists full prefill graphs only"),
        ("cuda_graph_backend_decode", "Foundry persists full decode graphs only"),
    ],
)
def test_non_full_backend_flag_falls_back(model_dir, persistence, field, match):
    """Nothing is declared: the record resolves as without the flag."""
    from sglang.srt.model_executor.cuda_graph_config import Backend

    sa = _resolve_unsupported(model_dir, persistence, match, **{field: Backend.BREAKABLE})
    declared = dict(sa._resolved_overrides)
    assert pins.RESOLUTION_SOURCE not in declared
    assert pins.PIN_RESOLUTION_SOURCE not in declared


@pytest.mark.parametrize(
    "json_cfg, match",
    [
        ({"prefill": {"backend": "breakable"}}, "prefill graphs only"),
        ({"decode": {"backend": "disabled"}}, "decode graphs only"),
    ],
)
def test_cuda_graph_config_json_is_validated(model_dir, persistence, json_cfg, match):
    """JSON outranks the per-phase flags, so the flag check alone misses it.
    (Passed as the parsed dict: parse_cuda_graph_config reads ``.items()``.)"""
    _resolve_unsupported(model_dir, persistence, match, cuda_graph_config=json_cfg)


@pytest.mark.parametrize("field", ["disable_cuda_graph", "disable_decode_cuda_graph"])
def test_disable_cuda_graph_falls_back_not_overridden(model_dir, persistence, field):
    sa = _resolve_unsupported(model_dir, persistence, "disable-cuda-graph", **{field: True})
    assert pins.RESOLUTION_SOURCE not in dict(sa._resolved_overrides)


def test_profile_cuda_graph_falls_back(model_dir, persistence):
    _resolve_unsupported(
        model_dir, persistence, "enable-profile-cuda-graph", enable_profile_cuda_graph=True
    )


def test_every_field_pin_is_applied_on_sglang_defaults(model_dir, persistence, capsys):
    sa = _resolve(model_dir)
    for pin in pins.FIELD_PINS:
        assert resolution_result(sa, pin.field) == pin.value, pin.field
    assert resolution_result(sa, pins.DSV4_ATTN_FIELD) == "auto"
    declared = dict(sa._resolved_overrides)[pins.PIN_RESOLUTION_SOURCE]
    assert {pin.field for pin in pins.FIELD_PINS} <= set(declared)
    err = capsys.readouterr().err
    for pin in pins.FIELD_PINS:
        assert f"[Foundry] pin: {pin.field}={pin.value!r}" in err, pin.field


def test_pinned_value_given_explicitly_is_accepted(model_dir, persistence):
    sa = _resolve(model_dir, disable_custom_all_reduce=True, enable_torch_symm_mem=True)
    assert resolution_result(sa, "disable_custom_all_reduce") is True
    assert resolution_result(sa, "enable_torch_symm_mem") is True


@pytest.mark.parametrize("field", ["enable_symm_mem", "enable_mscclpp"])
def test_explicit_contrary_flag_falls_back_in_resolution(model_dir, persistence, field):
    """Also the graph pins stay undeclared: the conflict is found before any
    declaration (pins.check_pinnable)."""
    flag = f"--{field.replace('_', '-')}=True"
    sa = _resolve_unsupported(model_dir, persistence, flag, **{field: True})
    assert pins.RESOLUTION_SOURCE not in dict(sa._resolved_overrides)


def test_nccl_cumem_is_pinned_by_resolution(model_dir, persistence):
    _resolve(model_dir)
    assert os.environ[pins.NCCL_CUMEM_PIN.name] == "0"
