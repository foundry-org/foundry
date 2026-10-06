# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Host-side pieces of the SAVE warm-up pool (warmup_pool.py): the saved-graph
pointer audit and the FlashInfer detection behind hybrid wrappers. No GPU.

    pytest tests/integration/sglang/test_sglang_warmup_pool.py -v
"""

import json
from types import SimpleNamespace

import pytest

warmup_pool = pytest.importorskip("foundry.integration.sglang.warmup_pool")

_POOL = [(0x7F0000000000, 0x7F0000100000)]
_IN = 0x7F0000000040
_OUT = 0x600000001000


def _word(address: int) -> str:
    return address.to_bytes(8, "little").hex()


def _graph(tmp_path, name, nodes):
    path = tmp_path / name
    path.write_text(json.dumps({"nodes": nodes}))
    return path


def test_clean_graph_passes(tmp_path):
    g = _graph(
        tmp_path,
        "graph_0_FULL_t1_r1_UX_pcN.json",
        [{"type": "KernelNode", "function_name": "k", "params": [{"value_hex": _word(_OUT)}]}],
    )
    assert warmup_pool.audit_graph_pointers([g], _POOL) == (1, 1)


def test_kernel_argument_into_the_pool_fails(tmp_path):
    g = _graph(
        tmp_path,
        "graph_0_FULL_t1_r1_UX_pcN.json",
        [{"type": "KernelNode", "function_name": "k", "params": [{"value_hex": _word(_IN)}]}],
    )
    with pytest.raises(RuntimeError, match="references into the released warm-up pool"):
        warmup_pool.audit_graph_pointers([g], _POOL)


def test_memcpy_address_into_the_pool_fails(tmp_path):
    g = _graph(tmp_path, "graph_1.json", [{"type": "MemcpyNode", "dstDevice": hex(_IN)}])
    with pytest.raises(RuntimeError):
        warmup_pool.audit_graph_pointers([g], _POOL)


def test_qknorm_scalar_words_are_not_pointers(tmp_path):
    words = [_OUT] * 9
    words[2] = _IN  # a scalar/padding word of QKNormParams that looks like an address
    hexed = "".join(_word(w) for w in words)
    g = _graph(
        tmp_path,
        "graph_2.json",
        [
            {
                "type": "KernelNode",
                "function_name": "fused_qknorm_warp<QKNormParams>",
                "params": [{"value_hex": hexed}],
            }
        ],
    )
    assert warmup_pool.audit_graph_pointers([g], _POOL) == (1, 1)
    words[0] = _IN  # a real pointer field
    g.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "type": "KernelNode",
                        "function_name": "fused_qknorm_warp<QKNormParams>",
                        "params": [{"value_hex": "".join(_word(w) for w in words)}],
                    }
                ]
            }
        )
    )
    with pytest.raises(RuntimeError):
        warmup_pool.audit_graph_pointers([g], _POOL)


def test_files_without_nodes_fail(tmp_path):
    g = _graph(tmp_path, "graph_3.json", [])
    with pytest.raises(RuntimeError, match="no graph nodes"):
        warmup_pool.audit_graph_pointers([g], _POOL)


def test_flashinfer_updater_found_behind_a_hybrid_wrapper():
    updater = SimpleNamespace(num_qo_heads=16)
    flashinfer = SimpleNamespace(indices_updater_decode=updater)
    hybrid = SimpleNamespace(full_attn_backend=flashinfer, linear_attn_backend=object())
    assert warmup_pool._flashinfer_decode_updater(flashinfer) is updater
    assert warmup_pool._flashinfer_decode_updater(hybrid) is updater
    assert warmup_pool._flashinfer_decode_updater(SimpleNamespace()) is None


def test_outside_the_two_pass_loop_there_is_no_phase():
    assert warmup_pool.current_phase() is None


def test_warm_policy_defaults_to_two_pass(monkeypatch):
    monkeypatch.delenv("FOUNDRY_SGLANG_WARM_POLICY", raising=False)
    assert warmup_pool.warm_policy() == warmup_pool.TWO_PASS
    monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POLICY", "")
    assert warmup_pool.warm_policy() == warmup_pool.TWO_PASS


def test_warm_policy_per_shape(monkeypatch):
    monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POLICY", "per_shape")
    assert warmup_pool.warm_policy() == warmup_pool.PER_SHAPE


def test_unknown_warm_policy_is_rejected(monkeypatch):
    monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POLICY", "per_bs")
    with pytest.raises(ValueError, match="FOUNDRY_SGLANG_WARM_POLICY"):
        warmup_pool.warm_policy()


def test_loop_without_save_or_load_runs_once_under_either_policy(monkeypatch):
    """Mode NONE: the loop runs once, no bootstrap, no phase."""
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.NONE
    )
    for policy in (warmup_pool.TWO_PASS, warmup_pool.PER_SHAPE):
        monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POLICY", policy)
        runs = []
        assert warmup_pool.run_capture_loop(object(), lambda: runs.append(1) or "r") == "r"
        assert runs == [1]
        assert warmup_pool.current_phase() is None


def test_per_shape_runs_the_loop_once_in_warm_and_capture_phase(monkeypatch):
    """LOAD under per_shape: one pass, phase WARM_AND_CAPTURE, no pool release."""
    monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POLICY", "per_shape")
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.LOAD
    )
    monkeypatch.setattr(warmup_pool, "bootstrap_persistent_resources", lambda runner: [])
    monkeypatch.setattr(warmup_pool.rt, "log_alloc_offset", lambda label: None)
    monkeypatch.setattr(warmup_pool, "get_config", lambda: None)
    phases = []
    warmup_pool.run_capture_loop(object(), lambda: phases.append(warmup_pool.current_phase()))
    assert phases == [warmup_pool.WARM_AND_CAPTURE]
    assert warmup_pool.current_phase() is None


def test_two_pass_runs_prepare_then_capture(monkeypatch):
    """LOAD under two_pass: preparation pass then capture pass."""
    monkeypatch.delenv("FOUNDRY_SGLANG_WARM_POLICY", raising=False)
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.LOAD
    )
    monkeypatch.setattr(warmup_pool, "bootstrap_persistent_resources", lambda runner: [])
    monkeypatch.setattr(warmup_pool.rt, "log_alloc_offset", lambda label: None)
    monkeypatch.setattr(warmup_pool, "get_config", lambda: None)
    phases = []
    warmup_pool.run_capture_loop(object(), lambda: phases.append(warmup_pool.current_phase()))
    assert phases == [warmup_pool.PREPARE, warmup_pool.CAPTURE]
