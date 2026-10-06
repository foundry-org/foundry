# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Host-side pieces of the SAVE warm-up pool (warmup_pool.py): loop state,
the trace knob, retained-block reporting and the FlashInfer detection behind
hybrid wrappers. No GPU.

    pytest tests/integration/sglang/test_sglang_warmup_pool.py -v
"""

from types import SimpleNamespace

import pytest

warmup_pool = pytest.importorskip("foundry.integration.sglang.warmup_pool")


def test_flashinfer_updater_found_behind_a_hybrid_wrapper():
    updater = SimpleNamespace(num_qo_heads=16)
    flashinfer = SimpleNamespace(indices_updater_decode=updater)
    hybrid = SimpleNamespace(full_attn_backend=flashinfer, linear_attn_backend=object())
    assert warmup_pool._flashinfer_decode_updater(flashinfer) is updater
    assert warmup_pool._flashinfer_decode_updater(hybrid) is updater
    assert warmup_pool._flashinfer_decode_updater(SimpleNamespace()) is None


def test_outside_the_loop_nothing_is_active():
    assert not warmup_pool.active()
    with pytest.raises(RuntimeError, match="outside run_capture_loop"):
        warmup_pool.warm_up("bs1", lambda: None, None, None)


def test_trace_knob(monkeypatch):
    monkeypatch.delenv("FOUNDRY_SGLANG_WARM_POOL_TRACE", raising=False)
    assert not warmup_pool.pool_trace_enabled()
    monkeypatch.setenv("FOUNDRY_SGLANG_WARM_POOL_TRACE", "1")
    assert warmup_pool.pool_trace_enabled()


def test_block_stack_formats_recorded_frames():
    block = {
        "frames": [
            {"filename": "a.py", "line": 3, "name": "f"},
            {"filename": "b.cpp", "line": 0, "name": "g"},
        ]
    }
    assert warmup_pool._block_stack(block) == "a.py:3 f\n      b.cpp:0 g"
    assert warmup_pool._block_stack({}) == ""


def test_loop_without_save_or_load_runs_as_it_is(monkeypatch):
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.NONE
    )
    runs = []
    assert warmup_pool.run_capture_loop(object(), lambda: runs.append(1) or "r") == "r"
    assert runs == [1]
    assert not warmup_pool.active()


def test_load_runs_the_loop_once_after_the_bootstrap(monkeypatch):
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.LOAD
    )
    monkeypatch.setattr(warmup_pool, "bootstrap_persistent_resources", lambda runner: ["x"])
    monkeypatch.setattr(warmup_pool.rt, "log_alloc_offset", lambda label: None)
    seen = []
    assert (
        warmup_pool.run_capture_loop(object(), lambda: seen.append(warmup_pool.active()) or "r")
        == "r"
    )
    assert seen == [True]
    assert not warmup_pool.active()


def test_nested_loops_are_rejected(monkeypatch):
    monkeypatch.setattr(
        warmup_pool, "get_graph_extension_mode", lambda: warmup_pool.CUDAGraphExtensionMode.LOAD
    )
    monkeypatch.setattr(warmup_pool, "bootstrap_persistent_resources", lambda runner: [])
    monkeypatch.setattr(warmup_pool.rt, "log_alloc_offset", lambda label: None)

    def inner():
        return warmup_pool.run_capture_loop(object(), lambda: None)

    with pytest.raises(RuntimeError, match="nested"):
        warmup_pool.run_capture_loop(object(), inner)
    assert not warmup_pool.active()
