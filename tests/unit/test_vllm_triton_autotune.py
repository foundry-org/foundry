# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""CPU-only regression checks for cached Triton selection during SAVE capture."""

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.runtime import triton_heuristics

pytest.importorskip("vllm")
hooks = pytest.importorskip("foundry.integration.vllm.hooks")


class _Cache:
    def __init__(self, best):
        self.best = best

    def read_best(self, _meta, _configs):
        return self.best

    def save(self):
        pass


def _autotuner(best="fast", compiled=("slow", "fast"), hook=True, descent=False):
    cache = _Cache(best)
    return SimpleNamespace(
        fn=lambda: None,
        filename="kernel.py",
        inductor_meta={"coordinate_descent_tuning": descent},
        compile_results=[SimpleNamespace(config=x) for x in compiled],
        launchers=[SimpleNamespace(config=x) for x in compiled],
        save_cache_hook=cache.save if hook else None,
    )


@pytest.fixture
def patched(monkeypatch):
    cls = triton_heuristics.CachingAutotuner
    original = cls.autotune_to_one_config
    monkeypatch.setattr(cls, "autotune_to_one_config", lambda self: "benchmark")
    monkeypatch.setattr(triton_heuristics, "triton_config_to_hashable", lambda x: x)
    monkeypatch.setattr(
        hooks, "get_graph_extension_mode", lambda: hooks.CUDAGraphExtensionMode.SAVE
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hooks._patch_triton_autotune()
    yield cls.autotune_to_one_config
    cls.autotune_to_one_config = original


def test_capture_uses_measured_winner_without_benchmark(patched):
    tuner = _autotuner()
    assert patched(tuner) is None
    assert [x.config for x in tuner.launchers] == ["fast"]


@pytest.mark.parametrize(
    ("tuner", "message"),
    [
        (_autotuner(hook=False), "no Triton autotune cache"),
        (_autotuner(best=None), "best_config missing"),
        (_autotuner(best="different"), "does not match one compiled launcher"),
        (_autotuner(compiled=("fast", "fast")), "2 matches"),
        (_autotuner(descent=True), "coordinate-descent autotune"),
    ],
)
def test_capture_fails_explicitly_without_usable_winner(patched, tuner, message):
    with pytest.raises(RuntimeError, match=message):
        patched(tuner)


def test_outside_capture_keeps_original_autotune(patched, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert patched(_autotuner()) == "benchmark"
