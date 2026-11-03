# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime contracts that hold on both sglang routes.

pytest tests/sgl/test_runtime_contracts.py -v
"""

import pytest

pytest.importorskip("sglang")


def test_triton_key_is_cached_process_wide():
    """The triton_key prewarm thread only helps if the main thread's later
    call hits the same lru_cache."""
    pytest.importorskip("triton")
    from foundry.integration.sglang.hooks import _resolve_triton_key

    assert hasattr(_resolve_triton_key(), "cache_info")


def test_foundry_graphs_support_backend_cleanup():
    """Upstream main's FullCudaGraphBackend.cleanup() calls graph.reset() on
    every graph in ``_graphs``; the graphs Foundry places there are
    foundry.ops.CUDAGraph objects (the fork base's cleanup() does not reset)."""
    from foundry import ops

    assert callable(getattr(ops.CUDAGraph, "reset", None))


def test_sglang_prebuilds_capture_state_in_its_warmup():
    """The dispatcher buffers / expert mappings and the logits all-gather state
    are sglang's to build before capture (BaseRunner._prepare_capture_state);
    check_capture_state_prebuilt reads only this public surface."""
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe.token_dispatcher import base as dispatcher_base
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    assert callable(BaseRunner._prepare_capture_state)
    assert callable(dispatcher_base.iter_dispatchers)
    assert isinstance(dispatcher_base.BaseDispatcher.prebuild_pending, property)
    assert isinstance(LogitsProcessor.logits_gatherer_pending, property)


def test_capture_state_check_names_unbuilt_dispatchers():
    """A dispatcher still pending at the first capture scope fails loudly by
    type name; once prebuilt the check passes."""
    import torch
    from foundry.integration.sglang.graph_ops import check_capture_state_prebuilt
    from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher

    class _LazyDispatcher(BaseDispatcher):
        built = False

        def dispatch(self, hidden_states, topk_output):
            raise NotImplementedError

        def combine(self, combine_input):
            raise NotImplementedError

        @property
        def prebuild_pending(self):
            return not self.built

        def prebuild(self):
            self.built = True

    model = torch.nn.Module()
    model.moe = torch.nn.Module()
    model.moe.dispatcher = _LazyDispatcher()
    with pytest.raises(RuntimeError, match="_LazyDispatcher"):
        check_capture_state_prebuilt(model)
    model.moe.dispatcher.prebuild()
    check_capture_state_prebuilt(model)
