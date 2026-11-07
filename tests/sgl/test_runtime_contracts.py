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
    capture_state_unsupported reads only this public surface."""
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe.token_dispatcher import base as dispatcher_base
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    assert callable(BaseRunner._prepare_capture_state)
    assert callable(dispatcher_base.iter_dispatchers)
    assert isinstance(dispatcher_base.BaseDispatcher.prebuild_pending, property)
    assert isinstance(dispatcher_base.BaseDispatcher.implements_prebuild, property)
    assert isinstance(LogitsProcessor.logits_gatherer_pending, property)


def test_barrier_work_is_waited_on_at_the_call():
    """A barrier Work dropped unwaited leaves its 4-byte CUDA tensor to the NCCL
    watchdog thread, whose release time shifts the host buffers of the graph
    layout; the wrapper must wait before returning the Work."""
    from foundry.integration.sglang.hooks import _waited

    class Work:
        waits = 0

        def wait(self):
            self.waits += 1

    work = Work()
    assert _waited(lambda self, *args, **kwargs: work)(None) is work
    assert work.waits == 1
    assert _waited(lambda self: None)(None) is None


def test_process_group_exposes_barrier():
    """_wait_on_barrier_works wraps ``barrier`` on torch's ProcessGroup class."""
    from torch._C import _distributed_c10d as c10d

    assert "barrier" in vars(c10d.ProcessGroup)


def test_cyclic_gc_is_held_until_the_layout_starts():
    """Between before_parallel_init and mark_layout_start only explicit collections
    may free reference cycles; the collector's own timing moves allocations."""
    import gc

    from foundry.integration.sglang import runtime as rt

    was_enabled = gc.isenabled()
    gc.enable()
    try:
        rt.hold_cyclic_gc()
        assert not gc.isenabled()
        rt.release_cyclic_gc()
        assert gc.isenabled()
        gc.disable()
        rt.hold_cyclic_gc()  # already off: not ours to turn back on
        rt.release_cyclic_gc()
        assert not gc.isenabled()
    finally:
        gc.enable() if was_enabled else gc.disable()
