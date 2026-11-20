# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""foundry.ops without pybind11: handle lifetime, concurrent calls (torch releases the GIL
around every op), argument shapes, and the guard against a second library copy. No GPU.

    pytest tests/unit/test_ops_handles.py -q
"""

import gc
import os
import shutil
import subprocess
import sys
import threading
import weakref

import pytest
import torch
from foundry import ops

graph_mod = __import__("foundry.graph", fromlist=["CUDAGraph"])


def _live():
    return tuple(torch.ops.foundry.live_handles())


def test_handle_lifetime_100k_cycles():
    gc.collect()
    graphs0, pending0 = _live()
    for i in range(100_000):
        cls = graph_mod.CUDAGraph if i % 2 else ops.CUDAGraph
        g = cls(keep_graph=bool(i % 3))
        assert _live()[0] == graphs0 + 1
        ref = weakref.ref(g)
        del g
        assert ref() is None
    gc.collect()
    assert _live() == (graphs0, pending0)


def test_reset_keeps_the_handle_and_destroys_once():
    graphs0, _ = _live()
    g = graph_mod.CUDAGraph()
    g.reset()
    g.reset()
    assert _live()[0] == graphs0 + 1
    fin = g._graph_finalizer
    del g
    assert not fin.alive
    assert _live()[0] == graphs0


def test_subclass_and_base_instances():
    g = graph_mod.CUDAGraph(True)
    assert isinstance(g, ops.CUDAGraph)
    g.extra = 1  # instance __dict__ as before
    assert ops.get_graph_ptr(g) != 0
    assert ops.get_graph_ptr(g) != ops.get_graph_ptr(ops.CUDAGraph())
    with pytest.raises(RuntimeError, match="without a preceding successful capture"):
        g.pool()
    assert g.loaded_events() == []
    assert '"enabled":false' in g.exec_share_stats()


def test_pending_graph_loads_is_not_constructible():
    with pytest.raises(TypeError):
        ops.PendingGraphLoads()
    with pytest.raises(TypeError):
        ops.CUDAGraph.finish_graph_loads(object())


def test_value_shapes():
    assert ops._i64(2**64 - 1) == -1
    assert ops._i64(2**63) == -(2**63)
    assert ops._i64(5) == 5
    assert ops.get_live_region_ranges() == []  # no region on this thread
    assert ops.get_current_alloc_offset() == 0
    assert ops.recorded_allocation_contains(0x1000, 8) is False
    with pytest.raises(RuntimeError, match="must be >= 0"):
        ops.recorded_allocation_contains(0x1000, -1)


def test_bind_events_rejects_other_objects():
    g = ops.CUDAGraph()
    with pytest.raises(RuntimeError, match="expected a torch.cuda.Event or an int handle"):
        g.bind_events({0: object()})


def test_concurrent_calls():
    """Ops touching shared native state from 8 Python threads at once (each op runs without
    the GIL): handle registry, recorded-allocation map, per-thread region state."""
    graphs0, pending0 = _live()
    errors = []
    barrier = threading.Barrier(8)

    def work():
        try:
            barrier.wait()
            for _ in range(5_000):
                g = graph_mod.CUDAGraph()
                ops.get_graph_ptr(g)
                g.loaded_events()
                g.exec_share_stats()
                assert ops.recorded_allocation_contains(0x1000, 8) is False
                assert ops.get_current_alloc_offset() == 0
                assert ops.get_live_region_ranges() == []
                del g
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    gc.collect()
    assert _live() == (graphs0, pending0)


def test_same_library_loads_once():
    path = os.path.join(os.path.dirname(ops.__file__), ops.LIBRARY_NAME)
    assert ops.load_library(path) is ops._C
    assert os.path.realpath(torch.ops.foundry.library_path()) == os.path.realpath(path)


def test_second_copy_is_refused(tmp_path):
    """A copy of the library from another path would crash inside torch.ops.load_library
    (same namespace registered twice); foundry refuses it first. Run in a child process so
    a regression cannot take the test session down."""
    src = os.path.join(os.path.dirname(ops.__file__), ops.LIBRARY_NAME)
    copy = tmp_path / ops.LIBRARY_NAME
    shutil.copy2(src, copy)
    code = (
        "import foundry.ops as o\n"
        "try:\n"
        f"    o.load_library({str(copy)!r})\n"
        "except ImportError as e:\n"
        "    print('REFUSED', e)\n"
        "else:\n"
        "    print('LOADED')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, (res.returncode, res.stderr[-2000:])
    assert "REFUSED" in res.stdout and "already registered by" in res.stdout, res.stdout
