# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The VMM allocation region (1 GPU): placement, preallocation, stop / resume,
live ranges and the allocation cursor. Each case runs in a child process with
libcuda_hook.so preloaded.

    pytest tests/core/test_allocation_region.py -q -s

The allocation timing benchmarks are in tools/bench/bench_preallocation.py.
"""

import json
import os
import sys

import pytest
import torch
from _helpers import run_phase

BASE_ADDR = 0x500000000000
REGION_SIZE = 1 << 30
MB = 1 << 20


def _in_region(t, base=BASE_ADDR, size=REGION_SIZE):
    return base <= t.data_ptr() and t.data_ptr() + t.numel() * t.element_size() <= base + size


def _assert_in_region(tensors, base=BASE_ADDR, size=REGION_SIZE):
    for i, t in enumerate(tensors):
        assert _in_region(t, base, size), (
            f"tensor {i} at 0x{t.data_ptr():x} is outside [0x{base:x}, 0x{base + size:x})"
        )


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _alloc_three():
    # 4 MB, 16 MB, 1 MB: different caching-allocator size classes.
    return [
        torch.randn(1000, 1000, device="cuda"),
        torch.randn(2000, 2000, device="cuda"),
        torch.randn(500, 500, device="cuda"),
    ]


def _run_placement(archive, api, prealloc):
    """Allocations land inside the region, through either API, with or without
    a preallocated span."""
    import foundry as fdry

    torch.cuda.init()
    prealloc_size = "128MB" if prealloc == "prealloc" else None

    def check():
        tensors = _alloc_three()
        _assert_in_region(tensors)
        assert tensors[0].data_ptr() < tensors[1].data_ptr(), "region addresses must increase"
        del tensors
        torch.cuda.empty_cache()

    if api == "ctx":
        with fdry.allocation_region(BASE_ADDR, "1GB", prealloc_size=prealloc_size):
            check()
    else:
        fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
        try:
            if prealloc_size is not None:
                assert fdry.preallocate_region(fdry.parse_size(prealloc_size)), "preallocation"
            check()
            if prealloc_size is not None:
                fdry.free_preallocated_region()
        finally:
            fdry.stop_allocation_region()
    print("[TEST] placement OK")


def _run_size_formats(archive):
    """Region bases as int or hex string, sizes as strings or bytes."""
    import foundry as fdry

    torch.cuda.init()
    cases = [
        (0x7F0000000000, "1GB"),
        (0x7F1000000000, "512MB"),
        (0x7F2000000000, "100MB"),
        ("0x7f3000000000", 50 * MB),
    ]
    for base, size in cases:
        base_int = int(base, 16) if isinstance(base, str) else base
        with fdry.allocation_region(base, size):
            t = torch.randn(100, 100, device="cuda")
            _assert_in_region([t], base_int, fdry.parse_size(size))
            del t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[TEST] size formats OK")


def _run_fallback(archive):
    """An allocation past the end of the preallocated span is mapped on demand
    (slow path), in the region and contiguous with the span."""
    import foundry as fdry

    torch.cuda.init()
    span = 32 * MB
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    try:
        start = fdry.get_current_alloc_offset()
        assert fdry.preallocate_region(span), "preallocation"
        a = torch.empty(span, dtype=torch.uint8, device="cuda")  # fills the span (fast path)
        b = torch.empty(span, dtype=torch.uint8, device="cuda")  # past it (slow path)
        _assert_in_region([a, b])
        assert a.data_ptr() == BASE_ADDR + start, hex(a.data_ptr())
        assert b.data_ptr() == a.data_ptr() + span, "slow path must continue at the span end"
        a.fill_(1)
        b.fill_(2)
        torch.cuda.synchronize()
        assert int(a.sum().item()) == span and int(b.sum().item()) == 2 * span
        del a, b
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()
    finally:
        fdry.stop_allocation_region()
    print("[TEST] fallback OK")


def _run_stop_resume(archive):
    """stop_allocation_region routes allocations (here a private MemPool) out
    of the region; resume_allocation_region continues at the region cursor."""
    import foundry as fdry

    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    t1 = torch.full((2 * MB,), 1.0, device="cuda")
    fdry.stop_allocation_region()
    pool = torch.cuda.memory.MemPool()
    with torch.cuda.use_mem_pool(pool):
        t2 = torch.full((2 * MB,), 2.0, device="cuda")
    fdry.resume_allocation_region()
    t3 = torch.full((2 * MB,), 3.0, device="cuda")
    _assert_in_region([t1, t3])
    assert not _in_region(t2), f"t2 at 0x{t2.data_ptr():x} was allocated while stopped"
    assert t3.data_ptr() > t1.data_ptr(), "resume must continue above the earlier allocations"
    fdry.stop_allocation_region()
    print("[TEST] stop/resume OK")


def _offsets_prefix():
    import foundry as fdry

    torch.cuda.init()
    torch.cuda.set_device(0)
    fdry.set_allocation_region(BASE_ADDR, REGION_SIZE)
    return fdry.get_current_alloc_offset()


def _covered(ranges, off, size):
    return any(o <= off and off + size <= o + n for o, n in ranges)


def _run_record(archive):
    """SAVE side of the sglang LOAD memory restore: A, B, C allocated, A freed;
    the live ranges cover B and C only."""
    import foundry as fdry

    start = _offsets_prefix()
    a = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    b = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    a_off = a.data_ptr() - BASE_ADDR
    del a
    torch.cuda.empty_cache()
    ranges = [list(r) for r in fdry.get_live_region_ranges()]
    final = fdry.get_current_alloc_offset()
    b_off, c_off = b.data_ptr() - BASE_ADDR, c.data_ptr() - BASE_ADDR
    assert not any(o <= a_off < o + n for o, n in ranges), f"freed A still live: {ranges}"
    assert _covered(ranges, b_off, b.numel()) and _covered(ranges, c_off, c.numel()), ranges
    assert final >= c_off + c.numel(), (final, c_off)
    with open(os.path.join(archive, "layout.json"), "w") as f:
        json.dump({"start": start, "ranges": ranges, "final": final, "b": b_off, "c": c_off}, f)
    fdry.stop_allocation_region()
    print("[TEST] record OK")


def _run_replay(archive):
    """LOAD side: map only the recorded live ranges, skip A's hole with
    set_current_alloc_offset, and B, C land at their SAVE addresses."""
    import foundry as fdry

    with open(os.path.join(archive, "layout.json")) as f:
        rec = json.load(f)
    start = _offsets_prefix()
    assert start == rec["start"], (start, rec["start"])
    ranges = [tuple(r) for r in rec["ranges"]]
    assert fdry.preallocate_ranges(ranges, rec["final"]), "preallocate_ranges"
    assert fdry.get_current_alloc_offset() == start, "preallocate_ranges must not move the cursor"
    fdry.set_current_alloc_offset(rec["b"])
    assert fdry.get_current_alloc_offset() == rec["b"]
    fdry.set_current_alloc_offset(start)  # backwards: refused, cursor unchanged
    assert fdry.get_current_alloc_offset() == rec["b"]
    b = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    assert b.data_ptr() - BASE_ADDR == rec["b"], hex(b.data_ptr())
    assert c.data_ptr() - BASE_ADDR == rec["c"], hex(c.data_ptr())
    assert fdry.get_current_alloc_offset() == rec["final"]
    assert [tuple(r) for r in fdry.get_live_region_ranges()] == ranges
    b.fill_(1)
    c.fill_(2)
    torch.cuda.synchronize()
    assert int(b.sum().item()) == b.numel() and int(c.sum().item()) == 2 * c.numel()
    del b, c
    torch.cuda.empty_cache()
    fdry.free_preallocated_region()
    fdry.stop_allocation_region()
    print("[TEST] replay OK")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prealloc", ["no_prealloc", "prealloc"])
@pytest.mark.parametrize("api", ["ctx", "set_stop"])
def test_allocations_land_in_region(workdir, api, prealloc):
    run_phase(__file__, "placement", workdir, api, prealloc)


def test_region_size_formats(workdir):
    run_phase(__file__, "size-formats", workdir)


def test_fallback_past_the_preallocated_span(workdir):
    run_phase(__file__, "fallback", workdir)


def test_stop_and_resume(workdir):
    run_phase(__file__, "stop-resume", workdir)


def test_live_ranges_and_cursor_restore_the_layout(workdir):
    run_phase(__file__, "record", workdir)
    run_phase(__file__, "replay", workdir)


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {
        "--placement": _run_placement,
        "--size-formats": _run_size_formats,
        "--fallback": _run_fallback,
        "--stop-resume": _run_stop_resume,
        "--record": _run_record,
        "--replay": _run_replay,
    }[mode](archive, *rest)
