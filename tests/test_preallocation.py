# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""
Test script for the preallocation API.

This tests the preallocate_region() API which enables fast allocations
by physically allocating memory upfront, allowing subsequent allocations
to just bump a pointer without VMM calls.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch


def _get_hook_so_path():
    import importlib.util

    spec = importlib.util.find_spec("foundry.ops")
    if not spec or not spec.origin:
        raise RuntimeError("foundry.ops not found; ensure setup.py develop/pip install completed")

    ops_so_path = Path(spec.origin).resolve()
    hook_so_path = ops_so_path.parent / "libcuda_hook.so"

    if not hook_so_path.exists():
        raise RuntimeError(f"libcuda_hook.so not found at {hook_so_path}")

    return str(hook_so_path)


def _run_core_basic_preallocation():
    import foundry as fdry

    torch.cuda.init()

    base_addr = 0x500000000000
    region_size = 1 * 1024 * 1024 * 1024  # 1GB
    prealloc_size = 256 * 1024 * 1024  # 256MB

    print("[TEST] Testing basic preallocation")

    fdry.set_allocation_region(base_addr, region_size)

    try:
        success = fdry.preallocate_region(prealloc_size)
        print(f"[TEST] Preallocation success: {success}")
        assert success, "Preallocation should succeed"

        tensors = []
        for i in range(5):
            t = torch.empty(1024, 1024, device="cuda")  # ~4MB each
            tensors.append(t)
            print(f"[TEST] Allocated tensor {i}: ptr=0x{t.data_ptr():x}")

            # Verify address is in region
            assert t.data_ptr() >= base_addr, "Address below region base"
            assert t.data_ptr() < base_addr + region_size, "Address above region end"

        del tensors
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()

    finally:
        fdry.stop_allocation_region()

    print("[TEST] test_basic_preallocation PASSED")


def _run_core_context_manager():
    import foundry as fdry

    torch.cuda.init()

    base_addr = 0x510000000000

    print("[TEST] Testing allocation_region with prealloc_size")

    with fdry.allocation_region(base_addr, "1GB", prealloc_size="128MB"):
        tensors = []
        for i in range(3):
            t = torch.empty(512, 512, device="cuda")
            tensors.append(t)
            print(f"[TEST] Allocated tensor {i}: ptr=0x{t.data_ptr():x}")

            assert t.data_ptr() >= base_addr, "Address below region base"

    print("[TEST] test_context_manager PASSED")


def _run_core_fallback():
    import foundry as fdry

    torch.cuda.init()

    base_addr = 0x520000000000
    region_size = 1 * 1024 * 1024 * 1024  # 1GB
    prealloc_size = 8 * 1024 * 1024  # Only 8MB preallocated

    print("[TEST] Testing fallback to slow path when preallocation exhausted")

    fdry.set_allocation_region(base_addr, region_size)

    try:
        success = fdry.preallocate_region(prealloc_size)
        assert success, "Preallocation should succeed"

        # Allocate more than preallocated (5 * 4MB = 20MB > 8MB)
        tensors = []
        for i in range(5):
            t = torch.empty(1024, 1024, device="cuda")  # ~4MB
            tensors.append(t)
            print(f"[TEST] Allocated tensor {i}: ptr=0x{t.data_ptr():x}")

        print(
            "[TEST] Successfully allocated beyond preallocated region, you are expected to see an error message from the hook"
        )

        del tensors
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()

    finally:
        fdry.stop_allocation_region()

    print("[TEST] test_fallback_to_slow_path PASSED")


def _run_core_performance():
    import random

    import foundry as fdry

    torch.cuda.init()
    torch.cuda.synchronize()

    random.seed(42)

    print("[TEST] Performance comparison: native vs VMM vs VMM+prealloc")

    # Generate allocation sizes
    num_allocs = 30
    alloc_sizes = []
    total_size = 0
    for _ in range(num_allocs):
        size = int(2 ** random.uniform(21, 30))  # 2MB to 1GB
        size = max(2 * 1024 * 1024, min(1024 * 1024 * 1024, size))
        size = ((size + (2 * 1024 * 1024 - 1)) // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
        alloc_sizes.append(size)
        total_size += size

    print(f"[TEST] Total: {total_size / (1024**3):.2f} GB across {num_allocs} allocations")

    # Test 1: Native cudaMalloc
    torch.cuda.synchronize()
    start = time.perf_counter()
    tensors = [torch.empty(s // 4, dtype=torch.float32, device="cuda") for s in alloc_sizes]
    torch.cuda.synchronize()
    native_time = time.perf_counter() - start
    del tensors
    torch.cuda.empty_cache()

    # Test 2: VMM slow path
    region_size = 16 * 1024 * 1024 * 1024
    fdry.set_allocation_region(0x530000000000, region_size)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        tensors = [torch.empty(s // 4, dtype=torch.float32, device="cuda") for s in alloc_sizes]
        torch.cuda.synchronize()
        slow_time = time.perf_counter() - start
        del tensors
        torch.cuda.empty_cache()
    finally:
        fdry.stop_allocation_region()

    # Test 3: VMM fast path
    fdry.set_allocation_region(0x550000000000, region_size)
    try:
        success = fdry.preallocate_region(total_size + 1024 * 1024 * 1024)
        assert success

        torch.cuda.synchronize()
        start = time.perf_counter()
        tensors = [torch.empty(s // 4, dtype=torch.float32, device="cuda") for s in alloc_sizes]
        torch.cuda.synchronize()
        fast_time = time.perf_counter() - start
        del tensors
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()
    finally:
        fdry.stop_allocation_region()

    print(f"[TEST] Native:    {native_time * 1000:.3f} ms")
    print(f"[TEST] VMM slow:  {slow_time * 1000:.3f} ms ({slow_time / native_time:.2f}x native)")
    print(f"[TEST] VMM fast:  {fast_time * 1000:.3f} ms ({fast_time / native_time:.2f}x native)")
    print(f"[TEST] Speedup:   {slow_time / fast_time:.1f}x (fast vs slow)")

    print("[TEST] test_performance_comparison PASSED")


def _run_core_boundary():
    import foundry as fdry

    torch.cuda.init()
    torch.cuda.synchronize()

    base_addr = 0x560000000000
    region_size = 20 * 1024 * 1024 * 1024  # 20GB
    prealloc_size = 8 * 1024 * 1024 * 1024  # 8GB
    alloc_size = 256 * 1024 * 1024  # 256MB each

    print("[TEST] Boundary test: 8GB preallocated, allocate to 16GB")

    fdry.set_allocation_region(base_addr, region_size)
    try:
        success = fdry.preallocate_region(prealloc_size)
        assert success

        tensors = []
        fast_times = []
        slow_times = []
        cumulative = 0
        target = 16 * 1024 * 1024 * 1024

        while cumulative < target:
            torch.cuda.synchronize()
            start = time.perf_counter()
            t = torch.empty(alloc_size // 4, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            tensors.append(t)
            cumulative += alloc_size

            if cumulative <= prealloc_size:
                fast_times.append(elapsed)
            else:
                slow_times.append(elapsed)

        avg_fast = sum(fast_times) / len(fast_times) if fast_times else 0
        avg_slow = sum(slow_times) / len(slow_times) if slow_times else 0

        print(f"[TEST] Fast path avg: {avg_fast:.3f} ms ({len(fast_times)} allocs)")
        print(f"[TEST] Slow path avg: {avg_slow:.3f} ms ({len(slow_times)} allocs)")
        if avg_fast > 0:
            print(f"[TEST] Speedup: {avg_slow / avg_fast:.1f}x")

        del tensors
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()

    finally:
        fdry.stop_allocation_region()

    print("[TEST] test_boundary_performance PASSED")


def _spawn_with_preload(test_mode):
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = so_path

    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{test_mode}"]
    subprocess.check_call(cmd, env=env)


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_basic_preallocation():
    """Test basic preallocation functionality."""
    _spawn_with_preload("basic")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_context_manager():
    """Test preallocated_allocation_region context manager."""
    _spawn_with_preload("context-manager")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_fallback_to_slow_path():
    """Test fallback when preallocated memory is exhausted."""
    _spawn_with_preload("fallback")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_performance_comparison():
    """Compare allocation performance: native vs VMM vs VMM+prealloc."""
    _spawn_with_preload("performance")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_boundary_performance():
    """Test fast->slow transition at preallocation boundary."""
    _spawn_with_preload("boundary")


def test_size_parsing():
    """Test size string parsing (no LD_PRELOAD needed)."""
    from foundry.allocation_region import parse_size

    assert parse_size(1024) == 1024
    assert parse_size("1KB") == 1024
    assert parse_size("1MB") == 1024 * 1024
    assert parse_size("1GB") == 1024 * 1024 * 1024
    assert parse_size("1.5GB") == int(1.5 * 1024 * 1024 * 1024)
    assert parse_size("512mb") == 512 * 1024 * 1024
    assert parse_size("  256 MB  ") == 256 * 1024 * 1024


if __name__ == "__main__":
    if "--basic" in sys.argv:
        _run_core_basic_preallocation()
    elif "--context-manager" in sys.argv:
        _run_core_context_manager()
    elif "--fallback" in sys.argv:
        _run_core_fallback()
    elif "--performance" in sys.argv:
        _run_core_performance()
    elif "--boundary" in sys.argv:
        _run_core_boundary()
    else:
        test_size_parsing()
        test_basic_preallocation()
        test_context_manager()
        test_fallback_to_slow_path()
        test_performance_comparison()
        test_boundary_performance()
