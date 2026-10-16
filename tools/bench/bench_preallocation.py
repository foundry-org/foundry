# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Allocation timing inside a Foundry VMM region (1 GPU; prints, asserts nothing).

  performance  30 random allocations of 2 MB - 1 GB: native cudaMalloc vs the
               region's slow path (cuMemCreate + cuMemMap per allocation) vs the
               fast path (pointer bump in a preallocated span)
  boundary     256 MB allocations up to --boundary-gb with --prealloc-gb
               preallocated: per-allocation time before and after the span end
               (maps --boundary-gb of physical memory)

Usage: python tools/bench/bench_preallocation.py [--cases performance,boundary]
                                                 [--prealloc-gb 8] [--boundary-gb 16]
libcuda_hook.so is preloaded automatically (the script re-executes itself).
"""

import argparse
import os
import random
import sys
import time

GB = 1 << 30
MB = 1 << 20


def _ensure_hook_preloaded():
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig

    hook = CUDAGraphExtensionConfig._detect_hook_so_path()
    if hook is None:
        sys.exit("libcuda_hook.so not found next to foundry.ops")
    preload = os.environ.get("LD_PRELOAD", "")
    if hook not in preload.split(":"):
        os.environ["LD_PRELOAD"] = f"{hook}:{preload}" if preload else hook
        os.execv(sys.executable, [sys.executable, *sys.argv])


def bench_performance():
    import foundry as fdry
    import torch

    torch.cuda.init()
    torch.cuda.synchronize()
    random.seed(42)
    sizes = []
    for _ in range(30):
        size = int(2 ** random.uniform(21, 30))  # 2 MB to 1 GB
        size = max(2 * MB, min(GB, size))
        sizes.append((size + 2 * MB - 1) // (2 * MB) * (2 * MB))
    total = sum(sizes)
    print(f"[bench] performance: {total / GB:.2f} GB across {len(sizes)} allocations")

    def timed():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tensors = [torch.empty(s // 4, dtype=torch.float32, device="cuda") for s in sizes]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        del tensors
        torch.cuda.empty_cache()
        return elapsed

    native = timed()
    fdry.set_allocation_region(0x530000000000, 16 * GB)
    try:
        slow = timed()
    finally:
        fdry.stop_allocation_region()
    fdry.set_allocation_region(0x550000000000, 16 * GB)
    try:
        assert fdry.preallocate_region(total + GB), "preallocate_region failed"
        fast = timed()
        fdry.free_preallocated_region()
    finally:
        fdry.stop_allocation_region()
    print(f"[bench] native   {native * 1000:.3f} ms")
    print(f"[bench] VMM slow {slow * 1000:.3f} ms ({slow / native:.2f}x native)")
    print(f"[bench] VMM fast {fast * 1000:.3f} ms ({fast / native:.2f}x native)")
    print(f"[bench] fast vs slow {slow / fast:.1f}x")


def bench_boundary(prealloc_gb: float, boundary_gb: float):
    import foundry as fdry
    import torch

    torch.cuda.init()
    torch.cuda.synchronize()
    prealloc = int(prealloc_gb * GB)
    target = int(boundary_gb * GB)
    alloc = 256 * MB
    print(f"[bench] boundary: {prealloc_gb} GB preallocated, allocating to {boundary_gb} GB")
    fdry.set_allocation_region(0x560000000000, target + 4 * GB)
    try:
        assert fdry.preallocate_region(prealloc), "preallocate_region failed"
        tensors, fast, slow = [], [], []
        while len(tensors) * alloc < target:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tensors.append(torch.empty(alloc // 4, dtype=torch.float32, device="cuda"))
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            (fast if len(tensors) * alloc <= prealloc else slow).append(ms)
        avg_fast = sum(fast) / len(fast) if fast else 0.0
        avg_slow = sum(slow) / len(slow) if slow else 0.0
        print(f"[bench] fast path avg {avg_fast:.3f} ms ({len(fast)} allocations)")
        print(f"[bench] slow path avg {avg_slow:.3f} ms ({len(slow)} allocations)")
        if avg_fast > 0:
            print(f"[bench] slow vs fast {avg_slow / avg_fast:.1f}x")
        del tensors
        torch.cuda.empty_cache()
        fdry.free_preallocated_region()
    finally:
        fdry.stop_allocation_region()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default="performance,boundary")
    parser.add_argument("--prealloc-gb", type=float, default=8)
    parser.add_argument("--boundary-gb", type=float, default=16)
    args = parser.parse_args()
    _ensure_hook_preloaded()
    for case in args.cases.split(","):
        if case == "performance":
            bench_performance()
        elif case == "boundary":
            bench_boundary(args.prealloc_gb, args.boundary_gb)
        else:
            parser.error(f"unknown case {case!r}")


if __name__ == "__main__":
    main()
