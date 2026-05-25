# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import subprocess
import sys
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


def _run_core():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device("cuda:0")

    base_addr = 0x7F0000000000
    region_size_str = "1GB"
    region_size = 1024 * 1024 * 1024

    print(f"[TEST] Setting allocation region: base={hex(base_addr)}, size={region_size_str}")

    allocated_addrs = []

    with fdry.allocation_region(base_addr, region_size_str):
        tensor1 = torch.randn(1000, 1000, device=device)
        addr1 = tensor1.data_ptr()
        allocated_addrs.append(addr1)
        print(f"[TEST] Allocated tensor1 at address: {hex(addr1)}")

        tensor2 = torch.randn(2000, 2000, device=device)
        addr2 = tensor2.data_ptr()
        allocated_addrs.append(addr2)
        print(f"[TEST] Allocated tensor2 at address: {hex(addr2)}")

        tensor3 = torch.randn(500, 500, device=device)
        addr3 = tensor3.data_ptr()
        allocated_addrs.append(addr3)
        print(f"[TEST] Allocated tensor3 at address: {hex(addr3)}")

    region_end = base_addr + region_size

    for i, addr in enumerate(allocated_addrs, 1):
        assert addr >= base_addr, (
            f"tensor{i} address {hex(addr)} is below region base {hex(base_addr)}"
        )
        assert addr < region_end, (
            f"tensor{i} address {hex(addr)} is above region end {hex(region_end)}"
        )
        print(
            f"[TEST] ✓ tensor{i} address {hex(addr)} is within region [{hex(base_addr)}, {hex(region_end)})"
        )

    assert allocated_addrs[0] < allocated_addrs[1], "Addresses should be increasing"
    print("[TEST] ✓ Addresses are increasing")

    print("[TEST] All allocations are within the specified region")
    print("[TEST] test_vmm_allocation PASSED")


def _run_core_without_region():
    torch.cuda.init()
    device = torch.device("cuda:0")

    base_addr = 0x7F0000000000
    region_size = 1024 * 1024 * 1024
    region_end = base_addr + region_size

    print("[TEST] Testing allocation WITHOUT allocation_region (should allocate outside)")

    tensor1 = torch.randn(1000, 1000, device=device)
    addr1 = tensor1.data_ptr()
    print(f"[TEST] Allocated tensor without region at address: {hex(addr1)}")

    is_in_region = addr1 >= base_addr and addr1 < region_end

    if is_in_region:
        print(
            f"[TEST] WARNING: Address {hex(addr1)} happened to be in [{hex(base_addr)}, {hex(region_end)}), but this is just coincidence"
        )
    else:
        print(
            f"[TEST] ✓ Address {hex(addr1)} is outside region [{hex(base_addr)}, {hex(region_end)}), as expected"
        )

    print("[TEST] test_vmm_allocation_without_region PASSED")


def _run_core_size_parsing():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device("cuda:0")

    test_cases = [
        (0x7F0000000000, "1GB"),
        (0x7F1000000000, "512MB"),
        (0x7F2000000000, "100MB"),
        ("0x7f3000000000", 50 * 1024 * 1024),
    ]

    for base, size in test_cases:
        base_int = int(base, 16) if isinstance(base, str) else base

        if isinstance(size, str):
            size_str = size
            size_bytes = fdry.parse_size(size)
        else:
            size_str = f"{size} bytes"
            size_bytes = size

        print(f"[TEST] Testing allocation region: base={hex(base_int)}, size={size_str}")

        with fdry.allocation_region(base, size):
            tensor = torch.randn(100, 100, device=device)
            addr = tensor.data_ptr()
            print(f"[TEST]   Allocated at {hex(addr)}")

            region_end = base_int + size_bytes
            assert addr >= base_int, f"Address {hex(addr)} below base {hex(base_int)}"
            assert addr < region_end, f"Address {hex(addr)} above end {hex(region_end)}"
            print(f"[TEST]   ✓ Address within region [{hex(base_int)}, {hex(region_end)})")

            del tensor

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[TEST] test_size_parsing PASSED")


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
def test_vmm_allocation():
    """Test that allocation_region can allocate memory at fixed addresses"""
    _spawn_with_preload("core")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_vmm_allocation_without_region():
    """Test that allocations without allocation_region work normally"""
    _spawn_with_preload("without-region")


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_size_parsing():
    """Test that different size formats work correctly"""
    _spawn_with_preload("size-parsing")


if __name__ == "__main__":
    if "--core" in sys.argv:
        _run_core()
    elif "--without-region" in sys.argv:
        _run_core_without_region()
    elif "--size-parsing" in sys.argv:
        _run_core_size_parsing()
    else:
        test_vmm_allocation()
        test_vmm_allocation_without_region()
        test_size_parsing()
