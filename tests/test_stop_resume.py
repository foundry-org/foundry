# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import subprocess
import sys
from pathlib import Path

import torch

BASE_ADDR = 0x7F0000000000
REGION_SIZE_STR = "1GB"


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


def _run_test():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.set_default_device(device)

    region_size = fdry.parse_size(REGION_SIZE_STR)

    print("[TEST] Setting up allocation region")
    fdry.set_allocation_region(BASE_ADDR, region_size)

    print("[TEST] Allocation 1 (Inside Region)")
    t1 = torch.full((2 * 1024 * 1024,), 1.0, device=device)
    print(
        f"[TEST] t1 size {t1.element_size() * t1.numel() / (1024 * 1024)} MB, address: {hex(t1.data_ptr())}"
    )

    print("[TEST] Stopping allocation region")
    fdry.stop_allocation_region()

    # Create a brand new pool (separate from default)
    pool = torch.cuda.memory.MemPool()  # separate pool id under the caching allocator

    # Allocate in the *new* pool
    with torch.cuda.use_mem_pool(pool, device=device):  # routes allocations to 'pool'
        print("[TEST] Allocation 2 (Stopped Region)")
        t2 = torch.full((2 * 1024 * 1024,), 2.0, device=device)
        print(
            f"[TEST] t2 size {t2.element_size() * t2.numel() / (1024 * 1024)} MB, address: {hex(t2.data_ptr())}"
        )

    print("[TEST] Resuming allocation region")
    fdry.resume_allocation_region()

    print("[TEST] Allocation 3 (Resumed Region)")
    t3 = torch.full((2 * 1024 * 1024,), 3.0, device=device)
    print(
        f"[TEST] t3 size {t3.element_size() * t3.numel() / (1024 * 1024)} MB, address: {hex(t3.data_ptr())}"
    )

    # Keep objects alive to ensure addresses are valid for printing
    return t1, t2, t3


def _spawn_with_preload():
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = so_path

    # Enable debug logs to see the hooks firing
    env["HOOK_DEBUG"] = "1"

    cmd = [sys.executable, str(Path(__file__).resolve()), "--run-inner"]
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    if "--run-inner" in sys.argv:
        _run_test()
    else:
        _spawn_with_preload()
