# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

BASE_ADDR = 0x7F0000000000
GRAPH_BASE_ADDR = 0x80000000000
REGION_SIZE_STR = "1GB"
ARCHIVE_DIR = "test_archive"
HOOK_ARCHIVE_DIR = "hook_archive"


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


@torch.compile
class DummyModel(nn.Module):
    def forward(self, x, y):
        return x + y


def _run_saving_run():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.set_default_device(device)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    region_size = fdry.parse_size(REGION_SIZE_STR)

    print("[TEST] Saving Run: Setting up allocation region")
    fdry.set_allocation_region(BASE_ADDR, region_size)

    print("[TEST] Saving Run: Allocating input tensors")
    input_tensor_a = torch.full((100, 100), 2.0, device=device)
    input_tensor_b = torch.full((100, 100), 3.0, device=device)

    print(f"[TEST] Saving Run: input_tensor_a address: {hex(input_tensor_a.data_ptr())}")
    print(f"[TEST] Saving Run: input_tensor_b address: {hex(input_tensor_b.data_ptr())}")

    model = DummyModel()
    model(input_tensor_a, input_tensor_b)
    torch.cuda.synchronize()

    fdry.set_allocation_region(GRAPH_BASE_ADDR, region_size)
    print("[TEST] Saving Run: Capturing CUDA graph")
    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        result = model(input_tensor_a, input_tensor_b)

    print("[TEST] Saving Run: Graph capture completed")
    print(f"[TEST] Saving Run: result tensor address: {hex(result.data_ptr())}")

    graph.replay()
    torch.cuda.synchronize()

    graph_json = os.path.join(ARCHIVE_DIR, "graph.json")

    print(f"[TEST] Saving Run: Saving graph with output tensor to {graph_json}")
    graph.save(graph_json, output_tensors=result)

    print("[TEST] Saving Run: Verifying output correctness")
    expected = input_tensor_a + input_tensor_b
    expected = expected.cpu()
    result_cpu = result.cpu()
    torch.testing.assert_close(result_cpu, expected, rtol=1e-5, atol=1e-5)
    print("[TEST] Saving Run: Output verification PASSED")

    fdry.stop_allocation_region()

    print("[TEST] Saving Run: Completed (binaries will be packed on exit)")


def _run_loading_run():
    import foundry as fdry

    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    if not os.path.exists(ARCHIVE_DIR):
        raise RuntimeError(f"Archive directory {ARCHIVE_DIR} not found")

    print(f"[TEST] Loading Run: Loading CUDA modules and libraries from {HOOK_ARCHIVE_DIR}")
    fdry.load_cuda_modules_and_libraries(HOOK_ARCHIVE_DIR)

    region_size = fdry.parse_size(REGION_SIZE_STR)

    print("[TEST] Loading Run: Setting up allocation region")
    fdry.set_allocation_region(BASE_ADDR, region_size)

    graph_json = os.path.join(ARCHIVE_DIR, "graph.json")

    print("[TEST] Loading Run: Allocating input tensors with different values")
    input_tensor_a = torch.full((100, 100), 5.0, device=device)
    input_tensor_b = torch.full((100, 100), 3.0, device=device)

    print(f"[TEST] Loading Run: input_tensor_a address: {hex(input_tensor_a.data_ptr())}")
    print(f"[TEST] Loading Run: input_tensor_b address: {hex(input_tensor_b.data_ptr())}")

    print(f"[TEST] Loading Run: Loading graph from {graph_json}")
    fdry.set_allocation_region(GRAPH_BASE_ADDR, region_size)
    graph, output_tensor = fdry.CUDAGraph.load(graph_json)

    print(
        f"[TEST] Loading Run: Reconstructed output tensor address: {hex(output_tensor.data_ptr())}"
    )
    print(
        f"[TEST] Loading Run: Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}"
    )

    print("[TEST] Loading Run: Replaying graph")
    graph.replay()
    torch.cuda.synchronize()

    print("[TEST] Loading Run: Verifying output correctness")
    expected = input_tensor_a + input_tensor_b
    expected = expected.cpu()
    output_cpu = output_tensor.cpu()
    torch.testing.assert_close(output_cpu, expected, rtol=1e-5, atol=1e-5)
    print("[TEST] Loading Run: Output verification PASSED")

    fdry.stop_allocation_region()

    print("[TEST] Loading Run: Completed successfully")


def _cleanup_archive():
    if os.path.exists(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
        print(f"[TEST] Cleaned up {ARCHIVE_DIR}")

    if os.path.exists(HOOK_ARCHIVE_DIR):
        shutil.rmtree(HOOK_ARCHIVE_DIR)
        print(f"[TEST] Cleaned up {HOOK_ARCHIVE_DIR}")


def _spawn_with_preload(launch_mode):
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = so_path

    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{launch_mode}"]
    subprocess.check_call(cmd, env=env)


@pytest.mark.filterwarnings("ignore:TORCH_CUDA_ARCH_LIST is not set")
def test_graph_save_and_load():
    """Test CUDA graph save/load with binary dumping and allocator replay"""
    print("\n[TEST] Starting test_graph_save_and_load")

    _cleanup_archive()

    print("[TEST] Running saving run (capture and save)")
    _spawn_with_preload("saving-run")

    print("[TEST] Running loading run (load and replay)")
    _spawn_with_preload("loading-run")

    _cleanup_archive()

    print("[TEST] test_graph_save_and_load PASSED")


if __name__ == "__main__":
    if "--saving-run" in sys.argv:
        _run_saving_run()
    elif "--loading-run" in sys.argv:
        _run_loading_run()
    elif "--cleanup" in sys.argv:
        _cleanup_archive()
    else:
        test_graph_save_and_load()
