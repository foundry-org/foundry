# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Async LOAD (1 GPU): SAVE 12 graphs of a stacked MLP, then LOAD them with
start_graph_builds(num_threads=6), CPU foreground work while Phase 2 builds,
finish_graph_loads, and replays checked against eager.

    pytest tests/core/test_async_load.py -q -s
"""

import os
import sys
import time

import torch
from _helpers import run_phase

BASE_ADDR = 0x2F0000000000
REGION_SIZE_STR = "2GB"
HOOK_ARCHIVE_DIR = "hook_archive"

NUM_GRAPHS = 12
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 8, 16, 32, 64]
HIDDEN_DIM = 1024
MLP_DIM = 4096
NUM_MLP_LAYERS = 4  # Stack multiple layers per graph for more kernel nodes


def init_cublas():
    _ = torch._C._cuda_getCurrentBlasHandle()


def _allocate_tensors(device):
    """Allocate stacked MLP weights and input in identical order/size for both runs.

    This ensures the VMM allocation offset matches between SAVE and LOAD.
    Each layer: output = GELU(x @ W1.T + b1) @ W2.T + b2 + x
    """
    layers = []
    for _ in range(NUM_MLP_LAYERS):
        w1 = torch.randn(MLP_DIM, HIDDEN_DIM, device=device)
        b1 = torch.randn(MLP_DIM, device=device)
        w2 = torch.randn(HIDDEN_DIM, MLP_DIM, device=device)
        b2 = torch.randn(HIDDEN_DIM, device=device)
        layers.append((w1, b1, w2, b2))
    max_batch = max(BATCH_SIZES)
    static_input = torch.randn(max_batch, HIDDEN_DIM, device=device)
    return layers, static_input


def _stacked_mlp(x, layers):
    """Multiple MLP layers with GELU activation and residual connections."""
    for w1, b1, w2, b2 in layers:
        hidden = x @ w1.T + b1
        hidden = torch.nn.functional.gelu(hidden)
        x = hidden @ w2.T + b2 + x
    return x


def _simulate_foreground_work():
    """Simulate foreground work (like weight loading) while graphs build in background.

    Uses CPU-bound work to keep the main thread busy without allocating GPU
    memory (which would advance the VMM pointer and break SAVE/LOAD address
    matching). In an engine the overlapped work is weight loading, which does
    allocate GPU memory, identically in SAVE and LOAD.
    """
    import numpy as np

    t0 = time.perf_counter()

    # Simulate weight deserialization: CPU tensor creation + processing
    for size in [512, 1024, 2048, 4096]:
        w = np.random.randn(size, size).astype(np.float32)
        # Simulate weight processing (transpose, quantize, etc.)
        w = w.T.copy()
        w = (w * 127).clip(-128, 127).astype(np.int8).astype(np.float32) / 127
        print(f"  [FG] Processed fake weight {size}x{size} on CPU")

    elapsed = time.perf_counter() - t0
    print(f"  [FG] Foreground work done in {elapsed * 1000:.1f}ms")


def _run_saving_run(archive):
    """Capture and save multiple graphs with different batch sizes."""
    import foundry as fdry

    print("[SAVE] Starting saving run")

    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.set_default_device(device)

    region_size = fdry.parse_size(REGION_SIZE_STR)
    fdry.set_allocation_region(BASE_ADDR, region_size)
    print(f"[SAVE] Allocation region set: base=0x{BASE_ADDR:x}, size={REGION_SIZE_STR}")

    layers, static_input = _allocate_tensors(device)
    print(f"[SAVE] layers[0].w1 address: 0x{layers[0][0].data_ptr():x}")
    print(f"[SAVE] static_input address: 0x{static_input.data_ptr():x}")

    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        init_cublas()
    print(f"[SAVE] cuBLAS initialized, {NUM_MLP_LAYERS} MLP layers per graph")

    for i, bs in enumerate(BATCH_SIZES):
        inp = static_input[:bs]

        graph = fdry.CUDAGraph()
        with fdry.graph(graph, stream=stream):
            output = _stacked_mlp(inp, layers)

        graph.replay()
        torch.cuda.synchronize()

        expected = _stacked_mlp(inp, layers)
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)

        graph_json = os.path.join(archive, f"graph_{i}.json")
        graph.save(graph_json, output_tensors=output)
        print(f"[SAVE] Saved graph_{i}.json (batch_size={bs}, output shape={output.shape})")

    fdry.stop_allocation_region()
    print("[SAVE] Saving run completed")


def _run_loading_run(archive):
    """Load graphs asynchronously and verify correctness."""
    import foundry as fdry

    print("[LOAD] Starting loading run")

    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.set_default_device(device)

    fdry.load_cuda_modules_and_libraries(HOOK_ARCHIVE_DIR)
    print(f"[LOAD] CUDA modules loaded from {HOOK_ARCHIVE_DIR}")

    region_size = fdry.parse_size(REGION_SIZE_STR)
    fdry.set_allocation_region(BASE_ADDR, region_size)
    print(f"[LOAD] Allocation region set: base=0x{BASE_ADDR:x}, size={REGION_SIZE_STR}")

    # Identical allocation sequence as saving run
    layers, static_input = _allocate_tensors(device)
    print(f"[LOAD] layers[0].w1 address: 0x{layers[0][0].data_ptr():x}")
    print(f"[LOAD] static_input address: 0x{static_input.data_ptr():x}")

    fdry.graph.default_capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(fdry.graph.default_capture_stream):
        init_cublas()
    print("[LOAD] cuBLAS initialized")

    graph_paths = [os.path.join(archive, f"graph_{i}.json") for i in range(NUM_GRAPHS)]

    # --- Phase 1: start_graph_builds (returns immediately) ---
    num_threads = 6
    print(f"\n[LOAD] Phase 1: start_graph_builds ({NUM_GRAPHS} graphs, {num_threads} threads)")
    t_start = time.perf_counter()
    pending = fdry.CUDAGraph.start_graph_builds(graph_paths, num_threads=num_threads)
    phase1_ms = (time.perf_counter() - t_start) * 1000
    print(f"[LOAD] Phase 1 returned in {phase1_ms:.1f}ms — Phase 2 building in background")

    # --- Foreground work while graphs build in background ---
    print("[LOAD] Running foreground work while graphs build in background...")
    _simulate_foreground_work()

    # --- finish_graph_loads: allocator replay + output tensor reconstruction ---
    print("[LOAD] Calling finish_graph_loads (allocator replay + output tensors)")
    t_finish = time.perf_counter()
    results = fdry.CUDAGraph.finish_graph_loads(pending)
    finish_ms = (time.perf_counter() - t_finish) * 1000
    print(f"[LOAD] finish_graph_loads completed in {finish_ms:.1f}ms")

    assert len(results) == NUM_GRAPHS, f"Expected {NUM_GRAPHS} results, got {len(results)}"

    # Output tensors are available after finish_graph_loads
    for i, (graph, output) in enumerate(results):
        bs = BATCH_SIZES[i]
        assert output.shape[0] == bs, f"Expected batch={bs}, got {output.shape[0]}"
    print(f"[LOAD] All {NUM_GRAPHS} output tensors reconstructed")

    # --- Replay + verify: blocks on ensure_loaded() if graph not yet built ---
    print("[LOAD] Replaying and verifying graphs...")
    for i, (graph, output) in enumerate(results):
        bs = BATCH_SIZES[i]
        t_replay = time.perf_counter()
        graph.replay()
        replay_ms = (time.perf_counter() - t_replay) * 1000
        torch.cuda.synchronize()

        expected = _stacked_mlp(static_input[:bs], layers)
        torch.cuda.synchronize()

        torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
        print(f"  Graph {i} (bs={bs}): replay {replay_ms:.1f}ms — PASSED")

    total_ms = (time.perf_counter() - t_start) * 1000
    print(
        f"\n[LOAD] All {NUM_GRAPHS} graphs verified. "
        f"Total: {total_ms:.1f}ms (phase1={phase1_ms:.1f}ms, finish={finish_ms:.1f}ms)"
    )

    fdry.stop_allocation_region()
    print("[LOAD] Loading run completed")


def test_async_graph_load(workdir):
    """Async CUDA graph loading with foreground work overlap."""
    run_phase(__file__, "save", workdir)
    assert (workdir / HOOK_ARCHIVE_DIR).is_dir(), "SAVE did not pack hook_archive"
    run_phase(__file__, "load", workdir)


if __name__ == "__main__":
    {"--save": _run_saving_run, "--load": _run_loading_run}[sys.argv[1]](sys.argv[2])
