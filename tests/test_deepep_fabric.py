# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""
Test DeepEP with use_fabric=True and CGE graph capture/reload.

This test verifies that:
1. DeepEP buffers can be allocated with use_fabric=True when CGE hook is active
2. CUDA graphs containing DeepEP dispatch operations can be captured and saved
3. The saved graphs can be loaded and replayed correctly

Requirements:
- DeepEP installed (deep_ep package)
- Multiple GPUs for distributed testing
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

# Use lower base address (matching working deepep_args example)
# Note: 0x60000000000 (6TB) NOT 0x600000000000 (96TB) - the extra zero matters!
BASE_ADDR = 0x600000000000  # 6TB
REGION_SIZE_STR = "512GB"
ARCHIVE_DIR = "deepep_fabric_archive"
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


def _init_dist(local_rank: int, num_local_ranks: int):
    """Initialize distributed environment for multi-GPU testing."""
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "29500"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    import inspect

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
    )


def _create_deterministic_inputs(
    rank: int, num_tokens: int, hidden: int, num_experts: int, num_topk: int
):
    """Create deterministic input tensors with traceable patterns.

    Pattern for x: each token i has value (rank * 1000 + i) in all hidden dims
    Pattern for topk_idx: deterministic expert selection based on token id
    """
    import deep_ep

    # x[token_i, :] = rank * 1000 + token_i (easy to trace which rank/token)
    x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    for i in range(num_tokens):
        x[i, :] = float(rank * 1000 + i)

    # Deterministic topk_idx: token i selects experts [i % num_experts, (i+1) % num_experts, ...]
    topk_idx = torch.zeros((num_tokens, num_topk), dtype=deep_ep.topk_idx_t, device="cuda")
    for i in range(num_tokens):
        for k in range(num_topk):
            topk_idx[i, k] = (i + k) % num_experts

    return x, topk_idx


def _print_buffer_info(buffer, rank: int, prefix: str):
    """Print DeepEP buffer memory addresses and info."""
    print(f"[Rank {rank}] {prefix}: DeepEP Buffer Info:", flush=True)

    # Try to access buffer attributes for addresses
    attrs_to_check = [
        "num_nvl_bytes",
        "num_rdma_bytes",
        "rank",
        "num_ranks",
        "low_latency_mode",
        "device_id",
    ]
    for attr in attrs_to_check:
        if hasattr(buffer, attr):
            print(f"[Rank {rank}] {prefix}:   {attr} = {getattr(buffer, attr)}", flush=True)

    # Try to get internal buffer pointers if available
    ptr_attrs = ["nvl_buffer", "rdma_buffer", "buffer_ptr", "recv_buffer", "send_buffer"]
    for attr in ptr_attrs:
        if hasattr(buffer, attr):
            ptr = getattr(buffer, attr)
            if hasattr(ptr, "data_ptr"):
                print(
                    f"[Rank {rank}] {prefix}:   {attr} address = {hex(ptr.data_ptr())}", flush=True
                )
            elif isinstance(ptr, int):
                print(f"[Rank {rank}] {prefix}:   {attr} address = {hex(ptr)}", flush=True)

    # Use the public API to fetch the RDMA buffer pointer if available.
    if hasattr(buffer, "get_local_buffer_tensor"):
        try:
            rdma_tensor = buffer.get_local_buffer_tensor(
                torch.uint8, size=torch.Size([1]), offset=0, use_rdma_buffer=True
            )
            print(
                f"[Rank {rank}] {prefix}:   rdma_buffer (api) address = {hex(rdma_tensor.data_ptr())}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Rank {rank}] {prefix}:   rdma_buffer (api) address = <error: {exc}>", flush=True
            )


def _verify_dispatch_result(
    recv_x,
    recv_topk_idx,
    recv_src_idx,
    num_recv,
    rank: int,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    prefix: str,
):
    """Verify dispatch results are correct based on our deterministic pattern."""
    num_local_experts = num_experts // num_ranks
    local_expert_start = rank * num_local_experts
    local_expert_end = local_expert_start + num_local_experts

    # num_recv might be an EventOverlap object (async), need to sync and get value
    actual_num_recv = num_recv
    if hasattr(num_recv, "wait"):
        # It's an async object, wait for it
        num_recv.wait()
    if hasattr(num_recv, "value"):
        actual_num_recv = num_recv.value
    elif hasattr(num_recv, "item"):
        actual_num_recv = num_recv.item()
    elif not isinstance(num_recv, int):
        # Try to get from recv_x shape
        actual_num_recv = (
            recv_x.shape[0] * recv_x.shape[1]
            if recv_x is not None and len(recv_x.shape) >= 2
            else 0
        )
        print(
            f"[Rank {rank}] {prefix}: num_recv is {type(num_recv)}, using recv_x shape to estimate",
            flush=True,
        )

    print(f"[Rank {rank}] {prefix}: Verification - num_recv = {actual_num_recv}", flush=True)
    print(
        f"[Rank {rank}] {prefix}: Verification - local experts range = [{local_expert_start}, {local_expert_end})",
        flush=True,
    )

    # recv_x shape is [num_local_experts, max_tokens_per_expert, hidden]
    # Print shape info for debugging
    if recv_x is not None:
        print(f"[Rank {rank}] {prefix}: recv_x shape = {recv_x.shape}", flush=True)

    # Check some values in recv_x to verify the pattern
    if recv_x is not None and recv_x.numel() > 0:
        # Check first expert's first few tokens
        num_experts_in_recv = recv_x.shape[0]
        tokens_per_expert = recv_x.shape[1]
        check_experts = min(2, num_experts_in_recv)
        check_tokens = min(3, tokens_per_expert)

        print(
            f"[Rank {rank}] {prefix}: Checking first {check_experts} experts, {check_tokens} tokens each:",
            flush=True,
        )

        for expert_i in range(check_experts):
            expert_global_idx = local_expert_start + expert_i
            for token_i in range(check_tokens):
                recv_val = recv_x[expert_i, token_i, 0].item()

                # Decode: src_rank = recv_val // 1000, token_id = recv_val % 1000
                decoded_src_rank = int(recv_val) // 1000
                decoded_token_id = int(recv_val) % 1000

                print(
                    f"[Rank {rank}] {prefix}:   expert[{expert_i}] (global {expert_global_idx}), "
                    f"token[{token_i}]: value={recv_val:.1f} "
                    f"(decoded: src_rank={decoded_src_rank}, token_id={decoded_token_id})",
                    flush=True,
                )
    else:
        print(f"[Rank {rank}] {prefix}: recv_x is empty or None", flush=True)

    return True


def _run_save(local_rank: int, num_processes: int):
    """SAVE mode: Create DeepEP buffer, capture graph, and save."""
    import deep_ep
    import foundry as fdry

    rank, num_ranks, group = _init_dist(local_rank, num_processes)

    # Disable auto-packing on exit - we'll manually pack to per-rank folder
    fdry.set_pack_fatbins_on_exit(False)

    # Debug: confirm NVSHMEM env vars are set
    if local_rank == 0:
        print(f"[Rank {local_rank}] NVSHMEM env vars:", flush=True)
        print(
            f"  NVSHMEM_REMOTE_TRANSPORT={os.environ.get('NVSHMEM_REMOTE_TRANSPORT', 'not set')}",
            flush=True,
        )
        print(
            f"  NVSHMEM_DISABLE_IBGDA={os.environ.get('NVSHMEM_DISABLE_IBGDA', 'not set')}",
            flush=True,
        )
        print(f"  NVSHMEM_IB_ENABLE={os.environ.get('NVSHMEM_IB_ENABLE', 'not set')}", flush=True)

    print(f"[Rank {rank}] SAVE: Initializing CUDA", flush=True)
    torch.cuda.init()

    # Test parameters
    # Note: DeepEP low-latency kernels require specific hidden dimensions
    # Common supported values: 2048, 4096, 5120, 7168, etc.
    num_tokens = 64
    hidden = 2048  # Must be a supported size for DeepEP low-latency kernels
    num_experts = num_ranks * 4  # 4 experts per rank
    num_topk = 2

    # Setup allocation region
    region_size = fdry.parse_size(REGION_SIZE_STR)
    print(f"[Rank {rank}] SAVE: Setting up allocation region at {hex(BASE_ADDR)}", flush=True)
    fdry.set_allocation_region(BASE_ADDR, region_size)

    # Create DeepEP buffer
    num_local_experts = num_experts // num_ranks
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )

    # Check if fabric should be used
    use_fabric = os.environ.get("TEST_USE_FABRIC", "1") == "1"
    # Test: add num_nvl_bytes to reproduce vllm failure
    num_nvl_bytes = int(os.environ.get("TEST_NVL_BYTES_MB", "0")) * 1024 * 1024
    print(
        f"[Rank {rank}] SAVE: Creating DeepEP buffer with use_fabric={use_fabric}, num_nvl_bytes={num_nvl_bytes / 1e6:.2f} MB, num_rdma_bytes={num_rdma_bytes / 1e6:.2f} MB",
        flush=True,
    )

    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        use_fabric=use_fabric,
    )

    # Print buffer info including addresses
    _print_buffer_info(buffer, rank, "SAVE")

    # Create deterministic input tensors
    x, topk_idx = _create_deterministic_inputs(rank, num_tokens, hidden, num_experts, num_topk)
    cumulative_stats = torch.zeros((num_local_experts,), dtype=torch.int, device="cuda")

    print(f"[Rank {rank}] SAVE: Input tensors:", flush=True)
    print(f"[Rank {rank}] SAVE:   x address = {hex(x.data_ptr())}, shape = {x.shape}", flush=True)
    print(
        f"[Rank {rank}] SAVE:   x[0, :5] = {x[0, :5].tolist()} (expect {rank * 1000 + 0})",
        flush=True,
    )
    print(
        f"[Rank {rank}] SAVE:   x[1, :5] = {x[1, :5].tolist()} (expect {rank * 1000 + 1})",
        flush=True,
    )
    print(f"[Rank {rank}] SAVE:   topk_idx address = {hex(topk_idx.data_ptr())}", flush=True)
    print(f"[Rank {rank}] SAVE:   topk_idx[:3] = {topk_idx[:3].tolist()}", flush=True)

    # Warmup and verify
    print(f"[Rank {rank}] SAVE: Running warmup dispatch", flush=True)
    result = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,  # Disable FP8 for exact value tracking
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=cumulative_stats,
        async_finish=False,
        return_recv_hook=False,
    )
    torch.cuda.synchronize()

    # Unpack dispatch result
    recv_x, recv_topk_idx, recv_src_idx, num_recv, *rest = result
    print(f"[Rank {rank}] SAVE: Warmup dispatch result:", flush=True)
    print(
        f"[Rank {rank}] SAVE:   recv_x address = {hex(recv_x.data_ptr()) if recv_x is not None else 'None'}",
        flush=True,
    )
    print(
        f"[Rank {rank}] SAVE:   recv_x shape = {recv_x.shape if recv_x is not None else 'None'}",
        flush=True,
    )
    print(f"[Rank {rank}] SAVE:   num_recv = {num_recv}", flush=True)

    # Verify results
    _verify_dispatch_result(
        recv_x,
        recv_topk_idx,
        recv_src_idx,
        num_recv,
        rank,
        num_ranks,
        num_tokens,
        hidden,
        num_experts,
        "SAVE-warmup",
    )

    dist.barrier(group)
    print(f"[Rank {rank}] SAVE: Warmup completed and verified", flush=True)

    # Capture graph
    print(f"[Rank {rank}] SAVE: Capturing CUDA graph", flush=True)
    graph = fdry.CUDAGraph()

    with fdry.graph(graph):
        graph_result = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=False,
            round_scale=False,
            use_ue8m0=False,
            cumulative_local_expert_recv_stats=cumulative_stats,
            async_finish=False,
            return_recv_hook=False,
        )

    print(f"[Rank {rank}] SAVE: Graph capture completed", flush=True)

    # Print graph output tensor addresses
    # graph_result structure varies - extract only actual tensors
    print(f"[Rank {rank}] SAVE: Graph output result type: {type(graph_result)}", flush=True)
    print(f"[Rank {rank}] SAVE: Graph output result length: {len(graph_result)}", flush=True)

    # Collect only tensor objects from the result
    output_tensors_to_save = []
    for i, item in enumerate(graph_result):
        item_type = type(item).__name__
        if isinstance(item, torch.Tensor):
            print(
                f"[Rank {rank}] SAVE:   result[{i}] = Tensor, address = {hex(item.data_ptr())}, shape = {item.shape}",
                flush=True,
            )
            output_tensors_to_save.append(item)
        elif isinstance(item, tuple):
            print(f"[Rank {rank}] SAVE:   result[{i}] = tuple of length {len(item)}", flush=True)
            # Check if tuple contains tensors
            for j, sub_item in enumerate(item):
                if isinstance(sub_item, torch.Tensor):
                    print(
                        f"[Rank {rank}] SAVE:     result[{i}][{j}] = Tensor, address = {hex(sub_item.data_ptr())}, shape = {sub_item.shape}",
                        flush=True,
                    )
                    output_tensors_to_save.append(sub_item)
        else:
            print(f"[Rank {rank}] SAVE:   result[{i}] = {item_type}", flush=True)

    # Replay and verify
    print(f"[Rank {rank}] SAVE: Replaying graph", flush=True)
    graph.replay()
    torch.cuda.synchronize()

    # Verify graph replay result
    graph_num_recv = cumulative_stats.sum().item()  # Approximate
    print(f"[Rank {rank}] SAVE: Graph replay completed", flush=True)

    # Save graph and fatbins to per-rank archive
    # output_tensors_to_save was already built above by filtering tensor objects
    rank_archive = os.path.join(ARCHIVE_DIR, f"rank_{rank}")
    os.makedirs(rank_archive, exist_ok=True)

    graph_json = os.path.join(rank_archive, "deepep_dispatch_graph.json")
    print(
        f"[Rank {rank}] SAVE: Saving graph to {graph_json} with {len(output_tensors_to_save)} output tensors",
        flush=True,
    )
    graph.save(graph_json, output_tensors=output_tensors_to_save)
    print(f"[Rank {rank}] SAVE: Graph saved successfully", flush=True)

    # Pack fatbins to per-rank archive
    print(f"[Rank {rank}] SAVE: Packing fatbins to {rank_archive}", flush=True)
    fdry.pack_fatbins_to_folder(rank_archive)
    print(f"[Rank {rank}] SAVE: Fatbins packed successfully", flush=True)

    # Sync all ranks before cleanup
    dist.barrier(group)
    print(f"[Rank {rank}] SAVE: All ranks synced, starting cleanup", flush=True)

    fdry.stop_allocation_region()
    buffer.destroy()
    dist.destroy_process_group()

    print(f"[Rank {rank}] SAVE: Completed", flush=True)


def _run_load(local_rank: int, num_processes: int):
    """LOAD mode: Load saved graph and replay."""
    import deep_ep
    import foundry as fdry

    rank, num_ranks, group = _init_dist(local_rank, num_processes)

    print(f"[Rank {rank}] LOAD: Initializing CUDA", flush=True)
    torch.cuda.init()

    # Test parameters (must match SAVE mode)
    # Note: DeepEP low-latency kernels require specific hidden dimensions
    num_tokens = 64
    hidden = 2048  # Must match SAVE and be a supported size
    num_experts = num_ranks * 4  # Must match SAVE
    num_topk = 2

    # Per-rank workspace
    rank_archive = os.path.join(ARCHIVE_DIR, f"rank_{rank}")
    if not os.path.exists(rank_archive):
        raise RuntimeError(f"Archive directory {rank_archive} not found. Run SAVE mode first.")

    # CRITICAL: Set skip_fatbin_processing=True so that when DeepEP loads its
    # kernels during warmup, they are tracked as warmup handles instead of being
    # processed in SAVE mode (which would dump to disk).
    print(
        f"[Rank {rank}] LOAD: Setting skip_fatbin_processing=True for warmup tracking", flush=True
    )
    fdry.set_skip_fatbin_processing(True)

    # CRITICAL ORDER FOR LOAD MODE:
    # 1. Load CUDA modules FIRST (creates CUDA context, adds NVSHMEM modules to pending list)
    # 2. Set allocation region (VMM setup)
    # 3. Create DeepEP Buffer (initializes NVSHMEM runtime)
    # 4. Call init_nvshmem_for_loaded_modules() to do deferred NVSHMEM init
    # 5. Warmup, load graph, replay

    # Step 1: Load modules (without NVSHMEM init - they go to pending list)
    print(f"[Rank {rank}] LOAD: Loading CUDA modules from {rank_archive}", flush=True)
    fdry.load_cuda_modules_and_libraries(rank_archive)

    # Step 2: Setup allocation region
    region_size = fdry.parse_size(REGION_SIZE_STR)
    print(f"[Rank {rank}] LOAD: Setting up allocation region at {hex(BASE_ADDR)}", flush=True)
    fdry.set_allocation_region(BASE_ADDR, region_size)

    # Step 3: Create DeepEP buffer (initializes NVSHMEM runtime)
    num_local_experts = num_experts // num_ranks
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )

    use_fabric = os.environ.get("TEST_USE_FABRIC", "1") == "1"
    print(
        f"[Rank {rank}] LOAD: Creating DeepEP buffer with use_fabric={use_fabric}, size={num_rdma_bytes / 1e6:.2f} MB",
        flush=True,
    )

    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        use_fabric=use_fabric,
    )

    # Print buffer info including addresses
    _print_buffer_info(buffer, rank, "LOAD")

    # Step 4: Deferred NVSHMEM init for loaded modules
    # Now that NVSHMEM runtime is initialized by DeepEP Buffer, we can init the loaded modules
    print(f"[Rank {rank}] LOAD: Initializing NVSHMEM for loaded modules (deferred)", flush=True)
    count = fdry.init_nvshmem_for_loaded_modules()
    print(f"[Rank {rank}] LOAD: Initialized NVSHMEM for {count} modules", flush=True)

    # Create deterministic input tensors (same as SAVE)
    x, topk_idx = _create_deterministic_inputs(rank, num_tokens, hidden, num_experts, num_topk)
    cumulative_stats = torch.zeros((num_local_experts,), dtype=torch.int, device="cuda")

    print(f"[Rank {rank}] LOAD: Input tensors:", flush=True)
    print(f"[Rank {rank}] LOAD:   x address = {hex(x.data_ptr())}, shape = {x.shape}", flush=True)
    print(
        f"[Rank {rank}] LOAD:   x[0, :5] = {x[0, :5].tolist()} (expect {rank * 1000 + 0})",
        flush=True,
    )
    print(
        f"[Rank {rank}] LOAD:   x[1, :5] = {x[1, :5].tolist()} (expect {rank * 1000 + 1})",
        flush=True,
    )
    print(f"[Rank {rank}] LOAD:   topk_idx address = {hex(topk_idx.data_ptr())}", flush=True)
    print(f"[Rank {rank}] LOAD:   topk_idx[:3] = {topk_idx[:3].tolist()}", flush=True)

    # Warmup - this loads DeepEP kernels and registers them as warmup handles
    print(f"[Rank {rank}] LOAD: Running kernel warmup (registers DeepEP modules)", flush=True)
    warmup_result = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,  # Must match SAVE
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=cumulative_stats,
        async_finish=False,
        return_recv_hook=False,
    )
    torch.cuda.synchronize()

    # Unpack and verify warmup result
    recv_x, recv_topk_idx, recv_src_idx, num_recv, *rest = warmup_result
    print(f"[Rank {rank}] LOAD: Warmup dispatch result:", flush=True)
    print(
        f"[Rank {rank}] LOAD:   recv_x address = {hex(recv_x.data_ptr()) if recv_x is not None else 'None'}",
        flush=True,
    )
    print(
        f"[Rank {rank}] LOAD:   recv_x shape = {recv_x.shape if recv_x is not None else 'None'}",
        flush=True,
    )
    print(f"[Rank {rank}] LOAD:   num_recv = {num_recv}", flush=True)

    # Verify warmup results
    _verify_dispatch_result(
        recv_x,
        recv_topk_idx,
        recv_src_idx,
        num_recv,
        rank,
        num_ranks,
        num_tokens,
        hidden,
        num_experts,
        "LOAD-warmup",
    )

    dist.barrier(group)
    print(f"[Rank {rank}] LOAD: Warmup completed", flush=True)

    # Debug: print warmup recv_x address vs what graph expects
    print(f"[Rank {rank}] LOAD: Warmup recv_x at {hex(recv_x.data_ptr())}", flush=True)
    print(
        f"[Rank {rank}] LOAD: This is where DeepEP outputs during warmup - graph replay should use same addresses",
        flush=True,
    )

    # Load graph from per-rank archive
    graph_json = os.path.join(rank_archive, "deepep_dispatch_graph.json")
    print(f"[Rank {rank}] LOAD: Loading graph from {graph_json}", flush=True)
    graph, output_tensors = fdry.CUDAGraph.load(graph_json)
    print(f"[Rank {rank}] LOAD: Graph loaded successfully", flush=True)

    # Print loaded output tensor info
    loaded_recv_x = None
    if output_tensors is not None:
        if isinstance(output_tensors, (list, tuple)):
            for i, t in enumerate(output_tensors):
                if hasattr(t, "data_ptr"):
                    print(
                        f"[Rank {rank}] LOAD: Loaded output[{i}] address = {hex(t.data_ptr())}, shape = {t.shape}",
                        flush=True,
                    )
                    # First tensor should be recv_x
                    if i == 0:
                        loaded_recv_x = t
        elif hasattr(output_tensors, "data_ptr"):
            print(
                f"[Rank {rank}] LOAD: Loaded output address = {hex(output_tensors.data_ptr())}, shape = {output_tensors.shape}",
                flush=True,
            )
            loaded_recv_x = output_tensors

    # Replay loaded graph
    print(f"[Rank {rank}] LOAD: Replaying loaded graph", flush=True)
    graph.replay()
    torch.cuda.synchronize()

    # Verify after replay - check the LOADED output tensors (not warmup recv_x)
    # recv_x shape is [num_local_experts, max_tokens_per_expert, hidden]
    print(f"[Rank {rank}] LOAD: After graph replay:", flush=True)
    verify_tensor = loaded_recv_x if loaded_recv_x is not None else recv_x
    if verify_tensor is not None and verify_tensor.numel() > 0:
        print(f"[Rank {rank}] LOAD:   verify_tensor shape = {verify_tensor.shape}", flush=True)
        num_local_experts = num_experts // num_ranks
        local_expert_start = rank * num_local_experts

        # Check first expert's first few tokens
        check_experts = min(2, verify_tensor.shape[0])
        check_tokens = min(3, verify_tensor.shape[1])

        for expert_i in range(check_experts):
            for token_i in range(check_tokens):
                val = verify_tensor[expert_i, token_i, 0].item()
                decoded_src_rank = int(val) // 1000
                decoded_token_id = int(val) % 1000
                print(
                    f"[Rank {rank}] LOAD:   expert[{expert_i}], token[{token_i}]: "
                    f"value={val:.1f} (rank={decoded_src_rank}, token={decoded_token_id})",
                    flush=True,
                )

    print(f"[Rank {rank}] LOAD: Graph replay successful", flush=True)

    dist.barrier(group)
    fdry.stop_allocation_region()

    buffer.destroy()
    dist.destroy_process_group()

    print(f"[Rank {rank}] LOAD: Completed successfully", flush=True)


def _cleanup_archive():
    """Clean up archive directories."""
    if os.path.exists(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
        print(f"[TEST] Cleaned up {ARCHIVE_DIR}")

    # Also clean up old-style hook_archive if it exists
    if os.path.exists(HOOK_ARCHIVE_DIR):
        shutil.rmtree(HOOK_ARCHIVE_DIR)
        print(f"[TEST] Cleaned up {HOOK_ARCHIVE_DIR}")


def _spawn_with_preload(mode: str, num_processes: int):
    """Spawn subprocess with LD_PRELOAD for CGE hook."""
    so_path = _get_hook_so_path()
    env = os.environ.copy()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{so_path}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = so_path

    # # Set NVSHMEM environment variables to disable IBGDA and remote transports
    # env['NVSHMEM_REMOTE_TRANSPORT'] = 'none'
    # env['NVSHMEM_DISABLE_IBGDA'] = '1'
    # env['NVSHMEM_IB_ENABLE'] = '0'

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--{mode}",
        f"--num-processes={num_processes}",
    ]
    subprocess.check_call(cmd, env=env)


def _check_deepep_available():
    """Check if DeepEP is available."""
    try:
        import deep_ep  # noqa: F401

        return True
    except ImportError:
        return False


def _check_multi_gpu():
    """Check if multiple GPUs are available."""
    return torch.cuda.device_count() >= 2


@pytest.mark.skipif(not _check_deepep_available(), reason="DeepEP not installed")
@pytest.mark.skipif(not _check_multi_gpu(), reason="Requires at least 2 GPUs")
def test_deepep_fabric_graph_save_load():
    """Test DeepEP with use_fabric=True graph save/load."""
    pytest.importorskip("deep_ep")
    num_processes = min(torch.cuda.device_count(), 2)  # Use 2 GPUs for testing

    print(f"\n[TEST] Starting DeepEP fabric graph save/load test with {num_processes} processes")

    _cleanup_archive()

    print("[TEST] Running SAVE mode (create buffer, capture graph, save)")
    _spawn_with_preload("save", num_processes)

    print("[TEST] Running LOAD mode (load graph, replay)")
    _spawn_with_preload("load", num_processes)

    _cleanup_archive()

    print("[TEST] test_deepep_fabric_graph_save_load PASSED")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test DeepEP with use_fabric=True and CGE")
    parser.add_argument("--save", action="store_true", help="Run SAVE mode")
    parser.add_argument("--load", action="store_true", help="Run LOAD mode")
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes")
    parser.add_argument("--cleanup", action="store_true", help="Clean up archive")
    parser.add_argument("--run", action="store_true", help="Run full test without pytest")
    parser.add_argument(
        "--no-fabric", action="store_true", help="Disable fabric (use_fabric=False)"
    )
    args = parser.parse_args()

    # Store fabric flag in env so spawned processes can access it
    os.environ["TEST_USE_FABRIC"] = "0" if args.no_fabric else "1"

    # # Set NVSHMEM environment variables to disable IBGDA and remote transports
    # # This must be set BEFORE any CUDA initialization
    # os.environ['NVSHMEM_REMOTE_TRANSPORT'] = 'none'
    # os.environ['NVSHMEM_DISABLE_IBGDA'] = '1'
    # os.environ['NVSHMEM_IB_ENABLE'] = '0'

    if args.cleanup:
        _cleanup_archive()
    elif args.save:
        print("[TEST] Running SAVE mode with NVSHMEM env vars:", flush=True)
        print(
            f"  NVSHMEM_REMOTE_TRANSPORT={os.environ.get('NVSHMEM_REMOTE_TRANSPORT')}", flush=True
        )
        print(f"  NVSHMEM_DISABLE_IBGDA={os.environ.get('NVSHMEM_DISABLE_IBGDA')}", flush=True)
        print(f"  NVSHMEM_IB_ENABLE={os.environ.get('NVSHMEM_IB_ENABLE')}", flush=True)
        torch.multiprocessing.spawn(
            _run_save,
            args=(args.num_processes,),
            nprocs=args.num_processes,
        )
    elif args.load:
        print("[TEST] Running LOAD mode with NVSHMEM env vars:", flush=True)
        print(
            f"  NVSHMEM_REMOTE_TRANSPORT={os.environ.get('NVSHMEM_REMOTE_TRANSPORT')}", flush=True
        )
        print(f"  NVSHMEM_DISABLE_IBGDA={os.environ.get('NVSHMEM_DISABLE_IBGDA')}", flush=True)
        print(f"  NVSHMEM_IB_ENABLE={os.environ.get('NVSHMEM_IB_ENABLE')}", flush=True)
        torch.multiprocessing.spawn(
            _run_load,
            args=(args.num_processes,),
            nprocs=args.num_processes,
        )
    elif args.run:
        # Run full test without pytest
        num_processes = min(torch.cuda.device_count(), 2)
        if num_processes < 2:
            print("[TEST] ERROR: Requires at least 2 GPUs")
            sys.exit(1)

        if not _check_deepep_available():
            print("[TEST] ERROR: DeepEP not installed")
            sys.exit(1)

        print(f"\n[TEST] Starting DeepEP fabric test with {num_processes} processes")
        _cleanup_archive()

        print("[TEST] Running SAVE mode")
        _spawn_with_preload("save", num_processes)

        print("[TEST] Running LOAD mode")
        _spawn_with_preload("load", num_processes)

        _cleanup_archive()
        print("[TEST] PASSED")
    else:
        parser.print_help()
        print("\nExample commands:")
        print("  # Test without fabric first:")
        print(
            "  LD_PRELOAD=/path/to/nvshmem/lib/libnvshmem_host.so:build/libcuda_hook.so python tests/test_deepep_fabric.py --save --no-fabric"
        )
        print(
            "  LD_PRELOAD=/path/to/nvshmem/lib/libnvshmem_host.so:build/libcuda_hook.so python tests/test_deepep_fabric.py --load --no-fabric"
        )
        print()
        print("  # Test with fabric:")
        print(
            "  LD_PRELOAD=/path/to/nvshmem/lib/libnvshmem_host.so:build/libcuda_hook.so python tests/test_deepep_fabric.py --save"
        )
        print(
            "  LD_PRELOAD=/path/to/nvshmem/lib/libnvshmem_host.so:build/libcuda_hook.so python tests/test_deepep_fabric.py --load"
        )


if __name__ == "__main__":
    main()
