# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SAVE / LOAD of a DeepEP low-latency dispatch graph (2 GPUs, deep_ep installed).

SAVE creates the DeepEP buffer inside the allocation region, captures one
low_latency_dispatch, replays it and saves it. LOAD (fresh processes) loads
the kernel images, recreates the buffer, runs the deferred NVSHMEM module
init (init_nvshmem_for_loaded_modules), restores the graph with
start_graph_builds + finish_graph_loads and replays it. Every replay must
deliver exactly the expected tokens to every local expert.

    pytest tests/multigpu/test_deepep.py -q -s
    pytest tests/multigpu/test_deepep.py -q -s -k nvl_ipc

Cases:
  ll                pure low latency: NVSHMEM heap only, no NVL buffer
  nvl_ipc           + 64 MB NVL buffer, exchanged through the hook's VMM-IPC path
  nvl_ipc_prealloc  as nvl_ipc, LOAD carves the NVL buffer from a 1024 MB preallocation
The NVSHMEM host library of the nvidia-nvshmem wheel is preloaded (as the
sglang integration does) and NVSHMEM_DISABLE_NCCL=1 is pinned. The buffer is
created without the use_fabric kwarg (cu12x DeepEP builds do not have it);
FOUNDRY_TEST_DEEPEP_FABRIC=1 passes use_fabric=True (needs IMEX / multi-node NVLink).
"""

import importlib.util
import inspect
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist
from _helpers import free_port, run_phase

BASE_ADDR = 0x600000000000  # 96 TB
REGION_SIZE_STR = "512GB"
NUM_RANKS = 2
NUM_TOKENS = 64
HIDDEN = 2048  # a hidden size the low-latency kernels support
EXPERTS_PER_RANK = 4
NUM_TOPK = 2
MB = 1 << 20
FABRIC_ENV = "FOUNDRY_TEST_DEEPEP_FABRIC"

CASES = {
    "ll": {"nvl_mb": 0, "load_prealloc_mb": 0},
    "nvl_ipc": {"nvl_mb": 64, "load_prealloc_mb": 0},
    "nvl_ipc_prealloc": {"nvl_mb": 64, "load_prealloc_mb": 1024},
}

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("deep_ep") is None, reason="deep_ep not installed"
)


def _env():
    env = {
        # As the sglang integration pins it (plugin.ENV_PINS) and the recipes run.
        "NVSHMEM_DISABLE_NCCL": "1",
        # NCCL buffers through the plain allocator.
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
        # NVSHMEM heap chunks on POSIX FD handles: no fabric handles anywhere.
        "NVSHMEM_CUMEM_HANDLE_TYPE": "FILE_DESCRIPTOR",
        "MASTER_PORT": str(free_port()),
    }
    # Single node: skip the InfiniBand discovery (recipe/sglang/set_env.sh), unless exported.
    env["NVSHMEM_REMOTE_TRANSPORT"] = os.environ.get("NVSHMEM_REMOTE_TRANSPORT", "none")
    env["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
    return env


def _init_dist(local_rank: int):
    params = {
        "backend": "nccl",
        "init_method": f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
        "world_size": NUM_RANKS,
        "rank": local_rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.new_group(list(range(NUM_RANKS)))


def _num_experts():
    return NUM_RANKS * EXPERTS_PER_RANK


def _inputs(rank: int):
    """x[i, :] = rank * NUM_TOKENS + i (exact in bf16); token i selects experts
    i % E, (i + 1) % E, ..."""
    import deep_ep

    tokens = torch.arange(NUM_TOKENS, device="cuda")
    x = (rank * NUM_TOKENS + tokens).to(torch.bfloat16)[:, None].expand(NUM_TOKENS, HIDDEN)
    topk = (tokens[:, None] + torch.arange(NUM_TOPK, device="cuda")[None, :]) % _num_experts()
    idx_t = getattr(deep_ep, "topk_idx_t", torch.int64)
    return x.contiguous(), topk.to(idx_t)


def _expected(rank: int):
    """Per local expert: the sorted source values it must receive."""
    out = []
    for e in range(EXPERTS_PER_RANK):
        g = rank * EXPERTS_PER_RANK + e
        values = [
            src * NUM_TOKENS + i
            for src in range(NUM_RANKS)
            for i in range(NUM_TOKENS)
            if any((i + k) % _num_experts() == g for k in range(NUM_TOPK))
        ]
        out.append(sorted(values))
    return out


def _verify(recv_x, recv_count, rank: int, what: str):
    torch.cuda.synchronize()
    counts = recv_count.tolist()
    for e, want in enumerate(_expected(rank)):
        assert counts[e] == len(want), f"[rank {rank}] {what}: expert {e} count {counts}"
        rows = recv_x[e, : counts[e]]
        assert torch.equal(rows, rows[:, :1].expand_as(rows)), (
            f"[rank {rank}] {what}: expert {e} received mixed rows"
        )
        got = sorted(int(v) for v in rows[:, 0].float().tolist())
        assert got == want, f"[rank {rank}] {what}: expert {e} received {got}, want {want}"


def _buffer(group, nvl_mb: int):
    import deep_ep

    extra = {"use_fabric": True} if os.environ.get(FABRIC_ENV) == "1" else {}
    return deep_ep.Buffer(
        group,
        num_nvl_bytes=nvl_mb * MB,
        num_rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(
            NUM_TOKENS, HIDDEN, NUM_RANKS, _num_experts()
        ),
        low_latency_mode=True,
        num_qps_per_rank=EXPERTS_PER_RANK,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
        **extra,
    )


def _dispatch(buffer, x, topk, stats):
    """(recv_x [local experts, NUM_TOKENS * NUM_RANKS, HIDDEN], recv_count, ...)."""
    return buffer.low_latency_dispatch(
        x,
        topk,
        NUM_TOKENS,
        _num_experts(),
        use_fp8=False,
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=stats,
        async_finish=False,
        return_recv_hook=False,
    )


def _tensors(result):
    """The tensors of a dispatch result, nested tuples flattened, in order."""
    out = []
    for item in result:
        if isinstance(item, torch.Tensor):
            out.append(item)
        elif isinstance(item, tuple):
            out.extend(t for t in item if isinstance(t, torch.Tensor))
    return out


def _rank_archive(archive, rank):
    return os.path.join(archive, f"rank_{rank}")


def _save_rank(local_rank: int, archive: str, case: str):
    import foundry as fdry

    cfg = CASES[case]
    rank, group = _init_dist(local_rank)
    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    buffer = _buffer(group, cfg["nvl_mb"])
    x, topk = _inputs(rank)
    stats = torch.zeros((EXPERTS_PER_RANK,), dtype=torch.int, device="cuda")

    recv_x, recv_count, *_ = _dispatch(buffer, x, topk, stats)  # eager warm-up
    _verify(recv_x, recv_count, rank, "SAVE eager")
    dist.barrier(group)

    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        result = _dispatch(buffer, x, topk, stats)
    outputs = _tensors(result)
    recv_x, recv_count = outputs[0], outputs[1]
    recv_x.zero_()
    recv_count.zero_()
    dist.barrier(group)
    graph.replay()
    _verify(recv_x, recv_count, rank, "SAVE replay")

    rank_archive = _rank_archive(archive, rank)
    os.makedirs(rank_archive, exist_ok=True)
    graph.save(os.path.join(rank_archive, "graph.json"), output_tensors=outputs)
    fdry.pack_fatbins_to_folder(rank_archive)
    # graph.save re-enables packing on exit; the ranks would race on ./hook_archive.
    fdry.set_pack_fatbins_on_exit(False)
    with open(os.path.join(rank_archive, "meta.json"), "w") as f:
        json.dump({"x": x.data_ptr(), "topk": topk.data_ptr()}, f)
    dist.barrier(group)
    fdry.stop_allocation_region()
    buffer.destroy()
    dist.destroy_process_group()


def _load_rank(local_rank: int, archive: str, case: str):
    import foundry as fdry

    cfg = CASES[case]
    rank, group = _init_dist(local_rank)
    torch.cuda.init()
    rank_archive = _rank_archive(archive, rank)
    # DeepEP's kernels load during the warm-up below: track them as warm-up
    # handles instead of processing them as SAVE would.
    fdry.set_skip_fatbin_processing(True)
    # Order: kernel images (NVSHMEM modules wait for their init), region,
    # optional preallocation, DeepEP buffer (starts the NVSHMEM runtime), then
    # the deferred NVSHMEM init of the loaded modules.
    fdry.load_cuda_modules_and_libraries(rank_archive)
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    if cfg["load_prealloc_mb"]:
        # The production LOAD shape: the NVL buffer is carved from one preallocated
        # chunk without its own cuMem handle (the hook's whole-chunk VMM-IPC export).
        assert fdry.preallocate_region(cfg["load_prealloc_mb"] * MB), "preallocate_region"
    buffer = _buffer(group, cfg["nvl_mb"])
    count = fdry.init_nvshmem_for_loaded_modules()
    print(f"[rank {rank}] NVSHMEM init for {count} loaded modules", flush=True)
    x, topk = _inputs(rank)
    with open(os.path.join(rank_archive, "meta.json")) as f:
        saved = json.load(f)
    assert (x.data_ptr(), topk.data_ptr()) == (saved["x"], saved["topk"]), (
        f"[rank {rank}] LOAD inputs not at the SAVE addresses"
    )
    stats = torch.zeros((EXPERTS_PER_RANK,), dtype=torch.int, device="cuda")

    recv_x, recv_count, *_ = _dispatch(buffer, x, topk, stats)  # registers DeepEP's modules
    _verify(recv_x, recv_count, rank, "LOAD eager")
    dist.barrier(group)

    # CUDAGraph.load does not merge the root-level common_kernel_node_attrs that
    # DeepEP's graph needs; the engines use start_graph_builds + finish_graph_loads.
    pending = fdry.CUDAGraph.start_graph_builds([os.path.join(rank_archive, "graph.json")])
    ((graph, outputs),) = fdry.CUDAGraph.finish_graph_loads(pending)
    recv_x, recv_count = outputs[0], outputs[1]
    for i in range(2):
        recv_x.zero_()
        recv_count.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)
        graph.replay()
        _verify(recv_x, recv_count, rank, f"LOAD replay {i + 1}")
    dist.barrier(group)
    fdry.stop_allocation_region()
    buffer.destroy()
    dist.destroy_process_group()


@pytest.mark.parametrize("case", list(CASES))
def test_deepep_low_latency_dispatch_save_load(workdir, case):
    run_phase(__file__, "save", workdir, case, env=_env(), preload_nvshmem=True)
    run_phase(__file__, "load", workdir, case, env=_env(), preload_nvshmem=True)


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    fn = {"--save": _save_rank, "--load": _load_rank}[mode]
    torch.multiprocessing.spawn(fn, args=(archive, *rest), nprocs=NUM_RANKS)
