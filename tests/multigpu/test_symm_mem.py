# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""SAVE / LOAD of a torch symmetric-memory all-reduce graph (2 GPUs).

Captures the op sequence of sglang's torch symm-mem all-reduce
(device_communicators/torch_symm_mem.py: copy-in -> all-reduce in place on
the persistent symmetric buffer -> copy-out) into a CUDA graph, saves it, and
LOADs it in fresh processes. The graph bakes the local buffer VA and the
peer-pointer device arrays: LOAD must place them (and the input / output
tensors) at the SAVE addresses and replay exact sums three times.

    pytest tests/multigpu/test_symm_mem.py -q -s
    pytest tests/multigpu/test_symm_mem.py -q -s -k two_shot

algo: two_shot (TP=2 on sm90), one_shot, multimem (skipped when the
rendezvous has no multicast object). load_api: single (CUDAGraph.load) or
parallel (start_graph_builds + finish_graph_loads). TEST_BUF_MB sets the
symmetric buffer size (default 8 MiB).
"""

import inspect
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist
from _helpers import (
    free_port,
    map_saved_layout,
    run_phase,
    save_layout,
)

BASE_ADDR = 0x600000000000
REGION_SIZE_STR = "32GB"
NUM_RANKS = 2
NO_MULTICAST = "NO_MULTICAST"
# NCCL buffers through the plain allocator, as the sglang integration pins it.
_ENV = {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0"}


def _init_dist(local_rank: int, num_ranks: int):
    params = {
        "backend": "nccl",
        "init_method": f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
        "world_size": num_ranks,
        "rank": local_rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def _buf_elems():
    return int(os.environ.get("TEST_BUF_MB", "8")) * 1024 * 1024 // 2  # bf16


def _make_input(rank: int, n: int) -> torch.Tensor:
    # Small integers: exact in bf16, rank-distinguishable, deterministic.
    base = torch.arange(n, device="cuda", dtype=torch.float32) % 13
    return (base + rank + 1).to(torch.bfloat16)


def _expected_sum(n: int, num_ranks: int) -> torch.Tensor:
    base = torch.arange(n, device="cuda", dtype=torch.float32) % 13
    return (base * num_ranks + sum(r + 1 for r in range(num_ranks))).to(torch.bfloat16)


def _run_allreduce(algo: str, buf: torch.Tensor, n: int, group_name: str):
    view = buf[:n]
    if algo == "multimem":
        torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
    elif algo == "one_shot":
        # Out of place: the result is a new tensor, copied back into the buffer.
        view.copy_(torch.ops.symm_mem.one_shot_all_reduce(view, "sum", group_name))
    else:
        torch.ops.symm_mem.two_shot_all_reduce_(view, "sum", group_name)


def _step(algo, buf, inp, out, n, group_name):
    buf[:n].copy_(inp)
    _run_allreduce(algo, buf, n, group_name)
    out.copy_(buf[:n])


def _verify(out: torch.Tensor, n: int, num_ranks: int, rank: int, what: str):
    expected = _expected_sum(n, num_ranks)
    bad = (out[:n].float() != expected.float()).nonzero()
    assert bad.numel() == 0, (
        f"[rank {rank}] {what}: all-reduce wrong at {bad[:5].flatten().tolist()}, "
        f"out[:5]={out[:5].float().tolist()} expected[:5]={expected[:5].float().tolist()}"
    )


def _setup_symm_buffer(group):
    """Allocate + rendezvous the persistent symmetric buffer (as engine init does)."""
    import torch.distributed._symmetric_memory as torch_symm_mem

    buf = torch_symm_mem.empty(
        _buf_elems(), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.bfloat16
    )
    hdl = torch_symm_mem.rendezvous(buf, group.group_name)
    info = {"buffer_va": buf.data_ptr()}
    for key in ("buffer_ptrs_dev", "signal_pad_ptrs_dev", "multicast_ptr"):
        info[key] = int(getattr(hdl, key, 0) or 0)
    return buf, info


def _rank_archive(archive, rank):
    return os.path.join(archive, f"rank_{rank}")


def _save_rank(local_rank: int, archive: str, algo: str):
    import foundry as fdry

    rank, num_ranks, group = _init_dist(local_rank, NUM_RANKS)
    torch.cuda.init()
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    n = _buf_elems()
    buf, info = _setup_symm_buffer(group)
    if algo == "multimem" and info["multicast_ptr"] == 0:
        if rank == 0:
            open(os.path.join(archive, NO_MULTICAST), "w").close()
        dist.barrier(group)
        fdry.stop_allocation_region()
        dist.destroy_process_group()
        return

    inp = _make_input(rank, n)
    out = torch.empty_like(inp)
    _step(algo, buf, inp, out, n, group.group_name)  # eager warm-up: loads the kernels
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "SAVE eager")
    dist.barrier(group)

    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        _step(algo, buf, inp, out, n, group.group_name)
    out.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    graph.replay()
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "SAVE replay")

    rank_archive = _rank_archive(archive, rank)
    os.makedirs(rank_archive, exist_ok=True)
    graph.save(os.path.join(rank_archive, "graph.json"), output_tensors=[out])
    fdry.pack_fatbins_to_folder(rank_archive)
    save_layout(rank_archive)
    # graph.save re-enables packing on exit; the ranks would race on ./hook_archive.
    fdry.set_pack_fatbins_on_exit(False)
    meta = {"info": info, "inp_va": inp.data_ptr(), "out_va": out.data_ptr()}
    with open(os.path.join(rank_archive, "meta.json"), "w") as f:
        json.dump(meta, f)
    dist.barrier(group)
    fdry.stop_allocation_region()
    dist.destroy_process_group()


def _load_rank(local_rank: int, archive: str, algo: str, load_api: str):
    import foundry as fdry

    rank, num_ranks, group = _init_dist(local_rank, NUM_RANKS)
    torch.cuda.init()
    rank_archive = _rank_archive(archive, rank)
    fdry.set_skip_fatbin_processing(True)
    fdry.load_cuda_modules_and_libraries(rank_archive)
    fdry.set_allocation_region(BASE_ADDR, fdry.parse_size(REGION_SIZE_STR))
    n = _buf_elems()
    buf, info = _setup_symm_buffer(group)
    inp = _make_input(rank, n)
    out = torch.empty_like(inp)

    # The graph bakes these addresses: LOAD must place them as SAVE did.
    with open(os.path.join(rank_archive, "meta.json")) as f:
        saved = json.load(f)
    want = {**saved["info"], "inp_va": saved["inp_va"], "out_va": saved["out_va"]}
    got = {**info, "inp_va": inp.data_ptr(), "out_va": out.data_ptr()}
    diverged = {k: (hex(want[k]), hex(got[k])) for k in want if want[k] != got[k]}
    assert not diverged, f"[rank {rank}] LOAD placement differs from SAVE: {diverged}"

    # Eager warm-up: loads the symm-mem kernels and validates the live rendezvous.
    _step(algo, buf, inp, out, n, group.group_name)
    torch.cuda.synchronize()
    _verify(out, n, num_ranks, rank, "LOAD eager")
    dist.barrier(group)

    graph_json = os.path.join(rank_archive, "graph.json")
    if load_api == "single":
        graph, outputs = fdry.CUDAGraph.load(graph_json)
    else:
        map_saved_layout(rank_archive)  # one_shot copies from graph-pool memory
        pending = fdry.CUDAGraph.start_graph_builds([graph_json])
        ((graph, outputs),) = fdry.CUDAGraph.finish_graph_loads(pending)
    loaded_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    assert loaded_out.data_ptr() == saved["out_va"], hex(loaded_out.data_ptr())
    for i in range(3):  # repeated replays catch state a single replay leaves corrupt
        loaded_out.zero_()
        torch.cuda.synchronize()
        dist.barrier(group)
        graph.replay()
        torch.cuda.synchronize()
        _verify(loaded_out, n, num_ranks, rank, f"LOAD replay {i + 1}")
    dist.barrier(group)
    fdry.stop_allocation_region()
    dist.destroy_process_group()


def _env():
    return {**_ENV, "MASTER_PORT": str(free_port())}


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One SAVE per algorithm, shared by both load APIs (None: no multicast)."""
    archives = {}

    def get(algo):
        if algo not in archives:
            work = tmp_path_factory.mktemp(f"symm_mem_{algo}")
            run_phase(__file__, "save", work, algo, env=_env())
            archives[algo] = None if (work / "archive" / NO_MULTICAST).exists() else work
        if archives[algo] is None:
            pytest.skip("the symmetric-memory rendezvous has no multicast object")
        return archives[algo]

    return get


@pytest.mark.parametrize("load_api", ["single", "parallel"])
@pytest.mark.parametrize("algo", ["two_shot", "one_shot", "multimem"])
def test_symm_mem_allreduce_save_load(saved, algo, load_api):
    run_phase(__file__, "load", saved(algo), algo, load_api, env=_env())


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    fn = {"--save": _save_rank, "--load": _load_rank}[mode]
    torch.multiprocessing.spawn(fn, args=(archive, *rest), nprocs=NUM_RANKS)
