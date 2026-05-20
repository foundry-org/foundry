# SPDX-License-Identifier: Apache-2.0
"""Graph capture / save / load mechanics.

Ports ``capture_or_load_graph``, ``start_graph_builds``,
``preload_all_graphs``, ``save_graph_manifest``, ``pack_fatbins`` and
helpers from ``vllm-cge/vllm/compilation/graph_extension.py:794–1156``.

The ``start_graph_builds`` call site has moved (audit fix): it is no
longer called from ``EngineCore._initialize_kv_caches`` (which only
worked under UniprocExecutor); it is called from
``_load_model_with_cge_overlap`` in the worker process so it works
across all executor types. See doc 05.
"""

from __future__ import annotations

import os
import re
import time

from vllm.logger import init_logger
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

import foundry as foundry_pkg
from foundry import ops as cge
from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.graph import graph as foundry_graph_ctx

from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)
from foundry.integration.vllm.runtime import get_state

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.utils.torch_utils import weak_ref_tensors

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

log = init_logger("vllm.foundry.graph_ops")


# (runtime_mode, batch_descriptor, piecewise_compile_index)
GraphIdentifier = tuple[CUDAGraphMode, BatchDescriptor, "int | None"]


# ---------------------------------------------------------------------------
# Identifier / filename / metadata serialization
# ---------------------------------------------------------------------------


def _build_graph_identifier(
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor,
    piecewise_compile_index: "int | None",
) -> GraphIdentifier:
    return (runtime_mode, batch_descriptor, piecewise_compile_index)


def _build_graph_filename(
    index: int,
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor,
    piecewise_compile_index: "int | None",
) -> str:
    mode = runtime_mode.name
    tokens = batch_descriptor.num_tokens
    reqs = (
        batch_descriptor.num_reqs
        if batch_descriptor.num_reqs is not None
        else "N"
    )
    uniform = "U" if batch_descriptor.uniform else "X"
    lora = "L" if batch_descriptor.has_lora else "X"
    pc = (
        piecewise_compile_index
        if piecewise_compile_index is not None
        else "N"
    )
    return f"graph_{index}_{mode}_t{tokens}_r{reqs}_{uniform}{lora}_pc{pc}.json"


_GRAPH_FILENAME_RE = re.compile(
    r"graph_(\d+)_([A-Z_]+)_t(\d+)_r(\d+|N)_([UX])([LX])_pc(\d+|N)\.json$"
)


def _parse_graph_filename(
    filename: str,
) -> "tuple[int, CUDAGraphMode, BatchDescriptor, int | None] | None":
    m = _GRAPH_FILENAME_RE.match(filename)
    if not m:
        return None
    index = int(m.group(1))
    mode = CUDAGraphMode[m.group(2)]
    num_tokens = int(m.group(3))
    num_reqs = int(m.group(4)) if m.group(4) != "N" else None
    uniform = m.group(5) == "U"
    has_lora = m.group(6) == "L"
    pc = int(m.group(7)) if m.group(7) != "N" else None
    bd = BatchDescriptor(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform=uniform,
        has_lora=has_lora,
    )
    return index, mode, bd, pc


# ---------------------------------------------------------------------------
# capture_or_load_graph — the single entry from CUDAGraphWrapper.__call__
# ---------------------------------------------------------------------------


def capture_or_load_graph(
    batch_descriptor: BatchDescriptor,
    runnable: Callable,
    weak_ref_output: bool,
    runtime_mode: CUDAGraphMode,
    graph_pool: "_POOL_HANDLE",
    runnable_args: tuple = (),
    runnable_kwargs: "dict | None" = None,
):
    """Capture (NONE/SAVE) or look up (LOAD) a CUDA graph for one batch.

    Returns a ``(graph, output)`` tuple. ``graph`` is either
    ``torch.cuda.CUDAGraph`` (NONE) or ``foundry.CUDAGraph`` (SAVE/LOAD).
    """
    if runnable_kwargs is None:
        runnable_kwargs = {}

    cge_mode = get_graph_extension_mode()

    if cge_mode == CUDAGraphExtensionMode.NONE:
        cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cudagraph, pool=graph_pool):
            output = runnable(*runnable_args, **runnable_kwargs)
            if weak_ref_output:
                output = weak_ref_tensors(output)
        return cudagraph, output

    cfg = get_config()
    state = get_state()
    if cfg is None or state is None:
        raise RuntimeError("Graph extension is not initialized")

    if cge_mode == CUDAGraphExtensionMode.SAVE:
        from vllm.compilation.piecewise_backend import PiecewiseBackend

        piecewise_compile_index: "int | None" = None
        if runtime_mode == CUDAGraphMode.PIECEWISE:
            assert isinstance(runnable, PiecewiseBackend), (
                "For PIECEWISE mode, runnable must be PiecewiseBackend"
            )
            piecewise_compile_index = runnable.piecewise_compile_index

        graph = FoundryCUDAGraph()
        with foundry_graph_ctx(graph, pool=graph_pool):
            output = runnable(*runnable_args, **runnable_kwargs)

        assert cfg.workspace_dir is not None
        # The filename alone encodes everything LOAD needs (mode, batch_descriptor,
        # piecewise index). The JSON body that foundry.graph.save writes is for
        # debugging only — we don't read it back on LOAD.
        filename = _build_graph_filename(
            state.capture_index,
            runtime_mode,
            batch_descriptor,
            piecewise_compile_index,
        )
        json_path = os.path.join(cfg.workspace_dir, filename)
        graph.save(json_path, output)
        state.capture_index += 1

        if weak_ref_output:
            output = weak_ref_tensors(output)
        return graph, output

    elif cge_mode == CUDAGraphExtensionMode.LOAD:
        from vllm.compilation.piecewise_backend import PiecewiseBackend

        piecewise_compile_index = None
        if runtime_mode == CUDAGraphMode.PIECEWISE:
            assert isinstance(runnable, PiecewiseBackend), (
                "For PIECEWISE mode, runnable must be PiecewiseBackend"
            )
            piecewise_compile_index = runnable.piecewise_compile_index

        identifier = _build_graph_identifier(
            runtime_mode, batch_descriptor, piecewise_compile_index
        )
        if identifier not in state.loaded_graphs:
            raise RuntimeError(
                f"Graph not found in preloaded graphs: {identifier}"
            )
        graph, output = state.loaded_graphs[identifier]
        return graph, output

    else:
        raise RuntimeError(
            f"capture_or_load_graph called with mode {cge_mode}, "
            "expected NONE, SAVE, or LOAD"
        )


# ---------------------------------------------------------------------------
# Two-phase LOAD pipeline: start_graph_builds + preload_all_graphs
# ---------------------------------------------------------------------------


_pending_graph_builds: tuple | None = None


def _scan_graph_files(
    workspace_dir: str,
) -> "list[tuple[int, str, tuple]]":
    """Return (index, filename, (mode, batch_descriptor, pc)) per graph file,
    sorted by index. Only the encoded-filename format is accepted — SAVE
    always writes that format, and LOAD never needs to read the JSON body."""
    graph_files: list[tuple[int, str, tuple]] = []
    for filename in os.listdir(workspace_dir):
        if not filename.startswith("graph_") or not filename.endswith(".json"):
            continue
        parsed = _parse_graph_filename(filename)
        if parsed is None:
            continue
        index, mode, batch_desc, pc = parsed
        graph_files.append((index, filename, (mode, batch_desc, pc)))
    graph_files.sort(key=lambda x: x[0])
    return graph_files


def start_graph_builds() -> None:
    """Phase 1: kick off background template builds.

    Called from ``_load_model_with_cge_overlap`` (worker process) on
    LOAD, after ``preallocate_for_load_mode`` for dense, or after
    NVSHMEM init + preallocate for EP.
    """
    global _pending_graph_builds
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None:
        return
    if cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    workspace_dir = cfg.workspace_dir
    if workspace_dir is None:
        return

    graph_files = _scan_graph_files(workspace_dir)
    json_paths = [
        os.path.join(workspace_dir, filename) for _, filename, _ in graph_files
    ]
    if not json_paths:
        return

    num_threads = 4
    s = time.perf_counter()
    log.info(
        "[CGE] Starting early graph builds (%d graphs, %d threads)",
        len(json_paths),
        num_threads,
    )
    pending = FoundryCUDAGraph.start_graph_builds(
        json_paths, num_threads=num_threads
    )
    e = time.perf_counter()
    log.info(
        "[CGE] start_graph_builds returned in %.3fs (building in background)",
        e - s,
    )
    _pending_graph_builds = (pending, graph_files)


def preload_all_graphs(
    graph_pool: "_POOL_HANDLE | None" = None,
) -> None:
    """Phase 2: NVSHMEM init for loaded modules, then finish_graph_loads.

    Called from the patched ``capture_model`` on LOAD. Populates
    ``state.loaded_graphs`` so ``capture_or_load_graph`` can satisfy
    inference-time lookups.
    """
    global _pending_graph_builds
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None:
        raise RuntimeError("Graph extension is not initialized")

    workspace_dir = cfg.workspace_dir
    if workspace_dir is None:
        raise RuntimeError("Workspace directory is not set")

    # Init NVSHMEM for any modules pending init from load_cuda_modules.
    t = time.perf_counter()
    n = cge.init_nvshmem_for_loaded_modules()
    log.info(
        "[CGE TIMING] init_nvshmem_for_loaded_modules: %.3f s (%d modules)",
        time.perf_counter() - t,
        n,
    )

    if _pending_graph_builds is None:
        log.warning(
            "[CGE] preload_all_graphs called but no pending graph builds. "
            "Was start_graph_builds() called first?"
        )
        return

    pending, graph_files = _pending_graph_builds
    _pending_graph_builds = None

    s = time.perf_counter()
    load_results = FoundryCUDAGraph.finish_graph_loads(pending)
    e = time.perf_counter()
    log.info(
        "[CGE] finish_graph_loads completed in %.3fs (%d graphs)",
        e - s,
        len(load_results),
    )

    for i, (_index, _filename, metadata) in enumerate(graph_files):
        graph, output = load_results[i]
        runtime_mode, batch_descriptor, piecewise_compile_index = metadata
        identifier = _build_graph_identifier(
            runtime_mode, batch_descriptor, piecewise_compile_index
        )
        state.loaded_graphs[identifier] = (graph, output)

    log.info(
        "Preloaded %d CUDA graphs from %s", len(graph_files), workspace_dir
    )


# ---------------------------------------------------------------------------
# SAVE post-capture: manifest + fatbin packing
# ---------------------------------------------------------------------------


def save_graph_manifest() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    foundry_pkg.save_graph_manifest(cfg.workspace_dir)


def pack_fatbins() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Graph extension is not initialized")
    cge.pack_fatbins_to_folder(cfg.workspace_dir)
    cge.set_pack_fatbins_on_exit(False)
