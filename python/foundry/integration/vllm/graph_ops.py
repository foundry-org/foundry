# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Graph capture / save / load mechanics.

Provides ``capture_or_load_graph``, ``start_graph_builds``,
``save_graph_manifest``, ``pack_fatbins`` and helpers.

LOAD uses per-graph finish: ``capture_or_load_graph`` calls
``FoundryCUDAGraph.finish_one_graph_load(pending, index)`` for the
current batch's identifier, so allocator-event replay interleaves with
the upstream capture loop's between-capture work.

``start_graph_builds`` is invoked from ``_load_model_with_overlap``
in the worker process (not ``EngineCore._initialize_kv_caches``) so it
works across all executor types. NVSHMEM has already been bootstrapped
by the time we are called (see ``_patch_prepare_comm_buffer``).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.utils.torch_utils import weak_ref_tensors

import foundry as foundry_pkg
from foundry import ops as fops
from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.graph import graph as foundry_graph_ctx
from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)
from foundry.integration.vllm.runtime import get_state

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE  # noqa: F401

log = init_logger("vllm.foundry.graph_ops")


# (runtime_mode, batch_descriptor, piecewise_compile_index)
GraphIdentifier = tuple[CUDAGraphMode, BatchDescriptor, "int | None"]


# ---------------------------------------------------------------------------
# Identifier / filename / metadata serialization
# ---------------------------------------------------------------------------


def _piecewise_index(runtime_mode: CUDAGraphMode, runnable: Callable) -> int | None:
    """``piecewise_compile_index`` if this is a PIECEWISE batch, else None.

    Used by both SAVE (when writing the per-graph filename) and LOAD (when
    looking up the pre-built template index).
    """
    if runtime_mode != CUDAGraphMode.PIECEWISE:
        return None
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    assert isinstance(runnable, PiecewiseBackend), (
        "For PIECEWISE mode, runnable must be PiecewiseBackend"
    )
    return runnable.piecewise_compile_index


def _build_graph_filename(
    index: int,
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor,
    piecewise_compile_index: int | None,
) -> str:
    mode = runtime_mode.name
    tokens = batch_descriptor.num_tokens
    reqs = batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else "N"
    uniform = "U" if batch_descriptor.uniform else "X"
    lora = "L" if batch_descriptor.has_lora else "X"
    pc = piecewise_compile_index if piecewise_compile_index is not None else "N"
    return f"graph_{index}_{mode}_t{tokens}_r{reqs}_{uniform}{lora}_pc{pc}.json"


_GRAPH_FILENAME_RE = re.compile(
    r"graph_(\d+)_([A-Z_]+)_t(\d+)_r(\d+|N)_([UX])([LX])_pc(\d+|N)\.json$"
)


def _parse_graph_filename(
    filename: str,
) -> tuple[int, CUDAGraphMode, BatchDescriptor, int | None] | None:
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
    graph_pool: _POOL_HANDLE,
    runnable_args: tuple = (),
    runnable_kwargs: dict | None = None,
):
    """Capture (NONE/SAVE) or look up (LOAD) a CUDA graph for one batch.

    Returns a ``(graph, output)`` tuple. ``graph`` is either
    ``torch.cuda.CUDAGraph`` (NONE) or ``foundry.CUDAGraph`` (SAVE/LOAD).
    """
    if runnable_kwargs is None:
        runnable_kwargs = {}

    mode = get_graph_extension_mode()

    if mode == CUDAGraphExtensionMode.NONE:
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

    piecewise_compile_index = _piecewise_index(runtime_mode, runnable)

    if mode == CUDAGraphExtensionMode.SAVE:
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

    if mode == CUDAGraphExtensionMode.LOAD:
        identifier: GraphIdentifier = (
            runtime_mode,
            batch_descriptor,
            piecewise_compile_index,
        )
        if _pending_graph_builds is None:
            raise RuntimeError(
                "capture_or_load_graph called on LOAD but no pending graph "
                "builds — was start_graph_builds() called first?"
            )
        index = _pending_graph_builds.identifier_to_index.get(identifier)
        if index is None:
            raise RuntimeError(f"Graph identifier not in pending builds: {identifier}")
        graph, output = FoundryCUDAGraph.finish_one_graph_load(_pending_graph_builds.pending, index)
        # Strong-reference the (graph, output) pair. The upstream
        # CUDAGraphWrapper stores `entry.output = weak_ref_tensors(output)`,
        # so without this dict the VMM-backed output tensors get GC'd between
        # captures and the next replay writes to freed memory → CUDA illegal
        # memory access. Don't drop this line.
        state.loaded_graphs[identifier] = (graph, output)
        return graph, output

    raise RuntimeError(
        f"capture_or_load_graph called with mode {mode}, expected NONE, SAVE, or LOAD"
    )


# ---------------------------------------------------------------------------
# LOAD pipeline: start_graph_builds (background) → per-graph finish on demand
#
# capture_or_load_graph calls finish_one_graph_load(pending, index) per
# batch as the patched capture_model's loop walks them. This keeps the
# foundry VMM cursor on LOAD walking the same trajectory it did on SAVE:
# allocations between captures (attention metadata, dummy inputs) happen
# on both sides in the same order, interleaved with replayed graph events.
# ---------------------------------------------------------------------------


@dataclass
class _PendingGraphBuilds:
    pending: object  # opaque PendingGraphLoads handle from C++
    identifier_to_index: dict[GraphIdentifier, int]


_pending_graph_builds: _PendingGraphBuilds | None = None


def _scan_graph_files(
    workspace_dir: str,
) -> list[tuple[int, str, tuple]]:
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

    Called from ``_load_model_with_overlap`` (worker process) on LOAD,
    AFTER ``do_original_load()`` completes. Background threads build
    cuGraph templates and prepare on-demand graphs concurrently with
    everything that happens between weight load and ``capture_model``
    (KV cache init etc). Running template builds in parallel with the
    weight load itself was tried earlier and dropped: it slowed weight
    load down (driver contention) without recovering meaningful time
    on the template side.

    By the time we are called, NVSHMEM is already bootstrapped — the
    DeepEP Buffer creation inside ``prepare_communication_buffer_for_model``
    runs inside ``do_original_load`` and ``_patch_prepare_comm_buffer``'s
    post-orig hook has flushed ``init_nvshmem_for_loaded_modules``.
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
    json_paths = [os.path.join(workspace_dir, filename) for _, filename, _ in graph_files]
    if not json_paths:
        return

    num_threads = 4
    s = time.perf_counter()
    log.info(
        "[foundry] Starting early graph builds (%d graphs, %d threads)",
        len(json_paths),
        num_threads,
    )
    pending = FoundryCUDAGraph.start_graph_builds(json_paths, num_threads=num_threads)
    log.info(
        "[foundry] start_graph_builds returned in %.3fs (building in background)",
        time.perf_counter() - s,
    )

    # Parsed-filename metadata is already in (runtime_mode, batch_descriptor,
    # piecewise_compile_index) shape — the same shape as the runtime identifier
    # built inside capture_or_load_graph — so just reuse it directly.
    identifier_to_index: dict[GraphIdentifier, int] = {
        metadata: i for i, (_index, _filename, metadata) in enumerate(graph_files)
    }

    _pending_graph_builds = _PendingGraphBuilds(
        pending=pending,
        identifier_to_index=identifier_to_index,
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
    fops.pack_fatbins_to_folder(cfg.workspace_dir)
    fops.set_pack_fatbins_on_exit(False)
