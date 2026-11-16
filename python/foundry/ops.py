# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry's native entry points.

The native part, ``libfoundry_ops.so`` next to this file, registers its functions as
torch custom ops in the ``foundry`` namespace through the LibTorch Stable ABI and has
no Python C API in it: ``torch.ops.load_library`` loads it and every call goes through
``torch.ops.foundry.<op>``, so one build serves every CPython and every torch >= 2.14.
This module keeps the Python surface: ``CUDAGraph`` and ``PendingGraphLoads`` are
classes over int64 handles of the native objects; tuples, output containers, events
and the torch version are handled here. torch releases the GIL around every op.
"""

from __future__ import annotations

import os
import re
import weakref
from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "CUDAGraph",
    "PendingGraphLoads",
    "event_elapsed_time",
    "event_query",
    "event_record",
    "event_synchronize",
    "free_preallocated_region",
    "get_current_alloc_offset",
    "get_graph_ptr",
    "get_live_region_ranges",
    "init_nvshmem_for_loaded_modules",
    "load_cuda_modules_and_libraries",
    "native_record_abort",
    "native_record_begin",
    "pack_fatbins_to_folder",
    "preallocate_cublas_workspaces",
    "preallocate_ranges",
    "preallocate_region",
    "recorded_allocation_contains",
    "resume_allocation_region",
    "set_allocation_region",
    "set_current_alloc_offset",
    "set_nvshmem_auto_init",
    "set_pack_fatbins_on_exit",
    "set_skip_fatbin_processing",
    "stop_allocation_region",
    "stream_wait_event",
]

LIBRARY_NAME = "libfoundry_ops.so"
NAMESPACE = "foundry"


def _registered_library(ns) -> str | None:
    """Path of the library that registered ``torch.ops.<NAMESPACE>``, None if nothing did."""
    try:
        op = ns.library_path
    except (AttributeError, RuntimeError):
        return None
    return op()


def load_library(path: str):
    """Load the native library at ``path`` once per process and return its op namespace.

    A second copy registering the same namespace would crash the process inside
    ``torch.ops.load_library`` (torch's stable ``StableLibrary`` ignores the
    "only one TORCH_LIBRARY per namespace" error), so a copy from another path is
    refused with ImportError before it is loaded.
    """
    path = os.path.realpath(path)
    ns = getattr(torch.ops, NAMESPACE)
    other = _registered_library(ns)
    if other is not None:
        if os.path.realpath(other) != path:
            raise ImportError(
                f"foundry.ops: the torch op namespace '{NAMESPACE}' is already registered by "
                f"{other}; a second Foundry library ({path}) cannot be loaded into the same "
                "process"
            )
        return ns
    try:
        torch.ops.load_library(path)
    except OSError as exc:
        raise ImportError(f"cannot load {path}: {exc}") from exc
    return ns


_C = load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), LIBRARY_NAME))

# Output kinds of save / load (csrc/binding.cpp kOut*).
_OUT_NONE, _OUT_SINGLE, _OUT_LIST, _OUT_TUPLE = 0, 1, 2, 3


def _i64(value: int) -> int:
    """A uint64 (seed, address) as the int64 bit pattern the op schema's ``int`` takes."""
    value = int(value)
    return value - (1 << 64) if value >= (1 << 63) else value


def _pool(pool: Sequence[int] | None) -> list[int] | None:
    return None if pool is None else [int(pool[0]), int(pool[1])]


def _outputs(kind: int, tensors: list[torch.Tensor]):
    if kind == _OUT_SINGLE:
        return tensors[0]
    if kind == _OUT_LIST:
        return list(tensors)
    if kind == _OUT_TUPLE:
        return tuple(tensors)
    return None


# ---------------------------------------------------------------- module functions


def native_record_begin() -> None:
    """SAVE: open the hook record window around torch's own capture (foundry/graph.py)."""
    _C.native_record_begin()


def native_record_abort() -> None:
    _C.native_record_abort()


def recorded_allocation_contains(addr: int, len: int) -> bool:
    """SAVE/LOAD: [addr, addr+len) lies in a live driver allocation Foundry recorded / replayed."""
    return _C.recorded_allocation_contains(addr, len)


def get_graph_ptr(graph: CUDAGraph) -> int:
    """Address of the native CUDAGraph object (hook integration)."""
    return _C.graph_ptr(graph._graph_handle)


def set_allocation_region(base: int, size: int) -> None:
    """Set the allocation region for the CUDA VMM hooks."""
    _C.set_allocation_region(base, size)


def stop_allocation_region() -> None:
    _C.stop_allocation_region()


def resume_allocation_region() -> None:
    _C.resume_allocation_region()


def preallocate_region(size: int) -> bool:
    """Preallocate memory in the allocation region for fast subsequent allocations."""
    return _C.preallocate_region(size)


def get_live_region_ranges() -> list[tuple[int, int]]:
    """Live (offset, size) ranges inside the allocation region, relative to its base."""
    flat = _C.get_live_region_ranges()
    return list(zip(flat[0::2], flat[1::2]))


def preallocate_ranges(ranges: Sequence[tuple[int, int]], end_offset: int) -> bool:
    """Map only the given live ranges and move the cursor to end_offset (sparse LOAD
    preallocation)."""
    flat = []
    for offset, size in ranges:
        flat += (int(offset), int(size))
    return _C.preallocate_ranges(flat, end_offset)


def free_preallocated_region() -> None:
    _C.free_preallocated_region()


def get_current_alloc_offset() -> int:
    """Current allocation offset from the region base address."""
    return _C.get_current_alloc_offset()


def set_current_alloc_offset(offset: int) -> None:
    _C.set_current_alloc_offset(offset)


def set_pack_fatbins_on_exit(enabled: bool) -> None:
    _C.set_pack_fatbins_on_exit(bool(enabled))


def set_skip_fatbin_processing(enabled: bool) -> None:
    """Skip heavy fatbin processing in LOAD mode for faster startup."""
    _C.set_skip_fatbin_processing(bool(enabled))


def set_nvshmem_auto_init(enabled: bool) -> None:
    """Enable/disable automatic NVSHMEM initialization for DeepEP kernels on load."""
    _C.set_nvshmem_auto_init(bool(enabled))


def init_nvshmem_for_loaded_modules() -> int:
    """Initialize NVSHMEM for the loaded modules that need it (after the NVSHMEM runtime is
    up, e.g. after DeepEP Buffer creation). Returns the number of modules initialized."""
    return _C.init_nvshmem_for_loaded_modules()


def pack_fatbins_to_folder(folder_path: str) -> None:
    _C.pack_fatbins_to_folder(os.fsdecode(folder_path))


def load_cuda_modules_and_libraries(archive_dir: str) -> None:
    _C.load_cuda_modules_and_libraries(os.fsdecode(archive_dir))


def preallocate_cublas_workspaces() -> None:
    """Pre-allocate the cuBLAS handle and workspace (= the unified cuBLASLt workspace) for
    the current stream."""
    # Running torch's major.minor ("2.14.1+cu130", "2.16.0.dev20261102", "2.16.0a0+git").
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if m is None:
        raise RuntimeError(
            f"preallocate_cublas_workspaces: cannot parse torch version {torch.__version__}"
        )
    major, minor = int(m[1]), int(m[2])
    _C.preallocate_cublas_workspaces(major > 2 or (major == 2 and minor >= 15))


# CUevent / CUstream handles as integers (foundry.graph.RestoredEvent). Driver errors raise
# RuntimeError with the error name; event_query maps CUDA_ERROR_NOT_READY to False.


def event_query(handle: int) -> bool:
    return _C.event_query(handle)


def event_synchronize(handle: int) -> None:
    _C.event_synchronize(handle)


def event_elapsed_time(start: int, end: int) -> float:
    return _C.event_elapsed_time(start, end)


def event_record(handle: int, stream: int) -> None:
    _C.event_record(handle, stream)


def stream_wait_event(stream: int, handle: int) -> None:
    _C.stream_wait_event(stream, handle)


# ---------------------------------------------------------------- handle classes

# Ops the finalizers call: bound here so that a finalizer never looks up torch.ops.
_graph_destroy = _C.graph_destroy
_pending_destroy = _C.pending_destroy
_graph_replay = _C.graph_replay


def _own(obj: Any, handle: int, destroy) -> weakref.finalize:
    """Destroy ``handle`` exactly once, when ``obj`` dies. Not at interpreter exit
    (atexit=False): the process tears the native objects down with it, as it did for
    pybind11 objects still alive then."""
    fin = weakref.finalize(obj, destroy, handle)
    fin.atexit = False
    return fin


class PendingGraphLoads:
    """Opaque handle of ``CUDAGraph.start_graph_builds``; pass it to ``finish_*``."""

    def __init__(self, *args, **kwargs):
        raise TypeError("foundry.ops.PendingGraphLoads: No constructor defined!")

    @classmethod
    def _wrap(cls, handle: int) -> PendingGraphLoads:
        obj = object.__new__(cls)
        obj._pending_handle = handle
        obj._finalizer = _own(obj, handle, _pending_destroy)
        return obj

    @property
    def build_seconds(self) -> float:
        """Seconds from start_graph_builds' entry to the end of Phase 2; waits for the
        background builds first (0 for an empty or failed build)."""
        return _C.pending_build_seconds(self._pending_handle)


def _pending(pending: PendingGraphLoads) -> int:
    if not isinstance(pending, PendingGraphLoads):
        raise TypeError(f"expected foundry.ops.PendingGraphLoads, got {type(pending).__name__}")
    return pending._pending_handle


class CUDAGraph:
    """A Foundry CUDA graph: a handle of the native ``foundry::CUDAGraph``.

    The native object lives until this Python object dies (or longer, while pending
    loads or other handles share it); ``reset()`` releases its CUDA graph resources.
    Subclassable (``foundry.graph.CUDAGraph``); graphs returned by ``load`` and
    ``finish_*`` are instances of this class.
    """

    def __new__(cls, keep_graph: bool = False):
        obj = super().__new__(cls)
        obj._bind(_C.graph_create(bool(keep_graph)))
        return obj

    def _bind(self, handle: int) -> None:
        self._graph_handle = handle
        self._graph_finalizer = _own(self, handle, _graph_destroy)

    @classmethod
    def _wrap(cls, handle: int) -> CUDAGraph:
        obj = object.__new__(CUDAGraph)
        obj._bind(handle)
        return obj

    def adopt_native_capture(
        self,
        raw_graph: int,
        pool: tuple[int, int],
        generators: Sequence[tuple[int, int, int, int, int, int]],
        rng_candidates: Sequence[int] = (),
    ) -> None:
        flat = [_i64(v) for gen in generators for v in gen]
        _C.graph_adopt_native_capture(
            self._graph_handle, raw_graph, _pool(pool), flat, [_i64(a) for a in rng_candidates]
        )

    def replay(self) -> None:
        _graph_replay(self._graph_handle)

    def exec_share_stats(self) -> str:
        return _C.graph_exec_share_stats(self._graph_handle)

    def loaded_events(self) -> list:
        """foundry.graph.LoadedEvent records, one per archived event_id (empty for a
        captured graph). Each RestoredEvent holds a reference to this graph."""
        from .graph import LoadedEvent, RestoredEvent

        ids, handles, flags, rec_counts, rec_nodes, wait_counts, wait_nodes = (
            _C.graph_loaded_events(self._graph_handle)
        )
        out = []
        r = w = 0
        for i, event_id in enumerate(ids):
            rec = list(rec_nodes[r : r + rec_counts[i]])
            wait = list(wait_nodes[w : w + wait_counts[i]])
            r += rec_counts[i]
            w += wait_counts[i]
            handle = handles[i]
            out.append(
                LoadedEvent(
                    event_id, handle, flags[i], rec, wait, RestoredEvent(handle, flags[i], self)
                )
            )
        return out

    def bind_events(self, events: dict) -> None:
        """{event_id: torch.cuda.Event | CUevent handle}"""
        ids, handles = [], []
        for key, value in events.items():
            if isinstance(value, int):
                handle = value
            else:
                if not hasattr(value, "cuda_event"):
                    raise RuntimeError("bind_events: expected a torch.cuda.Event or an int handle")
                handle = value.cuda_event
                if not handle:  # torch.cuda.Event creates its CUDA event on first record
                    value.record()
                    handle = value.cuda_event
            ids.append(int(key))
            handles.append(int(handle))
        _C.graph_bind_events(self._graph_handle, ids, handles)

    def reset(self) -> None:
        _C.graph_reset(self._graph_handle)

    def pool(self) -> tuple[int, int]:
        return tuple(_C.graph_pool(self._graph_handle))

    def enable_debug_mode(self) -> None:
        _C.graph_enable_debug_mode(self._graph_handle)

    def debug_dump(self, debug_path: str) -> None:
        _C.graph_debug_dump(self._graph_handle, os.fsdecode(debug_path))

    def raw_cuda_graph(self) -> int:
        return _C.graph_raw_cuda_graph(self._graph_handle)

    def save(self, json_path: str, output_tensors=None) -> None:
        if output_tensors is None:
            kind, tensors = _OUT_NONE, []
        elif isinstance(output_tensors, (list, tuple)):
            kind = _OUT_LIST if isinstance(output_tensors, list) else _OUT_TUPLE
            tensors = list(output_tensors)
        else:
            kind, tensors = _OUT_SINGLE, [output_tensors]
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    "output_tensors must be a Tensor, list[Tensor], or tuple[Tensor]: got "
                    f"{type(t).__name__}"
                )
        _C.graph_save(
            self._graph_handle,
            os.fsdecode(json_path),
            tensors,
            [int(t.requires_grad) for t in tensors],
            kind,
        )

    @staticmethod
    def load(json_path: str, pool: tuple[int, int] | None = None):
        """Returns (graph, outputs); outputs is None, a tensor, a list or a tuple."""
        handle, tensors, kind = _C.graph_load(os.fsdecode(json_path), _pool(pool))
        return CUDAGraph._wrap(handle), _outputs(kind, tensors)

    @staticmethod
    def start_exec_pool_prewarm(json_paths: Sequence[str]) -> None:
        _C.start_exec_pool_prewarm([os.fsdecode(p) for p in json_paths])

    @staticmethod
    def start_graph_builds(
        json_paths: Sequence[str], pool: tuple[int, int] | None = None, num_threads: int = 4
    ) -> PendingGraphLoads:
        handle = _C.start_graph_builds(
            [os.fsdecode(p) for p in json_paths], _pool(pool), num_threads
        )
        return PendingGraphLoads._wrap(handle)

    @staticmethod
    def finish_graph_loads(pending: PendingGraphLoads) -> list[tuple[CUDAGraph, Any]]:
        handles, tensors, counts, kinds = _C.finish_graph_loads(_pending(pending))
        out = []
        t = 0
        for handle, count, kind in zip(handles, counts, kinds):
            out.append((CUDAGraph._wrap(handle), _outputs(kind, tensors[t : t + count])))
            t += count
        return out

    @staticmethod
    def finish_one_graph_load(pending: PendingGraphLoads, index: int) -> tuple[CUDAGraph, Any]:
        h = _pending(pending)
        n = _C.pending_size(h)
        if not 0 <= index < n:
            raise IndexError(f"finish_one_graph_load: index {index} out of range (num_graphs={n})")
        handle, tensors, kind = _C.finish_one_graph_load(h, index)
        return CUDAGraph._wrap(handle), _outputs(kind, tensors)
