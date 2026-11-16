# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from . import ops

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

OutputTensors = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]


class RestoredEvent(torch.cuda.Event):
    """A CUevent LOAD created for a loaded graph's EventRecord/EventWait nodes,
    usable where a ``torch.cuda.Event`` is (``isinstance`` holds;
    ``stream.wait_event`` / ``stream.record_event`` call ``wait`` / ``record``).

    Every method acts on the LOAD-created handle through the driver
    (``foundry.ops.event_*``); the torch event of the base class is never
    created. The object holds a reference to the graph that owns the handle.
    ``elapsed_time`` on an event created with CU_EVENT_DISABLE_TIMING raises
    the driver's error. ``torch.cuda.Event.elapsed_time(restored)`` reads the
    base class's event of its argument (never created), so put the
    RestoredEvent first: ``restored.elapsed_time(end)``.
    """

    def __new__(cls, handle: int, flags: int = 0, owner: object = None):
        # The base event is lazy: it creates no CUDA event until its own record().
        return super().__new__(cls)

    def __init__(self, handle: int, flags: int = 0, owner: object = None):
        self._handle = handle
        self.flags = flags  # creation flags SAVE recorded (CU_EVENT_*)
        self._owner = owner

    @property
    def cuda_event(self) -> int:
        return self._handle

    def query(self) -> bool:
        return ops.event_query(self._handle)

    def synchronize(self) -> None:
        ops.event_synchronize(self._handle)

    def elapsed_time(self, end_event: torch.cuda.Event) -> float:
        return ops.event_elapsed_time(self._handle, end_event.cuda_event)

    def record(self, stream: torch.cuda.Stream | None = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        ops.event_record(self._handle, stream.cuda_stream)

    def wait(self, stream: torch.cuda.Stream | None = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        ops.stream_wait_event(stream.cuda_stream, self._handle)

    def ipc_handle(self):
        raise NotImplementedError("RestoredEvent: a LOAD-created event has no IPC handle")

    def __repr__(self) -> str:
        return f"<foundry.RestoredEvent {self._handle:#x} flags={self.flags:#x}>"


class LoadedEvent(NamedTuple):
    """One entry of ``loaded_events()``: an event LOAD rebuilt for a loaded
    graph's EventRecord/EventWait nodes.

    Loaded graphs (``CUDAGraph.load``, ``finish_graph_loads``) are
    ``foundry.ops.CUDAGraph`` objects; ``loaded_events()`` and
    ``bind_events(events)`` are methods of that class:

    - ``graph.loaded_events() -> list[LoadedEvent]``, one per archived event_id
      in order of first appearance (empty for a captured graph);
    - ``graph.bind_events({event_id: torch.cuda.Event | int})`` makes the
      graph's exec record / wait the given events (every record and wait node
      of each event_id) instead of the rebuilt ones. Keep the events alive while
      the graph replays. With FOUNDRY_ENABLE_EXEC_SHARE=1 the binding belongs
      to the graph, not the shared exec: every switch of the exec to the graph
      applies it, and other graphs of the group keep their own events.
    """

    event_id: int  # as in the archive
    handle: int  # CUevent LOAD created (kept alive with the graph)
    flags: int  # creation flags SAVE recorded (CU_EVENT_*)
    record_nodes: list[int]  # indices in the archive's node list
    wait_nodes: list[int]
    event: RestoredEvent  # the handle as a torch.cuda.Event


def _check_rng_buffers_loadable(rng: dict) -> None:
    """SAVE, graphs with RNG only: LOAD re-creates the seed/offset buffers the kernels read
    by repeating torch 2.14's allocation (CUDAGeneratorCaptureState::initialize: two int64[1]
    on the default stream, default pool), so their addresses match only if SAVE's came from
    there. torch builds that allocate them elsewhere (e.g. per-graph RNG state created in
    CUDAGraph::capture_begin on the capture stream) are refused here instead of loading
    silently wrong random numbers."""
    segments = torch.cuda.memory_snapshot()
    for name in ("seed_ptr", "offset_ptr"):
        addr = rng[name]
        seg = next(
            (s for s in segments if s["address"] <= addr < s["address"] + s["total_size"]), None
        )
        if seg is None:
            where = "not in a caching-allocator segment"
        else:
            stream, pool = seg.get("stream"), tuple(seg.get("segment_pool_id", (None, None)))
            if stream == 0 and pool == (0, 0):
                continue
            where = f"in a segment of stream {stream:#x}, pool {pool}"
        raise RuntimeError(
            f"foundry capture: unsupported: RNG in a graph on torch {torch.__version__}: the "
            f"default generator's capture {name[:-4]} buffer {addr:#x} is {where}, not in the "
            "default stream's default pool, so LOAD cannot reproduce its address"
        )


class CUDAGraph(ops.CUDAGraph):
    def __new__(cls, keep_graph: bool = False):
        return super().__new__(cls, keep_graph)

    # SAVE captures with torch's own torch.cuda.CUDAGraph(keep_graph=True) inside
    # foundry's hook record and hands the raw cudaGraph_t to foundry for analysis and
    # save. The torch graph owning the capture, and its RNG probe:
    _native_graph = None
    _native_rng = None

    def capture_begin(
        self, pool: _POOL_HANDLE | None = None, capture_error_mode: str = "global"
    ) -> None:
        if self._native_graph is not None:
            raise RuntimeError("This CUDAGraph instance already owns a captured graph.")
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        if not hasattr(gen, "philox_state"):
            raise RuntimeError(
                "foundry native capture: unsupported torch "
                f"{torch.__version__} (needs Generator.philox_state, torch >= 2.14)"
            )
        native = torch.cuda.CUDAGraph(keep_graph=True)
        # Hook record window: opened before torch's beginAllocateToPool, closed by
        # adopt_native_capture after torch's endAllocateToPool (same as capture_begin/_end).
        ops.native_record_begin()
        try:
            native.capture_begin(pool=pool, capture_error_mode=capture_error_mode)
        except BaseException:
            ops.native_record_abort()
            raise
        self._native_graph = native

    def capture_end(self) -> None:
        native = self._native_graph
        if native is None:
            raise RuntimeError("capture_end() without capture_begin()")
        device = torch.cuda.current_device()
        gen = torch.cuda.default_generators[device]
        try:
            # Still capturing: returns aliases of this capture's seed/offset buffers of the
            # default generator (creating them, like foundry's own capture_begin always does)
            # and the offset consumed so far = torch's wholegraph_increment for it.
            # v2.14.1 aten/src/ATen/cuda/CUDAGeneratorImpl.cpp:516-536.
            seed_buf, offset_buf, intragraph = gen.philox_state(0)
            native.capture_end()
        except BaseException:
            ops.native_record_abort()
            raise
        increment = int(intragraph.item())
        seed = gen.initial_seed()
        self._native_rng = {
            "seed_ptr": seed_buf.data_ptr(),
            "offset_ptr": offset_buf.data_ptr(),
            "increment": increment,
            "seed": seed,
        }
        del seed_buf, offset_buf
        if increment:
            try:
                _check_rng_buffers_loadable(self._native_rng)
            except BaseException:
                ops.native_record_abort()
                raise
        # Default generator of the capture device; id as foundry's registry gives the first
        # generator it sees (CUDAGeneratorStateRegistry::id_counter starts at 0).
        # NOT covered: RNG from any other generator (no public API lists the generators the
        # torch graph registered on 2.14.1); such a graph is saved without its RNG state.
        generators = [(device, seed, increment)]
        super().adopt_native_capture(native.raw_cuda_graph(), native.pool(), generators)

    def replay(self) -> None:
        if self._native_graph is not None:
            # torch's replay runs its own RNG prologue (fills the seed/offset buffers)
            self._native_graph.replay()
            return
        super().replay()

    def reset(self) -> None:
        if self._native_graph is not None:
            self._native_graph.reset()
            self._native_graph = None
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        if self._native_graph is not None:
            return self._native_graph.pool()
        return super().pool()

    def enable_debug_mode(self) -> None:
        return super().enable_debug_mode()

    def debug_dump(self, debug_path: str) -> None:
        return super().debug_dump(debug_path)

    def raw_cuda_graph(self) -> int:
        return super().raw_cuda_graph()

    def save(self, json_path: str, output_tensors: OutputTensors | None = None) -> None:
        return super().save(json_path, output_tensors)

    @staticmethod
    def load(
        json_path: str,
        pool: tuple[int, int] | None = None,
    ) -> CUDAGraph | tuple[CUDAGraph, OutputTensors]:
        if graph.default_capture_stream is None:
            graph.default_capture_stream = torch.cuda.Stream()

        torch.cuda.synchronize()

        stream_ctx = torch.cuda.stream(graph.default_capture_stream)
        with stream_ctx:
            cuda_graph, output_tensors = ops.CUDAGraph.load(json_path, pool)

        if output_tensors is None:
            return cuda_graph
        return cuda_graph, output_tensors

    @staticmethod
    def start_exec_pool_prewarm(json_paths: list[str]) -> None:
        """LOAD: grow the driver's graph-exec memory for these graphs on a
        background thread (kernel-only copies, instantiated then destroyed, never
        launched), so the later start_graph_builds instantiates reuse it instead of
        growing it one exec at a time. Needs the recorded binaries loaded."""
        ops.CUDAGraph.start_exec_pool_prewarm(json_paths)

    @staticmethod
    def start_graph_builds(
        json_paths: list[str],
        pool: tuple[int, int] | None = None,
        num_threads: int = 4,
    ):
        """Parse graph JSONs and start template building in background.

        Returns immediately with an opaque PendingGraphLoads handle.
        Template graphs are built on a background thread while the caller
        continues with subsequent initialization (KV cache, etc.).

        Call finish_graph_loads() later to wait for builds to complete
        and perform generator registration, allocator replay, and output
        tensor reconstruction.
        """
        if graph.default_capture_stream is None:
            graph.default_capture_stream = torch.cuda.Stream()

        torch.cuda.synchronize()

        stream_ctx = torch.cuda.stream(graph.default_capture_stream)
        with stream_ctx:
            return ops.CUDAGraph.start_graph_builds(json_paths, pool, num_threads)

    @staticmethod
    def finish_graph_loads(
        pending,
    ) -> list[tuple[CUDAGraph, OutputTensors | None]]:
        """Finish graph loads: replay allocator events and reconstruct output tensors.

        Takes the PendingGraphLoads handle from start_graph_builds().
        """
        return ops.CUDAGraph.finish_graph_loads(pending)

    @staticmethod
    def finish_one_graph_load(
        pending,
        index: int,
    ) -> tuple[CUDAGraph, OutputTensors | None]:
        """Finish a single graph by index. Interleaves allocator replay with
        between-capture work so the VMM cursor on LOAD walks the same
        trajectory it did on SAVE. First call waits on background build
        completion (idempotent thereafter). Caller must invoke in the same
        order SAVE captured the graphs.
        """
        return ops.CUDAGraph.finish_one_graph_load(pending, index)


class graph:
    default_capture_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: _POOL_HANDLE | None = None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ):
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> None:
        torch.cuda.synchronize()

        # NOTE(liuxs): no empty_cache here — foundry's allocator offset is
        # monotonically increasing, freeing between captures unmaps the VMM
        # range but doesn't rewind the cursor, so the next allocation lands
        # at a higher cursor and final_alloc_offset inflates past the real
        # working set (~194 GB observed vs ~80 GB real on Qwen3-30B-A3B EP2).
        # if torch.compiler.config.force_cudagraph_gc:
        #     gc.collect()
        # torch.cuda.empty_cache()

        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(
            *self.pool,
            capture_error_mode=self.capture_error_mode,
        )

    def __exit__(self, *args) -> None:
        self.cuda_graph.capture_end()
        self.stream_ctx.__exit__(*args)


def save_graph_manifest(
    archive_dir: str,
    enable_templates: bool = True,
    partition: Callable[[str], str] | None = None,
) -> None:
    """Write graph_manifest.json with topology groups and template assignments.

    With ``enable_templates=False`` every graph is its own topology group
    (template of itself, no on-demand members), so LOAD rebuilds each graph
    from its full node list instead of re-parameterizing a shared template.

    ``partition`` maps a graph filename to a partition name; graphs in
    different partitions never share a topology group. Use it when LOAD
    restores the partitions with separate ``start_graph_builds`` calls: a
    member's template must be in the same call (members are stored without
    their dependencies).

    Reads topology_key from each saved graph JSON, groups graphs by topology,
    picks the first graph in each group as the template, strips the
    dependencies array from on-demand (non-template) graphs to reduce file
    size, and writes the manifest.

    Should be called after all graphs are captured and saved.
    """
    # graph_<index>_*.json; not the manifest itself, which a repeated call
    # (one per capture scope) finds in the directory.
    graph_files = sorted(
        (
            f
            for f in os.listdir(archive_dir)
            if f.startswith("graph_") and f.endswith(".json") and f.split("_")[1].isdigit()
        ),
        key=lambda f: int(f.split("_")[1]),
    )
    if not graph_files:
        return

    # Read topology_key from each graph JSON
    topology_keys: dict[tuple[str, str], list[str]] = {}
    for filename in graph_files:
        json_path = os.path.join(archive_dir, filename)
        with open(json_path) as f:
            data = json.load(f)
        topo_key = data.get("topology_key", "")
        if not topo_key:
            return
        if not enable_templates:
            topo_key = f"{topo_key}#{filename}"
        part = partition(filename) if partition is not None else ""
        topology_keys.setdefault((part, topo_key), []).append(filename)

    # Build manifest
    groups = []
    templates = set()
    for (part, topo_key), filenames in topology_keys.items():
        group = {
            "topology_key": topo_key,
            "template": filenames[0],
            "members": filenames,
        }
        if part:
            group["partition"] = part
        groups.append(group)
        templates.add(filenames[0])

    manifest = {"topology_groups": groups}
    manifest_path = os.path.join(archive_dir, "graph_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Strip dependencies from on-demand graph files.
    # On-demand graphs only need nodes + common_kernel_node_attrs for
    # prepare_on_demand_graph; dependencies are only used by
    # build_graph_from_parsed (template builds).
    num_stripped = 0
    for filename in graph_files:
        if filename in templates:
            continue
        json_path = os.path.join(archive_dir, filename)
        with open(json_path) as f:
            data = json.load(f)
        if "dependencies" in data:
            del data["dependencies"]
            with open(json_path, "w") as f:
                json.dump(data, f)
            num_stripped += 1

    num_templates = len(groups)
    num_on_demand = sum(len(g["members"]) - 1 for g in groups)

    print(
        f"[foundry] Saved graph_manifest.json (templates={'on' if enable_templates else 'off'}): "
        f"{num_templates} topology groups "
        f"({num_templates} templates, {num_on_demand} on-demand, "
        f"{num_stripped} stripped)",
        file=sys.stderr,
    )
