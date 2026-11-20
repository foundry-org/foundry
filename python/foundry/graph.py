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


# Archive "placement" of the default generator's capture seed/offset buffers (csrc
# RngPlacement): inside a driver allocation Foundry recorded (torch allocated them in
# Foundry's RNG pool inside a record window); LOAD writes at the archived addresses.
_RNG_RECORDED = 1

# SAVE: one caching-allocator pool per device for torch's capture RNG state. Every
# capture runs inside torch.cuda.use_mem_pool(<this pool>) (opened before
# torch.cuda.CUDAGraph.capture_begin, closed after capture_end), so the int64[1]
# seed/offset tensors torch creates for the capture land here instead of torch's
# default pools, on every torch version: 2.14/2.15 create them lazily on the default
# stream (not capturing: the capture's pool filter misses, this thread's use_mem_pool
# filter matches); 2.16 pre-allocates them in capture_begin before the capture's
# routing starts. The pool's segments are allocated inside record windows (recorded
# alloc events), so LOAD's event replay maps them; torch's default pools are left
# exactly as LOAD (which does not capture) has them. Kept alive for the process.
_RNG_POOLS: dict[int, torch.cuda.MemPool] = {}


def _rng_pool(device: int) -> torch.cuda.MemPool:
    pool = _RNG_POOLS.get(device)
    if pool is None:
        with torch.cuda.device(device):
            pool = _RNG_POOLS[device] = torch.cuda.MemPool()
    return pool


def _unsupported_torch(what: str) -> RuntimeError:
    return RuntimeError(
        f"foundry capture: unsupported: torch {torch.__version__}: {what} (Foundry's RNG state "
        "detection depends on it)"
    )


class _Block(NamedTuple):
    pool: tuple[int, int]
    stream: int
    size: int
    requested: int


def _live_blocks() -> dict[int, _Block]:
    """address -> _Block of every live caching-allocator block of the current device. A
    snapshot without the fields this relies on raises instead of hiding blocks."""
    out = {}
    try:
        for seg in torch.cuda.memory_snapshot():
            pool = tuple(seg["segment_pool_id"])
            for b in seg["blocks"]:
                if b["state"] == "active_allocated":
                    out[b["address"]] = _Block(pool, seg["stream"], b["size"], b["requested_size"])
    except KeyError as e:
        raise _unsupported_torch(f"torch.cuda.memory_snapshot() has no {e} field") from None
    return out


def _stray_message(strays: dict[int, _Block]) -> str:
    return ", ".join(
        f"{a:#x} (block {b.size} B, requested {b.requested} B, stream {b.stream:#x})"
        for a, b in sorted(strays.items())
    )


# SAVE: what Foundry keeps alive in its RNG pool for the process, per device: the
# seed/offset pair of every capture that claimed one (philox_state aliases: the archived
# address of one graph's pair is never handed to another graph's state, and no pair can
# be freed inside a later capture and reused by that capture's state unseen) and what the
# RNG self-test left there. Every other live block of the pool (state of a refused
# capture) is unaccounted for.
_RNG_HELD: dict[int, list] = {}
# Per device: None = the RNG self-test passed, else its error message.
_RNG_SELF_TEST: dict[int, str | None] = {}


def _held_addresses(device: int) -> set[int]:
    return {
        t.data_ptr() if isinstance(t, torch.Tensor) else t
        for t in _RNG_HELD.get(device, [])
        if isinstance(t, (torch.Tensor, int))
    }


def _rng_self_test(device: int, pool: torch.cuda.MemPool) -> str | None:
    """Once per process and device, inside the first capture's record window (so LOAD maps
    what it allocates): torch must put a capture's default-generator RNG state into the
    pool routed with use_mem_pool, philox_state must return it, and the graph must hold its
    address where the kernel-parameter scan finds it. Everything stays alive (no reuse).
    Returns None, or what failed."""
    pool_id = tuple(pool.id)
    gen = torch.cuda.default_generators[device]
    before = set(_live_blocks())
    side = torch.cuda.Stream(device)
    native = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.use_mem_pool(pool):
        # The draw goes in place into a tensor of the RNG pool, on the default stream like
        # torch's state (one segment), so the graph's own pool stays empty.
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            out = torch.empty(4, device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        side.wait_stream(torch.cuda.default_stream(device))
    with torch.cuda.use_mem_pool(pool), torch.cuda.stream(side):
        native.capture_begin(capture_error_mode="relaxed")
        try:
            out.uniform_()
            seed_buf, offset_buf, intragraph = gen.philox_state(0)
        finally:
            native.capture_end()
    torch.cuda.synchronize(device)
    pair = (seed_buf.data_ptr(), offset_buf.data_ptr())
    new = {a: b for a, b in _live_blocks().items() if a not in before}
    held = _RNG_HELD.setdefault(device, [])
    held += [native, out, seed_buf, offset_buf]
    problems = []
    if not all(a in new and new[a].pool == pool_id for a in pair):
        problems.append(f"its seed/offset pair {[hex(a) for a in pair]} is not new in the routed pool")
    if int(intragraph.item()) <= 0:
        problems.append("philox_state reports no offset consumed by uniform_")
    if not ops.scan_graph_words(native.raw_cuda_graph(), list(pair)):
        problems.append("no kernel parameter of the graph holds the pair's addresses")
    held += list(new)  # addresses of everything the self-test left (all in the pool)
    if problems:
        return "; ".join(problems)
    return None


class CUDAGraph(ops.CUDAGraph):
    def __new__(cls, keep_graph: bool = False):
        return super().__new__(cls, keep_graph)

    # SAVE captures with torch's own torch.cuda.CUDAGraph(keep_graph=True) inside
    # foundry's hook record and hands the raw cudaGraph_t to foundry for analysis and
    # save. The torch graph owning the capture, and its RNG probe:
    _native_graph = None
    _native_rng = None
    _rng_route = None

    def capture_begin(
        self, pool: _POOL_HANDLE | None = None, capture_error_mode: str = "global"
    ) -> None:
        if self._native_graph is not None:
            raise RuntimeError("This CUDAGraph instance already owns a captured graph.")
        device = torch.cuda.current_device()
        gen = torch.cuda.default_generators[device]
        if not hasattr(gen, "philox_state"):
            raise RuntimeError(
                "foundry native capture: unsupported torch "
                f"{torch.__version__} (needs Generator.philox_state, torch >= 2.14)"
            )
        backend = torch.cuda.get_allocator_backend()
        if backend != "native":
            raise RuntimeError(
                f"foundry capture: unsupported: CUDA allocator backend {backend!r} "
                "(PYTORCH_CUDA_ALLOC_CONF backend:native is required: Foundry routes and finds "
                "torch's capture RNG state through caching-allocator pools)"
            )
        native = torch.cuda.CUDAGraph(keep_graph=True)
        self._native_capture_stream = torch.cuda.current_stream().cuda_stream
        rng_pool = _rng_pool(device)
        self._rng_pool_id = tuple(rng_pool.id)
        # Hook record window: opened before torch's beginAllocateToPool, closed by
        # adopt_native_capture after torch's endAllocateToPool (same as capture_begin/_end).
        ops.native_record_begin()
        try:
            if device not in _RNG_SELF_TEST:
                _RNG_SELF_TEST[device] = _rng_self_test(device, rng_pool)
            if _RNG_SELF_TEST[device] is not None:
                raise _unsupported_torch(
                    f"RNG self-test capture of Tensor.uniform_: {_RNG_SELF_TEST[device]}"
                )
            # Blocks live before the capture: anything new at capture end was created by
            # this capture (or by another thread meanwhile).
            self._blocks_before = _live_blocks()
        except BaseException:
            ops.native_record_abort()
            raise
        # Live RNG-pool blocks Foundry does not hold could be freed inside this capture and
        # their addresses reused by this capture's state, unseen by the difference above.
        held = _held_addresses(device)
        self._rng_unaccounted = {
            a for a, b in self._blocks_before.items() if b.pool == self._rng_pool_id and a not in held
        }
        # Opened before torch's capture scope: the allocator searches its routing scopes
        # last-in first-out, so the capture's own (capturing-stream) filter still wins for
        # everything captured; only this thread's allocations on non-capturing streams
        # (the RNG state) reach the RNG pool.
        route = torch.cuda.use_mem_pool(rng_pool)
        route.__enter__()
        try:
            native.capture_begin(pool=pool, capture_error_mode=capture_error_mode)
        except BaseException:
            route.__exit__(*sys.exc_info())
            ops.native_record_abort()
            raise
        self._rng_route = route
        self._native_graph = native

    def capture_end(self) -> None:
        native = self._native_graph
        if native is None:
            raise RuntimeError("capture_end() without capture_begin()")
        device = torch.cuda.current_device()
        gen = torch.cuda.default_generators[device]
        rng_pool_id = self._rng_pool_id
        before = self._blocks_before
        philox = None
        try:
            try:
                # Does the default generator hold RNG state for this capture? Torch creates
                # it in the RNG pool (2.14/2.15: lazily at the first RNG op, on the default
                # stream; 2.16: capture_begin's pre-allocated pairs); another thread creates
                # it in its default pool. Any new block in the RNG pool, any new int64[1]
                # block in a default pool, or an RNG-pool block that may have been freed and
                # reused inside the capture: ask. Still capturing, philox_state returns
                # aliases of the default generator's seed/offset buffers of this capture
                # (creating them in the RNG pool if absent) and the offset consumed so far =
                # torch's wholegraph_increment (v2.14.1 CUDAGeneratorImpl.cpp:516-536).
                # Otherwise no RNG ran in this capture and nothing is created.
                now = _live_blocks()
                new = {a: b for a, b in now.items() if a not in before}
                if (
                    self._rng_unaccounted
                    or any(b.pool == rng_pool_id for b in new.values())
                    or any(b.pool == (0, 0) and b.requested == 8 for b in new.values())
                ):
                    philox = gen.philox_state(0)
                native.capture_end()
            finally:
                self._rng_route.__exit__(None, None, None)
                self._rng_route = None
        except BaseException:
            ops.native_record_abort()
            raise
        graph_pool = tuple(native.pool())
        now = _live_blocks()
        new = {a: b for a, b in now.items() if a not in before}
        generators = []
        pair = set()
        if philox is not None:
            seed_buf, offset_buf, intragraph = philox
            increment = int(intragraph.item())
            seed = gen.initial_seed()
            self._native_rng = {
                "seed_ptr": seed_buf.data_ptr(),
                "offset_ptr": offset_buf.data_ptr(),
                "increment": increment,
                "seed": seed,
            }
            pair = {seed_buf.data_ptr(), offset_buf.data_ptr()}
            # Kept for the process: no later state (of this or another capture) can take
            # these addresses while this graph's archive refers to them.
            _RNG_HELD.setdefault(device, []).extend((seed_buf, offset_buf))
            del seed_buf, offset_buf, philox
            outside = [a for a in sorted(pair) if a not in now or now[a].pool != rng_pool_id]
            if outside:
                ops.native_record_abort()
                raise RuntimeError(
                    "foundry capture: unsupported: the default generator's capture RNG state "
                    f"{', '.join(hex(a) for a in outside)} is not in Foundry's RNG pool (RNG "
                    "drawn on the capturing stream from a thread other than the capturing one)"
                )
            generators.append(
                (
                    device,
                    seed,
                    increment,
                    _RNG_RECORDED,
                    self._native_rng["seed_ptr"],
                    self._native_rng["offset_ptr"],
                )
            )
        # Candidates: every RNG-pool block that may hold this capture's RNG state (created
        # during the capture, or live but not held by Foundry, so possibly freed and reused
        # inside it: compared by address that looks unchanged) and every int64[1] block
        # created during the capture outside the graph pool (RNG state another thread
        # created). adopt scans every kernel parameter (and copy/memset address) of the raw
        # graph for them by value: a read one that is not the default generator's pair is
        # refused; the pair must lie in a driver allocation Foundry recorded (the RNG pool's
        # segment), which LOAD maps by replaying the recorded events and writes (seed,
        # offset) into per replay. The graph pool holds the body's tensors (int64 scalars
        # too), never RNG state.
        candidates = sorted(
            {
                a
                for a, b in new.items()
                if b.pool == rng_pool_id or (b.requested == 8 and b.pool != graph_pool)
            }
            | {a for a in self._rng_unaccounted if a in now}
        )
        super().adopt_native_capture(native.raw_cuda_graph(), native.pool(), generators, candidates)
        # The RNG pool must hold nothing new from this capture but torch's RNG state: anything
        # else this thread allocated on a non-capturing stream inside the routing scope (in
        # the body, or in torch's Python capture-end hooks, which run after the capture's own
        # routing ended) is memory SAVE's later code may bake into graphs but that LOAD's
        # torch never allocates; an unread int64[1] there is another generator's state.
        strays = {a: b for a, b in new.items() if b.pool == rng_pool_id and a not in pair}
        if strays:
            raise RuntimeError(
                "foundry capture: unsupported: allocation(s) on a non-capturing stream during "
                "the capture (they landed in Foundry's RNG pool, not in the graph pool) or RNG "
                f"state of another generator: {_stray_message(strays)}"
            )

    def _abort_capture(self) -> None:
        """End a capture whose body raised without adopting it: torch's capture is ended
        (its errors are dropped, the body's exception is the one that propagates) and the
        hook record window is discarded. Memory the capture allocated stays allocated on
        SAVE but is in no archive, so later graphs that use it (an RNG-pool segment
        allocated in this window) are refused at SAVE."""
        native = self._native_graph
        if native is None:
            return
        try:
            native.capture_end()
        except Exception:
            pass
        finally:
            if self._rng_route is not None:
                self._rng_route.__exit__(None, None, None)
                self._rng_route = None
            ops.native_record_abort()

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

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.cuda_graph.capture_end()
            else:
                # The body raised: nothing of this capture is adopted or archived.
                self.cuda_graph._abort_capture()
        finally:
            self.stream_ctx.__exit__(exc_type, exc, tb)


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
