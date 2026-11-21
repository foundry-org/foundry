# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
from __future__ import annotations

import json
import os
import sys
import weakref
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


class CaptureUnsupported(RuntimeError):
    """SAVE refused a capture (or the save of a captured graph) that Foundry cannot persist
    correctly; the message starts with "foundry capture: unsupported:" (or "foundry save:
    unsupported:") and names the reason. Integrations fall back to native capture."""


_UNSUPPORTED_PREFIXES = ("foundry capture: unsupported:", "foundry save: unsupported:")


def _as_capture_unsupported(e: BaseException) -> BaseException:
    """The CaptureUnsupported for a refusal raised as a plain RuntimeError (the C++ checks)."""
    if isinstance(e, RuntimeError) and not isinstance(e, CaptureUnsupported):
        text = str(e)
        if any(p in text for p in _UNSUPPORTED_PREFIXES):
            return CaptureUnsupported(text[text.index("foundry ") :])
    return e


_RNG_RESERVED: set[int] = set()
_CAPTURES_BEGUN: set[int] = set()  # devices where a Foundry capture began


def _check_generator_contract(device: int) -> None:
    """LOAD takes (seed, offset) per replay from ``philox_state(inc)`` outside capture (the
    replay binding): it must return the generator's current seed and offset and advance the
    offset by exactly ``inc``. Checked once per device with public API; the generator's
    offset is restored, so later draws are unchanged."""
    gen = torch.cuda.default_generators[device]
    offset = gen.get_offset()
    seed_t, offset_t, _ = gen.philox_state(4)
    after = gen.get_offset()
    gen.set_offset(offset)
    seed = int(seed_t.item()) & 0xFFFFFFFFFFFFFFFF
    if seed != gen.initial_seed() or int(offset_t.item()) != offset or after != offset + 4:
        raise CaptureUnsupported(
            f"foundry capture: unsupported: torch {torch.__version__}: Generator.philox_state(4) "
            f"outside capture returned seed {seed} / offset {int(offset_t.item())} and left offset "
            f"{after}; Foundry's loaded graphs need seed {gen.initial_seed()}, offset {offset} and "
            f"offset {offset + 4} after it"
        )


def reserve_rng_pool() -> None:
    """SAVE and LOAD alike, at the same program point right after ``set_allocation_region``
    and before any capture (the integrations call it there): create the first segment of
    Foundry's RNG pool for the current device outside any capture, with the same torch
    allocation on both sides, so it lands at the same address and counts as recorded on
    both. Capture seed/offset pairs then live in it instead of in a segment of the first
    capture's record window (later segments, one per ~2048 live captures, are still
    allocated inside the capture that needs them and recorded when that graph is saved).
    The allocated block is kept for the process on both sides, so the segment survives any
    pool-release rule. LOAD allocates this one segment once; nothing per graph or replay.
    Also checks the generator contract LOAD relies on. Without the call the first segment
    comes from the first capture. A second call on the same device is a no-op."""
    device = torch.cuda.current_device()
    if device in _RNG_RESERVED:
        return
    if device in _CAPTURES_BEGUN:
        raise CaptureUnsupported(
            "foundry capture: unsupported: reserve_rng_pool() after a capture began on device "
            f"{device}: call it right after set_allocation_region, before any capture"
        )
    _check_generator_contract(device)
    pool = _rng_pool(device)
    pool_id = tuple(pool.id)

    def segments():
        try:
            return [
                (seg["address"], seg["total_size"], seg["is_expandable"])
                for seg in torch.cuda.memory_snapshot()
                if tuple(seg["segment_pool_id"]) == pool_id
            ]
        except KeyError as e:
            raise _unsupported_torch(f"torch.cuda.memory_snapshot() has no {e} field") from None

    if segments():
        raise CaptureUnsupported(
            "foundry capture: unsupported: reserve_rng_pool(): Foundry's RNG pool already has a "
            "segment it did not create here"
        )
    with torch.cuda.use_mem_pool(pool), torch.cuda.stream(torch.cuda.default_stream(device)):
        block = torch.empty(1, dtype=torch.int64, device=device)
    _RNG_HELD.setdefault(device, []).append(block)  # kept for the process, SAVE and LOAD
    reserved = segments()
    if any(expandable for _, _, expandable in reserved):
        raise CaptureUnsupported(
            "foundry capture: unsupported: expandable segments (PYTORCH_CUDA_ALLOC_CONF "
            "expandable_segments:True) in Foundry's RNG pool: a segment that grows in place is "
            "not a fixed reservation"
        )
    for address, size, _ in reserved:
        ops.note_reserved_allocation(address, size)
    _RNG_RESERVED.add(device)


def _unsupported_torch(what: str) -> RuntimeError:
    return CaptureUnsupported(
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


# SAVE, per device: the RNG-pool blocks accounted for. HELD: kept alive for the process
# (the archived seed/offset pairs, as philox_state aliases, and what the self-test left):
# never freed, so no later state takes an archived address. LIVE: pairs of captures that
# archive none, alive as long as torch's graph keeps its capture state (removed by a
# finaliser on that graph); a LIVE address that disappears during a capture was freed
# inside it and may hold that capture's state.
_RNG_HELD: dict[int, list] = {}
_RNG_LIVE: dict[int, set[int]] = {}
# Per device: None = the RNG self-test passed, else its error message.
_RNG_SELF_TEST: dict[int, str | None] = {}
# SAVE: adopted captures whose record window made driver allocations and that are not saved
# yet: capture sequence number -> bytes. LOAD replays the windows of the saved graphs in
# save order, so a graph must not be saved after such an earlier capture was skipped.
_UNSAVED_WINDOWS: dict[int, int] = {}
_CAPTURE_SEQ = [0]
_LAST_SAVED_SEQ = [-1]
# Set when a capture ended without being adopted after its record window allocated driver
# memory: SAVE's allocation cursor has moved past memory no archive records, which LOAD
# (where no capture is aborted) never allocates, so every later allocation and graph of
# this process would sit at other addresses than on LOAD.
_POISONED: str | None = None


def _held_addresses(device: int) -> set[int]:
    return {
        t.data_ptr() if isinstance(t, torch.Tensor) else t
        for t in _RNG_HELD.get(device, [])
        if isinstance(t, (torch.Tensor, int))
    }


def _rng_self_test(device: int, pool: torch.cuda.MemPool) -> str | None:
    """Once per process and device, at the end of the first capture, inside its record
    window and after that capture's own seed/offset pair exists (so it shares the RNG
    pool's segment and adds no segment of its own): torch must put a capture's
    default-generator state into the pool routed with use_mem_pool, philox_state must
    return it with the consumed offset, and the graph must hold its address where the
    kernel-parameter scan finds it. Everything stays alive. Returns None, or what failed."""
    pool_id = tuple(pool.id)
    gen = torch.cuda.default_generators[device]
    before = set(_live_blocks())
    side = torch.cuda.Stream(device)
    native = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.use_mem_pool(pool):
        # The draw goes in place into a tensor of the RNG pool, on the default stream like
        # torch's state (same segment), so the self-test graph's own pool stays empty.
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
    if any(
        seg["is_expandable"]
        for seg in torch.cuda.memory_snapshot()
        if tuple(seg["segment_pool_id"]) == pool_id
    ):
        return "expandable segments (PYTORCH_CUDA_ALLOC_CONF expandable_segments:True) in the pool"
    problems = []
    if not all(a in new and new[a].pool == pool_id for a in pair):
        problems.append(
            f"its seed/offset pair {[hex(a) for a in pair]} is not new in the routed pool"
        )
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
        if _POISONED is not None:
            raise CaptureUnsupported(f"foundry capture: unsupported: {_POISONED}")
        device = torch.cuda.current_device()
        gen = torch.cuda.default_generators[device]
        if not hasattr(gen, "philox_state"):
            raise CaptureUnsupported(
                f"foundry capture: unsupported: torch {torch.__version__} (needs "
                "Generator.philox_state, torch >= 2.14)"
            )
        backend = torch.cuda.get_allocator_backend()
        if backend != "native":
            raise CaptureUnsupported(
                f"foundry capture: unsupported: CUDA allocator backend {backend!r} "
                "(PYTORCH_CUDA_ALLOC_CONF backend:native is required: Foundry routes and finds "
                "torch's capture RNG state through caching-allocator pools)"
            )
        if _RNG_SELF_TEST.get(device):
            raise _unsupported_torch(
                f"RNG self-test capture of Tensor.uniform_: {_RNG_SELF_TEST[device]}"
            )
        _CAPTURES_BEGUN.add(device)
        native = torch.cuda.CUDAGraph(keep_graph=True)
        self._native_capture_stream = torch.cuda.current_stream().cuda_stream
        rng_pool = _rng_pool(device)
        self._rng_pool_id = tuple(rng_pool.id)
        # Blocks live before the capture: anything new at capture end was created by this
        # capture (or by another thread meanwhile).
        self._blocks_before = _live_blocks()
        self._rng_live_before = set(_RNG_LIVE.get(device, ()))
        self._alloc_offset_before = ops.get_current_alloc_offset()
        # Hook record window: opened before torch's beginAllocateToPool, closed by
        # adopt_native_capture after torch's endAllocateToPool (same as capture_begin/_end).
        ops.native_record_begin()
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
            self._record_abort()
            raise
        self._rng_route = route
        self._native_graph = native

    def _record_abort(self, discard_window: bool = True) -> None:
        """A capture failed: discard its record window (unless adopt already closed it; the
        refused graph is never saved either way). If the window allocated driver memory,
        SAVE can no longer match LOAD: refuse every later capture."""
        global _POISONED
        if discard_window:
            ops.native_record_abort()
        moved = ops.get_current_alloc_offset() - self._alloc_offset_before
        if moved and _POISONED is None:
            _POISONED = (
                f"an earlier capture of this process ended without being saved after its record "
                f"window allocated {moved} bytes of driver memory; LOAD cannot reproduce that "
                "allocation, so no later graph of this SAVE would load at its addresses "
                "(restart SAVE without the failing capture)"
            )

    def capture_end(self) -> None:
        native = self._native_graph
        if native is None:
            raise RuntimeError("capture_end() without capture_begin()")
        try:
            self._capture_end(native)
        except BaseException as e:
            self._record_abort(discard_window=not self._adopted)
            converted = _as_capture_unsupported(e)
            if converted is not e:
                raise converted from e
            raise

    _adopted = False

    def _capture_end(self, native) -> None:
        device = torch.cuda.current_device()
        gen = torch.cuda.default_generators[device]
        rng_pool_id = self._rng_pool_id
        before = self._blocks_before
        # Still capturing, on every torch version: philox_state returns aliases of the
        # default generator's seed/offset buffers of this capture (created now in the RNG
        # pool if no RNG op ran, by whichever thread created them otherwise) and the offset
        # consumed so far = torch's wholegraph_increment (v2.14.1 CUDAGeneratorImpl.cpp:516-536).
        probe_error = None
        try:
            try:
                seed_buf, offset_buf, intragraph = gen.philox_state(0)
            except RuntimeError as e:  # e.g. 2.16: "CUDA graph RNG state limit exceeded"
                probe_error = e
            native.capture_end()  # torch's capture is closed whatever the probe did
        finally:
            self._rng_route.__exit__(None, None, None)
            self._rng_route = None
        if probe_error is not None:
            raise CaptureUnsupported(
                "foundry capture: unsupported: torch refused the default generator's capture "
                f"RNG state: {probe_error}"
            ) from probe_error
        pair = (seed_buf.data_ptr(), offset_buf.data_ptr())
        if device not in _RNG_SELF_TEST:
            # After this capture's pair exists: the self-test shares its segment.
            _RNG_SELF_TEST[device] = _rng_self_test(device, _rng_pool(device))
            if _RNG_SELF_TEST[device] is not None:
                raise _unsupported_torch(
                    f"RNG self-test capture of Tensor.uniform_: {_RNG_SELF_TEST[device]}"
                )
        graph_pool = tuple(native.pool())
        now = _live_blocks()
        new = {a: b for a, b in now.items() if a not in before}
        increment = int(intragraph.item())
        seed = gen.initial_seed()  # diagnostic only: LOAD takes the seed from torch
        outside = [a for a in pair if a not in now or now[a].pool != rng_pool_id]
        if outside:
            raise CaptureUnsupported(
                "foundry capture: unsupported: the default generator's capture RNG state "
                f"{', '.join(hex(a) for a in outside)} is not in Foundry's RNG pool (RNG drawn "
                "on the capturing stream from a thread other than the capturing one)"
            )
        generators = [(device, seed, increment, _RNG_RECORDED, pair[0], pair[1])]
        # Candidates: blocks that may hold RNG state of this capture other than the default
        # pair: RNG-pool blocks created during the capture, RNG-pool pairs of other captures
        # freed during it (their address may have been reused unseen), and int64[1] blocks
        # created during it in torch's default pools (state another thread created). The
        # self-test's blocks are held and excluded. adopt scans every kernel parameter (and
        # copy/memset address) of the raw graph for them by value and refuses a read one; it
        # archives the default pair only if the graph reads it. The graph pool holds the
        # body's tensors (int64 scalars too), never RNG state; other user pools are not
        # touched by torch's RNG.
        held = _held_addresses(device)
        freed_live = self._rng_live_before - _RNG_LIVE.get(device, set())
        candidates = sorted(
            {
                a
                for a, b in new.items()
                if a not in held
                and (b.pool == rng_pool_id or (b.requested == 8 and b.pool == (0, 0)))
            }
            | {a for a in freed_live if a in now}
        )
        archived = super().adopt_native_capture(
            native.raw_cuda_graph(), graph_pool, generators, candidates
        )
        self._adopted = True
        self._capture_seq = _CAPTURE_SEQ[0]
        _CAPTURE_SEQ[0] += 1
        allocated = ops.get_current_alloc_offset() - self._alloc_offset_before
        if allocated:
            _UNSAVED_WINDOWS[self._capture_seq] = allocated
        if archived:
            self._native_rng = {
                "seed_ptr": pair[0],
                "offset_ptr": pair[1],
                "increment": increment,
                "seed": seed,
            }
            # Kept for the process: no later state can take an archived address.
            _RNG_HELD.setdefault(device, []).extend((seed_buf, offset_buf))
        else:
            # Alive as long as torch's graph keeps the capture state.
            live = _RNG_LIVE.setdefault(device, set())
            live.update(pair)
            weakref.finalize(native, live.difference_update, pair)
        del seed_buf, offset_buf
        # The RNG pool must hold nothing new from this capture but torch's RNG state: anything
        # else this thread allocated on a non-capturing stream inside the routing scope (in
        # the body, or in torch's Python capture-end hooks, which run after the capture's own
        # routing ended) is memory SAVE's later code may bake into graphs but that LOAD's
        # torch never allocates; an unread int64[1] there is another generator's state.
        strays = {
            a: b for a, b in new.items() if b.pool == rng_pool_id and a not in set(pair) | held
        }
        if strays:
            raise CaptureUnsupported(
                "foundry capture: unsupported: allocation(s) on a non-capturing stream during "
                "the capture (they landed in Foundry's RNG pool, not in the graph pool) or RNG "
                f"state of another generator: {_stray_message(strays)}"
            )

    def _abort_capture(self) -> None:
        """End a capture whose body raised without adopting it: torch's capture is ended
        (its errors are dropped, the body's exception is the one that propagates) and the
        hook record window is discarded (see _record_abort)."""
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
            self._record_abort()

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

    _capture_seq = None

    def save(self, json_path: str, output_tensors: OutputTensors | None = None) -> None:
        seq = self._capture_seq
        if seq is not None and seq < _LAST_SAVED_SEQ[0]:
            raise CaptureUnsupported(
                f"foundry save: unsupported: capture #{seq} saved after capture "
                f"#{_LAST_SAVED_SEQ[0]}: LOAD replays the saved graphs' windows in save order and "
                "its region cursor only moves forward (save graphs in capture order)"
            )
        if seq is not None:
            skipped = sorted(k for k in _UNSAVED_WINDOWS if k < seq)
            if skipped:
                k = skipped[0]
                raise CaptureUnsupported(
                    f"foundry save: unsupported: capture #{k} (before this graph, #{seq}) "
                    f"allocated {_UNSAVED_WINDOWS[k]} bytes of driver memory in its record window "
                    "and was not saved before this graph: LOAD replays the windows of the saved "
                    "graphs in save order and would place this graph's memory elsewhere (save "
                    "every capture whose window allocated, in capture order)"
                )
        try:
            super().save(json_path, output_tensors)
        except RuntimeError as e:
            converted = _as_capture_unsupported(e)
            if converted is not e:
                raise converted from e
            raise
        if seq is not None:
            _UNSAVED_WINDOWS.pop(seq, None)
            _LAST_SAVED_SEQ[0] = max(_LAST_SAVED_SEQ[0], seq)

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
        try:
            self.cuda_graph.capture_begin(
                *self.pool,
                capture_error_mode=self.capture_error_mode,
            )
        except BaseException:
            # __exit__ does not run when __enter__ raises: restore the caller's stream.
            self.stream_ctx.__exit__(*sys.exc_info())
            raise

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
