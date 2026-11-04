# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Events of a loaded graph: loaded_events() / bind_events() (1 GPU).

    pytest tests/core/test_loaded_events.py -q -s

SAVE captures elementwise steps separated by external event record/wait pairs
(timing and non-timing events alternating, the first and last timing), times
the span between the first and the last record node over a few replays with
the captured events (the native measurement, written into the archive) and
saves the graph. Archives: ``single`` (one graph) and ``group`` (the same chain
on two row counts plus a manifest: one template and one member). Each LOAD
must list every archived event with the flags SAVE recorded and the
record/wait node indices of the archive's node list, each as a RestoredEvent
the replay records (timing the span with the first and last), then bind two
timing torch.cuda.Events of its own to the first and last record nodes: the span
they measure must match SAVE's, and the outputs stay bit for bit SAVE's.

LOAD APIs: ``load`` (CUDAGraph.load), ``builds`` (start_graph_builds +
finish_graph_loads, eager; on ``group`` the member replays its own exec),
``builds_json`` (the same from a copy of the JSON without its .cugraph, so the
template is built from the JSON). ``share`` (FOUNDRY_ENABLE_EXEC_SHARE=1):
on ``single`` as above; on ``group`` the template and the member are bound
to the same two torch events and replayed in turn, each replay measuring
its own graph's span. ``share_member``: only the member is bound; the
template's replays record its own events (timed through its RestoredEvents)
and leave the member's alone.
"""

import json
import os
import shutil
import statistics
import sys

import pytest
import torch
from _helpers import (
    assert_bit_exact,
    load_reference,
    map_saved_layout,
    prefix_region,
    run_phase,
    save_layout,
    save_reference,
)

BASE_ADDR = 0x3E0000000000
REGION_SIZE_STR = "1GB"
NATIVE_JSON = "native_timing.json"
EVENT_CHAIN = 9  # odd: the first and the last event are timing events
COLS = 4096
ROWS = {"single": (2048,), "group": (2048, 1024)}  # fp32: 32 MiB at 2048 rows
TIMING_REPLAYS = 50
REL_TOL = 0.15
ABS_TOL_MS = 0.03
CU_EVENT_DEFAULT = 0x0
CU_EVENT_DISABLE_TIMING = 0x2
_EVENTS = []  # the captured record/wait nodes reference these: keep them alive


def _timing(i):
    return i % 2 == 0


def _body(x, n):
    s = torch.cuda.current_stream()
    y = x[:n]
    for i in range(EVENT_CHAIN):
        y = y * 1.5 + 1.0
        ev = torch.cuda.Event(enable_timing=_timing(i), external=True)
        _EVENTS.append(ev)
        ev.record(s)
        s.wait_event(ev)
    return y


def _graph_paths(archive, layout):
    if layout == "single":
        return [os.path.join(archive, "graph.json")]
    # save_graph_manifest expects graph_<index>_<suffix>.json
    return [os.path.join(archive, f"graph_{i}_r{n}.json") for i, n in enumerate(ROWS[layout])]


def _prefix():
    g, stream = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.full((max(ROWS["group"]), COLS), 0.5, device="cuda")
    with torch.cuda.stream(stream):
        _body(x, x.shape[0])  # warm-up: lazy module loads
    _EVENTS.clear()
    torch.cuda.synchronize()
    return x


def _replay_twice(graph, out):
    replays = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        replays.append(out.cpu().clone())
    return replays


def _span_ms(graph, start, end):
    """Median start->end elapsed time over TIMING_REPLAYS replays."""
    times = []
    for _ in range(TIMING_REPLAYS):
        graph.replay()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _run_save(archive, layout):
    import foundry as fdry

    x = _prefix()
    refs, native, graphs = [], [], []
    for path, n in zip(_graph_paths(archive, layout), ROWS[layout]):
        first = len(_EVENTS)
        graph = fdry.CUDAGraph()
        with fdry.graph(graph):
            out = _body(x, n)
        replays = _replay_twice(graph, out)
        assert torch.equal(replays[0], replays[1]), "output changed between replays"
        span = _span_ms(graph, _EVENTS[first], _EVENTS[-1])
        print(f"[test] {os.path.basename(path)}: native span {span * 1e3:.1f} us")
        graph.save(path, output_tensors=out)
        graphs.append(graph)
        refs.append(replays)
        native.append(span)
    if layout == "group":
        fdry.save_graph_manifest(archive)
    with open(os.path.join(archive, NATIVE_JSON), "w") as f:
        json.dump({"span_ms": native}, f)
    save_reference(archive, refs, [x])
    save_layout(archive)
    fdry.stop_allocation_region()


def _expected_events(path):
    """{event_id: (flags, record node indices, wait node indices)} from the JSON."""
    with open(path) as f:
        nodes = json.load(f)["nodes"]
    want = {}
    for idx, node in enumerate(nodes):
        kind = {"EventRecordNode": 1, "EventWaitNode": 2}.get(node["type"])
        if kind:
            p = node["params"]
            entry = want.setdefault(p["event_id"], (p["event_flags"], [], []))
            entry[kind].append(idx)
    return want


def _check_listing(graph, path):
    events = graph.loaded_events()
    expected = _expected_events(path)
    got = {e.event_id: (e.flags, e.record_nodes, e.wait_nodes) for e in events}
    assert got == expected, f"loaded_events {got} != archive {expected}"
    assert len(events) == EVENT_CHAIN, f"{len(events)} events, expected {EVENT_CHAIN}"
    for i, e in enumerate(events):
        assert e.flags == (CU_EVENT_DEFAULT if _timing(i) else CU_EVENT_DISABLE_TIMING), e
        assert len(e.record_nodes) == 1 and len(e.wait_nodes) == 1, e
        if i:
            assert e.record_nodes[0] > events[i - 1].record_nodes[0], "not in node order"
    handles = [e.handle for e in events]
    assert all(handles) and len(set(handles)) == len(handles), f"event handles {handles}"
    for e in events:
        assert isinstance(e.event, torch.cuda.Event), e
        assert e.event.cuda_event == e.handle and e.event.flags == e.flags, e
    return events


def _check_restored(graph, events):
    """The RestoredEvents of an unbound graph: the replay records them, the
    first->last span is timed with them, a timing-disabled one refuses
    elapsed_time, and stream.record_event / wait_event route to them."""
    import foundry as fdry

    first, second, last = events[0].event, events[1].event, events[-1].event
    assert isinstance(first, fdry.RestoredEvent) and "RestoredEvent" in repr(first), repr(first)
    graph.replay()
    last.synchronize()
    assert all(e.event.query() for e in events), "replay left an event unrecorded"
    assert first.elapsed_time(last) > 0, "restored span is not positive"
    with pytest.raises(RuntimeError, match="cuEventElapsedTime"):
        second.elapsed_time(last)  # CU_EVENT_DISABLE_TIMING
    with pytest.raises(NotImplementedError):
        first.ipc_handle()
    s = torch.cuda.current_stream()
    assert s.record_event(second) is second
    s.wait_event(second)
    second.synchronize()
    assert second.query()
    torch.cuda.synchronize()


def _check_span(label, bound, native):
    print(
        f"[test] {label}: bound span {bound * 1e3:.1f} us vs native {native * 1e3:.1f} us "
        f"({(bound / native - 1) * 100:+.1f} %)"
    )
    assert abs(bound - native) <= max(REL_TOL * native, ABS_TOL_MS), (
        f"{label}: bound span {bound:.4f} ms vs native {native:.4f} ms"
    )


def _replay_in_turn(loaded, rounds, after=None):
    """Replay the graphs in turn (graph 0, 1, ..., 0, 1, ...) ``rounds`` times,
    synchronizing after each replay; ``after(gi)`` runs after graph gi's."""
    for _ in range(rounds):
        for gi, (graph, _) in enumerate(loaded):
            graph.replay()
            torch.cuda.synchronize()
            if after:
                after(gi)


def _check_outputs_in_turn(loaded, want, what):
    replays = [[] for _ in loaded]
    _replay_in_turn(loaded, 2, lambda gi: replays[gi].append(loaded[gi][1].cpu().clone()))
    for gi, got in enumerate(replays):
        assert_bit_exact(got, want[gi], f"{what} graph {gi}")


def _share_alternating(loaded, listings, native, want):
    """Template and member bound to the same two torch events, replayed in turn:
    each replay's span is the replayed graph's."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for (graph, _), events in zip(loaded, listings):
        graph.bind_events({events[0].event_id: start, events[-1].event_id: end})
    spans = [[] for _ in loaded]
    _replay_in_turn(loaded, TIMING_REPLAYS, lambda gi: spans[gi].append(start.elapsed_time(end)))
    for gi, times in enumerate(spans):
        _check_span(f"share, in turn: graph {gi}", statistics.median(times), native[gi])
    _check_outputs_in_turn(loaded, want, "share, in turn")


def _share_member_only(loaded, listings, native, want):
    """Only the member bound: the template's replays keep recording its own
    events (RestoredEvents) and leave the member's bound ones alone."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    member = listings[1]
    loaded[1][0].bind_events({member[0].event_id: start, member[-1].event_id: end})
    tmpl_first, tmpl_last = listings[0][0].event, listings[0][-1].event
    spans = [[], []]

    def after(gi):
        if gi == 0:
            assert tmpl_first.query() and tmpl_last.query()
            spans[0].append(tmpl_first.elapsed_time(tmpl_last))
            if spans[1]:
                assert start.elapsed_time(end) == spans[1][-1], (
                    "the template's replay recorded the member's bound events"
                )
        else:
            spans[1].append(start.elapsed_time(end))

    _replay_in_turn(loaded, TIMING_REPLAYS, after)
    tmpl_span = statistics.median(spans[0])
    _check_span("share, member bound: template (own events)", tmpl_span, native[0])
    _check_span("share, member bound: member", statistics.median(spans[1]), native[1])
    _check_outputs_in_turn(loaded, want, "share, member bound")


def _run_load(archive, layout, load_api, share):
    import foundry as fdry

    fdry.load_cuda_modules_and_libraries("hook_archive")
    x = _prefix()
    want = load_reference(archive, [x])
    with open(os.path.join(archive, NATIVE_JSON)) as f:
        native = json.load(f)["span_ms"]
    paths = _graph_paths(archive, layout)
    if load_api == "builds_json":
        json_only = os.path.join(archive, "json_only")
        os.makedirs(json_only, exist_ok=True)
        manifest = os.path.join(archive, "graph_manifest.json")
        for p in paths + ([manifest] if os.path.exists(manifest) else []):
            shutil.copy(p, json_only)
        paths = [os.path.join(json_only, os.path.basename(p)) for p in paths]
    if load_api == "load":
        assert layout == "single"
        loaded = [fdry.CUDAGraph.load(paths[0])]
    else:
        map_saved_layout(archive)
        pending = fdry.CUDAGraph.start_graph_builds(paths)
        loaded = fdry.CUDAGraph.finish_graph_loads(pending)

    # every graph's events first: binding one graph must not touch another's
    listings = [_check_listing(graph, path) for (graph, _), path in zip(loaded, paths)]
    for (graph, _), events in zip(loaded, listings):
        _check_restored(graph, events)
        with pytest.raises(RuntimeError, match="no event_id"):
            graph.bind_events({EVENT_CHAIN + 100: torch.cuda.Event(enable_timing=True)})
    if share != "eager" and layout == "group":
        {"share": _share_alternating, "share_member": _share_member_only}[share](
            loaded, listings, native, want
        )
        for gi, (graph, _) in enumerate(loaded):
            assert [e.handle for e in graph.loaded_events()] == [e.handle for e in listings[gi]]
        fdry.stop_allocation_region()
        return
    timers = []
    for (graph, out), events in zip(loaded, listings):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        graph.bind_events({events[0].event_id: start, events[-1].event_id: end})
        timers.append((start, end))
    for gi, (graph, out) in enumerate(loaded):
        assert [e.handle for e in graph.loaded_events()] == [e.handle for e in listings[gi]]
        assert_bit_exact(_replay_twice(graph, out), want[gi], f"{load_api} graph {gi} replays")
    for gi, ((graph, out), (start, end)) in enumerate(zip(loaded, timers)):
        bound = _span_ms(graph, start, end)
        _check_span(f"{layout}/{load_api}/{share} {os.path.basename(paths[gi])}", bound, native[gi])
        assert_bit_exact(_replay_twice(graph, out), want[gi], f"graph {gi} after timing")
    fdry.stop_allocation_region()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    archives = {}

    def get(layout):
        if layout not in archives:
            work = tmp_path_factory.mktemp(f"loaded_events_{layout}")
            run_phase(__file__, "save", work, layout)
            assert (work / "hook_archive").is_dir(), "SAVE did not pack hook_archive"
            archives[layout] = work
        return archives[layout]

    return get


@pytest.mark.parametrize(
    "layout, load_api, share",
    [
        ("single", "load", "eager"),
        ("single", "builds", "eager"),
        ("single", "builds_json", "eager"),
        ("single", "builds", "share"),
        ("group", "builds", "eager"),
        ("group", "builds_json", "eager"),
        ("group", "builds", "share"),
        ("group", "builds", "share_member"),
    ],
)
def test_loaded_events(saved, layout, load_api, share):
    env = {"FOUNDRY_ENABLE_EXEC_SHARE": "1"} if share != "eager" else {}
    log = run_phase(__file__, "load", saved(layout), layout, load_api, share, env=env)
    if layout == "group" and share == "eager":
        assert "1 on-demand" in log, "group archive did not load as template + member"


if __name__ == "__main__":
    mode, archive, *rest = sys.argv[1:]
    {"--save": _run_save, "--load": _run_load}[mode](archive, *rest)
