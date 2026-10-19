# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

import foundry as foundry_pkg
from foundry import ops as cge
from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.graph import graph as foundry_graph_ctx
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_config,
)
from foundry.integration.sglang.nvtx import nvtx_range, nvtx_traced
from foundry.integration.sglang.runtime import get_state

logger = logging.getLogger(__name__)

_GRAPH_FILENAME_RE = re.compile(r"^graph_(?P<index>\d+)_FULL_t(?P<bs>\d+)_r\d+_UX_pcN\.json$")
# Decode-type graphs of every other capture scope kind (speculative decoding:
# target_verify, draft_decode, draft_extend, ...) and of target_decode graphs
# with a variant label: the kind and the ShapeKey's size (and variant label)
# are in the name. The plain target decode graphs keep the name above, so
# archives without speculative decoding are unchanged.
_KIND_GRAPH_FILENAME_RE = re.compile(
    r"^graph_(?P<index>\d+)_DECODE-(?P<kind>[a-z][a-z0-9_]*)_t(?P<size>\d+)_UX_pcN"
    r"(?:_v(?P<variant>[A-Za-z0-9]+))?\.json$"
)
# Full-backend prefill graphs: one per token bucket t and cached-prefix variant
# pc (prefix chunks, 0 = suffix-only), captured with r request slots. Their own
# phase tag keeps them out of the decode scan above, so decode-only archives
# and their consumers are unchanged.
_PREFILL_GRAPH_FILENAME_RE = re.compile(
    r"^graph_(?P<index>\d+)_PREFILL_t(?P<tokens>\d+)_r(?P<slots>\d+)_UX_pc(?P<chunks>\d+)\.json$"
)
# Per-rank SAVE record of what LOAD needs besides the graphs: the DP-gather
# flag the captured forwards latch and the prefill output structures.
_PREFILL_STATE_FILE = "prefill_graphs.json"
# Per-rank SAVE record of the decode-type capture scopes: their kinds in
# process order and the output structure of each decode-type graph. Absent in
# archives saved before 2.1 (one target_decode scope, logits-only outputs).
_DECODE_STATE_FILE = "decode_graphs.json"
# PrefillCudaGraphRunner's only ShapeKey variants (_chunked_prefix_variant).
_CHUNKED_PREFIX_LABEL = "chunked_prefix:"
# SAVE: output structure of each prefill graph whose output is not one tensor.
_prefill_output_specs: dict[str, Any] = {}
# SAVE: output structure of each decode-type graph, and the kinds of the
# decode-type scopes finished so far (both written to _DECODE_STATE_FILE).
_decode_output_specs: dict[str, Any] = {}
_saved_decode_kinds: list[str] = []

# The decode runner of a process without speculative decoding (and the kind
# an integration API 2.0 caller implies).
DEFAULT_DECODE_KIND = "target_decode"
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VARIANT_RE = re.compile(r"^[A-Za-z0-9]+$")
# Output of a pre-2.1 decode graph: LogitsProcessorOutput(next_token_logits).
_LEGACY_DECODE_OUTPUT_SPEC: Any = {"lpo": ["next_token_logits"]}


def check_decode_kind(kind: str | None) -> str:
    """The scope kind (None: :data:`DEFAULT_DECODE_KIND`); it goes into graph
    filenames and manifest partitions, so it must be a lowercase identifier."""
    if kind is None:
        return DEFAULT_DECODE_KIND
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError(f"[Foundry] invalid decode capture scope kind {kind!r}")
    return kind


def _decode_shape(key: Any) -> tuple[int, str | None]:
    """(size, variant label) of a decode-type ShapeKey(size, stream_idx,
    variant_label, attention_variant). Multi-stream (PD multiplexing) and
    attention graph variants are not persisted; a variant label (the EAGLE
    draft runner's greedy / sampling graph pair) is, by name."""
    if key.stream_idx is not None or key.attention_variant is not None:
        raise ValueError(f"Foundry SGLang save/load does not support graph variants: {key!r}")
    label = key.variant_label
    if label is not None and (not isinstance(label, str) or not _VARIANT_RE.match(label)):
        raise ValueError(
            f"Foundry SGLang save/load does not support decode variant label {label!r}: {key!r}"
        )
    return key.size, label


def _graph_filename(index: int, key: Any, kind: str = DEFAULT_DECODE_KIND) -> str:
    size, variant = _decode_shape(key)
    if kind == DEFAULT_DECODE_KIND and variant is None:
        return f"graph_{index}_FULL_t{size}_r{size}_UX_pcN.json"
    suffix = f"_v{variant}" if variant is not None else ""
    return f"graph_{index}_DECODE-{kind}_t{size}_UX_pcN{suffix}.json"


def _prefill_shape(key: Any) -> tuple[int, int]:
    """(num_tokens, prefix chunks) of a prefill ShapeKey."""
    if key.stream_idx is not None or key.attention_variant is not None:
        raise ValueError(f"Foundry SGLang save/load does not support graph variants: {key!r}")
    label = key.variant_label
    if label is None:
        return key.size, 0
    if label.startswith(_CHUNKED_PREFIX_LABEL):
        return key.size, int(label[len(_CHUNKED_PREFIX_LABEL) :])
    raise ValueError(f"Foundry SGLang save/load does not support prefill variant {label!r}")


def _prefill_graph_filename(index: int, key: Any, req_slots: int) -> str:
    tokens, chunks = _prefill_shape(key)
    return f"graph_{index}_PREFILL_t{tokens}_r{req_slots}_UX_pc{chunks}.json"


def graph_partition(filename: str) -> str:
    """Manifest partition: prefill graphs and each decode-type scope kind's
    graphs are restored by separate start_graph_builds calls (at their own
    runner's capture point), so they must not share a template. The plain
    target decode graphs keep the pre-2.1 partition name."""
    if _PREFILL_GRAPH_FILENAME_RE.match(filename):
        return "prefill"
    match = _KIND_GRAPH_FILENAME_RE.match(filename)
    if match:
        return f"decode:{match.group('kind')}"
    return "decode"


def _flatten_prefill_output(output: Any) -> tuple[Any, list[torch.Tensor]]:
    """Prefill graphs capture the transformer body, whose output is the hidden
    states: one tensor for most models, a nested tuple / list (auxiliary hidden
    states) or PPProxyTensors (non-last PP rank) otherwise. Returns a JSON spec
    of the structure and its tensors in depth-first order. Decode-type graphs
    use the same spec, plus ``{"lpo": [fields]}`` for a LogitsProcessorOutput
    (its tensor fields; the target-verify and draft-extend outputs carry the
    hidden states next to the logits) and tuples with None members (the EAGLE
    draft runner's (parent_list, top_scores_index, draft_tokens, draft_probs))."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    tensors: list[torch.Tensor] = []

    def walk(x: Any) -> Any:
        if x is None:
            return "N"
        if isinstance(x, torch.Tensor):
            tensors.append(x)
            return "T"
        if isinstance(x, LogitsProcessorOutput):
            names = []
            for f in dataclasses.fields(x):
                value = getattr(x, f.name)
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"SGLang CUDA graph output field LogitsProcessorOutput.{f.name} is a "
                        f"{type(value).__name__}; only tensor fields can be persisted"
                    )
                names.append(f.name)
                tensors.append(value)
            return {"lpo": names}
        if isinstance(x, PPProxyTensors):
            names = list(x.tensors)
            tensors.extend(x.tensors[n] for n in names)
            return {"pp": names}
        if isinstance(x, tuple):
            return {"tuple": [walk(v) for v in x]}
        if isinstance(x, list):
            return {"list": [walk(v) for v in x]}
        raise TypeError(f"Unsupported SGLang prefill CUDA graph output type: {type(x)!r}")

    return walk(output), tensors


def _unflatten_prefill_output(spec: Any, tensors: Any) -> Any:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    if tensors is None:
        tensors = []
    elif isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    it = iter(tensors)

    def build(s: Any) -> Any:
        if s == "N":
            return None
        if s == "T":
            return next(it)
        if "lpo" in s:
            fields = {name: next(it) for name in s["lpo"]}
            if "next_token_logits" not in fields:
                fields["next_token_logits"] = None
            return LogitsProcessorOutput(**fields)
        if "pp" in s:
            return PPProxyTensors({n: next(it) for n in s["pp"]})
        if "tuple" in s:
            return tuple(build(v) for v in s["tuple"])
        return [build(v) for v in s["list"]]

    out = build(spec)
    if next(it, None) is not None:
        raise RuntimeError("SGLang prefill CUDA graph restored more output tensors than recorded")
    return out


def _pack_output(output: Any) -> tuple[Any, Any]:
    """Decode-type graph output -> (structure spec, tensor(s) for graph.save):
    one tensor when the output holds one (the pre-2.1 layout of a
    logits-only output), else the list in depth-first order."""
    spec, tensors = _flatten_prefill_output(output)
    if not tensors:
        raise TypeError(f"SGLang decode CUDA graph output holds no tensor: {type(output)!r}")
    return spec, tensors[0] if len(tensors) == 1 else tensors


def _scan_graph_files(
    workspace_dir: str, kind: str | None = None
) -> list[tuple[int, str, dict[str, Any]]]:
    """Decode-type graph files in capture order, of every kind or of
    ``kind`` only; meta: index, kind, size, variant."""
    graph_files = []
    for filename in os.listdir(workspace_dir):
        match = _GRAPH_FILENAME_RE.match(filename)
        if match:
            meta = {
                "index": int(match.group("index")),
                "kind": DEFAULT_DECODE_KIND,
                "size": int(match.group("bs")),
                "variant": None,
            }
        else:
            match = _KIND_GRAPH_FILENAME_RE.match(filename)
            if not match:
                continue
            meta = {
                "index": int(match.group("index")),
                "kind": match.group("kind"),
                "size": int(match.group("size")),
                "variant": match.group("variant"),
            }
        if kind is not None and meta["kind"] != kind:
            continue
        graph_files.append((int(meta["index"]), filename, meta))
    graph_files.sort(key=lambda x: x[0])
    return graph_files


@nvtx_traced("foundry.save.capture_graph")
def capture_graph(pool, stream, forward_fn) -> tuple[Any, Any]:
    """SAVE: capture ``forward_fn`` on ``stream`` into ``pool`` as a new
    foundry graph; returns ``(graph, output)``."""
    graph = FoundryCUDAGraph()
    with foundry_graph_ctx(graph, pool=pool, stream=stream):
        out = forward_fn()
    return graph, out


def save_graph(
    graph,
    output: Any,
    key: Any,
    prefill_req_slots: int | None = None,
    kind: str = DEFAULT_DECODE_KIND,
) -> None:
    """Persist one captured graph. ``prefill_req_slots`` is set for the
    prefill runner's graphs (its fixed request-slot count) and selects the
    prefill filename and output packing; decode-type graphs are named after
    their scope ``kind`` and record their output structure."""
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    if prefill_req_slots is None:
        filename = _graph_filename(state.capture_index, key, kind)
        spec, packed_output = _pack_output(output)
        _decode_output_specs[filename] = spec
    else:
        filename = _prefill_graph_filename(state.capture_index, key, prefill_req_slots)
        spec, tensors = _flatten_prefill_output(output)
        if spec == "T":
            packed_output = tensors[0]
        else:
            packed_output = tensors or None
            _prefill_output_specs[filename] = spec
    graph_path = os.path.join(cfg.workspace_dir, filename)
    with nvtx_range("foundry.save.graph_save"):
        graph.save(graph_path, packed_output)

    state.capture_index += 1
    logger.info("[Foundry] Saved SGLang CUDA graph %s key=%s", filename, key)


@nvtx_traced("foundry.save.manifest")
def save_graph_manifest() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    foundry_pkg.save_graph_manifest(
        cfg.workspace_dir, enable_templates=cfg.graph_templates, partition=graph_partition
    )


def save_prefill_graph_state(*, req_slots: int, has_dp_gather: bool) -> None:
    """SAVE: write the rank's prefill record (see _PREFILL_STATE_FILE)."""
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    record = {
        "req_slots": req_slots,
        "prefill_graph_has_dp_gather": has_dp_gather,
        "output_specs": _prefill_output_specs,
    }
    with open(os.path.join(cfg.workspace_dir, _PREFILL_STATE_FILE), "w") as f:
        json.dump(record, f, indent=2)


def save_decode_graph_state(kind: str) -> None:
    """SAVE, at the end of each decode-type capture scope: append its kind
    and (re)write the rank's decode record (see _DECODE_STATE_FILE). The
    process does not know which scope is its last, so the record is complete
    after every scope."""
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    _saved_decode_kinds.append(kind)
    record = {"scopes": _saved_decode_kinds, "output_specs": _decode_output_specs}
    with open(os.path.join(cfg.workspace_dir, _DECODE_STATE_FILE), "w") as f:
        json.dump(record, f, indent=2)


_decode_record: dict[str, Any] | None = None


def load_decode_graph_state() -> dict[str, Any]:
    """LOAD: the rank's decode record; an archive saved before 2.1 has none
    and holds one target_decode scope with logits-only outputs."""
    global _decode_record
    if _decode_record is not None:
        return _decode_record
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")
    path = os.path.join(cfg.workspace_dir, _DECODE_STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            record = json.load(f)
    else:
        record = {"scopes": [DEFAULT_DECODE_KIND], "output_specs": {}}
    _decode_record = record
    return record


def _scan_prefill_graph_files(workspace_dir: str) -> list[tuple[int, str, dict[str, Any]]]:
    graph_files = []
    for filename in os.listdir(workspace_dir):
        match = _PREFILL_GRAPH_FILENAME_RE.match(filename)
        if not match:
            continue
        meta = {k: int(match.group(k)) for k in ("index", "tokens", "slots", "chunks")}
        graph_files.append((meta["index"], filename, meta))
    graph_files.sort(key=lambda x: x[0])
    return graph_files


@dataclass
class _RestoreTiming:
    """LOAD: timing of one runner's graph restore, logged by _log_restore."""

    # When this runner's restore started (its capture; its start_graph_builds
    # call follows immediately).
    t0: float
    # (start, end) of each restore_next_* call.
    calls: list[tuple[float, float]] = field(default_factory=list)


def _log_restore(what: str, n: int, pending: Any, timing: _RestoreTiming, loop_s: float) -> None:
    """Log the restore work and, separately, the capture loop around it.

    ``Loaded`` counts only foundry's work: the builds (Phase 1 + Phase 2, from
    this runner's restore start to the last build / instantiate) plus the
    part of each restore_next_* call after the builds finished (allocator
    replay, output reconstruction). A call's wait for the builds is already
    in the builds. The loop line is the upstream capture() wall around it;
    ``per-shape prep`` is that wall minus the time inside the restore calls
    (warmup, freeze_gc, dummy batches, attention metadata, ...). The builds
    run on the background thread during the loop, so restore + prep can
    exceed the loop wall."""
    build_end = timing.t0 + pending.build_seconds
    builds = max(0.0, build_end - timing.t0)
    handover = sum(max(0.0, end - max(start, build_end)) for start, end in timing.calls)
    in_calls = sum(end - start for start, end in timing.calls)
    # "prefill", "decode" (the target_decode scope, pre-2.1 wording) or the
    # decode-type scope kind.
    graphs = {"prefill": "SGLang prefill graphs", "decode": "SGLang graphs"}.get(
        what, f"SGLang {what} graphs"
    )
    logger.info(
        "[Foundry] Loaded %d %s in %.3fs (builds %.3fs, handover %.3fs)",
        n,
        graphs,
        builds + handover,
        builds,
        handover,
    )
    logger.info(
        "[Foundry] SGLang %s capture loop on LOAD: %.3fs (%d shapes, restore %.3fs, "
        "per-shape prep %.3fs)",
        what,
        loop_s,
        n,
        builds + handover,
        max(0.0, loop_s - in_calls),
    )


def has_prefill_graphs() -> bool:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return False
    return bool(_scan_prefill_graph_files(cfg.workspace_dir))


@dataclass
class _PrefillRestore:
    pending: Any
    graph_files: list[tuple[int, str, dict[str, Any]]]
    output_specs: dict[str, Any]
    timing: _RestoreTiming
    next: int = 0


_prefill_restore: _PrefillRestore | None = None


@nvtx_traced("foundry.graph_restore.prefill_start")
def start_prefill_graph_restore() -> dict[str, Any]:
    """LOAD, at the prefill runner's capture: start building the prefill
    graphs and return the rank's SAVE record. The graphs are then finished one
    at a time by restore_next_prefill_graph, from inside the upstream capture
    loop, so their allocator events replay at the points SAVE captured them."""
    global _prefill_restore
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        raise RuntimeError("Foundry SGLang graph extension is not initialized for LOAD")
    graph_files = _scan_prefill_graph_files(cfg.workspace_dir)
    record_path = os.path.join(cfg.workspace_dir, _PREFILL_STATE_FILE)
    if not graph_files or not os.path.exists(record_path):
        raise RuntimeError(
            f"Foundry archive {cfg.workspace_dir} has no prefill graphs: it was saved with "
            "prefill graphs disabled. Re-save with the same --cuda-graph-backend-prefill as LOAD."
        )
    with open(record_path) as f:
        record = json.load(f)

    # Graphs may reference NVSHMEM symbols; the DeepEP bootstrap ran before.
    cge.init_nvshmem_for_loaded_modules()
    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    t0 = time.perf_counter()
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    _prefill_restore = _PrefillRestore(
        pending=pending,
        graph_files=graph_files,
        output_specs=record.get("output_specs", {}),
        timing=_RestoreTiming(t0=t0),
    )
    logger.info(
        "[Foundry] Started SGLang prefill graph builds for %d graphs in %.3fs",
        len(paths),
        time.perf_counter() - t0,
    )
    return record


def restore_next_prefill_graph(key: Any, req_slots: int) -> tuple[Any, Any]:
    """LOAD: stands in for capturing prefill graph ``key``. The archive holds
    the graphs in capture order; the upstream loop asks for them in the same
    order, which is checked here."""
    r = _prefill_restore
    state = get_state()
    if r is None or state is None:
        raise RuntimeError("Foundry prefill graph restore was not started")
    tokens, chunks = _prefill_shape(key)
    if r.next >= len(r.graph_files):
        raise RuntimeError(
            f"Foundry archive holds {len(r.graph_files)} prefill graphs, but the capture loop "
            f"asks for more (t={tokens} pc={chunks}): the prefill capture shapes differ from SAVE"
        )
    _index, filename, meta = r.graph_files[r.next]
    if (meta["tokens"], meta["chunks"], meta["slots"]) != (tokens, chunks, req_slots):
        raise RuntimeError(
            f"Foundry prefill graph order mismatch: the capture loop asks for t={tokens} "
            f"pc={chunks} r={req_slots}, the next archived graph is {filename}. The prefill "
            "capture shapes (--cuda-graph-bs-prefill, full_prefill_max_req, chunked-prefix "
            "settings) differ between SAVE and LOAD."
        )
    t = time.perf_counter()
    graph, tensors = FoundryCUDAGraph.finish_one_graph_load(r.pending, r.next)
    r.timing.calls.append((t, time.perf_counter()))
    r.next += 1
    output = _unflatten_prefill_output(r.output_specs.get(filename, "T"), tensors)
    # Strong reference, like the decode graphs.
    state.loaded_graphs[("prefill", tokens, chunks)] = (graph, output)
    return graph, output


def finish_prefill_graph_restore(loop_s: float) -> None:
    """LOAD, after the upstream prefill capture loop (``loop_s`` is its
    wall): check that it restored every archived prefill graph."""
    global _prefill_restore
    r = _prefill_restore
    _prefill_restore = None
    state = get_state()
    if r is None or state is None:
        raise RuntimeError("Foundry prefill graph restore was not started")
    if r.next != len(r.graph_files):
        raise RuntimeError(
            f"Foundry archive holds {len(r.graph_files)} prefill graphs, the capture loop "
            f"restored {r.next}: the prefill capture shapes differ from SAVE"
        )
    state.prefill_graphs_restored = True
    _log_restore("prefill", r.next, r.pending, r.timing, loop_s)


@nvtx_traced("foundry.save.pack_fatbins")
def pack_fatbins() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    t0 = time.perf_counter()
    cge.pack_fatbins_to_folder(cfg.workspace_dir)
    cge.set_pack_fatbins_on_exit(False)
    logger.info(
        "[Foundry] SAVE packed the recorded binaries in %.1f ms", 1000 * (time.perf_counter() - t0)
    )


@nvtx_traced("foundry.graph_restore.prewarm")
def start_exec_pool_prewarm() -> None:
    """LOAD, at setup (right after the recorded binaries are loaded): start
    growing the driver's graph-exec memory for every archived graph (decode and
    prefill) on a background thread, while torch-distributed init, weight
    loading and the memory pools run. Phase 2's instantiates then reuse that
    memory instead of growing it per exec (~3/4 of an instantiate's cost with
    earlier execs alive). Nothing is launched and no torch/VMM allocation is
    made. Phase 2 never waits for it: one still running at the capture point
    is told to stop and Phase 2 instantiates without it."""
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return
    files = _scan_graph_files(cfg.workspace_dir) + _scan_prefill_graph_files(cfg.workspace_dir)
    if not files:
        return
    FoundryCUDAGraph.start_exec_pool_prewarm(
        [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in files]
    )


def bootstrap_collective_connections() -> None:
    """Connect every NCCL communicator the graphs may use, before capture, on
    SAVE and LOAD alike.

    NCCL sets a communicator up lazily: the first collective on it allocates
    its buffers and connects the peers, work that is illegal inside a
    capturing stream ("operation not permitted when stream is capturing",
    seen on the attention-TP sub-group's first reduce_scatter of
    Qwen3.5-35B-A3B attention-TP2 + EP4). Native sglang hides this behind
    its eager warmup forwards; Foundry runs none, so it issues one small and
    one large all-reduce, all-gather and reduce-scatter on each initialized
    group through sglang's own coordinators (so the same communicator
    objects get connected), plus the WORLD group the DP gather can use. The
    buffers this allocates go through the hook at the same sequence point on
    both modes.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    from sglang.srt.distributed import parallel_state as ps

    t0 = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    groups = []
    for getter in (
        ps.get_tp_group,
        ps.get_attn_tp_group,
        ps.get_moe_ep_group,
        ps.get_moe_tp_group,
        ps.get_moe_dp_group,
    ):
        try:
            group = getter()
        except Exception:
            continue
        if group is not None and group.world_size > 1 and group not in groups:
            groups.append(group)
    sizes = (16, 4 << 20)  # elements: LL-sized, and ring-sized (8 MB bf16)
    connected = []
    for group in groups:
        ws = group.world_size
        for n in sizes:
            group.all_reduce(torch.ones(n, dtype=torch.bfloat16, device=device))
            group.all_gather_into_tensor(
                torch.empty(ws * n, dtype=torch.bfloat16, device=device),
                torch.ones(n, dtype=torch.bfloat16, device=device),
            )
            group.reduce_scatter_tensor(
                torch.empty(n, dtype=torch.bfloat16, device=device),
                torch.ones(ws * n, dtype=torch.bfloat16, device=device),
            )
        connected.append(f"{group.unique_name}(ws={ws})")
    ws = dist.get_world_size()
    for n in sizes:
        dist.all_reduce(torch.ones(n, dtype=torch.bfloat16, device=device), group=dist.group.WORLD)
        dist.all_gather_into_tensor(
            torch.empty(ws * n, dtype=torch.bfloat16, device=device),
            torch.ones(n, dtype=torch.bfloat16, device=device),
            group=dist.group.WORLD,
        )
    torch.cuda.synchronize()
    logger.info(
        "[Foundry] SGLang collective connections bootstrapped in %.3fs: %s + WORLD(ws=%d)",
        time.perf_counter() - t0,
        ", ".join(connected) or "-",
        ws,
    )


def bootstrap_logits_gatherer(model) -> bool:
    """Create the logits all-gather's symmetric-memory state BEFORE capture, on
    SAVE and LOAD alike.

    sglang's ``LogitsProcessor`` gathers the TP-sharded logits through
    ``MultimemAllGatherer``: a torch symmetric-memory buffer plus a multicast
    mapping and signal pads, built lazily on the first *eager* call
    (``create_state``, a collective) and skipped under capture, where the
    gatherer falls back to an NCCL all-gather. Whether the multicast path is
    available is a host property (IMEX / fabric), so on a host with multicast
    any eager forward before capture activates the gatherer and every graph
    then bakes in its pointers; those allocations bypass cudaMalloc, so the
    hook never records them and LOAD cannot recreate them (illegal address at
    the first decode). Creating the state here, at the same sequence point in
    both modes, gives it the same addresses on both sides (its VA reservations
    are carved from the region) and lets the graphs capture the multicast
    gather exactly as native sglang does.

    Returns True when a state was (or already is) built, False when the
    gatherer is disabled, absent, or multicast is unavailable (NCCL fallback,
    identical on both modes).
    """
    import torch

    try:
        from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
            MultimemAllGatherer,
        )
    except Exception:
        return False
    built = False
    for module in model.modules():
        gatherer = getattr(module, "_logits_gatherer", None)
        if not isinstance(gatherer, MultimemAllGatherer):
            continue
        if gatherer._state is None:
            continue  # disabled by configuration
        if gatherer._state is not MultimemAllGatherer._UNINIT:
            # Built before this point by an eager forward: its addresses depend
            # on that forward's allocations, which LOAD does not repeat.
            logger.warning(
                "[Foundry] logits all-gather state was already built before the bootstrap; "
                "SAVE and LOAD may place it differently"
            )
            built = True
            continue
        lm_head = getattr(model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if weight is None or weight.dim() != 2:
            logger.warning("[Foundry] logits gatherer found but no lm_head weight; leaving it lazy")
            return False
        # The gather input is the per-rank logits shard: (tokens, vocab / tp).
        probe = torch.empty((1, weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
        t0 = time.perf_counter()
        state = gatherer._build(probe)
        if state is not MultimemAllGatherer._UNINIT:
            gatherer._state = state
        built = state is not None and state is not MultimemAllGatherer._UNINIT
        logger.info(
            "[Foundry] logits all-gather state %s pre-capture in %.3fs (shard width %d)",
            "built" if built else "not built (NCCL fallback)",
            time.perf_counter() - t0,
            weight.shape[0],
        )
    return built


def install_capture_autotune_guard() -> None:
    """SAVE-only: turn inductor's Triton autotuning inside the capture window
    into a clear error.

    A freshly compiled inductor kernel with several candidate configs is
    benchmarked on its first run (``CachingAutotuner.benchmark_all_configs``),
    and the benchmark synchronizes the device, which the capturing stream
    rejects ("operation not permitted when stream is capturing") and the whole
    capture then fails with an unrelated-looking error. With a warm inductor
    cache (``autotune_local_cache``) the best config is loaded and nothing is
    benchmarked, so SAVE succeeds; this is why SAVE passes after a plain sglang
    run of the same model on the same machine and fails cold. The guard does
    not change behaviour, it names the cause and the remedy.
    """
    try:
        import torch
        from torch._inductor.runtime import triton_heuristics
    except Exception as exc:  # pragma: no cover - inductor layout changed
        logger.warning("[Foundry] capture-time autotune guard not installed: %s", exc)
        return
    cls = triton_heuristics.CachingAutotuner
    orig = cls.benchmark_all_configs
    if getattr(orig, "_foundry_capture_guard", False):
        return

    def benchmark_all_configs(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "[Foundry] inductor wants to autotune a freshly compiled Triton kernel inside "
                "the capture window; its benchmark synchronizes the device, which capture "
                "forbids. Warm the JIT caches first: run plain sglang with the same decode-graph "
                "set once for this model on this machine (same SGLANG_CACHE_DIR / "
                "TORCHINDUCTOR_CACHE_DIR), then SAVE."
            )
        return orig(self, *args, **kwargs)

    benchmark_all_configs._foundry_capture_guard = True  # type: ignore[attr-defined]
    cls.benchmark_all_configs = benchmark_all_configs


def bootstrap_lazy_runtimes() -> None:
    """SAVE-only: perform, outside any stream capture, the one-time runtime
    initializations that capture rejects, so that the model's own compiles and
    JIT kernel loads can happen inside the captured forward and be recorded
    like every other capture-time event (no eager warmup forward).

    * torch.compile / inductor: its pattern-matcher lazy init traces example
      graphs and copies unpinned host tensors to the device, which fails inside
      capture ("Cannot copy between CPU and CUDA tensors during CUDA graph
      capture"). The three lazy_init entry points run it once; the tensors are
      transient. Dynamo tracing, inductor codegen and Triton compiles are host
      work and module loads, legal during capture.
    * DeepGEMM: the first JIT call builds its runtime, which calls
      cudaFree(nullptr) and probes nvcc; get_num_sms() goes through the same
      device object. Allocation-free.

    LOAD needs neither: restore compiles nothing, and LOAD's first eager
    request performs these inits outside capture, as native sglang does. The
    caller runs this with the allocation region suspended, so the transient
    tensors are plain driver allocations and the deterministic cursor is the
    same on SAVE and LOAD afterwards.
    """
    import torch

    t0 = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device())
    # inductor guards its pattern-matcher initializers with a functools.cache
    # keyed on the "input device" it derives from the compiled function's first
    # tensor input (None when there is none). Warm every key a compile can
    # produce: a trivial compile of a CUDA function runs the whole pipeline
    # (dynamo, joint-graph and post-grad passes, one Triton kernel) for the
    # CUDA key; the initializers are then run directly for the None and CPU
    # keys.
    torch.compile(lambda x: x * 2 + 1, dynamic=True)(torch.ones(8, device=device))
    # Only the joint-graph initializers are device-keyed (pad_mm / sfdp / misc
    # patterns trace example tensors on `input_device`, which is where the
    # unpinned host-to-device copy happens). pre_grad and post_grad register
    # global patterns once, and a second call raises "Duplicate pattern";
    # the trivial compile above already ran them. Guard every call on its
    # own: one failure must not skip the remaining keys (that skip let
    # GLM-5.3-Flash's DSA indexer compile run _sfdp_init inside capture).
    try:
        from torch._inductor.fx_passes import joint_graph

        keys = (None, torch.device("cpu"), torch.device("cuda"), device)
        for key in keys:
            try:
                joint_graph.lazy_init(key)
            except Exception as exc:
                logger.warning("[Foundry] inductor joint_graph.lazy_init(%s) failed: %s", key, exc)
    except Exception as exc:
        logger.warning("[Foundry] inductor lazy_init warm-up unavailable: %s", exc)
    install_capture_autotune_guard()
    try:
        import deep_gemm

        deep_gemm.get_num_sms()
    except Exception:
        pass
    logger.info("[Foundry] SGLang lazy runtimes bootstrapped in %.3fs", time.perf_counter() - t0)


def _find_dispatcher(model, cls) -> Any:
    """The first ``cls`` dispatcher on the model: ``module.dispatcher`` itself
    or one of its ``_inners`` (the two-batch-overlap wrapper holds the real
    dispatchers there)."""
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is None:
            continue
        for candidate in (dispatcher, *getattr(dispatcher, "_inners", [])):
            if isinstance(candidate, cls):
                return candidate
    return None


def bootstrap_deepep_buffer(model) -> bool:
    """Force the singleton DeepEP ``Buffer`` (NVSHMEM runtime + symmetric heap)
    to be created BEFORE the cuda-graph capture loop.

    sglang creates the DeepEP buffer lazily on the first MoE dispatch — normally
    during the two pre-capture warmup forwards. Foundry suppresses those warmups
    (for allocation determinism), which would push buffer creation into the
    captured forward, where ``deep_ep_cpp.Buffer(...)`` aborts with
    ``operation not permitted when stream is capturing``.

    Triggering it here (outside any stream capture) creates only the NVSHMEM
    runtime + symmetric heap — no model activations — so it stays symmetric
    across SAVE and LOAD and lands at the same VMM offset on both. The buffer is
    a process-wide singleton (``DeepEPBuffer._buffer``), so one creation per rank
    is enough. It is a collective over the EP group, so every rank must reach
    this point together — which they do, since ``capture`` runs on all ranks.

    Returns True if a buffer was (or already is) created, False if DeepEP is off.
    """
    try:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        backend = get_moe_a2a_backend()
    except Exception:
        return False
    # Outside the guard: a failure to build the buffer must surface here, not
    # resurface as a native abort when the capture forward retries lazily.
    if backend.is_deepep_v2():
        return _bootstrap_deepep_v2_buffer(model)
    if backend.is_mooncake():
        return _bootstrap_mooncake_buffer(model)
    if not backend.is_deepep():
        return False

    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPBuffer,
        DeepEPDispatcher,
    )

    # The singleton now lives on the runtime context's resources
    # (DeepEPBuffer._state().buffer), not a class attribute.
    if DeepEPBuffer._state().buffer is not None:
        return True

    deepep = _find_dispatcher(model, DeepEPDispatcher)
    # Prefer the low-latency impl (the mode foundry captures); the buffer is
    # sized for whichever impls exist, so either bootstraps the shared
    # singleton.
    impl = deepep and (
        getattr(deepep, "_low_latency_dispatcher", None)
        or getattr(deepep, "_normal_dispatcher", None)
    )
    if impl is not None:
        t0 = time.perf_counter()
        impl._get_buffer()
        logger.info(
            "[Foundry] Bootstrapped DeepEP buffer pre-capture in %.3fs",
            time.perf_counter() - t0,
        )
        return True

    logger.warning(
        "[Foundry] DeepEP backend active but no DeepEPDispatcher found on the "
        "model; buffer not bootstrapped (capture may fail inside stream capture)."
    )
    return False


def _bootstrap_mooncake_buffer(model) -> bool:
    """Mooncake EP counterpart (elastic EP): create the process-wide mooncake
    ``Buffer`` (RDMA-registered EP buffer) before capture / graph load, for the
    same SAVE/LOAD allocation-sequence parity reasons as DeepEP. Without it the
    first MoE dispatch -- and the torch.compile / DeepGEMM lazy init around it
    -- runs inside the captured forward on SAVE."""
    from sglang.srt.layers.moe.token_dispatcher.mooncake import (
        EPBuffer,
        MooncakeEPDispatcher,
    )

    if EPBuffer.get_existing_buffer() is not None:
        return True

    mk = _find_dispatcher(model, MooncakeEPDispatcher)
    impl = getattr(mk, "_low_latency_dispatcher", None)
    if impl is not None:
        t0 = time.perf_counter()
        impl._get_buffer()
        logger.info(
            "[Foundry] Bootstrapped Mooncake EP buffer pre-capture in %.3fs",
            time.perf_counter() - t0,
        )
        return True

    logger.warning(
        "[Foundry] Mooncake a2a backend active but no MooncakeEPDispatcher found; "
        "buffer not bootstrapped (capture may fail inside stream capture)."
    )
    return False


def _bootstrap_deepep_v2_buffer(model) -> bool:
    """DeepEP v2 counterpart: create the process-wide ``ElasticBuffer`` (its own
    NCCL communicator + symmetric-memory windows) before capture / graph load.

    sglang creates it lazily on the first dispatch, which on SAVE is inside the
    first captured forward and on LOAD would be the first request — so the
    window VA and NCCL state would sit at different allocation-sequence points
    on the two paths. Collective over the EP group, like the v1 bootstrap.
    """
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2Buffer,
        DeepEPv2Dispatcher,
    )

    if DeepEPv2Buffer._state().buffer is not None:
        return True

    v2 = _find_dispatcher(model, DeepEPv2Dispatcher)
    if v2 is not None:
        t0 = time.perf_counter()
        v2._impl._get_buffer()
        logger.info(
            "[Foundry] Bootstrapped DeepEP v2 ElasticBuffer pre-capture in %.3fs",
            time.perf_counter() - t0,
        )
        return True

    logger.warning(
        "[Foundry] DeepEP v2 backend active but no DeepEPv2Dispatcher found on the "
        "model; ElasticBuffer not bootstrapped."
    )
    return False


@dataclass
class _DecodeRestore:
    kind: str
    pending: Any
    graph_files: list[tuple[int, str, dict[str, Any]]]
    output_specs: dict[str, Any]
    timing: _RestoreTiming
    next: int = 0


_decode_restore: _DecodeRestore | None = None


def _shape_label(size: int, variant: str | None) -> str:
    return f"size={size}" + (f" variant={variant}" if variant is not None else "")


@nvtx_traced("foundry.graph_restore.decode_start")
def start_decode_graph_restore(kind: str = DEFAULT_DECODE_KIND) -> None:
    """LOAD, at a decode-type runner's capture: start building every graph of
    this scope ``kind`` in one ``start_graph_builds`` call (the manifest's
    template / on-demand linking needs a partition in one call; the builds
    run on foundry's background thread). The graphs are then finished one at
    a time by restore_next_decode_graph, from inside the upstream capture
    loop, so their allocator events replay at the points SAVE captured them,
    between the same per-shape eager work."""
    global _decode_restore
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        raise RuntimeError("Foundry SGLang graph extension is not initialized for LOAD")
    if _decode_restore is not None:
        raise RuntimeError(
            f"Foundry decode graph restore of {_decode_restore.kind} is still open "
            f"at the start of {kind}"
        )
    graph_files = _scan_graph_files(cfg.workspace_dir, kind)
    if not graph_files:
        raise RuntimeError(f"No Foundry SGLang {kind} graph files found in {cfg.workspace_dir}")

    # Graphs may reference NVSHMEM symbols; the DeepEP bootstrap ran before.
    # Single-GPU dense models have no NVSHMEM modules (no-op there).
    cge.init_nvshmem_for_loaded_modules()

    t0 = time.perf_counter()
    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    _decode_restore = _DecodeRestore(
        kind=kind,
        pending=pending,
        graph_files=graph_files,
        output_specs=load_decode_graph_state().get("output_specs", {}),
        timing=_RestoreTiming(t0=t0),
    )


def restore_next_decode_graph(key: Any) -> tuple[Any, Any]:
    """LOAD: stands in for capturing decode-type graph ``key``. The archive
    holds the scope's graphs in capture order (``reversed(capture_bs)``, and
    per size the runner's variants in its order); the upstream loop asks for
    them in the same order, which is checked here."""
    r = _decode_restore
    state = get_state()
    if r is None or state is None:
        raise RuntimeError("Foundry decode graph restore was not started")
    size, variant = _decode_shape(key)
    if r.next >= len(r.graph_files):
        raise RuntimeError(
            f"Foundry archive holds {len(r.graph_files)} {r.kind} graphs, but the capture loop "
            f"asks for more ({_shape_label(size, variant)}): the capture sizes "
            "(--cuda-graph-bs) or speculative settings differ from SAVE"
        )
    _index, filename, meta = r.graph_files[r.next]
    if (meta["size"], meta["variant"]) != (size, variant):
        raise RuntimeError(
            f"Foundry {r.kind} graph order mismatch: the capture loop asks for "
            f"{_shape_label(size, variant)}, the next archived graph is {filename}. The capture "
            "sizes (--cuda-graph-bs) or speculative settings differ between SAVE and LOAD."
        )
    t = time.perf_counter()
    with nvtx_range("foundry.graph_restore.finish_one"):
        graph, tensors = FoundryCUDAGraph.finish_one_graph_load(r.pending, r.next)
    r.timing.calls.append((t, time.perf_counter()))
    r.next += 1
    output = _unflatten_prefill_output(
        r.output_specs.get(filename, _LEGACY_DECODE_OUTPUT_SPEC), tensors
    )
    # Strong reference next to the backend's own.
    state.loaded_graphs[(r.kind, size, variant)] = (graph, output)
    return graph, output


def finish_decode_graph_restore(loop_s: float) -> None:
    """LOAD, after the upstream capture loop of a decode-type scope
    (``loop_s`` is its wall): check that it restored every archived graph of
    the scope's kind."""
    global _decode_restore
    r = _decode_restore
    _decode_restore = None
    if r is None:
        raise RuntimeError("Foundry decode graph restore was not started")
    if r.next != len(r.graph_files):
        raise RuntimeError(
            f"Foundry archive holds {len(r.graph_files)} {r.kind} graphs, the capture loop "
            f"restored {r.next}: the capture sizes (--cuda-graph-bs) or speculative settings "
            "differ from SAVE"
        )
    _log_restore(
        "decode" if r.kind == DEFAULT_DECODE_KIND else r.kind, r.next, r.pending, r.timing, loop_s
    )
