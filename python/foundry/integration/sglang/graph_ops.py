# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
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
from foundry.integration.sglang.runtime import get_state, load_warmup_state, save_warmup_state

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
# process order and the output structure of each decode-type graph.
_DECODE_STATE_FILE = "decode_graphs.json"
# PrefillCudaGraphRunner's only ShapeKey variants (_chunked_prefix_variant).
_CHUNKED_PREFIX_LABEL = "chunked_prefix:"
# SAVE: output structure of each prefill graph whose output is not one tensor.
_prefill_output_specs: dict[str, Any] = {}
# SAVE: output structure of each decode-type graph, and the kinds of the
# decode-type scopes finished so far (both written to _DECODE_STATE_FILE).
_decode_output_specs: dict[str, Any] = {}
_saved_decode_kinds: list[str] = []

# The decode runner of a process without speculative decoding (the kind a
# caller that passes none implies).
DEFAULT_DECODE_KIND = "target_decode"
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VARIANT_RE = re.compile(r"^[A-Za-z0-9]+$")


def check_decode_kind(kind: str | None) -> str:
    """The scope kind (None: :data:`DEFAULT_DECODE_KIND`); it goes into graph
    filenames and manifest partitions, so it must be a lowercase identifier."""
    if kind is None:
        return DEFAULT_DECODE_KIND
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError(f"[Foundry] invalid decode capture scope kind {kind!r}")
    return kind


def _is_graph_variant(key: Any) -> bool:
    """One of several graphs per shape: multi-stream (PD multiplexing;
    stream_idx left ShapeKey with upstream #43898), an attention variant or a
    sharded-verify logits variant."""
    return any(
        getattr(key, field, None) is not None
        for field in ("stream_idx", "attention_variant", "logits_variant")
    )


def _decode_shape(key: Any) -> tuple[int, str | None]:
    """(size, variant label) of a decode-type ShapeKey(size, variant_label,
    attention_variant, logits_variant; older sglang also stream_idx). Graph
    variants (:func:`_is_graph_variant`) are not persisted; a variant label
    (the EAGLE draft runner's greedy / sampling graph pair) is, by name."""
    if _is_graph_variant(key):
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
    if _is_graph_variant(key):
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
    target decode graphs keep the plain "decode" partition name."""
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
    one tensor when the output holds one, else the list in depth-first
    order."""
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
    """LOAD: the rank's decode record (scope kinds and output structures)."""
    global _decode_record
    if _decode_record is not None:
        return _decode_record
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")
    path = os.path.join(cfg.workspace_dir, _DECODE_STATE_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry decode record not found: {path} (re-save the archive with this version)"
        )
    with open(path) as f:
        record = json.load(f)
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
    # "prefill", "decode" (the target_decode scope) or another decode-type
    # scope kind.
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

    # Graphs may reference NVSHMEM symbols; sglang's warmup built the DeepEP buffer before.
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


def _load_flashinfer_tactics(path: str) -> tuple[bool, int]:
    """Install the tactics of ``path`` into FlashInfer's AutoTuner (its config
    lookup; a kernel's first call then picks the tuned tactic without tuning).
    Returns (accepted, entries loaded); ``accepted`` is False when the file's
    metadata (FlashInfer / CUDA / cuBLAS / cuDNN versions, GPU) differs from
    this process, in which case FlashInfer ignores the whole file."""
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    # _file_configs: the AutoTuner's table of file-loaded tactics (search_cache source 2).
    before = len(getattr(tuner, "_file_configs", {}))
    accepted = bool(tuner.load_configs(path))
    return accepted, len(getattr(tuner, "_file_configs", {})) - before


def bootstrap_flashinfer_tactics(cache_path: str | None) -> None:
    """Both modes, at the first capture scope (after the EagerRunner warmup,
    before any capture): give the captured kernels FlashInfer's tuned tactics
    without a tuning pass.

    Foundry pins ``--disable-flashinfer-autotune`` because the tuning pass
    benchmarks every tactic of every shape inside the recorded allocation
    sequence, with timing-dependent allocations SAVE and LOAD cannot repeat.
    The tactics themselves are static: a plain SGLang start of the same model
    tunes them once and writes its per-rank cache file (``cache_path``,
    SGLang's ``flashinfer_autotune_cache_path``; None when the model's kernels
    take no FlashInfer tactics). SAVE installs that file and keeps its content
    in the rank's ``warmup_state.json`` (section ``flashinfer_tactics``); LOAD
    installs that section. The tactics decide the kernels and their
    workspaces, so archived tactics LOAD cannot apply are an error, not a
    fallback. Without a cache file SAVE archives no section, and LOAD then
    runs FlashInfer's default tactics as SAVE did.
    """
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    if cfg.mode == CUDAGraphExtensionMode.LOAD:
        tactics = load_warmup_state().flashinfer_tactics
        if not tactics:
            if cache_path is not None:
                logger.info(
                    "[Foundry] FlashInfer tactics: the archive carries none; the kernels use "
                    "FlashInfer's default tactics, as SAVE did"
                )
            return
        # load_configs reads a file: hand it the archived section as one.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(tactics, f)
            path = f.name
        try:
            accepted, n = _load_flashinfer_tactics(path)
        finally:
            os.unlink(path)
        if not accepted or n == 0:
            raise RuntimeError(
                "[Foundry] this FlashInfer rejected the archive's tuned tactics "
                "(warmup_state.json flashinfer_tactics): its metadata (FlashInfer / CUDA / "
                "cuBLAS / cuDNN versions, GPU) differs from SAVE's. The tactics select the "
                "captured kernels and their workspaces, so LOAD needs SAVE's stack"
            )
        logger.info("[Foundry] FlashInfer tactics: %d loaded from the archive", n)
        return
    if cache_path is None:
        return  # the model's kernels take no FlashInfer tactics
    if not os.path.isfile(cache_path):
        logger.warning(
            "[Foundry] FlashInfer autotune is pinned off and no tuned-tactics cache exists at %s: "
            "the captured kernels use FlashInfer's default tactics (slower on FlashInfer MoE / "
            "GEMM paths). Tune once with a plain SGLang start of the same model and flags "
            "(autotune on writes that file), then SAVE",
            cache_path,
        )
        return
    accepted, n = _load_flashinfer_tactics(cache_path)
    if not accepted or n == 0:
        logger.warning(
            "[Foundry] FlashInfer tactics cache %s was rejected (metadata of another FlashInfer / "
            "CUDA / cuDNN / GPU) or is empty: the captured kernels use FlashInfer's default "
            "tactics. Re-tune with a plain SGLang start on this stack, then SAVE",
            cache_path,
        )
        return
    with open(cache_path) as f:
        tactics = json.load(f)
    state = load_warmup_state()
    state.flashinfer_tactics = tactics
    save_warmup_state(state)
    logger.info(
        "[Foundry] FlashInfer tactics: %d loaded from %s and archived in warmup_state.json",
        n,
        cache_path,
    )


def bootstrap_lazy_runtimes() -> None:
    """SAVE-only: DeepGEMM's one-time runtime init, outside any stream capture.

    The first DeepGEMM JIT call builds its runtime, which calls
    cudaFree(nullptr) and probes nvcc, both rejected inside a capture;
    get_num_sms() goes through the same device object. Allocation-free. LOAD
    does not need it: restore compiles nothing, and its first eager request
    performs the init outside capture, as native sglang does.

    torch.compile needs nothing here: TORCHDYNAMO_DISABLE=1 is pinned
    (pins.ENV_PINS), so no inductor compile runs in an engine.
    """
    t0 = time.perf_counter()
    try:
        import deep_gemm

        deep_gemm.get_num_sms()
    except Exception:
        pass
    logger.info("[Foundry] SGLang lazy runtimes bootstrapped in %.3fs", time.perf_counter() - t0)


def capture_state_unsupported(model) -> str | None:
    """Both modes, at the first capture scope: why sglang's warmup
    (``BaseRunner._prepare_capture_state``) could not have built the
    communication state the first forward otherwise creates lazily, or None.
    Built inside a capture, it records garbage (the expert mapping) or aborts
    (DeepEP buffers). A dispatcher class without the prebuild protocol
    (``BaseDispatcher.implements_prebuild``) builds it lazily by design; one
    that has it but is still ``prebuild_pending`` was skipped."""
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe.token_dispatcher.base import iter_dispatchers

    dispatchers = _all_dispatchers(iter_dispatchers(model))
    missing = sorted({type(d).__name__ for d in dispatchers if not d.implements_prebuild})
    if missing:
        return f"{', '.join(missing)} {'has' if len(missing) == 1 else 'have'} no prebuild"
    pending = sorted({type(d).__name__ for d in dispatchers if d.prebuild_pending})
    if any(isinstance(m, LogitsProcessor) and m.logits_gatherer_pending for m in model.modules()):
        pending.append("LogitsProcessor logits all-gather")
    if pending:
        return f"{', '.join(pending)} prebuild left state pending"
    return None


def _all_dispatchers(dispatchers) -> list:
    """``dispatchers`` and the dispatchers they wrap (attributes or list items,
    e.g. MaybeTboDeepEPDispatcher's inner ones), each once."""
    from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher

    found: dict[int, Any] = {}
    stack = list(dispatchers)
    while stack:
        dispatcher = stack.pop()
        if id(dispatcher) in found:
            continue
        found[id(dispatcher)] = dispatcher
        for value in vars(dispatcher).values():
            items = value if isinstance(value, (list, tuple)) else (value,)
            stack.extend(item for item in items if isinstance(item, BaseDispatcher))
    return list(found.values())


def agree_unsupported(step: str, reason: str | None) -> None:
    """Raise :class:`Unsupported` on every rank when any rank has a ``reason``.

    A rank that fell back alone would capture natively, with two eager warmup
    forwards per shape whose collectives the other ranks never issue, and the
    job would hang; the configuration is the same everywhere, so the ranks
    normally agree already and this only makes it certain. The vote runs on
    the world group's CPU (gloo) group: no device memory, so the allocation
    layout is the same as without it. A non-None vote gathers the reasons."""
    import torch
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() > 1:
        from sglang.srt.distributed import parallel_state as ps

        world = ps.get_world_group()
        vote = torch.tensor([0 if reason is None else 1], dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MAX, group=world.cpu_group)
        if not vote.item():
            return
        reasons: list = [None] * world.world_size
        dist.all_gather_object(reasons, reason, group=world.cpu_group)
        if reason is None:
            rank, first = next((r, text) for r, text in enumerate(reasons) if text is not None)
            reason = f"{first} (found on rank {rank})"
    if reason is not None:
        from foundry.integration.sglang.api import Unsupported

        raise Unsupported(step, reason)


def capture_state_objects(model) -> list:
    """SAVE: the objects whose attributes a capture must not create: every
    dispatcher of ``model`` (the wrapped ones too) and its logits processors.
    Off with ``FOUNDRY_SGLANG_CAPTURE_SNAPSHOT=0`` (diagnosis only)."""
    if os.environ.get("FOUNDRY_SGLANG_CAPTURE_SNAPSHOT") == "0":
        return []
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe.token_dispatcher.base import iter_dispatchers

    objects = _all_dispatchers(iter_dispatchers(model))
    return objects + [m for m in model.modules() if isinstance(m, LogitsProcessor)]


def attribute_snapshot(objects: list) -> list[tuple[frozenset, frozenset]]:
    """Per object: (names of all its attributes, names whose value is not
    None). Names only, no value is read beyond the None check."""
    return [
        (frozenset(vars(obj)), frozenset(k for k, v in vars(obj).items() if v is not None))
        for obj in objects
    ]


def created_attributes(objects: list, before: list) -> list[tuple[Any, str, bool]]:
    """(object, name, existed as None) of every attribute that went from absent
    or None to a value since ``before``."""
    created = []
    for obj, (names, set_names) in zip(objects, before):
        for name, value in vars(obj).items():
            if value is not None and name not in set_names:
                created.append((obj, name, name in names))
    return created


def reset_created_attributes(created: list) -> None:
    """Put the attributes back to absent / None, so the next eager forward
    builds them outside any capture, as natively."""
    for obj, name, existed in created:
        if existed:
            setattr(obj, name, None)
        else:
            delattr(obj, name)


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

    # Graphs may reference NVSHMEM symbols; sglang's warmup built the DeepEP buffer before.
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
    try:
        spec = r.output_specs[filename]
    except KeyError:
        raise RuntimeError(
            f"Foundry decode record has no output structure for {filename} "
            "(re-save the archive with this version)"
        ) from None
    output = _unflatten_prefill_output(spec, tensors)
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
