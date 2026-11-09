# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Save / restore of SGLang's breakable (BCG) prefill CUDA graphs.

A breakable graph is a chain of CUDA graph segments with an eager call (a
"break", ``eager_on_graph``: attention, linear attention, ...) between two
consecutive segments. SGLang's ``BreakableCudaGraphBackend.capture_one``
hands its capture body to :func:`capture_one` (integration API 2.5):

- SAVE runs the body once, without warm-up forwards, with every segment
  captured into a foundry graph. Each non-empty segment is archived as a graph
  file of its own (``graph_<i>_BCG_t<tokens>_s<segment>_UX_pc<chunks>.json``),
  in capture order, so LOAD replays the allocator events segment by segment.
  Each break is archived as data: the decorated function (module and
  qualified name), and its arguments as a structure whose leaves are tensors,
  submodules of the runner's model (by name) and scalars. The tensors (break
  arguments and the stored output, a slice of the backend's shared output
  buffer) are recorded as the output tensors of the shape's first non-empty
  segment, which archives their address, shape, stride and dtype.
- LOAD restores the segments where SAVE captured them, rebuilds every break as
  ``functools.partial`` of the same function over the archived arguments
  (tensors as views at the archived addresses, like the weak references the
  capture keeps), and runs no forward.

Request state is read inside a break body when it runs (``get_forward_batch``),
so nothing request-specific is archived. Breaks that run an ``on_capture``
host hook or allocate an ``output_factory`` buffer are not supported: their
closure is not the plain ``partial(run, ...)`` and SAVE refuses them.
"""

from __future__ import annotations

import enum
import functools
import importlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.integration.sglang.config import CUDAGraphExtensionMode, get_config
from foundry.integration.sglang.runtime import get_state

logger = logging.getLogger(__name__)

BCG_FILENAME_RE = re.compile(
    r"^graph_(?P<index>\d+)_BCG(?P<scope>\d+)_t(?P<tokens>\d+)_s(?P<seg>\d+)_UX_pc(?P<chunks>\d+)\.json$"
)
# Per-rank SAVE record of each breakable prefill scope (the target runner's,
# then e.g. an EAGLE draft runner's): request slots, the DP-gather latch, and
# per shape its segments, breaks and output structure.
STATE_FILE = "breakable_graphs.json"


def state_file(scope: int) -> str:
    return STATE_FILE if scope == 0 else f"breakable_graphs_{scope}.json"


def _workspace_dir() -> str:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")
    return cfg.workspace_dir


def scan_files(workspace_dir: str) -> list[tuple[int, str]]:
    """Archived BCG segment files in capture order."""
    out = []
    for filename in os.listdir(workspace_dir):
        m = BCG_FILENAME_RE.match(filename)
        if m:
            out.append((int(m.group("index")), filename))
    out.sort()
    return out


def has_graphs() -> bool:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return False
    return os.path.exists(os.path.join(cfg.workspace_dir, STATE_FILE))


# ---------------------------------------------------------------------------
# Break encoding
# ---------------------------------------------------------------------------


class UnsupportedBreak(RuntimeError):
    pass


def _break_parts(fn: Any) -> tuple[Any, tuple, dict]:
    """(decorated function, args, kwargs) of a break closure built by
    ``eager_on_graph`` without ``output_factory``: ``partial(run, *args,
    **kwargs)`` where ``run`` closes over the decorated function ``inner``."""
    if not isinstance(fn, functools.partial):
        raise UnsupportedBreak(
            f"break closure {getattr(fn, '__qualname__', fn)!r} is not a functools.partial "
            "(eager_on_graph with output_factory / pass_output); Foundry cannot archive it"
        )
    run = fn.func
    names = getattr(getattr(run, "__code__", None), "co_freevars", ())
    if "inner" not in names:
        raise UnsupportedBreak(f"break closure {run!r} does not close over 'inner'")
    inner = run.__closure__[names.index("inner")].cell_contents
    return inner, fn.args, dict(fn.keywords)


def _resolve_inner(module: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    # The decorator returns an _EagerOnGraphCallable; class attribute access
    # (instance None) gives it back unbound; its eager entry is the function.
    return getattr(obj, "_eager_call", obj)


def _inner_ref(inner: Any) -> dict[str, str]:
    module = getattr(inner, "__module__", None)
    qualname = getattr(inner, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        raise UnsupportedBreak(
            f"break function {qualname!r} in {module!r} is not importable by name"
        )
    try:
        resolved = _resolve_inner(module, qualname)
    except (ImportError, AttributeError) as exc:
        raise UnsupportedBreak(
            f"break function {module}.{qualname} does not resolve: {exc}"
        ) from exc
    if resolved is not inner:
        raise UnsupportedBreak(
            f"break function {module}.{qualname} resolves to {resolved!r}, not the captured "
            "function (on_capture hooks and re-decorated functions are not supported)"
        )
    return {"module": module, "qualname": qualname}


class _Encoder:
    """Argument structure -> JSON spec; tensors are appended to ``tensors``
    and referenced by index."""

    def __init__(self, module_names: dict[int, str], tensors: list[torch.Tensor]):
        self.module_names = module_names
        self.tensors = tensors

    def __call__(self, x: Any, path: str) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return {"v": x}
        if isinstance(x, torch.Tensor):
            if x.device.type != "cuda":
                raise UnsupportedBreak(f"break argument {path} is a {x.device.type} tensor")
            if x.numel() == 0:
                return {
                    "empty": {
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "dtype": str(x.dtype).removeprefix("torch."),
                    }
                }
            self.tensors.append(x)
            return {"t": len(self.tensors) - 1}
        if isinstance(x, torch.nn.Module):
            name = self.module_names.get(id(x))
            if name is None:
                raise UnsupportedBreak(
                    f"break argument {path} is a {type(x).__qualname__} that is not a "
                    "submodule of the runner's model"
                )
            return {"module": name}
        if isinstance(x, enum.Enum):
            return {
                "enum": {
                    "module": type(x).__module__,
                    "qualname": type(x).__qualname__,
                    "name": x.name,
                }
            }
        if isinstance(x, tuple):
            return {"tuple": [self(e, f"{path}[{i}]") for i, e in enumerate(x)]}
        if isinstance(x, list):
            return {"list": [self(e, f"{path}[{i}]") for i, e in enumerate(x)]}
        if isinstance(x, dict):
            if not all(isinstance(k, str) for k in x):
                raise UnsupportedBreak(f"break argument {path} is a dict with non-str keys")
            return {"dict": {k: self(v, f"{path}[{k!r}]") for k, v in x.items()}}
        raise UnsupportedBreak(
            f"break argument {path} is a {type(x).__module__}.{type(x).__qualname__}; "
            "Foundry archives tensors, model submodules, enums and scalars only"
        )


def _decode(spec: Any, tensors: list[torch.Tensor], modules: dict[str, Any], device) -> Any:
    if "v" in spec:
        return spec["v"]
    if "t" in spec:
        return tensors[spec["t"]]
    if "empty" in spec:
        e = spec["empty"]
        return torch.empty_strided(
            e["shape"], e["stride"], dtype=getattr(torch, e["dtype"]), device=device
        )
    if "module" in spec:
        return modules[spec["module"]]
    if "enum" in spec:
        e = spec["enum"]
        cls = _resolve_inner(e["module"], e["qualname"])
        return cls[e["name"]]
    if "tuple" in spec:
        return tuple(_decode(s, tensors, modules, device) for s in spec["tuple"])
    if "list" in spec:
        return [_decode(s, tensors, modules, device) for s in spec["list"]]
    if "dict" in spec:
        return {k: _decode(s, tensors, modules, device) for k, s in spec["dict"].items()}
    raise ValueError(f"[Foundry] bad BCG break spec {spec!r}")


def _run_break(inner, *args, **kwargs) -> None:
    out = inner(*args, **kwargs)
    assert out is None, f"{inner.__qualname__} is a graph break and must return None"


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------


_model: Any = None
_module_names: dict[int, str] = {}
_saved_shapes: list[dict[str, Any]] = []
# Index of the breakable prefill scope being run in this process (0, 1, ...).
_scope = -1
# LOAD: segment graphs restored over all scopes (the init summary's count,
# which on SAVE counts every captured graph).
_restored_segments = 0


def restored_segment_count() -> int:
    return _restored_segments


def begin_scope(model: Any) -> None:
    """At the prefill capture scope: the runner's model, whose submodules the
    breaks reference by name."""
    global _model, _module_names, _scope, _saved_shapes
    # Several scopes per process (EAGLE: the target's and the draft's prefill
    # runner), each archived and restored on its own, in the same order.
    _scope += 1
    _saved_shapes = []
    _model = model
    _module_names = {id(m): n for n, m in model.named_modules(remove_duplicate=False)}


def _shape(key: Any) -> tuple[int, int]:
    from foundry.integration.sglang.graph_ops import _is_graph_variant, _prefill_shape

    if _is_graph_variant(key):
        raise RuntimeError(
            f"[Foundry] shape {key} captures a graph variant; Foundry save/load does not "
            "support graph variants"
        )
    return _prefill_shape(key)


class _EmptySegment:
    """A segment that captured no kernels: SGLang replays it as a no-op."""

    def replay(self) -> None:
        pass

    def reset(self) -> None:
        pass


def save_shape(key: Any, capture) -> tuple[Any, Any]:
    """SAVE: capture one breakable shape and archive it."""
    from foundry.integration.sglang.graph_ops import _flatten_prefill_output

    state = get_state()
    ws = _workspace_dir()
    tokens, chunks = _shape(key)
    # The segments' RNG capture states come from the slab reserved at the
    # layout start (rng_slab), so the capture grows no torch pool.
    graph, stored = capture(segment_graph_factory=FoundryCUDAGraph)
    tensors: list[torch.Tensor] = []
    enc = _Encoder(_module_names, tensors)
    breaks = []
    for i, fn in enumerate(graph._break_fns):
        inner, args, kwargs = _break_parts(fn)
        breaks.append(
            {
                "fn": _inner_ref(inner),
                "args": enc(args, f"break[{i}].args"),
                "kwargs": enc(kwargs, f"break[{i}].kwargs"),
            }
        )
    out_spec, out_tensors = _flatten_prefill_output(stored)
    first_out = len(tensors)
    tensors.extend(out_tensors)

    segments: list[str | None] = []
    carrier = None
    for s, seg in enumerate(graph._segments):
        filename = f"graph_{state.capture_index}_BCG{_scope}_t{tokens}_s{s}_UX_pc{chunks}.json"
        carry = carrier is None
        try:
            seg.save(os.path.join(ws, filename), list(tensors) if carry and tensors else None)
        except RuntimeError as exc:
            # A segment that captured no kernels (back-to-back breaks) has no
            # nodes and cannot be archived; LOAD replays nothing in its place.
            if "has no nodes" not in str(exc):
                raise
            segments.append(None)
            continue
        state.capture_index += 1
        if carry:
            carrier = s
        segments.append(filename)
    if carrier is None:
        raise RuntimeError(f"[Foundry] breakable shape t={tokens} captured no kernels")
    for s, seg in enumerate(graph._segments):
        if segments[s] is None:
            graph._segments[s] = _EmptySegment()
    _saved_shapes.append(
        {
            "tokens": tokens,
            "chunks": chunks,
            "segments": segments,
            "carrier": carrier,
            "num_tensors": len(tensors),
            "breaks": breaks,
            "output": {"spec": out_spec, "first": first_out},
        }
    )
    logger.info(
        "[Foundry] Saved SGLang breakable prefill graph t=%d pc=%d: %d segments "
        "(%d empty), %d breaks, %d tensors",
        tokens,
        chunks,
        len(segments),
        segments.count(None),
        len(breaks),
        len(tensors),
    )
    return graph, stored


def save_state(*, req_slots: int, has_dp_gather: bool) -> None:
    record = {
        "req_slots": req_slots,
        "prefill_graph_has_dp_gather": has_dp_gather,
        "shapes": _saved_shapes,
    }
    with open(os.path.join(_workspace_dir(), state_file(_scope)), "w") as f:
        json.dump(record, f)


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------


@dataclass
class _Restore:
    record: dict[str, Any]
    pending: Any
    t0: float
    next_shape: int = 0
    next_file: int = 0
    in_calls: float = 0.0
    keep: list[Any] = field(default_factory=list)


_restore: _Restore | None = None


def start_restore() -> dict[str, Any]:
    global _restore
    from foundry import ops as cge

    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        raise RuntimeError("Foundry SGLang graph extension is not initialized for LOAD")
    path = os.path.join(cfg.workspace_dir, state_file(_scope))
    if not os.path.exists(path):
        raise RuntimeError(
            f"Foundry archive {cfg.workspace_dir} has no breakable prefill graphs for scope "
            f"{_scope}: it was saved with another --cuda-graph-backend-prefill or speculative "
            "decoding setting. Re-save with the same settings as LOAD."
        )
    with open(path) as f:
        record = json.load(f)
    files = [fn for shape in record["shapes"] for fn in shape["segments"] if fn is not None]
    on_disk = {
        fn
        for _, fn in scan_files(cfg.workspace_dir)
        if BCG_FILENAME_RE.match(fn)["scope"] == str(_scope)
    }
    if set(files) != on_disk:
        raise RuntimeError("[Foundry] breakable archive record and segment files disagree")
    cge.init_nvshmem_for_loaded_modules()
    t0 = time.perf_counter()
    pending = FoundryCUDAGraph.start_graph_builds(
        [os.path.join(cfg.workspace_dir, fn) for fn in files], num_threads=4
    )
    _restore = _Restore(record=record, pending=pending, t0=t0)
    logger.info(
        "[Foundry] Started SGLang breakable prefill graph builds: %d shapes, %d segments",
        len(record["shapes"]),
        len(files),
    )
    return record


def restore_shape(key: Any) -> tuple[Any, Any]:
    """LOAD: stands in for capturing breakable shape ``key``; no forward runs."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
    )

    from foundry.integration.sglang.graph_ops import _unflatten_prefill_output

    r = _restore
    if r is None:
        raise RuntimeError("Foundry breakable graph restore was not started")
    tokens, chunks = _shape(key)
    shapes = r.record["shapes"]
    if r.next_shape >= len(shapes):
        raise RuntimeError(
            f"Foundry archive holds {len(shapes)} breakable prefill graphs, the capture loop asks "
            f"for more (t={tokens} pc={chunks}): the prefill capture shapes differ from SAVE"
        )
    shape = shapes[r.next_shape]
    if (shape["tokens"], shape["chunks"]) != (tokens, chunks):
        raise RuntimeError(
            f"Foundry breakable graph order mismatch: the capture loop asks for t={tokens} "
            f"pc={chunks}, the archive's next shape is t={shape['tokens']} pc={shape['chunks']}"
        )
    t = time.perf_counter()
    graph = BreakableCUDAGraph(None)
    tensors: list[torch.Tensor] | None = None
    for s, filename in enumerate(shape["segments"]):
        if filename is None:
            graph._segments.append(_EmptySegment())
            continue
        seg, out = FoundryCUDAGraph.finish_one_graph_load(r.pending, r.next_file)
        r.next_file += 1
        graph._segments.append(seg)
        if s == shape["carrier"]:
            tensors = [] if out is None else ([out] if isinstance(out, torch.Tensor) else list(out))
    if tensors is None or len(tensors) != shape["num_tensors"]:
        raise RuntimeError(
            f"[Foundry] breakable shape t={tokens}: restored "
            f"{0 if tensors is None else len(tensors)} tensors, archive has {shape['num_tensors']}"
        )
    modules = dict(_model.named_modules(remove_duplicate=False)) if _model is not None else {}
    device = torch.device("cuda", torch.cuda.current_device())
    for b in shape["breaks"]:
        inner = _resolve_inner(b["fn"]["module"], b["fn"]["qualname"])
        args = _decode(b["args"], tensors, modules, device)
        kwargs = _decode(b["kwargs"], tensors, modules, device)
        graph._break_fns.append(functools.partial(_run_break, inner, *args, **kwargs))
    o = shape["output"]
    stored = _unflatten_prefill_output(o["spec"], tensors[o["first"] :])
    r.keep.append((graph, stored))
    r.next_shape += 1
    r.in_calls += time.perf_counter() - t
    return graph, stored


def finish_restore(loop_s: float) -> None:
    global _restore
    r = _restore
    _restore = None
    if r is None:
        raise RuntimeError("Foundry breakable graph restore was not started")
    n = len(r.record["shapes"])
    if r.next_shape != n:
        raise RuntimeError(
            f"Foundry archive holds {n} breakable prefill graphs, the capture loop restored "
            f"{r.next_shape}: the prefill capture shapes differ from SAVE"
        )
    state = get_state()
    state.loaded_graphs[("breakable", _scope)] = r.keep
    state.prefill_graphs_restored = True
    global _restored_segments
    _restored_segments += r.next_file
    logger.info(
        "[Foundry] Loaded %d SGLang breakable prefill graphs (%d segments) in %.3fs; capture "
        "loop on LOAD %.3fs (per-shape prep %.3fs)",
        n,
        r.next_file,
        r.in_calls,
        loop_s,
        max(0.0, loop_s - r.in_calls),
    )
