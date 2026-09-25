# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

import foundry as foundry_pkg
from foundry import ops as cge
from foundry.graph import CUDAGraph as FoundryCUDAGraph
from foundry.graph import graph as foundry_graph_ctx
from foundry.integration.sglang.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
)
from foundry.integration.sglang.nvtx import nvtx_range, nvtx_traced
from foundry.integration.sglang.runtime import get_state

logger = logging.getLogger(__name__)

_pending_graph_builds: tuple[Any, list[tuple[int, str, dict[str, Any]]]] | None = None
_GRAPH_FILENAME_RE = re.compile(r"^graph_(?P<index>\d+)_FULL_t(?P<bs>\d+)_r\d+_UX_pcN\.json$")
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
# PrefillCudaGraphRunner's only ShapeKey variants (_chunked_prefix_variant).
_CHUNKED_PREFIX_LABEL = "chunked_prefix:"
# SAVE: output structure of each prefill graph whose output is not one tensor.
_prefill_output_specs: dict[str, Any] = {}


def _batch_size_from_key(key: Any) -> int:
    if isinstance(key, int):
        return key
    # ShapeKey(size, stream_idx, variant_label, attention_variant) -- the last
    # field was called dsa_variant before sglang #39176. Phase 1 persists only
    # plain single-stream decode graphs, where size == bs.
    if hasattr(key, "size"):
        variant = getattr(key, "attention_variant", None) or getattr(key, "dsa_variant", None)
        if key.stream_idx is not None or key.variant_label is not None or variant is not None:
            raise ValueError(f"Foundry SGLang save/load does not support graph variants: {key!r}")
        return key.size
    key_str = str(key)
    for part in reversed(key_str.split("_")):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot derive batch size from SGLang CUDA graph key: {key!r}")


def _graph_filename(index: int, key: Any) -> str:
    batch_size = _batch_size_from_key(key)
    return f"graph_{index}_FULL_t{batch_size}_r{batch_size}_UX_pcN.json"


def _prefill_shape(key: Any) -> tuple[int, int]:
    """(num_tokens, prefix chunks) of a prefill ShapeKey."""
    variant = getattr(key, "attention_variant", None) or getattr(key, "dsa_variant", None)
    if key.stream_idx is not None or variant is not None:
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
    """Manifest partition: prefill and decode graphs are restored by separate
    start_graph_builds calls (at their own runner's capture point), so they
    must not share a template."""
    return "prefill" if _PREFILL_GRAPH_FILENAME_RE.match(filename) else "decode"


def _flatten_prefill_output(output: Any) -> tuple[Any, list[torch.Tensor]]:
    """Prefill graphs capture the transformer body, whose output is the hidden
    states: one tensor for most models, a nested tuple / list (auxiliary hidden
    states) or PPProxyTensors (non-last PP rank) otherwise. Returns a JSON spec
    of the structure and its tensors in depth-first order."""
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    tensors: list[torch.Tensor] = []

    def walk(x: Any) -> Any:
        if x is None:
            return "N"
        if isinstance(x, torch.Tensor):
            tensors.append(x)
            return "T"
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
        if "pp" in s:
            return PPProxyTensors({n: next(it) for n in s["pp"]})
        if "tuple" in s:
            return tuple(build(v) for v in s["tuple"])
        return [build(v) for v in s["list"]]

    out = build(spec)
    if next(it, None) is not None:
        raise RuntimeError("SGLang prefill CUDA graph restored more output tensors than recorded")
    return out


def _pack_output(output: Any) -> torch.Tensor:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if isinstance(output, LogitsProcessorOutput):
        if output.next_token_logits is None:
            raise TypeError("SGLang decode CUDA graph output has no next_token_logits")
        return output.next_token_logits

    if isinstance(output, torch.Tensor):
        return output

    raise TypeError(f"Unsupported SGLang CUDA graph output type: {type(output)!r}")


def _unpack_output(tensors: Any) -> Any:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if isinstance(tensors, (tuple, list)):
        if len(tensors) != 1:
            raise RuntimeError(f"Expected one SGLang CUDA graph output tensor, got {len(tensors)}")
        tensors = tensors[0]
    return LogitsProcessorOutput(next_token_logits=tensors)


def _scan_graph_files(workspace_dir: str) -> list[tuple[int, str, dict[str, Any]]]:
    graph_files = []
    for filename in os.listdir(workspace_dir):
        match = _GRAPH_FILENAME_RE.match(filename)
        if not match:
            continue
        meta = {
            "index": int(match.group("index")),
            "key": int(match.group("bs")),
        }
        graph_files.append((int(meta["index"]), filename, meta))
    graph_files.sort(key=lambda x: x[0])
    return graph_files


def create_device_graph():
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        return FoundryCUDAGraph()
    return torch.cuda.CUDAGraph()


@nvtx_traced("foundry.save.capture_graph")
def capture_graph(graph, pool, stream, run_once_fn):
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        with foundry_graph_ctx(graph, pool=pool, stream=stream):
            return run_once_fn()
    return None


def save_graph(graph, output: Any, key: Any, prefill_req_slots: int | None = None) -> None:
    """Persist one captured graph. ``prefill_req_slots`` is set for the
    prefill runner's graphs (its fixed request-slot count) and selects the
    prefill filename and output packing; decode graphs are unchanged."""
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    if prefill_req_slots is None:
        packed_output = _pack_output(output)
        filename = _graph_filename(state.capture_index, key)
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
    t0: float
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
        t0=t0,
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
    graph, tensors = FoundryCUDAGraph.finish_one_graph_load(r.pending, r.next)
    r.next += 1
    output = _unflatten_prefill_output(r.output_specs.get(filename, "T"), tensors)
    # Strong reference, like the decode graphs.
    state.loaded_graphs[("prefill", tokens, chunks)] = (graph, output)
    return graph, output


def finish_prefill_graph_restore() -> None:
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
    logger.info(
        "[Foundry] Loaded %d SGLang prefill graphs in %.3fs",
        r.next,
        time.perf_counter() - r.t0,
    )


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


@nvtx_traced("foundry.graph_restore.start")
def start_graph_builds() -> None:
    global _pending_graph_builds
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None or cfg.mode != CUDAGraphExtensionMode.LOAD:
        return

    graph_files = _scan_graph_files(cfg.workspace_dir)
    if not graph_files:
        raise RuntimeError(f"No Foundry SGLang graph files found in {cfg.workspace_dir}")

    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    t0 = time.perf_counter()
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    _pending_graph_builds = (pending, graph_files)
    logger.info(
        "[Foundry] Started SGLang graph builds for %d graphs in %.3fs",
        len(paths),
        time.perf_counter() - t0,
    )


@nvtx_traced("foundry.graph_restore.preload")
def preload_all_graphs() -> None:
    global _pending_graph_builds
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    if _pending_graph_builds is None:
        start_graph_builds()
    assert _pending_graph_builds is not None

    cge.init_nvshmem_for_loaded_modules()

    pending, graph_files = _pending_graph_builds
    _pending_graph_builds = None

    t0 = time.perf_counter()
    with nvtx_range("foundry.graph_restore.finish"):
        results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Finished SGLang graph loads for %d graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))


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


def bootstrap_logits_gatherer(cuda_graph_runner) -> bool:
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
    model = cuda_graph_runner.model_runner.model
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


def bootstrap_deepep_buffer(cuda_graph_runner) -> bool:
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
        return _bootstrap_deepep_v2_buffer(cuda_graph_runner)
    if backend.is_mooncake():
        return _bootstrap_mooncake_buffer(cuda_graph_runner)
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

    model = cuda_graph_runner.model_runner.model
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is None:
            continue
        # ``module.dispatcher`` is normally a MaybeTboDeepEPDispatcher wrapper
        # whose ``_inners`` hold the real DeepEPDispatcher(s); unwrap it. (Also
        # handle a bare DeepEPDispatcher for safety.)
        candidates = [dispatcher, *getattr(dispatcher, "_inners", [])]
        deepep = next((d for d in candidates if isinstance(d, DeepEPDispatcher)), None)
        if deepep is None:
            continue
        # Prefer the low-latency impl (the mode foundry captures); the buffer
        # is sized for whichever impls exist, so either bootstraps the shared
        # singleton.
        impl = getattr(deepep, "_low_latency_dispatcher", None) or getattr(
            deepep, "_normal_dispatcher", None
        )
        if impl is None:
            continue
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


def _bootstrap_mooncake_buffer(cuda_graph_runner) -> bool:
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

    model = cuda_graph_runner.model_runner.model
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is None:
            continue
        candidates = [dispatcher, *getattr(dispatcher, "_inners", [])]
        mk = next((d for d in candidates if isinstance(d, MooncakeEPDispatcher)), None)
        if mk is None:
            continue
        impl = getattr(mk, "_low_latency_dispatcher", None)
        if impl is None:
            continue
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


def _bootstrap_deepep_v2_buffer(cuda_graph_runner) -> bool:
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

    model = cuda_graph_runner.model_runner.model
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is None:
            continue
        candidates = [dispatcher, *getattr(dispatcher, "_inners", [])]
        v2 = next((d for d in candidates if isinstance(d, DeepEPv2Dispatcher)), None)
        if v2 is None:
            continue
        t0 = time.perf_counter()
        # Prototype bisect knob: build the buffer (NCCL comm, GIN/GDAKI context,
        # symmetric windows) with the hook's region suspended, so its
        # allocations go to plain CUDA memory. Non-deterministic across
        # SAVE/LOAD; only for isolating hook-vs-NCCL failures.
        outside_region = os.environ.get("FOUNDRY_V2_BUFFER_OUTSIDE_REGION") == "1"
        if outside_region:
            cge.stop_allocation_region()
        try:
            v2._impl._get_buffer()
        finally:
            if outside_region:
                cge.resume_allocation_region()
        logger.info(
            "[Foundry] Bootstrapped DeepEP v2 ElasticBuffer pre-capture in %.3fs%s",
            time.perf_counter() - t0,
            " (outside region)" if outside_region else "",
        )
        return True

    logger.warning(
        "[Foundry] DeepEP v2 backend active but no DeepEPv2Dispatcher found on the "
        "model; ElasticBuffer not bootstrapped."
    )
    return False


def initialize_attention_metadata_for_bs(cuda_graph_runner, bs: int) -> None:
    """Populate the backend's per-bs cuda-graph metadata for runtime replay.

    Drives the public capture-time entry point with a duck-typed batch
    carrying exactly the fields ``init_forward_metadata_out_graph`` reads.
    With ``in_capture=True`` FlashInfer's implementation first runs its
    allocation half (``_prepare_cuda_graph_metadata``: wrappers +
    ``_int_workspace_buffer``) and then the planner — the graph's runtime
    kernels reference these buffer addresses, so LOAD must re-run the same
    call before replay so the wrappers exist at deterministic VMM
    addresses. fa3-style backends allocate their metadata once in
    ``init_cuda_graph_state``; for them this only builds lightweight views
    and does not move the VMM cursor.
    """
    buffers = cuda_graph_runner.buffers
    attn_backend = cuda_graph_runner.attn_backend
    num_tokens = bs * cuda_graph_runner.captured_req_width
    encoder_lens = buffers.encoder_lens[:bs] if cuda_graph_runner.is_encoder_decoder else None
    spec_info = cuda_graph_runner.get_spec_info(num_tokens)
    forward_mode = cuda_graph_runner.capture_forward_mode

    fb = SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=bs,
        req_pool_indices=buffers.req_pool_indices[:bs],
        seq_lens=buffers.seq_lens[:bs],
        seq_lens_cpu=buffers.seq_lens_cpu[:bs],
        seq_lens_sum=int(buffers.seq_lens[:bs].sum().item()),
        encoder_lens=encoder_lens,
        spec_info=spec_info,
        out_cache_loc=buffers.out_cache_loc[:num_tokens],
        positions=buffers.positions[:num_tokens],
    )
    attn_backend.init_forward_metadata_out_graph(fb, in_capture=True)


def initialize_all_attention_metadata(cuda_graph_runner) -> None:
    """Pre-pass: populate ``decode_cuda_graph_metadata`` for all bs at once.

    Called on both SAVE and LOAD before the capture/load loop. Walking
    ``reversed(self.capture_bs)`` (largest first) matches SAVE's natural
    capture order; same order on both sides keeps the VMM cursor
    trajectory identical.
    """
    for bs in reversed(cuda_graph_runner.capture_bs):
        initialize_attention_metadata_for_bs(cuda_graph_runner, bs)


@nvtx_traced("foundry.graph_restore.load_all")
def load_all_graphs(cuda_graph_runner) -> None:
    """LOAD-time replacement for the upstream capture loop.

    All FlashInfer wrappers are pre-allocated by
    ``initialize_all_attention_metadata`` (called by the capture hook
    before this function), so the VMM cursor sits where SAVE recorded
    ``start_base_addr_0``. Load every graph in one
    ``start_graph_builds`` call — this is what enables template +
    on-demand linking in the manifest. ``finish_graph_loads`` then
    replays each graph's alloc events in sequence, advancing the
    cursor exactly the way SAVE did inside its capture loop.
    """
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    graph_files = _scan_graph_files(cfg.workspace_dir)
    if not graph_files:
        raise RuntimeError(f"No Foundry SGLang graph files found in {cfg.workspace_dir}")

    # NVSHMEM init runs once before any graph is finished/replayed — graphs may
    # reference NVSHMEM symbols. Single-GPU dense models have 0 NVSHMEM
    # modules, so this is a no-op there but kept for EP parity.
    cge.init_nvshmem_for_loaded_modules()

    global _pending_graph_builds
    t0 = time.perf_counter()
    if _pending_graph_builds is not None:
        # Builds were started at setup (start_graph_builds, right after the
        # binaries were loaded) and ran on the background thread during
        # torch-distributed init, weight loading and the memory-pool setup;
        # only the remainder is waited for here. finish_graph_loads replays
        # the allocator events and reconstructs output tensors, so the
        # cursor-dependent part still happens at this sequence point.
        pending, early_files = _pending_graph_builds
        _pending_graph_builds = None
        if [f for _, f, _ in early_files] != [f for _, f, _ in graph_files]:
            raise RuntimeError("Foundry: graph file list changed between setup and load")
        logger.info("[Foundry] Using the %d graph builds started at setup", len(graph_files))
    else:
        paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
        pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    with nvtx_range("foundry.graph_restore.finish"):
        results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Loaded %d SGLang graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))
