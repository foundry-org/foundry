# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import logging
import os
import re
import time
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
from foundry.integration.sglang.runtime import get_state

logger = logging.getLogger(__name__)

_pending_graph_builds: tuple[Any, list[tuple[int, str, dict[str, Any]]]] | None = None
_GRAPH_FILENAME_RE = re.compile(r"^graph_(?P<index>\d+)_FULL_t(?P<bs>\d+)_r\d+_UX_pcN\.json$")


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


def capture_graph(graph, pool, stream, run_once_fn):
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.SAVE:
        with foundry_graph_ctx(graph, pool=pool, stream=stream):
            return run_once_fn()
    return None


def save_graph(graph, output: Any, key: Any) -> None:
    cfg = get_config()
    state = get_state()
    if cfg is None or state is None or cfg.workspace_dir is None:
        raise RuntimeError("Foundry SGLang graph extension is not initialized")

    packed_output = _pack_output(output)
    filename = _graph_filename(state.capture_index, key)
    graph_path = os.path.join(cfg.workspace_dir, filename)
    graph.save(graph_path, packed_output)

    state.capture_index += 1
    logger.info("[Foundry] Saved SGLang CUDA graph %s key=%s", filename, key)


def save_graph_manifest() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    foundry_pkg.save_graph_manifest(cfg.workspace_dir, enable_templates=cfg.graph_templates)


def pack_fatbins() -> None:
    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return
    cge.pack_fatbins_to_folder(cfg.workspace_dir)
    cge.set_pack_fatbins_on_exit(False)


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
    results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Finished SGLang graph loads for %d graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))


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

    # NVSHMEM init runs once before any graph loads — graphs may reference
    # NVSHMEM symbols. Single-GPU dense models have 0 NVSHMEM modules, so
    # this is a no-op there but kept for EP parity.
    cge.init_nvshmem_for_loaded_modules()

    paths = [os.path.join(cfg.workspace_dir, filename) for _, filename, _ in graph_files]
    t0 = time.perf_counter()
    pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=4)
    results = FoundryCUDAGraph.finish_graph_loads(pending)
    logger.info(
        "[Foundry] Loaded %d SGLang graphs in %.3fs",
        len(results),
        time.perf_counter() - t0,
    )

    for i, (_index, _filename, meta) in enumerate(graph_files):
        graph, tensors = results[i]
        state.loaded_graphs[meta["key"]] = (graph, _unpack_output(tensors))
