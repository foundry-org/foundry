# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry CUDA graph save/load helpers for SGLang."""

from __future__ import annotations

import logging
import os
import re
import time
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
    foundry_pkg.save_graph_manifest(cfg.workspace_dir)


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


def initialize_attention_metadata_for_bs(cuda_graph_runner, bs: int) -> None:
    """Populate ``decode_cuda_graph_metadata[bs]`` for runtime replay.

    The FlashInfer wrappers and their internal ``_int_workspace_buffer``
    are constructed here, outside the captured graph. The graph's
    runtime kernels reference these buffer addresses, so LOAD must
    re-run the same call before runtime replay so the wrappers exist
    at deterministic VMM addresses.
    """
    buffers = cuda_graph_runner.buffers
    num_tokens = bs * cuda_graph_runner.num_tokens_per_bs
    encoder_lens = buffers.encoder_lens[:bs] if cuda_graph_runner.is_encoder_decoder else None
    spec_info = cuda_graph_runner.get_spec_info(num_tokens)
    cuda_graph_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
        bs,
        num_tokens,
        buffers.req_pool_indices[:bs],
        buffers.seq_lens[:bs],
        encoder_lens,
        cuda_graph_runner.capture_forward_mode,
        spec_info,
    )


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
