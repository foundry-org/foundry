# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Stable API for SGLang's in-tree Foundry adapter.

SGLang declares Foundry as an optional dependency and calls these functions
from its own call sites, through ``sglang.srt.utils.foundry_adapter``, when
``--cuda-graph-persistence {save,load}`` is given. Nothing here patches SGLang.
The resolution pins and rejects live in :mod:`.pins`, the call-site functions
in :mod:`.hooks`.

Compatibility: SGLang checks :data:`INTEGRATION_API_VERSION` at activation.
The major number changes when a function below is removed or changes its
signature or its call-site contract; the minor number when one is added.
2.0 is the contract below: every call site passes values (the model, the
draft flag, the pool config, the prefill request-slot count once per prefill
capture scope), never a runner or backend object. 2.1 adds the ``kind`` of a
decode-type capture scope (:func:`capture_scope`), so that the speculative
decoding runners (EAGLE target verify, draft decode, draft extend) are told
apart; a 2.0 caller passes none, which means the plain target decode runner.
2.2 adds ``flashinfer_tactics`` to :func:`capture_scope`. 2.3 adds
:func:`record_memory_pool_config`, one call at the end of
``_resolve_memory_pool_config``. 2.2 callers are not supported: the functions
only they called (``record_memory_pool_overrides``,
``before_alloc_memory_pool``, ``after_alloc_memory_pool``,
``validate_graph_config``, ``apply_env_pins``, ``after_parallel_init``,
``after_runner_distributed_init``) are gone.

Call sites (SGLang side, in process order):

===============================================  =========================================
SGLang site                                      function
===============================================  =========================================
resolution: ``handle_cuda_graph_persistence``    :func:`activate`, :func:`pin_server_args`
resolution: ``validate_cuda_graph_persistence``  :func:`validate_resolved_server_args`
``Engine`` launch, before the spawns             :func:`activate`
scheduler spawn                                  :func:`configure_subprocess`
DP-controller spawn                              :func:`configure_subprocess`
``run_scheduler_process``                        :func:`activate`
``bootstrap.init_parallel_runtime``              :func:`before_parallel_init`
``_resolve_memory_pool_config``                  :func:`replay_saved_memory_pool_config`
                                                 :func:`record_memory_pool_config` (2.3)
``BaseRunner._prepare_capture_state``            none (SGLang-owned prebuild of the
                                                 dispatchers and logits gatherers;
                                                 the first :func:`capture_scope`
                                                 checks them)
decode runner ``capture``                        :func:`capture_scope`
prefill runner ``capture``                       :func:`capture_scope`
``FullCudaGraphBackend.capture_one``             :func:`capture_one` (replaces the body)
===============================================  =========================================

Every function except :func:`activate` requires an
active process (:func:`activate` ran in it).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

INTEGRATION_API_VERSION = (2, 3)
MODES = ("save", "load")
FLAG = "--cuda-graph-persistence"

_active: tuple[str, str | None] | None = None


def activate(*, mode: str, config_path: str | None = None) -> None:
    """Make this process a Foundry SAVE or LOAD process.

    Loads the Foundry TOML at ``config_path`` (None: defaults, i.e. workspace
    ``foundry_archive`` in the working directory and the hook library of this
    installation); ``mode`` replaces the TOML's ``mode``. Applies the
    environment pins that do not depend on the resolved server args.
    Idempotent for the same arguments; called in the launcher (at the first
    resolution step) and at the top of every scheduler and DP-controller
    process, on the record it received."""
    global _active
    if mode not in MODES:
        raise ValueError(f"{FLAG} must be one of {MODES}, got {mode!r}")
    key = (mode, os.path.abspath(config_path) if config_path else None)
    if _active is not None:
        if _active != key:
            raise RuntimeError(
                f"[Foundry] already active with mode={_active[0]} config={_active[1]}; "
                f"cannot re-activate with mode={key[0]} config={key[1]}"
            )
        return

    from foundry.integration.sglang import pins
    from foundry.integration.sglang.config import (
        get_graph_extension_mode,
        get_workspace_root,
        load_graph_extension_config,
    )

    if config_path is not None and not os.path.isfile(config_path):
        raise FileNotFoundError(f"{FLAG}-config {config_path!r} does not exist")

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    load_graph_extension_config(config_path, mode=mode)
    pins.apply_env_pins()
    _active = key
    if t0_ns:
        logger.info(
            "[Foundry] SGLang spawn -> activate: %.1f ms",
            (time.perf_counter_ns() - int(t0_ns)) / 1e6,
        )
    logger.info(
        "[Foundry] SGLang graph persistence active: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )
    # Logging is not configured yet in the launcher when resolution runs.
    print(
        f"[Foundry] cuda graph persistence active: pid={os.getpid()} mode={mode} "
        f"config={config_path}",
        file=sys.stderr,
        flush=True,
    )


def _require_active() -> None:
    if _active is None:
        raise RuntimeError("[Foundry] not active in this process: call activate() first")


# ---------------------------------------------------------------------------
# Server-args resolution (launcher)
# ---------------------------------------------------------------------------


def pin_server_args(server_args: Any) -> None:
    """Before ``apply_inkling_prefill_cuda_graph_default``: pin decode ``full``,
    prefill ``full`` / ``disabled`` (default ``disabled``), profiling and
    FlashInfer autotune off, and every flag that selects state Foundry cannot
    replay (``pins.FIELD_PINS``); conflicting user values raise."""
    _require_active()
    from foundry.integration.sglang import pins

    pins.pin_graph_fields(server_args)
    pins.pin_unreplayable_fields(server_args)


def validate_resolved_server_args(server_args: Any) -> None:
    """After ``handle_other_validations``: reject what Foundry cannot persist
    (speculative decoding other than EAGLE / EAGLE3, LoRA, PD multiplexing,
    elastic-EP recapture,
    post-capture KV sizing), re-check the graph config and the strict pins
    after the later cascades, and pin the environment including
    ``NCCL_CUMEM_ENABLE`` (left alone for ``--moe-a2a-backend deepep_v2``)."""
    _require_active()
    from foundry.integration.sglang import pins

    pins.reject_unsupported_features(server_args)
    pins.validate_resolved_graph_config(server_args)
    pins.validate_pinned_fields(server_args)
    pins.apply_env_pins(server_args)


# ---------------------------------------------------------------------------
# Process spawn
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def configure_subprocess(server_args: Any = None):
    """Wrap ``Process.start()`` of a scheduler or DP-controller process: the
    child gets ``LD_PRELOAD`` (Foundry's CUDA hook, plus DeepEP's NVSHMEM host
    library and the optional udev-wait shim when configured) and
    ``FOUNDRY_MODE``; the parent's environment is restored on exit. With
    ``server_args``, first checks that the record was resolved with Foundry's
    graph pins and re-asserts the environment pins (both permanent and
    idempotent)."""
    _require_active()
    from foundry.integration.sglang import hooks, pins
    from foundry.integration.sglang import runtime as rt

    if server_args is not None:
        hooks._check_resolved_graph_config(server_args)
    pins.apply_env_pins(server_args)
    with rt.subprocess_preload_env():
        yield


# ---------------------------------------------------------------------------
# Scheduler process
# ---------------------------------------------------------------------------


def before_parallel_init(device: str) -> None:
    """Head of ``bootstrap.init_parallel_runtime``: bind this rank's VMM region
    (on its own device) before any communicator exists; LOAD also restores the
    CUDA modules and starts the graph-exec pool prewarm."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.before_parallel_init(device)


def replay_saved_memory_pool_config():
    """Head of ``KVCacheConfigurator._resolve_memory_pool_config``. LOAD returns
    the saved ``MemoryPoolConfig`` (the caller returns it and skips memory
    profiling) after replaying SAVE's context overrides; SAVE returns None."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.replay_saved_memory_pool_config()


def record_memory_pool_config(memory_pool_config: Any) -> None:
    """End of ``KVCacheConfigurator._resolve_memory_pool_config`` (2.3), with
    the ``MemoryPoolConfig`` it returns: SAVE keeps the context overrides the
    resolver issued and writes both into the rank's ``warmup_state.json`` for
    LOAD (whose resolver returns early through
    :func:`replay_saved_memory_pool_config`). The resolver runs for the target
    runner only (draft workers reuse its config), so no draft flag is needed."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.record_memory_pool_config(memory_pool_config)


def capture_scope(
    phase: str,
    *,
    model: Any,
    req_slots: int | None = None,
    kind: str | None = None,
    flashinfer_tactics: str | None = None,
):
    """Context manager around the body of ``DecodeCudaGraphRunner.capture``
    (``phase="decode"``, also reached by its subclasses: the EAGLE draft and
    draft-extend runners) or ``PrefillCudaGraphRunner.capture``
    (``phase="prefill"``, with the runner's fixed request-slot count
    ``req_slots``, recorded with every prefill graph captured or restored in
    the scope). ``model`` is the runner's loaded model: the first scope checks
    that sglang's warmup prebuilt its dispatchers and logits gatherer.
    ``kind`` (2.1, decode only) names the decode-type graph set, a lowercase
    identifier derived from values: ``<role>_<mode>`` with role ``target`` /
    ``draft`` (the runner's draft-worker flag) and mode
    ``decode``, ``verify`` (TARGET_VERIFY), ``extend`` (DRAFT_EXTEND_V2) or
    the forward mode's name; None means ``target_decode``. A process runs at
    most one decode-type scope per kind (EAGLE: ``target_verify``,
    ``draft_decode``, ``draft_extend``, in that order); each kind's graphs
    are archived and restored on their own, and LOAD must run the kinds SAVE
    ran, in the same order. SGLang's capture loop runs inside on both modes;
    the scope adds the bootstraps and the layout start (first scope), LOAD's
    preallocation and restore bookkeeping, and SAVE's manifest, fatbin pack
    and region layout (rewritten at the end of every decode-type scope).
    ``flashinfer_tactics`` (2.2): the path of the runner's FlashInfer
    tuned-tactics cache file (SGLang's ``flashinfer_autotune_cache_path``), or
    None when its kernels take no FlashInfer tactics. FlashInfer autotune is
    pinned off; SAVE installs the file's tactics and keeps them in the rank's
    ``warmup_state.json``, LOAD installs them from there, so both run the same
    kernels without a tuning pass. The file need not exist (SAVE then warns
    and the archive carries no tactics)."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.capture_scope(phase, model, req_slots, kind, flashinfer_tactics)


def capture_one(shape_key: Any, forward_fn, *, pool: Any, stream: Any):
    """Replaces the body of ``FullCudaGraphBackend.capture_one`` (inside a
    :func:`capture_scope`) and returns ``(graph, output)`` for SGLang to
    store: SAVE captures the shape on ``stream`` into ``pool`` without warm-up
    forwards and archives it; LOAD restores the archived graph and runs no
    forward."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.capture_one(shape_key, forward_fn, pool=pool, stream=stream)
