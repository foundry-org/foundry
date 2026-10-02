# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Stable API for SGLang's in-tree Foundry adapter (dependency route).

SGLang declares Foundry as an optional dependency and calls these functions
from its own call sites, through ``sglang.srt.utils.foundry_adapter``, when
``--cuda-graph-persistence {save,load}`` is given. Nothing here patches SGLang
and nothing reads ``FOUNDRY_GRAPH_EXTENSION_CONFIG`` (that variable belongs to
the general-plugin route, ``foundry_sglang_plugin``, which refuses to run on
an SGLang that has the adapter).

The implementation is shared with the plugin route: the resolution pins and
rejects live in :mod:`.plugin`, the call-site functions in :mod:`.hooks`, so
both routes run the same code at the same sequence points.

Compatibility: SGLang checks :data:`INTEGRATION_API_VERSION` at activation.
The major number changes when a function below is removed or changes its
signature or its call-site contract; the minor number when one is added.

Call sites (SGLang side, in process order):

==========================================  ===================================
SGLang site                                 function
==========================================  ===================================
resolution, before the prefill default      :func:`pin_server_args`
resolution, after ``handle_cuda_graph_config``  :func:`validate_graph_config`
resolution, after ``handle_other_validations``  :func:`validate_resolved_server_args`
``_set_envs_and_config`` (launcher)         :func:`apply_env_pins`
scheduler / DP-controller spawn             :func:`configure_subprocess`
``bootstrap.init_parallel_runtime``         :func:`before_parallel_init`,
                                            :func:`after_parallel_init`
``ModelRunner.init_torch_distributed``      :func:`after_runner_distributed_init`
``_resolve_memory_pool_config``             :func:`begin_memory_pool_resolution`,
                                            :func:`end_memory_pool_resolution`
``ModelRunner.alloc_memory_pool``           :func:`before_alloc_memory_pool`,
                                            :func:`after_alloc_memory_pool`
decode / prefill runner ``capture``         :func:`capture_scope`
``FullCudaGraphBackend.capture_one``        :func:`capture_one`
``_resolve_shared_read_ends``               :func:`shared_read_ends_override`
==========================================  ===================================

Every function except :func:`activate` and :func:`is_active` requires an
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

INTEGRATION_API_VERSION = (1, 0)
MODES = ("save", "load")
FLAG = "--cuda-graph-persistence"

_active: tuple[str, str | None] | None = None


def is_active() -> bool:
    return _active is not None


def mode() -> str | None:
    return None if _active is None else _active[0]


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

    from foundry.integration.sglang import hooks, plugin
    from foundry.integration.sglang.config import (
        get_graph_extension_mode,
        get_workspace_root,
        load_graph_extension_config,
    )

    if hooks._INSTALLED or plugin._ACTIVATED:
        raise RuntimeError(
            "[Foundry] the general plugin already patched this process "
            f"({plugin.CONFIG_ENV} is set); use either {FLAG} or the plugin, not both"
        )
    if config_path is not None and not os.path.isfile(config_path):
        raise FileNotFoundError(f"{FLAG}-config {config_path!r} does not exist")

    t0_ns = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    load_graph_extension_config(config_path, mode=mode)
    plugin.use_flag_labels(FLAG)
    plugin.apply_env_pins()
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
    replay (``plugin.FIELD_PINS``); conflicting user values raise."""
    _require_active()
    from foundry.integration.sglang import plugin

    plugin.pin_graph_fields(server_args)
    plugin.pin_unreplayable_fields(server_args)


def validate_graph_config(server_args: Any) -> None:
    """After ``handle_cuda_graph_config``: the resolved ``cuda_graph_config``
    (which also covers ``--cuda-graph-config`` JSON) must be decode ``full``
    and prefill ``full`` / ``disabled``."""
    _require_active()
    from foundry.integration.sglang import plugin

    plugin.validate_resolved_graph_config(server_args)


def validate_resolved_server_args(server_args: Any) -> None:
    """After ``handle_other_validations``: reject what Foundry cannot persist
    (speculative decoding, LoRA, PD multiplexing, elastic-EP recapture,
    post-capture KV sizing), re-check the graph config and the strict pins
    after the later cascades, and pin the environment including
    ``NCCL_CUMEM_ENABLE`` (left alone for ``--moe-a2a-backend deepep_v2``)."""
    _require_active()
    from foundry.integration.sglang import plugin

    plugin.reject_unsupported_features(server_args)
    plugin.validate_resolved_graph_config(server_args)
    plugin.validate_pinned_fields(server_args)
    plugin.apply_env_pins(server_args)


def apply_env_pins(server_args: Any = None) -> None:
    """Set ``plugin.ENV_PINS`` (and, with ``server_args``, ``NCCL_CUMEM_ENABLE``)
    in ``os.environ``; a contrary user value raises. Idempotent."""
    _require_active()
    from foundry.integration.sglang import plugin

    plugin.apply_env_pins(server_args)


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
    idempotent, as on the plugin route)."""
    _require_active()
    from foundry.integration.sglang import hooks, plugin
    from foundry.integration.sglang import runtime as rt

    if server_args is not None:
        hooks._check_resolved_graph_config(server_args)
    plugin.apply_env_pins(server_args)
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


def after_parallel_init() -> None:
    """End of ``bootstrap.init_parallel_runtime`` (allocation-offset log)."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.after_parallel_init()


def after_runner_distributed_init(model_runner: Any) -> None:
    """End of ``ModelRunner.init_torch_distributed`` (target runner only): move
    the allocation cursor to the scratch boundary so weight loading starts at
    the same offset on SAVE and LOAD."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.after_runner_distributed_init(model_runner)


def begin_memory_pool_resolution():
    """Head of ``KVCacheConfigurator._resolve_memory_pool_config``. LOAD returns
    the saved ``MemoryPoolConfig`` (the caller returns it and skips memory
    profiling) after replaying SAVE's context overrides; SAVE returns None."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.begin_memory_pool_resolution()


def end_memory_pool_resolution() -> None:
    """End of ``_resolve_memory_pool_config`` (SAVE keeps the overrides the
    resolver issued)."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.end_memory_pool_resolution()


def before_alloc_memory_pool(model_runner: Any) -> None:
    """Head of ``ModelRunner.alloc_memory_pool``."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.before_alloc_memory_pool(model_runner)


def after_alloc_memory_pool(model_runner: Any) -> None:
    """End of ``ModelRunner.alloc_memory_pool`` (SAVE writes the warmup state)."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.after_alloc_memory_pool(model_runner)


def capture_scope(runner: Any):
    """Context manager around the body of ``DecodeCudaGraphRunner.capture`` or
    ``PrefillCudaGraphRunner.capture``. SGLang's capture loop runs inside on
    both modes; the scope adds the pre-capture bootstraps and the layout start
    (first runner), LOAD's preallocation and restore bookkeeping, and SAVE's
    manifest, fatbin pack and region layout."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.capture_scope(runner)


def capture_one(backend: Any, shape_key: Any, forward_fn) -> None:
    """Replaces the body of ``FullCudaGraphBackend.capture_one``: SAVE captures
    the shape without warm-up forwards and archives it; LOAD restores the
    archived graph for the shape."""
    _require_active()
    from foundry.integration.sglang import hooks

    hooks.capture_one(backend, shape_key, forward_fn)


def shared_read_ends_override(runner: Any, attn_backend: Any, forward_mode: Any):
    """``DecodeCudaGraphRunner._resolve_shared_read_ends``: LOAD fences the
    scheduler's shared-buffer writes after replay (POST_REPLAY) where upstream
    would fall back to PRE_REPLAY. None keeps upstream's answer."""
    _require_active()
    from foundry.integration.sglang import hooks

    return hooks.shared_read_ends_override(runner, attn_backend, forward_mode)
