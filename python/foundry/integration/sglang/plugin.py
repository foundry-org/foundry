# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Foundry as an SGLang plugin.

Activation: ``FOUNDRY_GRAPH_EXTENSION_CONFIG=<foundry toml>`` in the launcher's
environment (spawned processes inherit it). SGLang's ``load_plugins()`` runs
the ``foundry`` entry point (``foundry_sglang_plugin:load``) in the launcher,
before the server args are built, and at the top of every scheduler process.
Each run of :func:`activate` registers the resolution hooks below and installs
the runtime patches (``hooks.install``).

The data-parallel controller process does not call ``load_plugins()``. It
needs nothing from Foundry: the only patch it used to receive (LD_PRELOAD for
the scheduler spawn) is covered by the environment it inherits from the
launcher, whose ``Engine._launch_scheduler_processes`` sets it before the
controller is spawned.

Resolution hooks (``register_resolution_hook`` on whitelisted steps, so the
pins land at the same pipeline positions as the former in-tree step):

- ``apply_inkling_prefill_cuda_graph_default`` (wrapped, pins first): the
  step the in-tree ``handle_graph_extension`` ran right before. Decode is
  pinned to ``full``, prefill to ``disabled`` unless ``full`` / ``disabled`` was
  given, profiling and FlashInfer autotune off (SAVE and LOAD must allocate
  identically). Conflicting flags raise instead of being overridden.
- ``handle_cuda_graph_config`` (wrapped, validates after): the resolved
  config, which also covers ``--cuda-graph-config`` JSON (it outranks the
  per-phase flags).
- ``handle_other_validations`` (wrapped, validates after): features Foundry
  does not support, once speculative decoding, LoRA and elastic EP are
  resolved; the graph config is checked again after the later cascades.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_ENV = "FOUNDRY_GRAPH_EXTENSION_CONFIG"
# Declaration source recorded in the resolution stash.
RESOLUTION_SOURCE = "foundry_graph_extension"
_FLAG = f"Foundry graph persistence ({CONFIG_ENV})"
# Whitelisted resolution steps the plugin wraps (register_resolution_hooks);
# the preflight checks that the installed sglang still offers them.
RESOLUTION_HOOK_STEPS = (
    "apply_inkling_prefill_cuda_graph_default",
    "handle_cuda_graph_config",
    "handle_other_validations",
)
_ACTIVATED = False


def activate(cfg_path: str) -> None:
    """Plugin body; called by ``foundry_sglang_plugin.load`` when the env is set.

    ``load_plugins()`` logs and swallows exceptions from a plugin, which would
    leave SGLang running natively while the user asked for SAVE or LOAD, so a
    failure here exits the process (SystemExit is not an ``Exception``)."""
    global _ACTIVATED
    if _ACTIVATED:
        return
    try:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"{cfg_path!r} does not exist")
        register_resolution_hooks()
        from foundry.integration.sglang.hooks import install

        install(cfg_path)
    except Exception as exc:
        raise SystemExit(
            f"[Foundry] sglang plugin failed to activate ({CONFIG_ENV}={cfg_path}): {exc!r}"
        ) from exc
    _ACTIVATED = True
    # Logging is not configured yet in every process that loads plugins.
    print(
        f"[Foundry] sglang plugin active: pid={os.getpid()} config={cfg_path}",
        file=sys.stderr,
        flush=True,
    )


def register_resolution_hooks() -> None:
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    wrappers = (
        _pin_then_previous,
        _previous_then_validate_graph_config,
        _previous_then_reject_unsupported,
    )
    for name, wrapper in zip(RESOLUTION_HOOK_STEPS, wrappers, strict=True):
        register_resolution_hook(name)(wrapper)


# ---------------------------------------------------------------------------
# Resolution-hook wrappers: fn(server_args, previous)
# ---------------------------------------------------------------------------


def _pin_then_previous(server_args: Any, previous) -> None:
    pin_graph_fields(server_args)
    previous(server_args)


def _previous_then_validate_graph_config(server_args: Any, previous) -> None:
    previous(server_args)
    validate_resolved_graph_config(server_args)


def _previous_then_reject_unsupported(server_args: Any, previous) -> None:
    previous(server_args)
    reject_unsupported_features(server_args)
    validate_resolved_graph_config(server_args)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _backend():
    from sglang.srt.model_executor.cuda_graph_config import Backend

    return Backend


def pin_graph_fields(server_args: Any) -> None:
    """Former in-tree ``handle_graph_extension`` (fork 4f018fd052), hardened:
    decode ``full``; prefill ``full`` or ``disabled`` (default ``disabled``, not
    the platform default, which on CUDA is ``breakable``; leaving it unset
    would also let a model default such as Inkling's pick it); profiling and
    FlashInfer autotune off."""
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    Backend = _backend()
    cfg = resolving_view(server_args)

    in_tree = getattr(cfg, "foundry_graph_extension_config_path", None)
    if in_tree and os.path.abspath(in_tree) != os.path.abspath(os.environ.get(CONFIG_ENV, "")):
        raise ValueError(
            f"--foundry-graph-extension-config-path={in_tree!r} and {CONFIG_ENV} "
            "name different Foundry configs; set one of them"
        )
    disabled = getattr(cfg, "disable_cuda_graph", False)
    if disabled or getattr(cfg, "disable_decode_cuda_graph", False):
        raise ValueError(
            "--disable-cuda-graph / --disable-decode-cuda-graph cannot be combined with "
            f"{_FLAG}: "
            "Foundry saves and restores the decode CUDA graphs; unset "
            f"{CONFIG_ENV} to run without graphs"
        )
    decode = getattr(cfg, "cuda_graph_backend_decode", None)
    if decode not in (None, Backend.FULL):
        raise ValueError(
            f"--cuda-graph-backend-decode={decode!r} is not supported with {_FLAG}: "
            "Foundry persists full decode graphs only; use 'full' or leave it unset"
        )
    prefill = getattr(cfg, "cuda_graph_backend_prefill", None)
    if prefill not in (None, Backend.FULL, Backend.DISABLED):
        raise ValueError(
            f"--cuda-graph-backend-prefill={prefill!r} is not supported with {_FLAG}: "
            "Foundry persists full prefill graphs only; use 'full' or 'disabled'"
        )
    if getattr(cfg, "enable_profile_cuda_graph", False):
        raise ValueError(
            f"--enable-profile-cuda-graph is not supported with {_FLAG}: profiling "
            "adds allocations to the capture that LOAD does not reproduce"
        )
    fields = dict(
        cuda_graph_backend_decode=Backend.FULL,
        enable_profile_cuda_graph=False,
        disable_flashinfer_autotune=True,
    )
    if prefill is None:
        fields["cuda_graph_backend_prefill"] = Backend.DISABLED
    declare_resolution(server_args, RESOLUTION_SOURCE, **fields)


def validate_resolved_graph_config(server_args: Any) -> None:
    """The resolved ``cuda_graph_config`` (JSON > per-phase flags > legacy
    flags > defaults, then the compatibility cascades) must still be decode
    ``full`` and prefill ``full`` / ``disabled``."""
    from sglang.srt.arg_groups.overrides import resolving_view

    Backend = _backend()
    graph_cfg = resolving_view(server_args).cuda_graph_config
    if graph_cfg is None:
        return
    decode = graph_cfg.decode.backend
    prefill = graph_cfg.prefill.backend
    if decode != Backend.FULL:
        raise ValueError(
            f"cuda_graph_config[decode].backend={decode!r} is not supported with {_FLAG}: "
            "Foundry persists full decode graphs only (check --cuda-graph-config, "
            "and disaggregation roles that turn decode graphs off)"
        )
    if prefill not in (Backend.FULL, Backend.DISABLED):
        raise ValueError(
            f"cuda_graph_config[prefill].backend={prefill!r} is not supported with {_FLAG}: "
            "Foundry persists full prefill graphs only; use 'full' or 'disabled'"
        )


def _elastic_recapture_path_exists() -> bool:
    try:
        from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
    except ImportError:
        return False
    return hasattr(cuda_graph_setup, "recapture_elastic_cuda_graph")


def reject_unsupported_features(server_args: Any) -> None:
    """Features whose graphs Foundry cannot save or restore, rejected once
    they are resolved. Pipeline parallelism is allowed: the workspace rank
    includes the PP rank and non-last-rank outputs (PPProxyTensors) are
    persisted, but it has not been exercised on a GPU."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    unsupported = []
    spec = getattr(cfg, "speculative_algorithm", None)
    if spec and str(spec).upper() != "NONE":
        unsupported.append(
            f"speculative decoding (--speculative-algorithm={spec}): target-verify and "
            "draft graphs are not persisted"
        )
    if getattr(cfg, "enable_lora", False) or getattr(cfg, "lora_paths", None):
        unsupported.append("LoRA (--enable-lora / --lora-paths): not persisted or tested")
    if getattr(cfg, "enable_pdmux", False):
        unsupported.append("PD multiplexing (--enable-pdmux): multi-stream graphs")
    max_ep = getattr(cfg, "max_ep_size", None)
    if (
        getattr(cfg, "elastic_ep_backend", None) is not None
        and max_ep is not None
        and max_ep > cfg.tp_size
        and _elastic_recapture_path_exists()
    ):
        unsupported.append(
            f"elastic-EP CUDA-graph recapture (--elastic-ep-backend with --max-ep-size={max_ep} "
            f"> tp size {cfg.tp_size}): graphs recaptured after a scale event cannot be restored"
        )
    if os.environ.get("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING", "").lower() in ("1", "true"):
        unsupported.append(
            "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING: the KV pool is re-sized from post-capture "
            "free memory, which LOAD cannot reproduce"
        )
    if unsupported:
        raise ValueError(
            f"not supported with the Foundry graph extension ({CONFIG_ENV}): "
            + "; ".join(unsupported)
        )
