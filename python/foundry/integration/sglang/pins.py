# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Server-args and environment pins and rejects of the Foundry SGLang
integration, called through ``api.py`` from SGLang's resolution steps:

- ``handle_cuda_graph_persistence`` (``api.pin_server_args``, before
  ``apply_inkling_prefill_cuda_graph_default``): decode is pinned to ``full``,
  prefill to ``disabled`` unless ``full`` / ``disabled`` was given, profiling
  and FlashInfer autotune off (the tuning pass allocates timing-dependently
  inside the recorded sequence; the tuned tactics are installed from SGLang's
  cache file / the archive instead, see
  ``graph_ops.bootstrap_flashinfer_tactics``). Conflicting flags are not
  overridden: they raise :class:`~foundry.integration.sglang.api.Unsupported`
  before anything is declared, and SGLang continues natively. The same step
  pins the flags that select state Foundry cannot replay (:data:`FIELD_PINS`,
  see "Pins" below).
- ``validate_cuda_graph_persistence`` (``api.validate_resolved_server_args``,
  after ``handle_other_validations``): features Foundry does not support
  (speculative decoding outside SUPPORTED_SPECULATIVE_ALGORITHMS), once speculative
  decoding, LoRA and elastic EP are resolved; the resolved graph config
  (which also covers ``--cuda-graph-config`` JSON) and the pinned flags are
  checked again after the later cascades, and ``NCCL_CUMEM_ENABLE`` is pinned
  once the MoE all-to-all backend is known.

Every conflict and reject here raises ``Unsupported`` (``step="server args"``,
``"environment"`` for the environment pins): a native run of the same record
works, so SGLang falls back to it with the reason.

Pins. State that cannot be made static across processes is not supported;
Foundry uses the cuMem-backed alternatives instead. A graph restored on LOAD
replays only what was recorded inside the capture region: a buffer
registration, IPC handle exchange or pool carve-out made outside it (at
``graph_capture()`` exit, at first use inside the forward, by a second graph
or allocator) does not exist in the LOAD process, and the restored kernels
read addresses nobody set up. Every such feature that a flag or an
environment variable can switch off is pinned here rather than left to the
recipe: each pin prints one ``[Foundry] pin:`` line, and a value the user
set explicitly to the contrary makes the engine fall back with the reason
(never a silent override). Features with no Foundry logic at all fall back
too (:func:`reject_unsupported_features`).

Environment pins (:data:`ENV_PINS`) are applied in ``api.activate``, which
runs in the launcher at the first resolution step (so the later resolution
steps that read them see the pinned values) and before any engine process is
spawned (spawned processes inherit ``os.environ``; values that sglang caches
at import time, such as ``SGLANG_JIT_DEEPGEMM_PRECOMPILE``, are only correct
this way). They are re-asserted by ``api.validate_resolved_server_args``, at
the top of every scheduler process (``activate`` runs there too) and at
each spawn site (``api.configure_subprocess``).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from foundry.integration.sglang.api import Unsupported

logger = logging.getLogger(__name__)

# Declaration source recorded in the resolution stash.
RESOLUTION_SOURCE = "foundry_graph_extension"
# Source of the unreplayable-state pins (FIELD_PINS), kept apart so that each
# declaration can be read back on its own.
PIN_RESOLUTION_SOURCE = "foundry_graph_extension_pins"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _backend():
    from sglang.srt.model_executor.cuda_graph_config import Backend

    return Backend


def _raise_unsupported(errors: list[str], step: str = "server args") -> None:
    if errors:
        raise Unsupported(step, "; ".join(errors))


def _graph_field_conflicts(server_args: Any) -> list[str]:
    """Explicit graph flags that contradict the graph pins."""
    from sglang.srt.arg_groups.overrides import resolving_view

    Backend = _backend()
    cfg = resolving_view(server_args)
    errors = []
    disabled = getattr(cfg, "disable_cuda_graph", False)
    if disabled or getattr(cfg, "disable_decode_cuda_graph", False):
        # No decode graphs: nothing to persist, and not a configuration error.
        raise Unsupported(
            "server args",
            "--disable-cuda-graph / --disable-decode-cuda-graph: no decode CUDA graphs to persist",
            quiet=True,
        )
    decode = getattr(cfg, "cuda_graph_backend_decode", None)
    if decode not in (None, Backend.FULL):
        errors.append(
            f"--cuda-graph-backend-decode={decode!r}: Foundry persists full decode graphs only"
        )
    prefill = getattr(cfg, "cuda_graph_backend_prefill", None)
    if prefill not in (None, Backend.FULL, Backend.DISABLED):
        errors.append(
            f"--cuda-graph-backend-prefill={prefill!r}: Foundry persists full prefill graphs "
            "only ('full' or 'disabled')"
        )
    if getattr(cfg, "enable_profile_cuda_graph", False):
        errors.append(
            "--enable-profile-cuda-graph: profiling adds allocations to the capture that LOAD "
            "does not reproduce"
        )
    return errors


def check_pinnable(server_args: Any) -> None:
    """Every conflict of :func:`pin_graph_fields` and
    :func:`pin_unreplayable_fields`, raised before either declares anything,
    so that a fallback resolves the record as without the flag."""
    _raise_unsupported(_graph_field_conflicts(server_args) + _unreplayable_plan(server_args)[0])


def pin_graph_fields(server_args: Any) -> None:
    """Decode ``full``; prefill ``full`` or ``disabled`` (default ``disabled``, not
    the platform default, which on CUDA is ``breakable``; leaving it unset
    would also let a model default such as Inkling's pick it); profiling and
    FlashInfer autotune off (tuned tactics come from the cache file / archive)."""
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    _raise_unsupported(_graph_field_conflicts(server_args))
    Backend = _backend()
    fields = dict(
        cuda_graph_backend_decode=Backend.FULL,
        enable_profile_cuda_graph=False,
        disable_flashinfer_autotune=True,
    )
    if getattr(resolving_view(server_args), "cuda_graph_backend_prefill", None) is None:
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
    errors = []
    if decode != Backend.FULL:
        errors.append(
            f"cuda_graph_config[decode].backend={decode!r}: Foundry persists full decode graphs "
            "only (check --cuda-graph-config, and disaggregation roles that turn decode graphs "
            "off)"
        )
    if prefill not in (Backend.FULL, Backend.DISABLED):
        errors.append(
            f"cuda_graph_config[prefill].backend={prefill!r}: Foundry persists full prefill "
            "graphs only ('full' or 'disabled')"
        )
    _raise_unsupported(errors)


def _elastic_recapture_path_exists() -> bool:
    try:
        from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
    except ImportError:
        return False
    return hasattr(cuda_graph_setup, "recapture_elastic_cuda_graph")


# Speculative decoding persisted by Foundry: the EAGLE worker's target-verify,
# draft-decode and draft-extend graphs (one decode-type capture scope each,
# hooks._decode_capture_scope). NEXTN resolves to EAGLE before this check.
# The other algorithms (STANDALONE, NGRAM, DFLASH, DSPARK, UNO, FROZEN_KV_MTP
# and plugin-registered ones) run other workers and runners and stay
# rejected.
# EAGLE-family (target-verify + draft-decode + draft-extend graph sets) and the
# verify-only algorithms whose target and draft run the plain decode runner in
# TARGET_VERIFY mode (kinds target_verify / draft_verify).
SUPPORTED_SPECULATIVE_ALGORITHMS = ("EAGLE", "EAGLE3", "NEXTN", "DFLASH", "NGRAM", "DSPARK")


def reject_unsupported_features(server_args: Any) -> None:
    """Features whose graphs Foundry cannot save or restore, rejected once
    they are resolved. Pipeline parallelism is allowed: the workspace rank
    includes the PP rank and non-last-rank outputs (PPProxyTensors) are
    persisted, but it has not been exercised on a GPU."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    unsupported = []
    spec = getattr(cfg, "speculative_algorithm", None)
    spec = str(spec).upper() if spec else "NONE"
    if spec != "NONE" and spec not in SUPPORTED_SPECULATIVE_ALGORITHMS:
        unsupported.append(
            f"speculative decoding (--speculative-algorithm={spec}): only "
            f"{'/'.join(SUPPORTED_SPECULATIVE_ALGORITHMS)} graphs are persisted"
        )
    if spec != "NONE":
        if getattr(cfg, "speculative_adaptive", False):
            unsupported.append(
                "adaptive speculative decoding (--speculative-adaptive): it captures one "
                "graph set per step count and switches between them"
            )
        if getattr(cfg, "enable_multi_layer_eagle", False):
            unsupported.append(
                "multi-layer EAGLE (--enable-multi-layer-eagle): its runners are not persisted"
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
    if (
        getattr(cfg, "moe_a2a_backend", None) == "nixl"
        and os.environ.get("FOUNDRY_DIAG_NIXL_OUTSIDE_REGION") != "1"
    ):
        unsupported.append(
            "NIXL EP (--moe-a2a-backend nixl): its buffers come from cuMemAlloc (no fabric "
            "handles), which UCX cuda_ipc cannot map inside the region, and outside it they "
            "are driver-placed, so the restored dispatch/combine params would be stale"
        )
    if os.environ.get("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING", "").lower() in ("1", "true"):
        unsupported.append(
            "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING: the KV pool is re-sized from post-capture "
            "free memory, which LOAD cannot reproduce"
        )
    _raise_unsupported(unsupported)


# ---------------------------------------------------------------------------
# Pins: configs that select state Foundry cannot replay (module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldPin:
    """A ServerArgs field Foundry declares to ``value``. ``default`` is
    sglang's default (fa090f7755): a raw value other than the pin and the
    default can only come from the user, and raises. ``strict`` pins are
    re-checked after the later resolution steps; a non-strict one may be
    turned off by sglang (the fallback is still replayable)."""

    field: str
    value: Any
    default: Any
    flag: str
    reason: str
    strict: bool = True


FIELD_PINS: tuple[FieldPin, ...] = (
    FieldPin(
        "disable_custom_all_reduce",
        True,
        False,
        "--disable-custom-all-reduce",
        "custom all-reduce (v1 and v2) registers the graphs' buffers over CUDA IPC when "
        "graph_capture() exits, outside the recorded region",
    ),
    FieldPin(
        "enable_torch_symm_mem",
        True,
        False,
        "--enable-torch-symm-mem",
        "the in-graph all-reduce uses torch symmetric memory, a cuMem buffer Foundry places "
        "at the same address on SAVE and LOAD",
        strict=False,
    ),
    FieldPin(
        "enable_symm_mem",
        False,
        False,
        "--enable-symm-mem",
        "NCCL symmetric-memory windows are registered at first use inside the forward and "
        "need NCCL's cuMem buffers",
    ),
    FieldPin(
        "enable_nccl_nvls",
        False,
        False,
        "--enable-nccl-nvls",
        "NVLS multicast buffers are registered at first use inside the forward",
    ),
    FieldPin(
        "enable_mscclpp",
        False,
        False,
        "--enable-mscclpp",
        "MSCCL++ registers its buffers at first use inside the forward",
    ),
    FieldPin(
        "enable_two_batch_overlap",
        False,
        False,
        "--enable-two-batch-overlap",
        "not validated with Foundry (children micro-batch graphs and DeepEP capture events); "
        "pinned off until a TBO row is validated",
    ),
    FieldPin(
        "enable_memory_saver",
        False,
        False,
        "--enable-memory-saver",
        "torch_memory_saver owns the pools (and, with SGLANG_MEMORY_SAVER_CUDA_GRAPH, the "
        "graph memory) outside Foundry's region",
    ),
)

# --dsv4-attn-backend: 'auto' (resolves to flashmla) or 'flashmla'; never trtllm.
DSV4_ATTN_FIELD = "dsv4_attn_backend"
DSV4_ATTN_REASON = (
    "the trtllm DeepSeek-V4 attention backend asserts at its first call, which falls "
    "inside the capture, and needs an eager bootstrap Foundry does not run"
)


@dataclass(frozen=True)
class EnvPin:
    name: str
    value: str
    reason: str


ENV_PINS: tuple[EnvPin, ...] = (
    # NCCL registers user buffers of graph-captured collectives (above a size
    # threshold) and the kernels then read peers' remote addresses from an
    # array NCCL fills on the host at capture time. A restored graph replays
    # those kernels without the registration, so the array holds garbage at
    # LOAD (illegal address in the DP-attention all-gather at bs>=4 with NCCL
    # 2.30). Keep every size on the unregistered path.
    EnvPin(
        "NCCL_GRAPH_REGISTER",
        "0",
        "NCCL registers the user buffers of graph-captured collectives on the host at capture "
        "time; a restored graph replays the kernels without the registration",
    ),
    EnvPin(
        "NCCL_LOCAL_REGISTER",
        "0",
        "same as NCCL_GRAPH_REGISTER for buffers registered outside a graph",
    ),
    EnvPin(
        "NCCL_NVLS_ENABLE",
        "0",
        "NVLS multicast buffers are mapped with driver flags Foundry's VMM region does not "
        "carry and are registered outside the recorded region",
    ),
    # NVSHMEM builds one NCCL communicator per team (5 on a 4-rank node, ~0.4 s
    # each, ~1.5 GB of region memory) while DeepEP creates its buffer; the
    # low-latency kernels never use them.
    EnvPin(
        "NVSHMEM_DISABLE_NCCL",
        "1",
        "NVSHMEM's per-team NCCL communicators cost ~2.4 s in the DeepEP bootstrap and are "
        "unused by the low-latency kernels",
    ),
    EnvPin(
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
        "0",
        "the DeepGEMM precompile sweep runs on the first rank inside the first capture: it "
        "synchronizes the device and allocates scratch only SAVE sees; kernels still JIT "
        "per shape",
    ),
    EnvPin(
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH",
        "0",
        "the memory-saver graph context owns the graph memory outside Foundry's region",
    ),
    EnvPin(
        "SGLANG_ENABLE_METADATA_GLUE_GRAPH",
        "0",
        "the attention-metadata prep is captured into a second graph Foundry does not save",
    ),
    EnvPin(
        "SGLANG_ENABLE_GRAPH_POOL_PRECARVE",
        "0",
        "only upstream capture_one runs the precarve (measured over its two eager warmups, "
        "minted at the first capture); Foundry's capture_one replaces it on SAVE and LOAD, so "
        "the flag would be a no-op and the graph pool would not match a native run's layout",
    ),
    EnvPin(
        "SGLANG_ENABLE_GRAPH_POOL_BORROW",
        "0",
        "eager allocations would borrow free graph-pool extents whose addresses the restored "
        "graphs reference",
    ),
    # Built before capture by _pre_initialize_dcp_push_combine, independent of
    # the --disable-custom-all-reduce pin.
    EnvPin(
        "SGLANG_ENABLE_DCP_PUSH_COMBINE",
        "0",
        "the DCP push-combine decode path builds its own CustomAllReduceV2, whose push "
        "workspace is registered over CUDA IPC outside the recorded region",
    ),
    # Read by torch.compile when it decorates (sglang applies it at import),
    # so it must be in the environment before the scheduler imports sglang;
    # the scheduler also sets torch._dynamo.config.disable
    # (hooks.before_parallel_init), which dynamo reads per frame.
    EnvPin(
        "TORCHDYNAMO_DISABLE",
        "1",
        "SAVE runs each shape's first forward inside the capture and torch >= 2.14's dynamo "
        "refuses to compile while a stream is capturing; sglang's compiled helpers run eagerly",
    ),
)

# NCCL_CUMEM_ENABLE=0 unless the MoE all-to-all is DeepEP v2, whose NCCL
# windows need cuMem (sglang defaults it to 1 for deepep_v2); pinned during
# resolution, once the backend is known, and at the spawn sites.
NCCL_CUMEM_PIN = EnvPin(
    "NCCL_CUMEM_ENABLE",
    "0",
    "NCCL's cuMem buffers (P2P, NVLS) are mapped with driver flags Foundry's VMM region "
    "does not carry; the plain allocator keeps them at deterministic offsets",
)
_CUMEM_EXEMPT_A2A = ("deepep_v2",)
# Set once the launcher applied the env pins: descendants log only changes.
_ENV_PINNED_MARK = "FOUNDRY_SGLANG_ENV_PINNED"
_CUMEM_EXEMPT_LOGGED = False
_KEPT_LOGGED: set[str] = set()


def _pin_log(message: str) -> None:
    # Logging is not configured yet in the launcher when resolution runs.
    print(f"[Foundry] pin: {message}", file=sys.stderr, flush=True)


def _contrary(pin: FieldPin, raw: Any) -> bool:
    return raw != pin.value and raw != pin.default


def no_in_graph_all_reduce(server_args: Any) -> bool:
    """Whether no captured forward contains an all-reduce, so the torch
    symmetric-memory pin (which only selects the all-reduce backend,
    sglang parallel_state.py:546, 1012-1105) has nothing to apply to:
    attention DP over the whole TP world (attention TP size 1, so the attention
    o_proj and the dense layers reduce over a group of one), experts reached by
    an all-to-all dispatcher over the whole TP world (no MoE TP reduce), the
    LM head on the attention-TP group (--enable-dp-lm-head: no vocab gather over
    the TP world), and no speculative draft model. A TorchSymmMemCommunicator is
    otherwise built at group creation and its rendezvous needs distinct device
    ordinals, which the one-visible-device-per-process launch does not have."""
    from foundry.integration.sglang.config import _attn_dp_size

    tp = int(getattr(server_args, "tp_size", 1) or 1)
    if tp <= 1:
        return False
    a2a = getattr(server_args, "moe_a2a_backend", None)
    return (
        _attn_dp_size(server_args) == tp
        and bool(getattr(server_args, "enable_dp_lm_head", False))
        and a2a not in (None, "none")
        and int(getattr(server_args, "ep_size", 1) or 1) == tp
        and getattr(server_args, "speculative_algorithm", None) is None
    )


def _unreplayable_plan(server_args: Any) -> tuple[list[str], dict[str, Any], list[str]]:
    """(conflicts, fields to declare, pin log lines) of :data:`FIELD_PINS` and
    the DeepSeek-V4 attention backend. Raw (user) values are read from
    ``server_args`` itself: declarations never change the raw inputs. Fields
    the installed sglang does not have are skipped."""
    errors = []
    fields: dict[str, Any] = {}
    lines = []
    for pin in FIELD_PINS:
        if not hasattr(server_args, pin.field):
            continue
        # DIAGNOSTIC ONLY (eep-archive-share study): two ranks per GPU / one GPU per
        # process, where torch symmetric memory's rendezvous rejects shared ordinals.
        if (
            pin.field == "enable_torch_symm_mem"
            and os.environ.get("FOUNDRY_DIAG_SKIP_SYMM_MEM_PIN") == "1"
        ):
            continue
        if pin.field == "enable_torch_symm_mem" and no_in_graph_all_reduce(server_args):
            lines.append(
                "enable_torch_symm_mem not pinned: no captured forward all-reduces (attention "
                "TP 1, expert all-to-all, --enable-dp-lm-head)"
            )
            continue
        raw = getattr(server_args, pin.field)
        if _contrary(pin, raw):
            errors.append(f"{pin.flag}={raw!r}: {pin.reason}")
            continue
        fields[pin.field] = pin.value
        if raw != pin.value:
            state = f"overrides sglang default {pin.default!r}"
        elif raw == pin.default:
            state = "sglang default"
        else:
            state = "as given"
        lines.append(f"{pin.field}={pin.value!r} ({state}): {pin.reason}")
    if hasattr(server_args, DSV4_ATTN_FIELD):
        raw = getattr(server_args, DSV4_ATTN_FIELD)
        if raw == "trtllm":
            errors.append(f"--dsv4-attn-backend=trtllm: {DSV4_ATTN_REASON}")
        else:
            value = raw or "auto"
            fields[DSV4_ATTN_FIELD] = value
            lines.append(f"{DSV4_ATTN_FIELD}={value!r} (not trtllm): {DSV4_ATTN_REASON}")
    return errors, fields, lines


def pin_unreplayable_fields(server_args: Any) -> None:
    """Declare :data:`FIELD_PINS` and the DeepSeek-V4 attention backend; a
    contrary user value raises ``Unsupported`` before anything is declared."""
    from sglang.srt.arg_groups.overrides import declare_resolution

    errors, fields, lines = _unreplayable_plan(server_args)
    _raise_unsupported(errors)
    for line in lines:
        _pin_log(line)
    if fields:
        declare_resolution(server_args, PIN_RESOLUTION_SOURCE, **fields)


def _last_source(server_args: Any, field: str) -> str:
    for source, declared in reversed(getattr(server_args, "_resolved_overrides", None) or ()):
        if field in declared:
            return source
    return "the raw input"


def validate_pinned_fields(server_args: Any) -> None:
    """After the later resolution steps (model overrides, platform fallbacks):
    a strict pin must still hold. A later sglang step that switched one back
    raises (naming it) rather than being overridden after other fields were
    derived from it."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    errors = []
    for pin in FIELD_PINS:
        if not hasattr(server_args, pin.field):
            continue
        value = getattr(cfg, pin.field)
        if value == pin.value:
            continue
        source = _last_source(server_args, pin.field)
        if pin.strict:
            errors.append(f"{pin.field}={value!r} (set by {source}): {pin.reason}")
        else:
            _pin_log(f"{pin.field}={value!r} (set by {source}); the collective falls back to NCCL")
    if hasattr(server_args, DSV4_ATTN_FIELD) and getattr(cfg, DSV4_ATTN_FIELD) == "trtllm":
        errors.append(
            f"{DSV4_ATTN_FIELD}='trtllm' (set by {_last_source(server_args, DSV4_ATTN_FIELD)}): "
            f"{DSV4_ATTN_REASON}"
        )
    _raise_unsupported(errors)


def _normalize_env(value: str) -> str:
    v = value.strip().lower()
    if v in ("0", "false", "no", "n", "off"):
        return "0"
    if v in ("1", "true", "yes", "y", "on"):
        return "1"
    return v


def _a2a_backend(server_args: Any) -> str | None:
    try:
        from sglang.srt.arg_groups.overrides import resolving_view

        value = getattr(resolving_view(server_args), "moe_a2a_backend", None)
    except ImportError:
        value = getattr(server_args, "moe_a2a_backend", None)
    return None if value is None else str(getattr(value, "value", value))


def apply_env_pins(server_args: Any = None) -> None:
    """Set :data:`ENV_PINS` in ``os.environ`` (and :data:`NCCL_CUMEM_PIN` when
    ``server_args`` is given, i.e. once the all-to-all backend is known). An
    unset variable is set; one set to the pinned value is kept; one set to
    anything else raises ``Unsupported`` before any variable is set.
    Idempotent: activation, resolution and every spawn site call it."""
    global _CUMEM_EXEMPT_LOGGED
    pins = list(ENV_PINS)
    if server_args is not None:
        a2a = _a2a_backend(server_args)
        if a2a not in _CUMEM_EXEMPT_A2A:
            pins.append(NCCL_CUMEM_PIN)
        elif not _CUMEM_EXEMPT_LOGGED:
            _CUMEM_EXEMPT_LOGGED = True
            _pin_log(
                f"{NCCL_CUMEM_PIN.name} left to sglang: --moe-a2a-backend {a2a} needs NCCL's "
                "cuMem windows"
            )
    first = os.environ.get(_ENV_PINNED_MARK) is None
    errors = [
        f"{pin.name}={os.environ[pin.name]!r} (Foundry needs {pin.value}): {pin.reason}"
        for pin in pins
        if pin.name in os.environ and _normalize_env(os.environ[pin.name]) != pin.value
    ]
    _raise_unsupported(errors, step="environment")
    for pin in pins:
        current = os.environ.get(pin.name)
        if current is None:
            os.environ[pin.name] = pin.value
            _pin_log(f"{pin.name}={pin.value} (was unset): {pin.reason}")
        elif (first or pin is NCCL_CUMEM_PIN) and pin.name not in _KEPT_LOGGED:
            _KEPT_LOGGED.add(pin.name)
            _pin_log(f"{pin.name}={pin.value} (already set): {pin.reason}")
    os.environ[_ENV_PINNED_MARK] = "1"
