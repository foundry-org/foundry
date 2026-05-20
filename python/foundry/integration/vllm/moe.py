# SPDX-License-Identifier: Apache-2.0
"""MoE / EP split-load helpers.

Ports ``collect_moe_quant_metadata``, ``inject_moe_quant_config``,
``reset_moe_quant_config``, ``reinit_moe_quant_config``, and the
``_split_load_model`` orchestrator from
``vllm-cge/vllm/compilation/graph_extension.py:127–266``
and ``vllm-cge/vllm/v1/worker/gpu_model_runner.py:3316–3419``.

Phase 2 functionality. Stays installed but inactive on dense models —
``_load_model_with_cge_overlap`` only calls ``_split_load_model`` when
``moe_quant_metadata`` is present in ``warmup_state.json``.

NOTE: this module's body executes at import time only — function calls
are deferred until a MoE checkpoint actually loads. So importing it on
a dense run is harmless even if some imports inside the functions
fail.
"""

from __future__ import annotations

from vllm.logger import init_logger
import time

import torch

from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
)
from foundry.integration.vllm.runtime import (
    preallocate_for_load_mode,
    set_moe_quant_metadata,
)

log = init_logger("vllm.foundry.moe")

# Set to True by split_load_model right before returning so that the very
# next call to ``prepare_communication_buffer_for_model`` (made by
# upstream ``GPUModelRunner.load_model`` post-load) is suppressed — we
# already prepared the comm buffers in split-load step 2 and a second
# call would re-init NVSHMEM.
_suppress_next_prepare_comm = False


def should_suppress_prepare_comm() -> bool:
    global _suppress_next_prepare_comm
    if _suppress_next_prepare_comm:
        _suppress_next_prepare_comm = False
        return True
    return False


def mark_prepare_comm_done() -> None:
    global _suppress_next_prepare_comm
    _suppress_next_prepare_comm = True


def collect_moe_quant_metadata(model: torch.nn.Module) -> dict:
    """Extract ``quant_dtype`` / ``block_shape`` / ``per_act_token_quant`` from
    the first ``FusedMoE`` layer's ``moe_quant_config``."""
    for module in model.modules():
        if module.__class__.__name__ in ("FusedMoE", "SharedFusedMoE"):
            qc = getattr(module.quant_method, "moe_quant_config", None)
            if qc is not None:
                metadata = {
                    "quant_dtype": str(qc.quant_dtype)
                    if qc.quant_dtype is not None
                    else None,
                    "block_shape": qc.block_shape,
                    "per_act_token_quant": qc.per_act_token_quant,
                }
                set_moe_quant_metadata(metadata)
                log.info("[CGE] Collected MoE quant metadata: %s", metadata)
                return metadata
    log.info("[CGE] No FusedMoE layers with quant metadata found")
    return {}


def inject_moe_quant_config(
    model: torch.nn.Module, metadata: dict
) -> None:
    """Pre-set a lightweight ``FusedMoEQuantConfig`` on every FusedMoE layer.

    This lets ``prepare_communication_buffer_for_model`` size NVSHMEM
    buffers correctly before weight loading, since
    ``maybe_make_prepare_finalize`` only needs ``quant_dtype`` /
    ``block_shape`` / ``per_act_token_quant`` (no actual weight tensors).
    """
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

    quant_dtype_str = metadata.get("quant_dtype")
    if quant_dtype_str and quant_dtype_str.startswith("torch."):
        quant_dtype = getattr(torch, quant_dtype_str.split(".", 1)[1])
    else:
        quant_dtype = None
    block_shape = metadata.get("block_shape")
    per_act_token_quant = metadata.get("per_act_token_quant", False)

    lightweight_config = FusedMoEQuantConfig.make(
        quant_dtype=quant_dtype,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant,
    )

    count = 0
    for module in model.modules():
        if module.__class__.__name__ in ("FusedMoE", "SharedFusedMoE"):
            module.quant_method.moe_quant_config = lightweight_config
            count += 1
    log.info(
        "[CGE] Injected lightweight quant config on %d FusedMoE layers "
        "(quant_dtype=%s, block_shape=%s, per_act_token=%s)",
        count,
        quant_dtype,
        block_shape,
        per_act_token_quant,
    )


def reset_moe_quant_config(model: torch.nn.Module) -> None:
    for module in model.modules():
        if module.__class__.__name__ in ("FusedMoE", "SharedFusedMoE"):
            module.quant_method.moe_quant_config = None


def reinit_moe_quant_config(model: torch.nn.Module) -> None:
    """Re-initialize full ``moe_quant_config`` (with scale tensors) post-load."""
    count = 0
    for module in model.modules():
        if module.__class__.__name__ in ("FusedMoE", "SharedFusedMoE"):
            qm = module.quant_method
            old_qm = getattr(qm, "old_quant_method", None)
            if old_qm is not None and qm.moe_quant_config is None:
                full_config = old_qm.get_fused_moe_quant_config(module)
                qm.moe_quant_config = full_config
                modular_kernel = getattr(qm, "fused_experts", None)
                if modular_kernel is not None:
                    gemm_impl = getattr(
                        modular_kernel, "fused_experts", None
                    )
                    if gemm_impl is not None and hasattr(
                        gemm_impl, "quant_config"
                    ):
                        gemm_impl.quant_config = full_config
                count += 1
            elif hasattr(module, "ensure_moe_quant_config_init"):
                module.ensure_moe_quant_config_init()
                count += 1
    if count > 0:
        log.info(
            "[CGE] Re-initialized full quant config on %d FusedMoE layers",
            count,
        )


def split_load_model(
    runner,
    model_loader,
    cge_mode,
    moe_quant_metadata: dict,
):
    """EP/MoE split-load: init structure, inject quant, prep comm buffers,
    preallocate, **start_graph_builds (audit-fix)**, load weights, reinit.

    Called from ``_load_model_with_cge_overlap`` when MoE quant metadata
    exists in ``warmup_state.json``.
    """
    from vllm.distributed import prepare_communication_buffer_for_model
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    # Local import to avoid circularity at module import.
    from foundry.integration.vllm.graph_ops import start_graph_builds

    device_config = runner.vllm_config.device_config
    load_config = runner.vllm_config.load_config
    load_device = (
        device_config.device
        if load_config.device is None
        else load_config.device
    )
    target_device = torch.device(load_device)

    with set_default_torch_dtype(runner.model_config.dtype):
        # Step 1: structure only, no weights
        t = time.perf_counter()
        log.info("[CGE] Split load: initializing model (no weights)")
        with target_device:
            model = initialize_model(
                vllm_config=runner.vllm_config,
                model_config=runner.model_config,
            )
        log.info(
            "[CGE TIMING] Split load step 1 (init model): %.3f s",
            time.perf_counter() - t,
        )

        # Step 2: lightweight quant + prep comm buffers (NVSHMEM init)
        t = time.perf_counter()
        inject_moe_quant_config(model, moe_quant_metadata)
        prepare_communication_buffer_for_model(model)
        reset_moe_quant_config(model)
        runner._comm_buffers_prepared = True
        log.info(
            "[CGE TIMING] Split load step 2 (inject quant + comm buffer): "
            "%.3f s",
            time.perf_counter() - t,
        )

        # Step 2.5: preallocate + start graph builds (LOAD only).
        # Must come AFTER NVSHMEM init (it does its own cuMemCreate that
        # would collide with preallocation).
        if cge_mode == CUDAGraphExtensionMode.LOAD:
            t = time.perf_counter()
            preallocate_for_load_mode()
            log.info(
                "[CGE TIMING] Split load step 2.5a (preallocate): %.3f s",
                time.perf_counter() - t,
            )
            # AUDIT FIX: kick off Phase 1 here so background build threads
            # overlap with the IO-bound load_weights step below.
            start_graph_builds()

        # Step 3: load weights
        t = time.perf_counter()
        log.info("[CGE] Split load: loading weights")
        model_loader.load_weights(model, runner.model_config)
        process_weights_after_loading(
            model, runner.model_config, target_device
        )
        log.info(
            "[CGE TIMING] Split load step 3 (load weights): %.3f s",
            time.perf_counter() - t,
        )

    # Step 4: reinit full quant config (with real scale tensors)
    t = time.perf_counter()
    reinit_moe_quant_config(model)
    log.info(
        "[CGE TIMING] Split load step 4 (reinit quant config): %.3f s",
        time.perf_counter() - t,
    )

    # Step 5 (SAVE only): collect metadata for next pass / LOAD
    if cge_mode == CUDAGraphExtensionMode.SAVE:
        collect_moe_quant_metadata(model)

    # Signal to the patched prepare_communication_buffer_for_model that
    # we already prepared comm buffers in step 2 — upstream's post-load
    # call (line 4834 of gpu_model_runner.py) must be suppressed to
    # avoid re-initializing NVSHMEM.
    mark_prepare_comm_done()
    return model.eval()
