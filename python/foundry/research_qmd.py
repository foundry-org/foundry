"""Conservative experiment: repair a persistent exec of actual LOADed graphs.

This is opt-in research infrastructure, not the scheduler overlap hot path.
The original LOAD template exec is never updated. A distinct exec is registered
while pristine from a clone of the real LOAD builder; member changes use
Foundry's C++ archive-param rewrite, then the pinned standalone guard and repair.
All GPU work is quiesced before mutations; keep the engine, allocations, modules,
communication objects and all LOAD graphs alive through the final synchronization.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from ._qmd_research.graph_probe import DriverGraph
from ._qmd_research.graph_diff import diff_graphs

_lock = threading.RLock()
_states = {}
_driver = None
_owners = []
_receipt_sequence = 0


def _fail_stop(exc):
    # No CUDA queries, cleanup, destructors or allocator work after a failed
    # transaction. A multi-rank caller must also have an external watchdog.
    try:
        os.write(2, (f"FOUNDRY_QMD_BRIDGE_FATAL {type(exc).__name__}: {exc}\n").encode())
    finally:
        os._exit(86)


def _driver_instance():
    global _driver
    if _driver is None:
        _driver = DriverGraph()
        mode = os.environ.get("FOUNDRY_QMD_RESEARCH_MODE", "repair")
        if mode not in ("repair", "unpatched"):
            raise ValueError("FOUNDRY_QMD_RESEARCH_MODE must be repair or unpatched")
        if mode == "repair" and not (
                _driver.repair_version == 3 and _driver.repair_writes_enabled
                and _driver.repair_capture_expect_memcpy is not None):
            raise RuntimeError("Strict helper-roles r2 repair ABI/write mode is required")
    return _driver


def _guard(diff):
    """Same full DAG/nonkernel gate as standalone, plus equal launch attrs.

    Foundry's coarse SAVE key is not permission to ignore changed attributes:
    execUpdate does not necessarily propagate them. Reject before update.
    """
    return bool(diff.get("aligned") and diff.get("nonkernel_compatible")
                and diff.get("kernel_nodes_compared", 0)
                    + diff.get("nonkernel_nodes_compared", 0)
                    == diff.get("source", {}).get("node_count")
                and diff.get("device_updatable_nodes", 0) == 0
                and not diff.get("changed_attribute_node_counts"))


def _receipt(record):
    global _receipt_sequence
    _receipt_sequence += 1
    record = {"sequence": _receipt_sequence, "pid": os.getpid(), **record}
    encoded = json.dumps(record, sort_keys=True)
    print("FOUNDRY_QMD_BRIDGE " + encoded, flush=True)
    directory = os.environ.get("FOUNDRY_QMD_RECEIPT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"bridge-{os.getpid()}-{_receipt_sequence:06d}.json"
        with target.open("x") as output:
            output.write(encoded + "\n")
    return record


def retain_owners(*owners):
    """Retain the engine/runner/graphs for the whole bounded experiment."""
    with _lock:
        _owners.extend(owners)


def state(graph):
    """CPU-only snapshot after prepare/replay; this does not change the graph."""
    with _lock:
        info = graph._research_info()
        if not info:
            return None
        value = _states.get(info["shared_graph"])
        if value is None:
            return {**info, "initialized": False}
        return {**info, "initialized": True,
                **{key: value[key] for key in (
                    "candidate_exec", "source_graph", "initial_source_graph",
                    "current_member_id", "template_member_id", "mode",
                    "last_update", "last_receipt", "census", "initial_flags")}}


def _prepare(graph):
    import torch
    info = graph._research_info()
    if not info:
        raise ValueError("Experimental repair only accepts actual Foundry LOAD shared graphs")
    if info["has_rng"]:
        raise ValueError("This deterministic comparison bridge does not yet accept RNG graphs")
    key = info["shared_graph"]
    # C++ LOAD stores the main_ctx passed explicitly to both binary and JSON
    # cuGraphAddMemcpyNode creation calls. This witness is not inferred from a
    # NULL public getter and does not authorize arbitrary external graphs.
    memcpy_creation_context = {
        "provenance": "foundry_explicit_cuGraphAddMemcpyNode_ctx",
        "context": info["context"], "builder_graph": key, "device": info["device"],
    }
    cached = _states.get(key)
    if cached is not None and (
            cached["builder_origin"] != memcpy_creation_context
            or cached["source_graph"] not in cached["sources"]
            or cached["initial_source_graph"] not in cached["sources"]):
        raise RuntimeError("Owned source clone does not descend from this LOAD builder/context/device")
    # Same-member replay has no prepare, update, upload or synchronization.
    if cached is not None and cached["current_member_id"] == info["graph_id"]:
        if info["current_params_id"] != info["graph_id"]:
            raise RuntimeError("External writer changed the shared builder behind the bridge")
        return cached
    driver = _driver_instance()
    mode = os.environ.get("FOUNDRY_QMD_RESEARCH_MODE", "repair")
    repair = mode == "repair"
    with torch.cuda.device(info["device"]):
        # This bounded experiment establishes replay parity, not overlap safety.
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        if cached is None:
            initial = driver.clone(key)
            census = driver.census(initial)
            eligibility = diff_graphs(driver, initial, initial, allow_dag=True,
                                      memcpy_creation_context={**memcpy_creation_context,
                                          "source_graph": initial, "target_graph": initial})
            if not _guard(eligibility):
                _receipt({"event": "registration_guard_rejected", "info": info,
                          "eligibility": eligibility})
                raise RuntimeError("Loaded pristine graph failed the strict public DAG/operation guard")
            flags = driver.flags(info["template_exec"])
            candidate, instantiate_us = driver.instantiate(
                initial, flags, register=repair,
                expected_kernel_count=census["node_types"].get("0", 0),
                allow_nonkernel=True)
            driver.upload(candidate, stream)
            torch.cuda.synchronize()
            cached = {"candidate_exec": candidate, "source_graph": initial,
                      "initial_source_graph": initial,
                      "current_member_id": info["current_params_id"],
                      "template_member_id": info["current_params_id"],
                      "mode": mode, "last_update": None, "census": census,
                      "initial_flags": flags, "owners": [graph], "sources": [initial],
                      "builder_origin": memcpy_creation_context}
            cached["last_receipt"] = _receipt({
                "event": "registered", "origin": "actual_foundry_archive_load",
                "separate_candidate_exec": True, "original_template_exec_untouched": True,
                "candidate_exec": candidate, "source_graph": initial, "info": info,
                "census": census, "mode": mode, "instantiate_us": instantiate_us,
                "initial_upload_synchronized": True,
                "counters": driver.repair_counter_snapshot()})
            _states[key] = cached
        if cached["mode"] != mode:
            raise RuntimeError("Cannot change repair mode inside a process")
        if cached["current_member_id"] != info["graph_id"]:
            previous_id = cached["current_member_id"]
            start = time.perf_counter_ns()
            graph._research_rewrite()  # actual archive node params, not a native capture
            target = driver.clone(key)
            cached["sources"].append(target)  # keep all source/target graph handles alive
            cached["owners"].append(graph)
            diff = diff_graphs(driver, cached["source_graph"], target, allow_dag=True,
                               memcpy_creation_context={**memcpy_creation_context,
                                   "source_graph": cached["source_graph"], "target_graph": target})
            if not _guard(diff):
                _receipt({"event": "update_guard_rejected", "info": info,
                          "from_member": previous_id, "to_member": info["graph_id"],
                          "diff": diff})
                raise RuntimeError("Loaded member transition failed strict full DAG/attribute/operation guard")
            update = driver.update(cached["candidate_exec"], target, repair_allowed=repair)
            if not update["success"]:
                raise RuntimeError(f"Loaded execUpdate failed: {update}")
            driver.upload(cached["candidate_exec"], stream)
            torch.cuda.synchronize()
            cached.update(source_graph=target, current_member_id=info["graph_id"],
                          last_update=update)
            cached["last_receipt"] = _receipt({
                "event": "updated", "from_member": previous_id,
                "to_member": info["graph_id"], "candidate_exec": cached["candidate_exec"],
                "source_graph": target, "shared_graph": key, "mode": mode,
                "actual_cpp_archive_rewrite": True, "diff": diff, "update": update,
                "post_update_upload_synchronized": True,
                "transaction_wall_us": (time.perf_counter_ns() - start) / 1000})
        return cached


def prepare(graph):
    """Register/update the true LOAD candidate without launching it."""
    with _lock:
        try:
            _prepare(graph)
            return state(graph)
        except BaseException as exc:
            _fail_stop(exc)


def replay(graph):
    """Called by the opt-in C++ binding on the real Foundry replay path."""
    with _lock:
        try:
            current = _prepare(graph)
            graph._research_replay_exec(current["candidate_exec"])
        except BaseException as exc:
            _fail_stop(exc)
