"""Bounded probes of a live vLLM model-forward CUDA graph.

Call only at a real replay boundary, after the runner has populated that
batch's metadata. Source graphs must have been captured with keep_graph=True.
This module does not install hooks, change drivers, or launch at import time.
"""

from __future__ import annotations

import ctypes as C
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


class UpdateInfo(C.Structure):
    _fields_ = [("result", C.c_int), ("errorNode", C.c_void_p),
                ("errorFromNode", C.c_void_p)]


class EdgeData(C.Structure):
    _fields_ = [("from_port", C.c_ubyte), ("to_port", C.c_ubyte),
                ("type", C.c_ubyte), ("reserved", C.c_ubyte * 5)]


class KernelParams(C.Structure):
    _fields_ = [("func", C.c_void_p),
                ("gridDimX", C.c_uint), ("gridDimY", C.c_uint), ("gridDimZ", C.c_uint),
                ("blockDimX", C.c_uint), ("blockDimY", C.c_uint), ("blockDimZ", C.c_uint),
                ("sharedMemBytes", C.c_uint), ("kernelParams", C.c_void_p),
                ("extra", C.c_void_p), ("kern", C.c_void_p), ("ctx", C.c_void_p)]


class DriverGraph:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.probe_arm = getattr(C.CDLL(None), "foundry_qmd_probe_arm", None)
        self.repair_gate = getattr(C.CDLL(None), "foundry_qmd_repair_gate", None)
        self.repair_finish = getattr(C.CDLL(None), "foundry_qmd_repair_finish", None)
        process = C.CDLL(None)
        version_fn = getattr(process, "foundry_qmd_repair_version", None)
        self.repair_version = 2 if self.repair_gate is not None else 0
        self.repair_writes_enabled = None
        self.repair_registered = set()
        self.repair_needs_upload = set()
        self.repair_invalid = set()
        self.repair_counter_schema = None
        self.repair_counter_error = None
        self.repair_get_counters_fn = None
        self.last_repair_counters = None
        self.repair_dump_inventory_fn = None
        self.repair_capture_expect_memcpy = None
        if version_fn is not None:
            version_fn.argtypes, version_fn.restype = [], C.c_int
            self.repair_version = version_fn()
        if self.repair_version == 3:
            for name, types in (("writes_enabled", []), ("capture_begin", []),
                                ("capture_expect_functions", [C.POINTER(C.c_void_p), C.c_uint, C.c_uint]),
                                ("capture_end", [C.c_void_p, C.c_uint]),
                                ("prepare", [C.c_void_p]), ("forget", [C.c_void_p])):
                fn = getattr(process, "foundry_qmd_repair_" + name)
                fn.argtypes, fn.restype = types, C.c_int
                setattr(self, "repair_" + name, fn)
            self.repair_writes_enabled = bool(self.repair_writes_enabled())
            memcpy_fn = getattr(process, "foundry_qmd_repair_capture_expect_memcpy", None)
            if memcpy_fn is not None:
                memcpy_fn.argtypes, memcpy_fn.restype = [C.c_uint], C.c_int
                self.repair_capture_expect_memcpy = memcpy_fn
            dump_fn = getattr(process, "foundry_qmd_repair_dump_inventory", None)
            if dump_fn is not None:
                dump_fn.argtypes, dump_fn.restype = [C.c_void_p, C.c_char_p], C.c_int
                self.repair_dump_inventory_fn = dump_fn
            schema_fn = getattr(process, "foundry_qmd_repair_counter_schema", None)
            counters_fn = getattr(process, "foundry_qmd_repair_get_counters", None)
            if schema_fn is not None and counters_fn is not None:
                try:
                    schema_fn.argtypes, schema_fn.restype = [], C.c_char_p
                    raw = schema_fn()
                    if not raw or len(raw) > 32768:
                        raise ValueError("Counter schema is missing or oversized")
                    schema = json.loads(raw.decode("utf-8"))
                    names = schema.get("names")
                    if (schema.get("version") not in (1, 2) or not isinstance(names, list)
                            or not 1 <= len(names) <= 256
                            or not all(isinstance(n, str) and n for n in names)
                            or len(set(names)) != len(names)):
                        raise ValueError("Unsupported or malformed counter schema")
                    counters_fn.argtypes = [C.POINTER(C.c_uint64), C.c_uint]
                    counters_fn.restype = C.c_int
                    self.repair_counter_schema, self.repair_get_counters_fn = schema, counters_fn
                except Exception as exc:
                    self.repair_counter_error = repr(exc)
        if self.repair_finish is not None:
            self.repair_finish.argtypes = []
            self.repair_finish.restype = C.c_int
        if self.repair_gate is not None:
            self.repair_gate.argtypes = [C.c_int]
            self.repair_gate.restype = None
        self.probe_phase = 0
        if self.probe_arm is not None:
            self.probe_arm.argtypes = [C.c_int]
            self.probe_arm.restype = None
        self._bind("cuGraphInstantiateWithFlags", [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong])
        self._bind("cuGraphExecUpdate_v2", [C.c_void_p, C.c_void_p, C.POINTER(UpdateInfo)])
        self._bind("cuGraphLaunch", [C.c_void_p, C.c_void_p])
        self._bind("cuGraphUpload", [C.c_void_p, C.c_void_p])
        self._bind("cuGraphExecDestroy", [C.c_void_p])
        self._bind("cuGraphClone", [C.POINTER(C.c_void_p), C.c_void_p])
        self._bind("cuGraphDestroy", [C.c_void_p])
        self._bind("cuGraphKernelNodeGetParams_v2", [C.c_void_p, C.POINTER(KernelParams)])
        self._bind("cuGraphKernelNodeSetParams_v2", [C.c_void_p, C.POINTER(KernelParams)])
        self._bind("cuGraphKernelNodeGetAttribute", [C.c_void_p, C.c_int, C.c_void_p])
        self._bind("cuGraphExecGetFlags", [C.c_void_p, C.POINTER(C.c_ulonglong)])
        self._bind("cuGraphGetNodes", [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)])
        self._bind("cuGraphNodeGetType", [C.c_void_p, C.POINTER(C.c_int)])
        self._bind("cuGraphGetEdges_v2", [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(EdgeData), C.POINTER(C.c_size_t)])
        self._bind("cuDriverGetVersion", [C.POINTER(C.c_int)])
        self._bind("cuGetErrorName", [C.c_int, C.POINTER(C.c_char_p)])

    def _bind(self, name, types):
        fn = getattr(self.lib, name)
        fn.argtypes = types
        fn.restype = C.c_int

    def check(self, rc, operation):
        if rc:
            name = C.c_char_p()
            self.lib.cuGetErrorName(rc, C.byref(name))
            raise RuntimeError(f"{operation}: {rc} {name.value!r}")

    def instantiate(self, graph, flags=0, register=False, expected_kernel_count=None, allow_nonkernel=False):
        out = C.c_void_p()
        start = time.perf_counter_ns()
        if not register:
            self.check(self.lib.cuGraphInstantiateWithFlags(C.byref(out), graph, flags), "cuGraphInstantiateWithFlags")
            return out.value, (time.perf_counter_ns() - start) / 1000
        if self.repair_version != 3 or not expected_kernel_count:
            raise ValueError("Exec inventory registration requires v3 and a positive expected kernel count")
        expect_memcpy = getattr(self, "repair_capture_expect_memcpy", None)
        if expect_memcpy is not None:
            public_functions, pdl_edges, memcpy_count = self._public_function_inventory(
                graph, int(expected_kernel_count), allow_nonkernel, include_memcpy_count=True)
        else:
            public_functions, pdl_edges = self._public_function_inventory(graph, int(expected_kernel_count), allow_nonkernel)
        function_array = (C.c_void_p * len(public_functions))(*public_functions)
        rc = self.repair_capture_begin()
        if rc:
            raise RuntimeError(f"v3 capture_begin failed: {rc}")
        try:
            status = self.repair_capture_expect_functions(function_array, len(public_functions), pdl_edges)
            if status:
                raise RuntimeError(f"v3 capture_expect_functions failed before instantiation: {status}")
            if expect_memcpy is not None:
                # Zero is an explicit census, distinct from an omitted ABI call.
                status = expect_memcpy(memcpy_count)
                if status:
                    raise RuntimeError(f"v3 capture_expect_memcpy failed before instantiation: {status}")
            self.check(self.lib.cuGraphInstantiateWithFlags(C.byref(out), graph, flags), "cuGraphInstantiateWithFlags")
        except BaseException:
            # This call resets capture state and intentionally returns -1.
            self.repair_capture_end(None, 0)
            raise
        status = self.repair_capture_end(out.value, int(expected_kernel_count))
        if status:
            # Inventory failure quarantines the owned handle. Do not invoke
            # further CUDA operations on an executable with rejected internals.
            self.repair_invalid.add(out.value)
            raise RuntimeError(f"v3 capture_end failed before any launch; candidate quarantined without CUDA cleanup: {status}")
        self.repair_registered.add(out.value)
        return out.value, (time.perf_counter_ns() - start) / 1000

    def _public_function_inventory(self, graph, expected_kernel_count, allow_nonkernel=False,
                                   include_memcpy_count=False):
        """Read exact public CUfunction multiplicities and PDL edge count.

        The v3 C inventory independently requires an exact public-function
        multiset, either alone or plus exactly PDL-edges + 1 nodes of a single
        nonpublic helper function. No public kernel is dropped and helper
        counts are never guessed in Python. The optional helper-role ABI also
        receives the actual public memcpy count (including zero); its separate
        C predicates identify helper roles. With allow_nonkernel=True the
        caller must first validate nonkernel operations, their lifetime and
        compatibility (memcpy is restricted by the upper layer to plain D2D).
        Skipping a public nonkernel node here does not expand C helper modes.
        """
        count = C.c_size_t()
        self.check(self.lib.cuGraphGetNodes(graph, None, C.byref(count)), "inventory cuGraphGetNodes(count)")
        if not 0 < count.value < 8192:
            raise ValueError("v3 public node inventory exceeds the bounded C inventory")
        capacity = count.value
        nodes = (C.c_void_p * capacity)()
        self.check(self.lib.cuGraphGetNodes(graph, nodes, C.byref(count)), "inventory cuGraphGetNodes")
        if count.value != capacity:
            raise RuntimeError("Graph node count changed during inventory enumeration")
        functions, memcpy_count = [], 0
        for node in nodes:
            kind = C.c_int()
            self.check(self.lib.cuGraphNodeGetType(node, C.byref(kind)), "inventory cuGraphNodeGetType")
            if kind.value != 0:
                if allow_nonkernel and kind.value in (1, 2, 5, 6, 7):
                    memcpy_count += kind.value == 1
                    continue
                raise ValueError("v3 registration currently requires all public nodes to be kernels")
            params = KernelParams()
            self.check(self.lib.cuGraphKernelNodeGetParams_v2(node, C.byref(params)), "inventory cuGraphKernelNodeGetParams_v2")
            if not params.func:
                raise ValueError("v3 function inventory requires a nonnull public CUfunction")
            functions.append(params.func)
        if len(functions) != expected_kernel_count:
            raise ValueError(f"Public kernel inventory count differs from census: {len(functions)} != {expected_kernel_count}")
        census = self.census(graph)
        pdl_edges = census["programmatic_edges"]
        if census["nodes"] != capacity or len(functions) + pdl_edges + 1 > 8192:
            raise ValueError("Public graph changed or internal helper count exceeds the bounded inventory")
        return (functions, pdl_edges, memcpy_count) if include_memcpy_count else (functions, pdl_edges)

    def flags(self, executable):
        flags = C.c_ulonglong()
        self.check(self.lib.cuGraphExecGetFlags(executable, C.byref(flags)), "cuGraphExecGetFlags")
        return flags.value

    def repair_counter_snapshot(self):
        """Optional read-only process counters; absent APIs preserve old runs."""
        fn = getattr(self, "repair_get_counters_fn", None)
        if fn is None:
            error = getattr(self, "repair_counter_error", None)
            return {"available": False, "error": error} if error else None
        schema = self.repair_counter_schema
        try:
            values = (C.c_uint64 * len(schema["names"]))()
            status = fn(values, len(values))
            if status:
                return {"available": False, "schema": schema, "api_status": status}
            return {"available": True, "schema": schema, "api_status": status,
                    "values": dict(zip(schema["names"], map(int, values)))}
        except Exception as exc:
            return {"available": False, "schema": schema, "error": repr(exc)}

    def dump_inventory(self, executable, path, allow_pending_upload=False):
        """Diagnostic-only host read; caller must synchronize prior GPU work.

        Explicit allow_pending_upload inspects finished host repair state before
        upload. Such a snapshot cannot prove that GPU QMD bytes were uploaded.
        """
        fn = getattr(self, "repair_dump_inventory_fn", None)
        if fn is None:
            raise RuntimeError("The explicitly requested diagnostic inventory API is unavailable")
        if executable not in self.repair_registered:
            raise RuntimeError("Diagnostic inventory requires a registered owned executable")
        pending = executable in self.repair_needs_upload
        if executable in self.repair_invalid or (pending and not allow_pending_upload):
            raise RuntimeError("Diagnostic dump requires a valid, explicitly uploaded executable")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        status = fn(executable, str(path).encode("utf-8"))
        if status:
            raise RuntimeError(f"Diagnostic inventory dump failed: {status}; partial file may remain: {path}")
        return {"path": str(path), "api_status": status,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "registered_exec": executable, "read_only": True,
                "host_descriptor_snapshot": True, "explicit_upload_pending": pending,
                "allow_pending_upload": allow_pending_upload, "gpu_full_qmd_upload_proven": False}

    def dump_cb0_parameters(self, executable, inventory, indices=None):
        """Opt-in bounded parameter bytes, never buffers pointed to by parameters.

        Call after upload and GPU synchronization, immediately after the owned
        exec inventory dump. No QMD byte-layout assumptions are used here.
        """
        if executable not in self.repair_registered:
            raise RuntimeError("CB0 diagnostic requires a registered owned executable")
        if executable in self.repair_invalid or executable in self.repair_needs_upload:
            raise RuntimeError("CB0 diagnostic requires a valid uploaded executable")
        if inventory.get("registered_exec") != executable:
            raise ValueError("CB0 inventory executable identity differs")
        path = Path(inventory["path"])
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("CB0 inventory exceeds 32 MiB input bound")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != inventory["sha256"]:
            raise ValueError("CB0 source inventory hash changed")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not records or records[0].get("event") != "inventory" or records[0].get("schema_version") != 1:
            raise ValueError("Unsupported CB0 inventory header")
        header, nodes = records[0], records[1:]
        if int(header["exec"], 0) != executable or not 0 < header["node_count"] <= 8192:
            raise ValueError("Invalid CB0 inventory executable or node count")
        if len(nodes) != header["node_count"] or any(n.get("event") != "node" for n in nodes):
            raise ValueError("Incomplete CB0 node inventory")
        if [n.get("index") for n in nodes] != list(range(len(nodes))):
            raise ValueError("CB0 inventory indices are not complete and unique")
        selected = list(range(len(nodes))) if indices is None else list(indices)
        if not selected or len(set(selected)) != len(selected) or any(
                type(i) is not int or not 0 <= i < len(nodes) for i in selected):
            raise ValueError("CB0 selected indices must be unique existing node indices")

        # Coalesce final QMD CB0 and ordinary parameter storage by base address.
        # This preserves the padded QMD tail (e.g. 4864 versus 4624 bytes), while
        # recording ordinary storage as a prefix when both addresses match.
        regions, selected_nodes = {}, []
        for index in selected:
            node = nodes[index]
            item = {key: node[key] for key in ("index", "node", "function", "helper")}
            item["regions"] = {}
            for label, prefix in (("final_cb0", "cb0"), ("ordinary_cb0", "ordinary_cb0")):
                address, size = int(node[prefix + "_address"], 0), node[prefix + "_size"]
                if type(size) is not int or not 0 <= size <= 65536:
                    raise ValueError(f"CB0 node {index} {label} exceeds 64 KiB per-region bound")
                if not 0 <= address < 2**64 or (size and (not address or address + size > 2**64)):
                    raise ValueError(f"CB0 node {index} {label} has invalid address")
                item["regions"][label] = {"address": hex(address), "size": size}
                if size:
                    regions[address] = max(regions.get(address, 0), size)
            selected_nodes.append(item)
        total = sum(regions.values())
        if total > 32 * 1024 * 1024:
            raise ValueError("CB0 diagnostic exceeds 32 MiB total copy bound")
        self._bind("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t])
        name_fn = getattr(self.lib, "cuFuncGetName", None)
        if name_fn is not None:
            name_fn.argtypes, name_fn.restype = [C.POINTER(C.c_char_p), C.c_void_p], C.c_int
        binary_path, manifest_path = path.with_suffix(".cb0.bin"), path.with_suffix(".cb0.json")
        manifest = {"schema_version": 1, "complete": False, "registered_exec": executable,
                    "inventory_path": str(path), "inventory_sha256": inventory["sha256"],
                    "binary_path": str(binary_path), "selected_indices": selected,
                    "bounds": {"per_region_bytes": 65536, "total_bytes": 32 * 1024 * 1024},
                    "planned_bytes": total, "copied_bytes": 0, "regions": [], "nodes": selected_nodes,
                    "read_only": True, "gpu_synchronized_before_dump": True,
                    "scope": "Final kernel parameter storage only; pointer arguments are retained as bytes and never dereferenced"}
        # Exclusive 0600 artifacts preserve failed diagnostics and cannot replace
        # earlier source evidence. Preflight all sizes before the first GPU read.
        manifest_fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(manifest_fd, "w") as output:
            try:
                binary_fd = os.open(binary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(binary_fd, "wb") as binary:
                    region_by_address = {}
                    for address, size in regions.items():
                        host = C.create_string_buffer(size)
                        self.check(self.lib.cuMemcpyDtoH_v2(host, address, size), "diagnostic cuMemcpyDtoH_v2(CB0)")
                        data = host.raw
                        region = {"address": hex(address), "size": size, "offset": binary.tell(),
                                  "sha256": hashlib.sha256(data).hexdigest()}
                        binary.write(data)
                        region_by_address[address] = (region, data)
                        manifest["regions"].append(region)
                        manifest["copied_bytes"] += size
                    for node in selected_nodes:
                        for region in node["regions"].values():
                            if region["size"]:
                                stored, data = region_by_address[int(region["address"], 0)]
                                region.update(offset=stored["offset"], sha256=hashlib.sha256(data[:region["size"]]).hexdigest())
                        if name_fn is not None:
                            name = C.c_char_p()
                            status = name_fn(C.byref(name), int(node["function"], 0))
                            node["function_name_api_status"] = status
                            node["function_name"] = name.value.decode("utf-8", "replace") if not status and name.value else None
                manifest["complete"] = True
                manifest["binary_sha256"] = hashlib.sha256(binary_path.read_bytes()).hexdigest()
            except BaseException as exc:
                manifest["error"] = repr(exc)
                raise
            finally:
                json.dump(manifest, output, indent=2)
                output.write("\n")
        return {"manifest_path": str(manifest_path), "binary_path": str(binary_path),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "binary_sha256": manifest["binary_sha256"], "copied_bytes": manifest["copied_bytes"],
                "selected_indices": selected, "read_only": True, "complete": True}

    def _repair_counter_delta(self, before, after, executable, graph):
        if before is None and after is None:
            return None
        result = {"before": before, "after": after, "delta": None, "available": False,
                  "exec": executable, "target_graph": graph, "update_phase": self.probe_phase,
                  "scope": "Process-global monotonic counters across this serialized prepare/update/finish window"}
        if before and after and before.get("available") and after.get("available"):
            if before["schema"] == after["schema"]:
                delta = {name: after["values"][name] - before["values"][name]
                         for name in before["schema"]["names"]}
                result.update(delta=delta, available=True, monotonic=all(value >= 0 for value in delta.values()))
            else:
                result["error"] = "Counter schema changed within an update"
        return result

    def update(self, executable, graph, allow_failure=False, repair_allowed=False, refresh=None):
        info = UpdateInfo()
        self.probe_phase += 1
        if executable in self.repair_invalid:
            raise RuntimeError("Refusing to update a candidate whose repair validation already failed")
        is_v3 = repair_allowed and self.repair_version == 3
        if refresh is not None and repair_allowed and not is_v3:
            raise ValueError("Diagnostic node refresh supports unpatched or v3 transactions only")
        if is_v3 and executable not in self.repair_registered:
            raise RuntimeError("v3 update requires registration from pristine instantiation")
        if self.probe_arm is not None:
            self.probe_arm(self.probe_phase)
        start = time.perf_counter_ns()
        counter_before = self.repair_counter_snapshot() if is_v3 else None
        self.last_repair_counters = None
        prepared = False
        finish_status = None
        refresh_result = None
        rc = None
        try:
            if is_v3:
                prepare_status = self.repair_prepare(executable)
                if prepare_status:
                    self.repair_invalid.add(executable)
                    raise RuntimeError(f"v3 prepare failed before CUDA update: {prepare_status}")
                prepared = True
            elif self.repair_gate is not None:
                self.repair_gate(int(repair_allowed))
            rc = self.lib.cuGraphExecUpdate_v2(executable, graph, C.byref(info))
            if refresh is not None and rc == 0 and info.result == 0:
                refresh_result = refresh(executable)
        except BaseException:
            if is_v3 or refresh is not None:
                self.repair_invalid.add(executable)
            raise
        finally:
            if self.repair_gate is not None:
                self.repair_gate(0)
            if self.probe_arm is not None:
                self.probe_arm(0)
            # v3 owns a transaction/inventory scope. It MUST be closed even on
            # an API error or Python exception; no failed candidate may launch.
            if prepared:
                try:
                    finish_status = self.repair_finish()
                except BaseException:
                    self.repair_invalid.add(executable)
                    raise
                if rc or rc is None or info.result or finish_status:
                    self.repair_invalid.add(executable)
            if refresh is not None and (rc or rc is None or info.result):
                self.repair_invalid.add(executable)
            if is_v3:
                self.last_repair_counters = self._repair_counter_delta(
                    counter_before, self.repair_counter_snapshot(), executable, graph)
                # Preserve diagnostics even when an API/guard error raises and
                # SGLang terminates workers without executing C destructors.
                if self.last_repair_counters is not None:
                    print("FOUNDRY_QMD_COUNTERS " + json.dumps(self.last_repair_counters, sort_keys=True),
                          file=sys.stderr, flush=True)
        if repair_allowed and self.repair_finish is not None and not is_v3:
            finish_status = self.repair_finish()
        if repair_allowed and (rc or info.result or finish_status):
            self.repair_invalid.add(executable)
            raise RuntimeError(f"Repair final verification failed before launch: api={rc}, result={info.result}, finish={finish_status}")
        if is_v3 or refresh is not None:
            self.repair_needs_upload.add(executable)
        us = (time.perf_counter_ns() - start) / 1000
        result = {"api_result": rc, "update_result": info.result,
                  "error_node": info.errorNode, "error_from_node": info.errorFromNode,
                  "cpu_us": us, "success": rc == 0 and info.result == 0}
        if self.probe_arm is not None:
            result["qmd_probe_phase"] = self.probe_phase
        if self.repair_gate is not None:
            result["prefetch_repair_allowed"] = repair_allowed
            result["prefetch_repair_finish_status"] = finish_status
            result["prefetch_repair_version"] = self.repair_version
            result["prefetch_repair_writes_enabled"] = self.repair_writes_enabled
            result["explicit_upload_required"] = is_v3 or refresh is not None
        if refresh is not None:
            result["diagnostic_node_refresh"] = refresh_result
            result["explicit_upload_required"] = True
        if self.last_repair_counters is not None:
            result["prefetch_repair_counters"] = self.last_repair_counters
        # A topology/function mismatch is an expected cross-batch outcome.
        if rc not in (0, 910) or (not allow_failure and not result["success"]):
            self.check(rc or 910, "cuGraphExecUpdate_v2")
        return result

    def launch(self, executable, stream):
        if executable in self.repair_invalid or executable in self.repair_needs_upload:
            raise RuntimeError("Refusing candidate launch: repair failed or required explicit upload has not completed")
        self.check(self.lib.cuGraphLaunch(executable, stream), "cuGraphLaunch")

    def upload(self, executable, stream):
        if executable in self.repair_invalid:
            raise RuntimeError("Refusing upload after repair validation failure")
        self.check(self.lib.cuGraphUpload(executable, stream), "cuGraphUpload")
        self.repair_needs_upload.discard(executable)

    def destroy(self, executable):
        if executable in self.repair_registered:
            status = self.repair_forget(executable)
            if status:
                raise RuntimeError(f"v3 forget failed before executable destruction: {status}")
            self.repair_registered.remove(executable)
        self.check(self.lib.cuGraphExecDestroy(executable), "cuGraphExecDestroy")
        self.repair_invalid.discard(executable)
        self.repair_needs_upload.discard(executable)

    def clone(self, graph):
        out = C.c_void_p()
        self.check(self.lib.cuGraphClone(C.byref(out), graph), "cuGraphClone")
        return out.value

    def geometry_variant(self, graph, limit=None):
        """Mutate a private source clone only. It must NEVER be launched.

        Doubling grid X preserves existing cluster divisibility. Cooperative
        kernels are excluded because their grids can have occupancy limits.
        Instantiation alone validates this source; the exec is updated back to
        the original valid source before any GPU launch.
        """
        count = C.c_size_t()
        self.check(self.lib.cuGraphGetNodes(graph, None, C.byref(count)), "cuGraphGetNodes(count)")
        nodes = (C.c_void_p * count.value)()
        self.check(self.lib.cuGraphGetNodes(graph, nodes, C.byref(count)), "cuGraphGetNodes")
        changed = []
        skipped = Counter()
        for index, node in enumerate(nodes):
            kind = C.c_int()
            self.check(self.lib.cuGraphNodeGetType(node, C.byref(kind)), "cuGraphNodeGetType")
            if kind.value != 0:
                continue
            attribute = (C.c_ulonglong * 8)()
            self.check(self.lib.cuGraphKernelNodeGetAttribute(node, 2, attribute), "cuGraphKernelNodeGetAttribute(cooperative)")
            if attribute[0] & 0xffffffff:
                skipped["cooperative"] += 1
                continue
            params = KernelParams()
            self.check(self.lib.cuGraphKernelNodeGetParams_v2(node, C.byref(params)), "cuGraphKernelNodeGetParams_v2")
            if not 0 < params.gridDimX < (1 << 29):
                skipped["grid_range"] += 1
                continue
            old_grid = [params.gridDimX, params.gridDimY, params.gridDimZ]
            params.gridDimX *= 2
            self.check(self.lib.cuGraphKernelNodeSetParams_v2(node, C.byref(params)), "cuGraphKernelNodeSetParams_v2")
            changed.append({"node_index": index, "original_grid": old_grid,
                            "never_launched_source_grid": [params.gridDimX, params.gridDimY, params.gridDimZ]})
            if limit is not None and len(changed) >= limit:
                break
        if not changed:
            raise RuntimeError("No eligible kernel geometry changed; refusing to call this a real update")
        return {"modified_nodes": len(changed), "changes": changed, "skipped": dict(skipped),
                "source_launched": False}

    def census(self, graph):
        count = C.c_size_t()
        self.check(self.lib.cuGraphGetNodes(graph, None, C.byref(count)), "cuGraphGetNodes(count)")
        nodes = (C.c_void_p * count.value)()
        self.check(self.lib.cuGraphGetNodes(graph, nodes, C.byref(count)), "cuGraphGetNodes")
        types = Counter()
        for node in nodes:
            kind = C.c_int()
            self.check(self.lib.cuGraphNodeGetType(node, C.byref(kind)), "cuGraphNodeGetType")
            types[str(kind.value)] += 1
        n_edges = C.c_size_t()
        self.check(self.lib.cuGraphGetEdges_v2(graph, None, None, None, C.byref(n_edges)), "cuGraphGetEdges_v2(count)")
        sources = (C.c_void_p * n_edges.value)()
        destinations = (C.c_void_p * n_edges.value)()
        edges = (EdgeData * n_edges.value)()
        # CUDA may reject nonnull zero-length buffers for an edgeless graph.
        # The count query is already complete; do not issue an empty read.
        if n_edges.value:
            capacity = n_edges.value
            self.check(self.lib.cuGraphGetEdges_v2(graph, sources, destinations, edges, C.byref(n_edges)), "cuGraphGetEdges_v2")
            if n_edges.value != capacity:
                raise RuntimeError("Graph edge count changed during census")
        kinds = Counter(f"{e.from_port}:{e.to_port}:{e.type}:" + bytes(e.reserved).hex() for e in edges)
        return {"nodes": count.value, "node_types": dict(types), "edges": n_edges.value,
                "programmatic_edges": sum(e.type == 1 for e in edges),
                "edge_data_histogram_from_to_type_reserved": dict(kinds)}

    def version(self):
        value = C.c_int()
        self.check(self.lib.cuDriverGetVersion(C.byref(value)), "cuDriverGetVersion")
        return value.value


def _tensors(value, torch, prefix="output"):
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _tensors(child, torch, f"{prefix}.{key}")]
    if isinstance(value, (tuple, list)):
        return [item for index, child in enumerate(value) for item in _tensors(child, torch, f"{prefix}.{index}")]
    if hasattr(value, "tensors") and isinstance(value.tensors, dict):
        return _tensors(value.tensors, torch, prefix)
    return []


def _summary(samples):
    return {"samples": samples, "median": statistics.median(samples),
            "minimum": min(samples), "maximum": max(samples)} if samples else None


def _description(entry):
    value = getattr(entry, "batch_descriptor", None)
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else repr(value)


def probe(entry, all_entries, torch, output_dir, config=None):
    """Probe a populated real decode batch, preserving its original executable.

    The outputs are full model hidden states, not sampled tokens or logits.
    Repeated launches write the same token's KV cache locations; native repeat
    validation explicitly checks whether this is stable for the current model.
    The overlap test repeats the same target graph: it validates API/GPU overlap,
    not a real scheduler or changing-batch metadata producer.
    """
    defaults = dict(blocks=5, launches_per_block=8, warmup=3,
                    cpu_update_reps=8, overlap_steps=8, overlap_blocks=3,
                    cross_candidates=4, rtol=1e-3, atol=1e-3)
    cfg = {**defaults, **(config or {})}
    for key in ("blocks", "launches_per_block", "cpu_update_reps", "overlap_steps"):
        if not 1 <= int(cfg[key]) <= 128:
            raise ValueError(f"{key} must be 1..128")
    if not 0 <= int(cfg["overlap_blocks"]) <= 128:
        raise ValueError("overlap_blocks must be 0..128; zero disables overlap tests")
    target_desc = _description(entry)
    tag = cfg.get("tag", "batch-" + hashlib.sha256(repr(target_desc).encode()).hexdigest()[:12])
    path = Path(output_dir) / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserving existing report: {path}")
    report = {"status": "started", "target": target_desc, "config": cfg,
              "scope": "Actual vLLM full model-forward graph; hidden-state outputs. No logits/sampling timing.",
              "limitations": ["Native CUDA captures, not a Foundry SAVE/LOAD archive reconstruction.",
                              "Replay/update overlap uses repeated target inputs and does not simulate scheduler work.",
                              "Process-local diagnostic/repair libraries, if any, are recorded separately from the unpatched baseline.",
                              "Output checks cover full returned hidden-state tensors, not all internal KV-cache storage."],
              "timing_units": "microseconds"}
    driver = DriverGraph()
    if isinstance(target_desc, dict):
        driver.probe_phase = int(target_desc.get("num_tokens", 0)) * 1000
    report["descriptor_probe_active"] = driver.probe_arm is not None
    report["prefetch_repair_library_loaded"] = driver.repair_gate is not None
    own_execs = []
    own_graphs = []
    stream = torch.cuda.current_stream()
    stream_handle = int(stream.cuda_stream)
    tensor_outputs = _tensors(entry.output, torch)
    if not tensor_outputs:
        raise ValueError(f"No output tensors in {type(entry.output)!r}")

    def fresh(graph, flags):
        handle, cpu_us = driver.instantiate(graph, flags)
        own_execs.append(handle)
        return handle, cpu_us

    def event():
        return torch.cuda.Event(enable_timing=True)

    def launch(handle):
        driver.launch(handle, stream_handle)

    def single(handle):
        start, end = event(), event()
        start.record(stream)
        cpu_start = time.perf_counter_ns()
        launch(handle)
        cpu_us = (time.perf_counter_ns() - cpu_start) / 1000
        end.record(stream)
        end.synchronize()
        return {"gpu_us": start.elapsed_time(end) * 1000, "launch_cpu_us": cpu_us}

    def validate(handle, references):
        # Qwen's returned hidden states are outputs, not token/position inputs.
        # Native repeat validation fails first if a future model aliases inputs.
        poisoned = []
        with torch.no_grad():
            for name, tensor in tensor_outputs:
                if tensor.is_floating_point() or tensor.is_complex():
                    tensor.fill_(float("nan"))
                    poisoned.append(name)
        launch(handle)
        stream.synchronize()
        values = []
        passed = True
        for (name, tensor), reference in zip(tensor_outputs, references):
            observed = tensor.detach().clone()
            both_finite = bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(observed).all().item())
            exact = bool(torch.equal(reference, observed))
            close = bool(torch.allclose(reference, observed, rtol=cfg["rtol"], atol=cfg["atol"], equal_nan=False))
            absolute = (reference.to(torch.float64) - observed.to(torch.float64)).abs()
            max_abs = float(absolute.max().item()) if absolute.numel() else 0.0
            max_abs = max_abs if math.isfinite(max_abs) else None
            mismatch = int(torch.count_nonzero(reference != observed).item())
            values.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "elements": tensor.numel(), "finite": both_finite,
                           "bitwise_equal": exact, "allclose": close,
                           "mismatched_elements": mismatch, "max_abs": max_abs})
            passed &= both_finite and close
        stream.synchronize()
        return {"pass": bool(passed), "poisoned_with_nan": poisoned, "tensors": values}

    try:
        # Input copies/metadata preparation must complete before direct launches.
        torch.cuda.synchronize()
        target = int(entry.cudagraph.raw_cuda_graph())
        native = int(entry.cudagraph.raw_cuda_graph_exec())
        if not target or not native:
            raise ValueError("Retained raw source graph and instantiated original executable are required")
        flags = driver.flags(native)
        report.update(driver_api_version=driver.version(), torch_version=str(torch.__version__),
                      torch_cuda_version=str(torch.version.cuda), device=torch.cuda.get_device_name(),
                      native_instantiate_flags=flags, graph=driver.census(target))
        report["output_tensors"] = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
                                     "device": str(t.device), "elements": t.numel()} for name, t in tensor_outputs]
        dedicated, dedicated_us = fresh(target, flags)
        updated, updated_us = fresh(target, flags)
        report["instantiate_cpu_us"] = {"dedicated": dedicated_us, "to_be_updated": updated_us}
        report["initial_same_source_update"] = driver.update(updated, target)
        report["same_source_update_control_note"] = "Identical source can take a no-op/equality path; geometry_update is the mandatory real-update test."
        # Measure the first launch separately before any steady-state warmup.
        report["first_launch"] = {"dedicated": single(dedicated), "same_source_update": single(updated)}
        launch(native)
        stream.synchronize()
        references = [tensor.detach().clone() for _, tensor in tensor_outputs]
        stream.synchronize()
        correctness = report["correctness"] = {}
        for name, handle in (("native_repeat", native), ("dedicated", dedicated), ("same_source_update", updated)):
            correctness[name] = validate(handle, references)
            if not correctness[name]["pass"]:
                raise RuntimeError(f"Output validation failed: {name}; refusing performance measurements")
        handles = {"native": native, "dedicated": dedicated, "same_source_update": updated}
        # Force an actual grid change without ever running an invalid model grid.
        # This does not rely on same-source update taking the slow update path.
        geometry_attempts = report["geometry_update_attempts"] = []
        for limit in (None, 1):
            geometry_report = {"mutation_limit": limit}
            geometry_attempts.append(geometry_report)
            variant = driver.clone(target)
            own_graphs.append(variant)
            geometry_report["variant"] = driver.geometry_variant(variant, limit=limit)
            try:
                geometry_exec, creation_us = fresh(variant, flags)
            except RuntimeError as exc:
                geometry_report["instantiate_error"] = repr(exc)
                continue
            geometry_report["instantiate_cpu_us"] = creation_us
            result = driver.update(geometry_exec, target, allow_failure=True, repair_allowed=True)
            geometry_report["restore_original_geometry_update"] = result
            if not result["success"]:
                continue
            geometry_report["first_launch_after_restoring_original_geometry"] = single(geometry_exec)
            geometry_report["correctness"] = validate(geometry_exec, references)
            if not geometry_report["correctness"]["pass"]:
                raise RuntimeError("Geometry update back to actual graph produced incorrect hidden states")
            correctness["geometry_update"] = geometry_report["correctness"]
            handles["geometry_update"] = geometry_exec
            uploaded_exec, _ = fresh(variant, flags)
            driver.upload(uploaded_exec, stream_handle)
            stream.synchronize()
            geometry_report["post_upload_restore"] = driver.update(uploaded_exec, target, repair_allowed=True)
            correctness["geometry_after_upload"] = validate(uploaded_exec, references)
            if not correctness["geometry_after_upload"]["pass"]:
                raise RuntimeError("Post-upload geometry update produced incorrect hidden states")
            handles["geometry_after_upload"] = uploaded_exec
            break
        if "geometry_update" not in handles:
            raise RuntimeError("Could not instantiate and restore a real geometry-changing update; no valid update performance claim")
        entries = list(all_entries.values()) if isinstance(all_entries, dict) else list(all_entries)
        candidates = [candidate for candidate in entries if candidate is not entry and getattr(candidate, "cudagraph", None) is not None]
        candidates.sort(key=lambda candidate: (abs(getattr(candidate.batch_descriptor, "num_tokens", 0) - target_desc.get("num_tokens", 0)), getattr(candidate.batch_descriptor, "num_tokens", 0)))
        cross_reports = report["cross_batch_updates"] = []
        for candidate in candidates[:int(cfg["cross_candidates"])]:
            source = int(candidate.cudagraph.raw_cuda_graph())
            candidate_report = {"source": _description(candidate), "source_graph": driver.census(source)}
            from graph_diff import diff_graphs
            candidate_report["parameter_diff"] = diff_graphs(driver, source, target)
            difference = candidate_report["parameter_diff"]
            fixed_family = bool(difference.get("aligned")
                                and not (set(difference.get("changed_field_node_counts", {})) - {"grid"})
                                and not difference.get("changed_attribute_node_counts")
                                and difference.get("kernel_nodes_compared") == report["graph"]["nodes"]
                                and difference.get("device_updatable_nodes", 0) == 0)
            candidate_report["public_fixed_family_check"] = fixed_family
            allow_family_repair = fixed_family and driver.repair_finish is not None
            cross_reports.append(candidate_report)
            cross_exec, creation_us = fresh(source, flags)
            candidate_report["instantiate_cpu_us"] = creation_us
            result = driver.update(cross_exec, target, allow_failure=True, repair_allowed=allow_family_repair)
            candidate_report["update"] = result
            if result["success"]:
                candidate_report["first_launch"] = single(cross_exec)
                candidate_report["correctness"] = validate(cross_exec, references)
                if not candidate_report["correctness"]["pass"]:
                    raise RuntimeError("Cross-batch update produced incorrect hidden states")
                if "cross_batch_update" not in handles:
                    handles["cross_batch_update"] = cross_exec
                    report["timed_cross_batch_source"] = candidate_report["source"]
                    uploaded_exec, _ = fresh(source, flags)
                    driver.upload(uploaded_exec, stream_handle)
                    stream.synchronize()
                    candidate_report["post_upload_update"] = driver.update(uploaded_exec, target, repair_allowed=allow_family_repair)
                    correctness["cross_batch_after_upload"] = validate(uploaded_exec, references)
                    if not correctness["cross_batch_after_upload"]["pass"]:
                        raise RuntimeError("Post-upload cross-batch update produced incorrect hidden states")
                    handles["cross_batch_after_upload"] = uploaded_exec
        samples = []
        for _ in range(int(cfg["cpu_update_reps"])):
            stream.synchronize()
            samples.append(driver.update(updated, target)["cpu_us"])
        report["same_source_update_cpu_us"] = _summary(samples)
        for handle in handles.values():
            for _ in range(int(cfg["warmup"])):
                launch(handle)
        stream.synchronize()
        timings = report["steady_replay"] = {name: [] for name in handles}
        names = list(handles)
        rng = random.Random(int(cfg.get("timing_order_seed", 20260926)))
        for block in range(int(cfg["blocks"])):
            order = list(names)
            rng.shuffle(order)
            for name in order:
                start, end = event(), event()
                stream.synchronize()
                wall_start = time.perf_counter_ns()
                start.record(stream)
                for _ in range(int(cfg["launches_per_block"])):
                    launch(handles[name])
                end.record(stream)
                end.synchronize()
                wall_us = (time.perf_counter_ns() - wall_start) / 1000
                gpu_us = start.elapsed_time(end) * 1000
                timings[name].append({"block": block, "order": order, "launches": cfg["launches_per_block"],
                                      "gpu_total_us": gpu_us, "gpu_per_launch_us": gpu_us / cfg["launches_per_block"],
                                      "wall_total_us": wall_us, "wall_per_launch_us": wall_us / cfg["launches_per_block"]})
        report["steady_summary_gpu_us"] = {name: _summary([sample["gpu_per_launch_us"] for sample in values]) for name, values in timings.items()}
        overlap = report["lookahead_one"] = {name: [] for name in
                     ("dedicated", "updated_no_host_update", "updated_overlap", "updated_serial")}
        steps = int(cfg["overlap_steps"])
        for block in range(int(cfg["overlap_blocks"])):
            order = list(overlap) if block % 2 == 0 else list(reversed(overlap))
            for name in order:
                handle = dedicated if name == "dedicated" else updated
                # Normalize updated executable before each timed sequence.
                if handle == updated:
                    driver.update(updated, target)
                launch(handle)
                stream.synchronize()
                start = event()
                done = [event() for _ in range(steps)]
                # Event allocation and lazy creation are outside the timed span.
                start.record(stream)
                for marker in done:
                    marker.record(stream)
                stream.synchronize()
                host_updates = []
                pending = []
                wall_start = time.perf_counter_ns()
                start.record(stream)
                launch(handle)
                done[0].record(stream)
                for step in range(1, steps):
                    if name == "updated_serial":
                        done[step - 1].synchronize()
                    elif step >= 2:
                        # At most previous step may remain pending when updating.
                        done[step - 2].synchronize()
                    if name in ("updated_overlap", "updated_serial"):
                        host_updates.append(driver.update(handle, target)["cpu_us"])
                        pending.append(not done[step - 1].query())
                    launch(handle)
                    done[step].record(stream)
                done[-1].synchronize()
                wall_us = (time.perf_counter_ns() - wall_start) / 1000
                gpu_us = start.elapsed_time(done[-1]) * 1000
                overlap[name].append({"block": block, "order": order, "steps": steps,
                                      "gpu_total_us": gpu_us, "gpu_per_step_us": gpu_us / steps,
                                      "wall_total_us": wall_us, "wall_per_step_us": wall_us / steps,
                                      "update_cpu_us": host_updates,
                                      "predecessor_pending_after_update": pending})
        report["lookahead_summary_wall_us"] = {name: _summary([sample["wall_per_step_us"] for sample in values]) for name, values in overlap.items()}
        report["correctness_after_timing"] = validate(updated, references)
        if not report["correctness_after_timing"]["pass"]:
            raise RuntimeError("Output validation failed after timing")
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        cleanup_errors = []
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            cleanup_errors.append(f"synchronize: {exc!r}")
        for handle in reversed(own_execs):
            try:
                driver.destroy(handle)
            except Exception as exc:
                cleanup_errors.append(f"destroy: {exc!r}")
        for graph in reversed(own_graphs):
            try:
                driver.check(driver.lib.cuGraphDestroy(graph), "cuGraphDestroy(private clone)")
            except Exception as exc:
                cleanup_errors.append(f"destroy graph: {exc!r}")
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return report
