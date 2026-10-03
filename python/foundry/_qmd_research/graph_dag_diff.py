"""Read-only, conservative typed-DAG matching and nonkernel launch guards.

No array-order inference: correspondence must follow unique forward/backward
structural labels and independently preserve every edge and its eight bytes.
Symmetric ambiguous graphs are reported unsupported, not arbitrarily paired.
"""
import ctypes as C
import hashlib
from collections import Counter, defaultdict, deque

from .graph_probe import EdgeData


class MemsetParams(C.Structure):
    _fields_ = [("dst", C.c_uint64), ("pitch", C.c_size_t),
                ("value", C.c_uint), ("elementSize", C.c_uint),
                ("width", C.c_size_t), ("height", C.c_size_t)]


class Memcpy3D(C.Structure):
    """Public CUDA_MEMCPY3D_v2 ABI; no driver-private structure reads."""
    _fields_ = [
        ("srcXInBytes", C.c_size_t), ("srcY", C.c_size_t),
        ("srcZ", C.c_size_t), ("srcLOD", C.c_size_t),
        ("srcMemoryType", C.c_int), ("srcHost", C.c_void_p),
        ("srcDevice", C.c_uint64), ("srcArray", C.c_void_p),
        ("reserved0", C.c_void_p), ("srcPitch", C.c_size_t),
        ("srcHeight", C.c_size_t),
        ("dstXInBytes", C.c_size_t), ("dstY", C.c_size_t),
        ("dstZ", C.c_size_t), ("dstLOD", C.c_size_t),
        ("dstMemoryType", C.c_int), ("dstHost", C.c_void_p),
        ("dstDevice", C.c_uint64), ("dstArray", C.c_void_p),
        ("reserved1", C.c_void_p), ("dstPitch", C.c_size_t),
        ("dstHeight", C.c_size_t),
        ("WidthInBytes", C.c_size_t), ("Height", C.c_size_t),
        ("Depth", C.c_size_t),
    ]


class MemcpyNodeParams(C.Structure):
    _fields_ = [("flags", C.c_int), ("reserved", C.c_int),
                ("copyCtx", C.c_void_p), ("copyParams", Memcpy3D)]


class GraphNodeUnion(C.Union):
    _fields_ = [("reserved1", C.c_longlong * 29), ("memcpy", MemcpyNodeParams)]


class GraphNodeParams(C.Structure):
    _anonymous_ = ("params",)
    _fields_ = [("type", C.c_int), ("reserved0", C.c_int * 3),
                ("params", GraphNodeUnion), ("reserved2", C.c_longlong)]


class MemLocation(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class MemAllocationFlags(C.Structure):
    _fields_ = [("compressionType", C.c_ubyte), ("gpuDirectRDMACapable", C.c_ubyte),
                ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]


class MemAllocationProp(C.Structure):
    _fields_ = [("type", C.c_int), ("requestedHandleTypes", C.c_int),
                ("location", MemLocation), ("win32HandleMetaData", C.c_void_p),
                ("allocFlags", MemAllocationFlags)]


MEMCPY_CALLER_OBLIGATIONS = [
    "Retain both source/target raw graphs, graph pools, source/destination allocations, "
    "and communication-buffer owners until all executable launches using them finish.",
    "Do not free, recycle, unmap/remap, change access permissions, or replace backing "
    "allocations while the compared graphs/executables may use these pointers.",
    "Serialize access to each graph/exec and wait for all prior uses of the candidate "
    "exec and shared communication buffers before this experimental private repair; "
    "perform required upload before replay. This guard does not authorize overlapping repair.",
    "Check cuGraphExecUpdate's API result and update-result enum; this conservative "
    "preflight cannot guarantee the driver accepts every otherwise compatible update.",
]


def _struct_values(value):
    return {field: getattr(value, field) for field, _ in value._fields_}


def _vmm_parameters(driver, address, width, device):
    """Prove a NULL-context operand is a mapped device VMM allocation.

    Retain is paired with release in finally, including property-query failures.
    This temporarily increments one handle reference; no mapping/access/allocation
    is changed. The returned snapshot is not a retained application lifetime.
    """
    result = {"retain_api_result": None, "properties_api_result": None,
              "release_api_result": None, "access": []}
    if C.sizeof(MemLocation) != 8 or C.sizeof(MemAllocationProp) != 32:
        return {**result, "inspection_error": "Unsupported VMM property ABI"}
    try:
        for name, arguments in (
            ("cuMemRetainAllocationHandle", [C.POINTER(C.c_uint64), C.c_void_p]),
            ("cuMemGetAllocationPropertiesFromHandle", [C.POINTER(MemAllocationProp), C.c_uint64]),
            ("cuMemRelease", [C.c_uint64]),
            ("cuMemGetAccess", [C.POINTER(C.c_ulonglong), C.POINTER(MemLocation), C.c_uint64]),
        ):
            driver._bind(name, arguments)
        handle = C.c_uint64()
        rc = driver.lib.cuMemRetainAllocationHandle(C.byref(handle), address)
        result.update(retain_api_result=int(rc), allocation_handle=handle.value)
        if rc:
            return result
        try:
            prop = MemAllocationProp()
            rc = driver.lib.cuMemGetAllocationPropertiesFromHandle(C.byref(prop), handle.value)
            result.update(properties_api_result=int(rc), properties={
                "type": prop.type, "requested_handle_types": prop.requestedHandleTypes,
                "location_type": prop.location.type, "location_id": prop.location.id,
                "win32_metadata": prop.win32HandleMetaData,
                "compression_type": prop.allocFlags.compressionType,
                "gpu_direct_rdma_capable": prop.allocFlags.gpuDirectRDMACapable,
                "usage": prop.allocFlags.usage, "reserved": list(prop.allocFlags.reserved),
            })
            location = MemLocation(1, device)
            for pointer in dict.fromkeys((address, address + width - 1)):
                if not 0 < pointer < 2**64:
                    continue
                flags = C.c_ulonglong()
                rc = driver.lib.cuMemGetAccess(C.byref(flags), C.byref(location), pointer)
                result["access"].append({"address": pointer, "api_result": int(rc), "flags": flags.value})
        finally:
            result["release_api_result"] = int(driver.lib.cuMemRelease(handle.value))
    except Exception as exc:
        result["inspection_error"] = repr(exc)
    return result


def _memcpy_parameters(driver, node):
    """Inspect public parameters and pointer metadata, without changing state.

    RANGE_START_ADDR/SIZE describe reserved VA on VMM; they are diagnostics,
    never bounds proof. MAPPING_BASE_ADDR/SIZE or cuMemGetAddressRange supply
    actual mapped bounds. Neither query proves application object lifetime.
    """
    if (C.sizeof(C.c_void_p), C.sizeof(Memcpy3D), C.sizeof(MemcpyNodeParams),
            C.sizeof(GraphNodeParams)) != (8, 200, 216, 256):
        return {"inspection_error": "Unsupported public CUDA memcpy ABI layout"}
    bindings = (
        ("cuGraphMemcpyNodeGetParams", [C.c_void_p, C.POINTER(Memcpy3D)]),
        ("cuGraphNodeGetParams", [C.c_void_p, C.POINTER(GraphNodeParams)]),
        ("cuCtxGetCurrent", [C.POINTER(C.c_void_p)]),
        ("cuCtxGetDevice", [C.POINTER(C.c_int)]),
    )
    try:
        for name, arguments in bindings:
            driver._bind(name, arguments)
        p, generic = Memcpy3D(), GraphNodeParams()
        legacy_rc = driver.lib.cuGraphMemcpyNodeGetParams(node, C.byref(p))
        generic_rc = driver.lib.cuGraphNodeGetParams(node, C.byref(generic))
        context, device = C.c_void_p(), C.c_int(-1)
        context_rc = driver.lib.cuCtxGetCurrent(C.byref(context))
        device_rc = driver.lib.cuCtxGetDevice(C.byref(device))
    except Exception as exc:
        return {"inspection_error": repr(exc)}
    result = {
        "params_api_result": int(legacy_rc), "params": _struct_values(p),
        "generic_api_result": int(generic_rc),
        "generic_params": _struct_values(generic.memcpy.copyParams),
        "node_type": generic.type, "copy_context": generic.memcpy.copyCtx,
        "flags": generic.memcpy.flags, "reserved": generic.memcpy.reserved,
        "current_context": {"api_result": int(context_rc), "value": context.value},
        "current_device": {"api_result": int(device_rc), "value": device.value},
        "operands": {},
    }
    if legacy_rc or generic_rc:
        return result
    for side in ("src", "dst"):
        address = getattr(p, side + "Device") + getattr(p, side + "XInBytes")
        data = {"effective_address": address, "attributes": {}}
        result["operands"][side] = data
        if getattr(p, side + "MemoryType") != 2 or not 0 < address < 2**64:
            continue  # Never query a host/array operand as a CUDA device pointer.
        for name, attribute, value_type in (
            ("context", 1, C.c_void_p), ("memory_type", 2, C.c_int),
            ("buffer_id", 7, C.c_uint64), ("is_managed", 8, C.c_uint),
            ("device_ordinal", 9, C.c_int), ("reserved_base", 11, C.c_uint64),
            ("reserved_size", 12, C.c_size_t), ("mapped", 13, C.c_uint),
            ("mapping_size", 18, C.c_size_t), ("mapping_base", 19, C.c_uint64),
            ("memory_block_id", 20, C.c_uint64),
        ):
            value = value_type()
            rc = driver.lib.cuPointerGetAttribute(C.byref(value), attribute, address)
            data["attributes"][name] = {"api_result": int(rc), "value": value.value}
        attributes = data["attributes"]
        if not attributes["mapping_base"]["api_result"] and not attributes["mapping_size"]["api_result"]:
            data["mapped_range"] = {
                "method": "CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR/SIZE",
                "api_result": 0, "base": attributes["mapping_base"]["value"],
                "size": attributes["mapping_size"]["value"],
            }
        else:
            base, size = C.c_uint64(), C.c_size_t()
            try:
                driver._bind("cuMemGetAddressRange_v2", [C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), C.c_uint64])
                rc = driver.lib.cuMemGetAddressRange_v2(C.byref(base), C.byref(size), address)
                data["mapped_range"] = {"method": "cuMemGetAddressRange_v2", "api_result": int(rc),
                                         "base": base.value, "size": size.value}
            except Exception as exc:
                data["mapped_range"] = {"method": "cuMemGetAddressRange_v2", "api_result": None,
                                         "error": repr(exc)}
        if not attributes["context"]["api_result"] and not attributes["context"]["value"]:
            data["vmm"] = _vmm_parameters(driver, address, p.WidthInBytes, device.value)
    return result


def _memset_vmm_parameters(driver, params):
    """Inspect a fixed 1D memset destination with a successful NULL context.

    CUDA VMM pointers can legitimately have no allocation context. As with the
    existing memcpy guard, prove actual mapped bounds and physical backing;
    reserved VA alone is never evidence of mapped storage. This does not change
    or retain application ownership beyond these read-only handle queries.
    """
    try:
        driver._bind("cuCtxGetCurrent", [C.POINTER(C.c_void_p)])
        driver._bind("cuCtxGetDevice", [C.POINTER(C.c_int)])
        driver._bind("cuPointerGetAttribute", [C.c_void_p, C.c_int, C.c_uint64])
        context, device = C.c_void_p(), C.c_int(-1)
        context_rc = driver.lib.cuCtxGetCurrent(C.byref(context))
        device_rc = driver.lib.cuCtxGetDevice(C.byref(device))
        address, width = params["dst"], params["width"] * params["elementSize"]
        operand = {"effective_address": address, "attributes": {}}
        result = {"current_context": {"api_result": int(context_rc), "value": context.value},
                  "current_device": {"api_result": int(device_rc), "value": device.value},
                  "byte_width": width, "operand": operand}
        if not 0 < address < 2**64 or not 0 < width < 2**64:
            return result
        for name, attribute, value_type in (
                ("context", 1, C.c_void_p), ("memory_type", 2, C.c_int),
                ("buffer_id", 7, C.c_uint64), ("is_managed", 8, C.c_uint),
                ("device_ordinal", 9, C.c_int), ("mapped", 13, C.c_uint),
                ("mapping_size", 18, C.c_size_t), ("mapping_base", 19, C.c_uint64),
                ("memory_block_id", 20, C.c_uint64)):
            value = value_type()
            rc = driver.lib.cuPointerGetAttribute(C.byref(value), attribute, address)
            operand["attributes"][name] = {"api_result": int(rc), "value": value.value}
        attrs = operand["attributes"]
        if not attrs["mapping_base"]["api_result"] and not attrs["mapping_size"]["api_result"]:
            operand["mapped_range"] = {"method": "CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR/SIZE",
                "api_result": 0, "base": attrs["mapping_base"]["value"],
                "size": attrs["mapping_size"]["value"]}
        else:
            base, size = C.c_uint64(), C.c_size_t()
            driver._bind("cuMemGetAddressRange_v2", [C.POINTER(C.c_uint64), C.POINTER(C.c_size_t), C.c_uint64])
            rc = driver.lib.cuMemGetAddressRange_v2(C.byref(base), C.byref(size), address)
            operand["mapped_range"] = {"method": "cuMemGetAddressRange_v2", "api_result": int(rc),
                                        "base": base.value, "size": size.value}
        operand["vmm"] = _vmm_parameters(driver, address, width, device.value)
        return result
    except Exception as exc:
        return {"inspection_error": repr(exc)}


def _memset_rejections(source, target):
    """Fixed-parameter memset; NULL context requires an identical VMM proof."""
    if source != target:
        return ["Memset parameters, destination, context or VMM backing changed"]
    context = source.get("allocation_context", {})
    if context.get("api_result") != 0:
        return ["Memset allocation context query failed"]
    if context.get("context"):
        return []  # Preserve the original non-NULL-context exact-match policy.
    problems = []
    proof = source.get("vmm_proof", {})
    current = proof.get("current_context", {})
    device = proof.get("current_device", {})
    address, width = source.get("dst", 0), source.get("width", 0) * source.get("elementSize", 0)
    if (source.get("height") != 1 or source.get("pitch") != 0
            or source.get("elementSize") not in (1, 2, 4) or not 0 < width < 2**64
            or proof.get("byte_width") != width):
        problems.append("Memset VMM extension only accepts a nonzero 1D unpitched fixed region")
    if (proof.get("inspection_error") or current.get("api_result") != 0 or not current.get("value")
            or device.get("api_result") != 0 or not isinstance(device.get("value"), int)
            or device.get("value", -1) < 0):
        problems.append("Memset VMM current context/device is unavailable")
    operand = proof.get("operand", {})
    attrs = operand.get("attributes", {})
    if operand.get("effective_address") != address:
        problems.append("Memset VMM destination inspection differs")
    for name, expected in {"context": None, "memory_type": 2, "is_managed": 0,
                           "mapped": 1, "device_ordinal": device.get("value")}.items():
        actual = attrs.get(name, {})
        if actual.get("api_result") != 0 or actual.get("value") != expected:
            problems.append(f"Memset VMM {name} is unavailable or unsupported")
    for name in ("buffer_id", "memory_block_id"):
        identity = attrs.get(name, {})
        if identity.get("api_result") != 0 or not identity.get("value"):
            problems.append(f"Memset VMM {name} identity is unavailable")
    bound = operand.get("mapped_range", {})
    base, size = bound.get("base", 0), bound.get("size", 0)
    if (bound.get("api_result") != 0 or not 0 < base < 2**64 or not 0 < size < 2**64
            or base + size > 2**64 or not base <= address < base + size
            or width > base + size - address or address + width > 2**64):
        problems.append("Memset full range is not proven mapped within bounds")
    vmm = operand.get("vmm", {})
    prop = vmm.get("properties", {})
    if (vmm.get("inspection_error") or any(vmm.get(name) != 0 for name in
            ("retain_api_result", "properties_api_result", "release_api_result"))
            or not vmm.get("allocation_handle") or prop.get("type") != 1
            or prop.get("location_type") != 1 or prop.get("location_id") != device.get("value")
            or prop.get("win32_metadata") or prop.get("usage") != 0 or any(prop.get("reserved", [1]))):
        problems.append("Memset NULL allocation context lacks proven device VMM backing")
    access = vmm.get("access", [])
    if (not access or any(item.get("api_result") != 0 or item.get("flags") != 3 for item in access)
            or {item.get("address") for item in access} != {address, address + width - 1}):
        problems.append("Memset VMM current-device READWRITE access is unproven")
    return problems


def _memcpy_rejections(source, target):
    """Strict, single-device 1D D2D subset of CUDA graph-update rules."""
    problems = []
    for label, data in (("source", source), ("target", target)):
        if "inspection_error" in data:
            problems.append(f"{label} memcpy parameters unavailable: {data['inspection_error']}")
            continue
        if data["params_api_result"] or data["generic_api_result"]:
            problems.append(f"{label} memcpy public parameter query failed")
            continue
        p = data["params"]
        if data["node_type"] != 1 or data["flags"] or data["reserved"]:
            problems.append(f"{label} memcpy generic type/flags/reserved are unsupported")
        if p != data["generic_params"]:
            problems.append(f"{label} memcpy legacy/generic public parameters disagree")
        if (not data["copy_context"] or data["current_context"]["api_result"]
                or data["current_device"]["api_result"]
                or data["copy_context"] != data["current_context"]["value"]):
            problems.append(f"{label} memcpy copy context is unavailable or is not the current context")
        if p["WidthInBytes"] <= 0 or p["Height"] != 1 or p["Depth"] != 1:
            problems.append(f"{label} memcpy must be nonzero 1D (Height=Depth=1)")
        if p["reserved0"] or p["reserved1"]:
            problems.append(f"{label} memcpy reserved pointers must be null")
        for side in ("src", "dst"):
            if p[side + "MemoryType"] != 2:
                problems.append(f"{label} {side} operand is not CU_MEMORYTYPE_DEVICE")
                continue
            if (p[side + "Host"] or p[side + "Array"] or not p[side + "Device"]
                    or any(p[side + suffix] for suffix in ("Y", "Z", "LOD"))
                    or p[side + "Height"] not in (0, 1)
                    or p[side + "Pitch"] not in (0, p[side + "XInBytes"] + p["WidthInBytes"])):
                problems.append(f"{label} {side} operand is not a supported linear device region")
            operand = data["operands"].get(side, {})
            attributes = operand.get("attributes", {})
            expected = {"memory_type": 2,
                        "is_managed": 0, "mapped": 1,
                        "device_ordinal": data["current_device"]["value"]}
            for name, value in expected.items():
                observed = attributes.get(name, {})
                if observed.get("api_result") != 0 or observed.get("value") != value:
                    problems.append(f"{label} {side} allocation {name} is unavailable or unsupported")
            context = attributes.get("context", {})
            if context.get("api_result") != 0:
                problems.append(f"{label} {side} allocation context query failed")
            elif context.get("value"):
                if context["value"] != data["copy_context"]:
                    problems.append(f"{label} {side} allocation context differs from copy context")
            else:
                vmm = operand.get("vmm", {})
                prop = vmm.get("properties", {})
                if (vmm.get("inspection_error") or any(vmm.get(name) != 0 for name in
                        ("retain_api_result", "properties_api_result", "release_api_result"))
                        or not vmm.get("allocation_handle") or prop.get("type") != 1
                        or prop.get("location_type") != 1
                        or prop.get("location_id") != data["current_device"]["value"]
                        or prop.get("win32_metadata") or prop.get("usage") != 0
                        or any(prop.get("reserved", [1]))):
                    problems.append(f"{label} {side} NULL allocation context lacks proven device VMM backing")
                access = vmm.get("access", [])
                if (not access or any(item["api_result"] or item["flags"] != 3 for item in access)
                        or {item["address"] for item in access} != {
                            operand.get("effective_address", 0),
                            operand.get("effective_address", 0) + p["WidthInBytes"] - 1}):
                    problems.append(f"{label} {side} VMM current-device READWRITE access is unproven")
                for name in ("buffer_id", "memory_block_id"):
                    identity = attributes.get(name, {})
                    if identity.get("api_result") != 0 or not identity.get("value"):
                        problems.append(f"{label} {side} VMM {name} identity is unavailable")
            bound = operand.get("mapped_range", {})
            address, width = operand.get("effective_address", 0), p["WidthInBytes"]
            base, size = bound.get("base", 0), bound.get("size", 0)
            if (bound.get("api_result") != 0 or not 0 < base < 2**64 or not 0 < size < 2**64
                    or base + size > 2**64 or not base <= address < base + size
                    or width > base + size - address or address + width > 2**64):
                problems.append(f"{label} {side} full copy range is not proven mapped within bounds")
        src = data["operands"].get("src", {}).get("effective_address", 0)
        dst = data["operands"].get("dst", {}).get("effective_address", 0)
        if src and dst and max(src, dst) < min(src, dst) + p["WidthInBytes"]:
            problems.append(f"{label} memcpy source/destination ranges overlap")
    if problems:
        return problems
    if source["copy_context"] != target["copy_context"]:
        problems.append("memcpy copy context changed")
    for side in ("src", "dst"):
        for name in ("context", "device_ordinal", "memory_type", "is_managed"):
            if source["operands"][side]["attributes"][name] != target["operands"][side]["attributes"][name]:
                problems.append(f"memcpy {side} allocation {name} changed")
        a, b = source["operands"][side], target["operands"][side]
        if "vmm" in a or "vmm" in b:
            # A narrow VMM extension: offsets/width may change, but the same
            # physical allocation and actual mapped interval must be retained.
            for name in ("buffer_id", "memory_block_id"):
                if a["attributes"][name] != b["attributes"][name]:
                    problems.append(f"memcpy {side} VMM {name} changed")
            if (a["mapped_range"] != b["mapped_range"]
                    or a.get("vmm", {}).get("allocation_handle") != b.get("vmm", {}).get("allocation_handle")
                    or a.get("vmm", {}).get("properties") != b.get("vmm", {}).get("properties")):
                problems.append(f"memcpy {side} VMM physical backing or mapping changed")
    return problems


def structural_labels(kinds, edges):
    """Pure CPU typed-DAG labels; input nodes can be any hashable identities."""
    incoming, outgoing = {n: [] for n in kinds}, {n: [] for n in kinds}
    for source, target, annotation in edges:
        if source not in kinds or target not in kinds:
            raise ValueError("Edge endpoint missing from node inventory")
        outgoing[source].append((target, annotation))
        incoming[target].append((source, annotation))
    degree = {n: len(incoming[n]) for n in kinds}
    queue = deque(n for n in kinds if degree[n] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target, _ in outgoing[node]:
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    if len(order) != len(kinds):
        raise ValueError("Directed cycle in graph")
    def digest(value):
        return hashlib.sha256(repr(value).encode()).hexdigest()
    forward, backward = {}, {}
    for node in order:
        forward[node] = digest((kinds[node], sorted((annotation, forward[p]) for p, annotation in incoming[node])))
    for node in reversed(order):
        backward[node] = digest((kinds[node], sorted((annotation, backward[p]) for p, annotation in outgoing[node])))
    return {n: (forward[n], backward[n]) for n in kinds}


def map_typed_dags(source_kinds, source_edges, target_kinds, target_edges, identical=False):
    """Return unique proven mapping, or a scoped rejection reason."""
    try:
        a = structural_labels(source_kinds, source_edges)
        b = structural_labels(target_kinds, target_edges)
    except ValueError as exc:
        return None, str(exc)
    if identical:
        mapping = {n: n for n in source_kinds}
    else:
        ag, bg = defaultdict(list), defaultdict(list)
        for node, label in a.items(): ag[label].append(node)
        for node, label in b.items(): bg[label].append(node)
        if {k: len(v) for k, v in ag.items()} != {k: len(v) for k, v in bg.items()}:
            return None, "Typed forward/backward DAG structure differs"
        if any(len(v) != 1 for v in ag.values()):
            return None, "Symmetric nodes have ambiguous structural correspondence"
        mapping = {members[0]: bg[label][0] for label, members in ag.items()}
    if set(mapping) != set(source_kinds) or set(mapping.values()) != set(target_kinds):
        return None, "Mapping does not cover both node inventories bijectively"
    if any(source_kinds[a] != target_kinds[b] for a, b in mapping.items()):
        return None, "Mapped node types differ"
    if Counter((mapping[a], mapping[b], annotation) for a, b, annotation in source_edges) != Counter(target_edges):
        return None, "Independent mapped-edge multiset verification failed"
    return mapping, None


def _read_dag(driver, graph):
    count = C.c_size_t()
    driver.check(driver.lib.cuGraphGetNodes(graph, None, C.byref(count)), "DAG node count")
    if not 0 < count.value <= 8192:
        raise ValueError("DAG node inventory outside bounded 1..8192 range")
    nodes = (C.c_void_p * count.value)()
    driver.check(driver.lib.cuGraphGetNodes(graph, nodes, C.byref(count)), "DAG nodes")
    kinds = {}
    for node in nodes:
        kind = C.c_int()
        driver.check(driver.lib.cuGraphNodeGetType(node, C.byref(kind)), "DAG node type")
        kinds[node] = kind.value
    count = C.c_size_t()
    driver.check(driver.lib.cuGraphGetEdges_v2(graph, None, None, None, C.byref(count)), "DAG edge count")
    if count.value > 65536:
        raise ValueError("DAG edge count exceeds bounded inventory")
    sources, targets = (C.c_void_p * count.value)(), (C.c_void_p * count.value)()
    annotations = (EdgeData * count.value)()
    driver.check(driver.lib.cuGraphGetEdges_v2(graph, sources, targets, annotations, C.byref(count)), "DAG edges")
    return kinds, [(a, b, bytes(edge).hex()) for a, b, edge in zip(sources, targets, annotations)]


def align_dags(driver, source, target):
    a, ae = _read_dag(driver, source)
    b, be = _read_dag(driver, target)
    mapping, reason = map_typed_dags(a, ae, b, be, identical=source == target)
    common = {"comparison": "Unique typed-DAG structural correspondence plus independent full edge-byte multiset check",
              "acyclicity_required": True, "raw_node_order_used_for_matching": False}
    ai = {**common, "node_count": len(a), "edge_count": len(ae), "strict_linear_chain": False}
    bi = {**common, "node_count": len(b), "edge_count": len(be), "strict_linear_chain": False}
    for info, kinds, edges in ((ai, a, ae), (bi, b, be)):
        info["raw_nodes_in_query_order"] = [{"handle": node, "type": kind} for node, kind in kinds.items()]
        info["raw_edges_in_query_order"] = [{"source": x, "target": y, "edge_data_bytes": annotation}
                                             for x, y, annotation in edges]
        info["raw_inventory_independently_queried"] = True
    if mapping is None:
        ai["reason"] = bi["reason"] = reason
        return ai, None, None, bi, None, None
    labels = structural_labels(a, ae)
    # The secondary node handle only orders an identical self-graph's symmetric
    # nodes. It never establishes correspondence between distinct graphs.
    order = sorted(a, key=lambda n: (labels[n], n))
    target_order = [mapping[n] for n in order]
    ai["dag_correspondence_verified"] = bi["dag_correspondence_verified"] = True
    ai["node_handles_in_correspondence_order"] = order
    bi["node_handles_in_correspondence_order"] = target_order
    positions = {n: i for i, n in enumerate(order)}
    target_positions = {n: i for i, n in enumerate(target_order)}
    ai["edges_in_correspondence_positions"] = sorted((positions[x], positions[y], e) for x, y, e in ae)
    bi["edges_in_correspondence_positions"] = sorted((target_positions[x], target_positions[y], e) for x, y, e in be)
    return ai, order, [a[n] for n in order], bi, target_order, [b[n] for n in target_order]


def compare_nonkernels(driver, source_order, target_order, kinds):
    driver._bind("cuGraphMemsetNodeGetParams", [C.c_void_p, C.POINTER(MemsetParams)])
    driver._bind("cuGraphEventRecordNodeGetEvent", [C.c_void_p, C.POINTER(C.c_void_p)])
    driver._bind("cuGraphEventWaitNodeGetEvent", [C.c_void_p, C.POINTER(C.c_void_p)])
    driver._bind("cuPointerGetAttribute", [C.c_void_p, C.c_int, C.c_uint64])
    entries, problems = [], []
    source_events, target_events = defaultdict(list), defaultdict(list)
    def parameters(node, kind):
        if kind == 1:
            return _memcpy_parameters(driver, node)
        if kind == 2:
            p = MemsetParams()
            driver.check(driver.lib.cuGraphMemsetNodeGetParams(node, C.byref(p)), "nonkernel memset params")
            values = {field: getattr(p, field) for field, _ in p._fields_}
            context = C.c_void_p()
            rc = driver.lib.cuPointerGetAttribute(C.byref(context), 1, p.dst)
            values["allocation_context"] = {"api_result": rc, "context": context.value}
            if rc == 0 and not context.value:
                values["vmm_proof"] = _memset_vmm_parameters(driver, values)
            return values
        if kind in (6, 7):
            event = C.c_void_p()
            fn = driver.lib.cuGraphEventWaitNodeGetEvent if kind == 6 else driver.lib.cuGraphEventRecordNodeGetEvent
            driver.check(fn(node, C.byref(event)), "nonkernel event handle")
            return {"event": event.value}
        if kind == 5:
            return {}
        return {"unsupported_node_type": kind}
    for position, (sn, tn, kind) in enumerate(zip(source_order, target_order, kinds)):
        if kind == 0:
            continue
        a, b = parameters(sn, kind), parameters(tn, kind)
        entry = {"position": position, "node_type": kind, "source": a, "target": b,
                 "parameters_identical": a == b}
        entries.append(entry)
        if kind == 1:
            reasons = _memcpy_rejections(a, b)
            entry["memcpy_compatible"] = not reasons
            entry["rejection_reasons"] = reasons
            problems.extend(f"Memcpy at position {position}: {reason}" for reason in reasons)
        elif kind == 2:
            reasons = _memset_rejections(a, b)
            entry["memset_compatible"] = not reasons
            entry["rejection_reasons"] = reasons
            problems.extend(f"Memset at position {position}: {reason}" for reason in reasons)
        elif kind in (6, 7):
            if not a["event"] or not b["event"]:
                problems.append(f"Null event at position {position}")
            source_events[a["event"]].append((position, kind))
            target_events[b["event"]].append((position, kind))
        elif kind != 5:
            problems.append(f"Unsupported nonkernel type {kind} at position {position}")
    renamings = []
    for source_event, usage in source_events.items():
        matches = [event for event, target_usage in target_events.items() if usage == target_usage]
        if len(matches) != 1:
            problems.append(f"Event identity/usage equivalence differs for source event {source_event}")
            continue
        target_event = matches[0]
        if source_event != target_event:
            # A graph's public parameters cannot prove absence of external
            # users of a recorded event. Record even an internally paired
            # rename, but keep the repair gate closed pending explicit proof.
            renamings.append({"source": source_event, "target": target_event, "usage": usage})
            problems.append(f"Event handle changed for mapped usage {usage}; external ownership is not established")
    return {"nonkernel_nodes_compared": len(entries), "nonkernel_compatible": not problems,
            "nonkernel_nodes": entries, "nonkernel_event_renamings": renamings,
            "nonkernel_rejection_reasons": problems,
            "memcpy_caller_obligations": MEMCPY_CALLER_OBLIGATIONS if 1 in kinds else [],
            "memcpy_lifetime_or_mapping_stability_proven_by_public_getters": False,
            "nonkernel_policy": "Known node types only; memcpy permits nonzero single-device 1D D2D address/width changes with identical copy contexts/devices and validated mapped bounds. Non-NULL allocation contexts must match copy context. A successful NULL context requires proven same device VMM allocation handle, properties, mapping, buffer/block identity and device READWRITE access; retain/release references are balanced. Exact memset parameters/context/address; successful NULL memset context additionally needs fixed 1D region, mapped bounds, same VMM handle/properties/buffer/block and READWRITE access. Exact event handles with matching mapped usage. No external event ownership or memory lifetime assumption."}
