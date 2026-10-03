"""Read-only comparison of kernel parameters in strictly linear CUDA graphs.

Node enumeration order is not treated as correspondence. We reconstruct and
validate a single chain, including node types and all eight edge-data bytes,
before comparing kernel parameters by chain position. No GPU work or mutation.
"""

import ctypes as C
import struct
from collections import Counter

from .graph_probe import EdgeData, KernelParams


def _linearize(driver, graph):
    n = C.c_size_t()
    driver.check(driver.lib.cuGraphGetNodes(graph, None, C.byref(n)), "diff cuGraphGetNodes(count)")
    nodes = (C.c_void_p * n.value)()
    driver.check(driver.lib.cuGraphGetNodes(graph, nodes, C.byref(n)), "diff cuGraphGetNodes")
    n_edges = C.c_size_t()
    driver.check(driver.lib.cuGraphGetEdges_v2(graph, None, None, None, C.byref(n_edges)), "diff cuGraphGetEdges(count)")
    sources = (C.c_void_p * n_edges.value)()
    targets = (C.c_void_p * n_edges.value)()
    edges = (EdgeData * n_edges.value)()
    driver.check(driver.lib.cuGraphGetEdges_v2(graph, sources, targets, edges, C.byref(n_edges)), "diff cuGraphGetEdges")
    result = {"node_count": n.value, "edge_count": n_edges.value, "strict_linear_chain": False}
    if n.value == 0 or n_edges.value != n.value - 1:
        result["reason"] = "Graph is empty or edge count is not node count minus one"
        return result, None, None
    incoming = {node: 0 for node in nodes}
    outgoing = {}
    for source, target, edge in zip(sources, targets, edges):
        if source not in incoming or target not in incoming:
            result["reason"] = "Edge endpoint is not in node enumeration"
            return result, None, None
        incoming[target] += 1
        if source in outgoing or incoming[target] != 1:
            result["reason"] = "Graph has a branch or a merge"
            return result, None, None
        outgoing[source] = (target, bytes(edge).hex())
    roots = [node for node, degree in incoming.items() if degree == 0]
    if len(roots) != 1:
        result["reason"] = "Graph does not have exactly one root"
        return result, None, None
    order, edge_order = [], []
    seen = set()
    cursor = roots[0]
    while cursor is not None:
        if cursor in seen:
            result["reason"] = "Cycle found during chain traversal"
            return result, None, None
        seen.add(cursor)
        order.append(cursor)
        successor = outgoing.get(cursor)
        if successor is None:
            break
        cursor, annotation = successor
        edge_order.append(annotation)
    if len(order) != n.value:
        result["reason"] = "Root chain does not cover every graph node"
        return result, None, None
    kinds = []
    for node in order:
        kind = C.c_int()
        driver.check(driver.lib.cuGraphNodeGetType(node, C.byref(kind)), "diff cuGraphNodeGetType")
        kinds.append(kind.value)
    result.update(strict_linear_chain=True, node_types_in_chain_order=kinds,
                  edge_data_bytes_in_chain_order=edge_order)
    return result, order, kinds


def diff_graphs(driver, source, target, allow_dag=False, memcpy_creation_context=None):
    """Return JSON-serializable field differences, or an alignment rejection.

    This reports host-visible node parameters and selected graph attributes.
    It cannot establish the driver's executable QMD or prefetch state.
    """
    driver._bind("cuFuncGetName", [C.POINTER(C.c_char_p), C.c_void_p])
    source_info, source_order, source_kinds = _linearize(driver, source)
    target_info, target_order, target_kinds = _linearize(driver, target)
    dag_mode = False
    if allow_dag and (source_order is None or target_order is None):
        from .graph_dag_diff import align_dags
        (source_info, source_order, source_kinds,
         target_info, target_order, target_kinds) = align_dags(driver, source, target)
        dag_mode = True
    report = {"source": source_info, "target": target_info,
              "aligned": False, "comparison": "Strict unique-chain correspondence; never cuGraphGetNodes array position",
              "limitations": ["Kernel argument contents are not compared.",
                              "Selected host-visible attributes are not a full launch-descriptor dump.",
                              "Field differences alone do not identify the cause of replay slowdown."]}
    if dag_mode:
        report["comparison"] = "Unique typed-DAG structural correspondence and independently verified mapped edge multiset"
    report["dag_extension_enabled"] = bool(allow_dag)
    if source_order is None or target_order is None:
        report["reason"] = source_info.get("reason") or target_info.get("reason") or "A graph is not a validated unique chain"
        return report
    if len(source_order) != len(target_order):
        report["reason"] = "Node counts differ"
        return report
    if source_kinds != target_kinds:
        report["reason"] = "Node types differ along the chains"
        return report
    if not dag_mode and source_info["edge_data_bytes_in_chain_order"] != target_info["edge_data_bytes_in_chain_order"]:
        report["reason"] = "Edge annotations differ along the chains"
        report["different_edge_positions"] = [i for i, (a, b) in enumerate(zip(
            source_info["edge_data_bytes_in_chain_order"], target_info["edge_data_bytes_in_chain_order"])) if a != b]
        return report
    report["aligned"] = True
    if allow_dag:
        from .graph_dag_diff import compare_nonkernels
        report.update(compare_nonkernels(driver, source_order, target_order, source_kinds,
                                         memcpy_creation_context=memcpy_creation_context))
    names = {}

    def function_name(function):
        if not function:
            return None
        if function not in names:
            name = C.c_char_p()
            rc = driver.lib.cuFuncGetName(C.byref(name), function)
            names[function] = name.value.decode("utf-8", "replace") if rc == 0 and name.value else f"unavailable_api_result_{rc}"
        return names[function]

    def parameters(node):
        params = KernelParams()
        driver.check(driver.lib.cuGraphKernelNodeGetParams_v2(node, C.byref(params)), "diff cuGraphKernelNodeGetParams_v2")
        values = {"function": params.func, "kernel": params.kern, "context": params.ctx,
                  "function_name": function_name(params.func),
                  "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                  "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                  "shared_bytes": params.sharedMemBytes}
        attrs = {}
        # 6 may be rejected for graph-node queries on some driver versions.
        # Record its API result instead of assuming support or ignoring it.
        for attr_id, attr_name in ((1, "access_policy_window"), (2, "cooperative"), (4, "cluster_dimensions"),
                                   (5, "cluster_scheduling_policy"), (6, "programmatic_stream_serialization"),
                                   (8, "priority"), (9, "mem_sync_domain_map"), (10, "mem_sync_domain"),
                                   (11, "preferred_cluster_dimensions"), (13, "device_updatable"),
                                   (14, "shared_memory_carveout")):
            buf = (C.c_ulonglong * 8)()
            rc = driver.lib.cuGraphKernelNodeGetAttribute(node, attr_id, buf)
            if rc:
                attrs[attr_name] = {"api_result": rc}
            else:
                ints = C.cast(buf, C.POINTER(C.c_uint))
                if attr_id == 1:
                    value = list(struct.unpack_from("<QQfii", bytes(buf)))
                elif attr_id in (4, 11):
                    value = [ints[i] for i in range(3)]
                elif attr_id == 9:
                    value = list(bytes(buf)[:2])
                else:
                    value = ints[0]
                attrs[attr_name] = {"api_result": 0, "value": value}
        values["attributes"] = attrs
        return values

    counts = Counter()
    attribute_counts = Counter()
    function_pairs = Counter()
    changes = []
    kernel_nodes = 0
    device_updatable_nodes = 0
    for index, (source_node, target_node, kind) in enumerate(zip(source_order, target_order, source_kinds)):
        if kind != 0:
            continue
        kernel_nodes += 1
        a, b = parameters(source_node), parameters(target_node)
        device_updatable_nodes += int(any(p["attributes"]["device_updatable"].get("value", 0) != 0 for p in (a, b)))
        fields = [key for key in ("function", "kernel", "context", "grid", "block", "shared_bytes") if a[key] != b[key]]
        attrs_changed = [name for name in a["attributes"] if a["attributes"][name] != b["attributes"][name]]
        counts.update(fields)
        attribute_counts.update(attrs_changed)
        if a["function"] != b["function"]:
            function_pairs[(a["function_name"], b["function_name"])] += 1
        if fields or attrs_changed:
            changes.append({"chain_position": index, "changed_fields": fields,
                            "changed_attributes": attrs_changed, "source": a, "target": b})
    report.update(kernel_nodes_compared=kernel_nodes, changed_node_count=len(changes),
                  device_updatable_nodes=device_updatable_nodes,
                  changed_field_node_counts=dict(counts), changed_attribute_node_counts=dict(attribute_counts),
                  changed_function_name_pairs=[{"source": a, "target": b, "nodes": count}
                                               for (a, b), count in function_pairs.items()],
                  changed_nodes=changes)
    return report
