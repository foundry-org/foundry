# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Microbenchmark: does cuGraphExecUpdate leave an executable graph
persistently slower, and which update class triggers it?

Foundry's on-demand replay updates a template exec on every batch-size
switch; serving A/Bs show a once-updated exec replays ~6% slower than a
freshly instantiated one. This isolates the driver behavior — no torch.

Method: linear graph of N tiny kernels (launch-overhead dominated, like a
decode step). For each update class, the penalty is
    median(after-update replay time) / median(fresh-exec-of-same-graph) - 1
so changed work (e.g. grid dims) cancels out. Rounds interleave the two
execs to cancel clock/contention drift.

Cases (--cases, comma-separated; prefer one per process for isolation):
  core        the six primary update classes: noop / args / grid / func-swap /
              5x switch roundtrip / per-node exec update  [default]
  upload      args update then cuGraphUpload (does upload re-bake?)
  fractions   args changed on 1/36/180 nodes (linearity in dirty count)
  renoop      real update then a no-op update (does re-update re-bake?)
  sanity      fresh exec vs fresh exec (expect ~0%)
  threshold   all-dirty penalty at 64..1024 nodes (per-exec size tiers)
  childsplit  chain split into <=128-node child graphs inside ONE exec
  attribution GPU event time vs CPU enqueue time (which side pays?)
  bigkernel   same update but ~us-scale kernels (does execution hide it?)
  devupdate   pristine replay cost of device-updatable nodes + host-update rc
  enable      SetEnabled toggle demotion + disabled-node replay cost
  conditional IF/SWITCH conditional node body overhead vs flat graph
  instcost    cuGraphInstantiate wall time + first-launch upload stall
  overlap     replay latency while another thread instantiates (+upload)

Key facts these cases established (H200, r595.58.03): updates applied before
the exec's first upload demote every byte-changed node (~0.1 us/node/replay,
permanent); after upload, argument changes are free but launch-config
changes (grid/block/shmem/func) still demote; execs <=128 nodes never
demote. See docs/exec-update-penalty.md.

Usage: python tools/bench/bench_exec_update.py [--cases core] [--nodes 365]
                                               [--replays 3000] [--rounds 5]
"""

import argparse
import ctypes as C
import statistics
import threading
import time

PTX = rb"""
.version 8.0
.target sm_90
.address_size 64

.visible .entry bump(.param .u64 p, .param .u32 v) {
  .reg .u64 %rd<3>;
  .reg .u32 %r<3>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  cvta.to.global.u64 %rd2, %rd1;
  red.global.add.u32 [%rd2], %r1;
  ret;
}

.visible .entry bump2(.param .u64 p, .param .u32 v) {
  .reg .u64 %rd<3>;
  .reg .u32 %r<3>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  cvta.to.global.u64 %rd2, %rd1;
  red.global.add.u32 [%rd2], %r1;
  ret;
}

.visible .entry spin(.param .u64 p, .param .u32 v) {
  .reg .u64 %rd<3>;
  .reg .u32 %r<6>;
  .reg .pred %p1;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [v];
  cvta.to.global.u64 %rd2, %rd1;
  and.b32 %r3, %r1, 15;
  add.u32 %r3, %r3, 8000;
  mov.u32 %r2, 0;
SPIN_LOOP:
  add.u32 %r2, %r2, 1;
  setp.lt.u32 %p1, %r2, %r3;
  @%p1 bra SPIN_LOOP;
  add.u32 %r1, %r1, %r2;
  red.global.add.u32 [%rd2], %r1;
  ret;
}
"""

CU_LAUNCH_ATTR_DEVICE_UPDATABLE = 13  # CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
CU_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2


class InstParams(C.Structure):
    """CUDA_GRAPH_INSTANTIATE_PARAMS for cuGraphInstantiateWithParams."""

    _fields_ = [
        ("flags", C.c_uint64),
        ("hUploadStream", C.c_void_p),
        ("hErrNode_out", C.c_void_p),
        ("result_out", C.c_int),
    ]


class LaunchAttr(C.Structure):
    """CUlaunchAttribute."""

    _fields_ = [("id", C.c_int), ("pad", C.c_char * 4), ("value", C.c_ubyte * 64)]


class LaunchConfig(C.Structure):
    """CUlaunchConfig."""

    _fields_ = [
        ("gridDimX", C.c_uint),
        ("gridDimY", C.c_uint),
        ("gridDimZ", C.c_uint),
        ("blockDimX", C.c_uint),
        ("blockDimY", C.c_uint),
        ("blockDimZ", C.c_uint),
        ("sharedMemBytes", C.c_uint),
        ("hStream", C.c_void_p),
        ("attrs", C.POINTER(LaunchAttr)),
        ("numAttrs", C.c_uint),
    ]


cuda = C.CDLL("libcuda.so.1")


def chk(res, what):
    if res != 0:
        s = C.c_char_p()
        cuda.cuGetErrorString(res, C.byref(s))
        raise RuntimeError(f"{what}: {res} {s.value}")


class KParams(C.Structure):
    _fields_ = [
        ("func", C.c_void_p),
        ("gridDimX", C.c_uint),
        ("gridDimY", C.c_uint),
        ("gridDimZ", C.c_uint),
        ("blockDimX", C.c_uint),
        ("blockDimY", C.c_uint),
        ("blockDimZ", C.c_uint),
        ("sharedMemBytes", C.c_uint),
        ("kernelParams", C.POINTER(C.c_void_p)),
        ("extra", C.POINTER(C.c_void_p)),
        ("kern", C.c_void_p),
        ("ctx", C.c_void_p),
    ]


class Bench:
    def __init__(self, n_nodes, grid, kernel=b"bump", kernel2=b"bump2"):
        self.N = n_nodes
        self.grid = grid
        chk(cuda.cuInit(0), "cuInit")
        dev = C.c_int()
        chk(cuda.cuDeviceGet(C.byref(dev), 0), "cuDeviceGet")
        self.ctx = C.c_void_p()
        chk(cuda.cuDevicePrimaryCtxRetain(C.byref(self.ctx), dev), "primaryCtxRetain")
        chk(cuda.cuCtxSetCurrent(self.ctx), "ctxSetCurrent")
        mod = C.c_void_p()
        chk(cuda.cuModuleLoadData(C.byref(mod), PTX), "moduleLoadData")
        self.fn = C.c_void_p()
        chk(cuda.cuModuleGetFunction(C.byref(self.fn), mod, kernel), "getFunction")
        self.fn2 = C.c_void_p()
        chk(cuda.cuModuleGetFunction(C.byref(self.fn2), mod, kernel2), "getFunction2")
        self.dbuf = C.c_uint64()
        chk(cuda.cuMemAlloc_v2(C.byref(self.dbuf), 8), "memAlloc")
        self.stream = C.c_void_p()
        chk(cuda.cuStreamCreate(C.byref(self.stream), 0), "streamCreate")
        self.ev0, self.ev1 = C.c_void_p(), C.c_void_p()
        chk(cuda.cuEventCreate(C.byref(self.ev0), 0), "eventCreate")
        chk(cuda.cuEventCreate(C.byref(self.ev1), 0), "eventCreate")

        self.arg_ptr = [C.c_uint64(self.dbuf.value) for _ in range(self.N)]
        self.arg_val = [C.c_uint32(1) for _ in range(self.N)]
        self.arg_arrays = [
            (C.c_void_p * 2)(
                C.cast(C.byref(self.arg_ptr[i]), C.c_void_p),
                C.cast(C.byref(self.arg_val[i]), C.c_void_p),
            )
            for i in range(self.N)
        ]

        self.graph = C.c_void_p()
        chk(cuda.cuGraphCreate(C.byref(self.graph), 0), "graphCreate")
        self.nodes = []
        prev = None
        for i in range(self.N):
            node = C.c_void_p()
            p = self.node_params(i)
            depa = (C.c_void_p * 1)(prev) if prev else None
            chk(
                cuda.cuGraphAddKernelNode(
                    C.byref(node), self.graph, depa, C.c_size_t(1 if prev else 0), C.byref(p)
                ),
                "addKernelNode",
            )
            self.nodes.append(node)
            prev = node

    def node_params(self, i, grid=None, func=None):
        p = KParams()
        C.memset(C.byref(p), 0, C.sizeof(p))
        p.func = C.cast(func if func is not None else self.fn, C.c_void_p)
        p.gridDimX, p.gridDimY, p.gridDimZ = grid or self.grid, 1, 1
        p.blockDimX, p.blockDimY, p.blockDimZ = 64, 1, 1
        p.kernelParams = C.cast(self.arg_arrays[i], C.POINTER(C.c_void_p))
        return p

    def instantiate(self):
        e = C.c_void_p()
        chk(
            cuda.cuGraphInstantiateWithFlags(C.byref(e), self.graph, C.c_ulonglong(0)),
            "instantiate",
        )
        return e

    def instantiate_upload(self, upload_stream):
        """Instantiate with CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD on a stream."""
        p = InstParams()
        p.flags = CU_GRAPH_INSTANTIATE_FLAG_UPLOAD
        p.hUploadStream = upload_stream
        e = C.c_void_p()
        chk(cuda.cuGraphInstantiateWithParams(C.byref(e), self.graph, C.byref(p)), "instWithParams")
        return e

    def whole_graph_update(self, e):
        info = (C.c_void_p * 4)()
        chk(cuda.cuGraphExecUpdate_v2(e, self.graph, info), "execUpdate")

    def set_all(self, grid=None, func=None, bump_args=False):
        for i, node in enumerate(self.nodes):
            if bump_args:
                self.arg_val[i].value += 1
            p = self.node_params(i, grid=grid, func=func)
            chk(cuda.cuGraphKernelNodeSetParams_v2(node, C.byref(p)), "nodeSetParams")

    def time_once(self, e, replays):
        for _ in range(50):
            chk(cuda.cuGraphLaunch(e, self.stream), "graphLaunch")
        chk(cuda.cuStreamSynchronize(self.stream), "streamSync")
        chk(cuda.cuEventRecord(self.ev0, self.stream), "eventRecord")
        for _ in range(replays):
            chk(cuda.cuGraphLaunch(e, self.stream), "graphLaunch")
        chk(cuda.cuEventRecord(self.ev1, self.stream), "eventRecord")
        chk(cuda.cuStreamSynchronize(self.stream), "streamSync")
        ms = C.c_float()
        chk(cuda.cuEventElapsedTime(C.byref(ms), self.ev0, self.ev1), "elapsed")
        return ms.value * 1000.0 / replays

    def enqueue_cost(self, e, launches=100):
        """CPU wall time per cuGraphLaunch enqueue, queue kept shallow."""
        for _ in range(50):
            chk(cuda.cuGraphLaunch(e, self.stream), "graphLaunch")
        chk(cuda.cuStreamSynchronize(self.stream), "streamSync")
        t0 = time.perf_counter()
        for _ in range(launches):
            chk(cuda.cuGraphLaunch(e, self.stream), "graphLaunch")
        t1 = time.perf_counter()
        chk(cuda.cuStreamSynchronize(self.stream), "streamSync")
        return (t1 - t0) * 1e6 / launches

    def compare(self, tag, updated_exec, replays, rounds, fresh=None):
        """Interleaved A/B: updated exec vs fresh exec of the SAME graph.

        Pass fresh explicitly when the graph does not allow another
        instantiation (device-updatable nodes) — e.g. an exec from a
        second, identically-built graph.
        """
        own_fresh = fresh is None
        if own_fresh:
            fresh = self.instantiate()
        upd, frs = [], []
        for _ in range(rounds):
            upd.append(self.time_once(updated_exec, replays))
            frs.append(self.time_once(fresh, replays))
        mu, mf = statistics.median(upd), statistics.median(frs)
        print(
            f"{tag:<28s} updated={mu:7.2f}us fresh={mf:7.2f}us "
            f"penalty={100.0 * (mu - mf) / mf:+5.1f}%"
        )
        if own_fresh:
            cuda.cuGraphExecDestroy(fresh)
        return mu, mf


def case_sanity(args, R, K):
    b = Bench(args.nodes, args.grid)
    e = b.instantiate()
    b.compare("fresh vs fresh (sanity)", e, R, K)
    cuda.cuGraphExecDestroy(e)


def case_threshold(args, R, K):
    """Find the node count where the dirty-node penalty switches on."""
    for n in (64, 96, 128, 160, 192, 256, 320, 365, 512, 1024):
        b = Bench(n, args.grid)
        e = b.instantiate()
        b.set_all(bump_args=True)
        b.whole_graph_update(e)
        mu, mf = b.compare(f"args all-dirty n={n}", e, R, K)
        print(f"    -> {(mu - mf) / n * 1000:+.1f}ns per dirty node")
        cuda.cuGraphExecDestroy(e)


def case_childsplit(args, R, K):
    """Same chain but partitioned into <=128-node child-graph nodes: does the
    per-child size stay under the penalty threshold after exec update?"""
    chunk = 128
    b = Bench(args.nodes, args.grid)  # provides ctx/module/arg buffers; flat graph unused
    parent = C.c_void_p()
    chk(cuda.cuGraphCreate(C.byref(parent), 0), "graphCreate(parent)")
    child_nodes = []
    prev = None
    i = 0
    while i < b.N:
        sub = C.c_void_p()
        chk(cuda.cuGraphCreate(C.byref(sub), 0), "graphCreate(sub)")
        sprev = None
        for j in range(i, min(i + chunk, b.N)):
            node = C.c_void_p()
            p = b.node_params(j)
            depa = (C.c_void_p * 1)(sprev) if sprev else None
            chk(
                cuda.cuGraphAddKernelNode(
                    C.byref(node), sub, depa, C.c_size_t(1 if sprev else 0), C.byref(p)
                ),
                "addKernelNode(sub)",
            )
            sprev = node
        cnode = C.c_void_p()
        depa = (C.c_void_p * 1)(prev) if prev else None
        chk(
            cuda.cuGraphAddChildGraphNode(
                C.byref(cnode), parent, depa, C.c_size_t(1 if prev else 0), sub
            ),
            "addChildGraphNode",
        )
        cuda.cuGraphDestroy(sub)  # the child node owns a clone
        child_nodes.append(cnode)
        prev = cnode
        i += chunk
    e = C.c_void_p()
    chk(cuda.cuGraphInstantiateWithFlags(C.byref(e), parent, C.c_ulonglong(0)), "instantiate")
    # Mutate args of every kernel node inside the CLONED children, then
    # whole-graph exec update against the parent.
    for cnode in child_nodes:
        g = C.c_void_p()
        chk(cuda.cuGraphChildGraphNodeGetGraph(cnode, C.byref(g)), "childGetGraph")
        cnt = C.c_size_t()
        chk(cuda.cuGraphGetNodes(g, None, C.byref(cnt)), "getNodes(count)")
        arr = (C.c_void_p * cnt.value)()
        chk(cuda.cuGraphGetNodes(g, arr, C.byref(cnt)), "getNodes")
        for k in range(cnt.value):
            b.arg_val[k].value += 1
            p = b.node_params(k)
            # arr[k] is a bare int; re-wrap so ctypes passes a full 64-bit handle
            chk(
                cuda.cuGraphKernelNodeSetParams_v2(C.c_void_p(arr[k]), C.byref(p)),
                "nodeSetParams(child)",
            )
    info = (C.c_void_p * 4)()
    chk(cuda.cuGraphExecUpdate_v2(e, parent, info), "execUpdate(parent)")
    b.graph = parent  # so compare()'s fresh control instantiates the parent
    mu, mf = b.compare(f"child-split {chunk}/node chunks", e, R, K)
    print(f"    -> fresh child-split={mf:.2f}us (flat-graph fresh reference ~{0.79 * b.N:.0f}us)")
    cuda.cuGraphExecDestroy(e)


def case_attribution(args, R, K):
    b = Bench(args.nodes, args.grid)
    e = b.instantiate()
    b.set_all(bump_args=True)
    b.whole_graph_update(e)
    fresh = b.instantiate()
    gu, cu, gf, cf = [], [], [], []
    for _ in range(K):
        gu.append(b.time_once(e, R))
        gf.append(b.time_once(fresh, R))
        cu.append(b.enqueue_cost(e))
        cf.append(b.enqueue_cost(fresh))
    m = statistics.median
    print(
        f"GPU event us/replay: updated={m(gu):7.2f} fresh={m(gf):7.2f} delta={m(gu) - m(gf):+6.2f}"
    )
    print(
        f"CPU enqueue us/launch: updated={m(cu):7.2f} fresh={m(cf):7.2f} "
        f"delta={m(cu) - m(cf):+6.2f}"
    )
    cuda.cuGraphExecDestroy(fresh)
    cuda.cuGraphExecDestroy(e)


def case_bigkernel(args, R, K):
    b = Bench(args.nodes, 1, kernel=b"spin", kernel2=b"spin")
    e = b.instantiate()
    b.set_all(bump_args=True)
    b.whole_graph_update(e)
    r = max(50, R // 40)  # ~15us kernels: keep runtime sane
    mu, mf = b.compare("args all-dirty, spin kernels", e, r, K)
    print(f"    -> kernel ~{mf / args.nodes:.2f}us each; delta {mu - mf:+.2f}us/replay")
    cuda.cuGraphExecDestroy(e)


def _capture_chain(b, device_updatable):
    """Capture b.N sequential cuLaunchKernelEx calls into a CUgraph.
    The device-updatable attribute can only be requested at capture time."""
    chk(cuda.cuStreamBeginCapture_v2(b.stream, 0), "beginCapture")  # GLOBAL mode
    for i in range(b.N):
        cfg = LaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = b.grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 64, 1, 1
        cfg.hStream = b.stream
        if device_updatable:
            # Fresh attr per launch: the driver RETURNS the devNode handle in
            # the attr value; a stale non-NULL devNode fails the next launch.
            attr = LaunchAttr()
            attr.id = CU_LAUNCH_ATTR_DEVICE_UPDATABLE
            attr.value[0] = 1  # deviceUpdatableKernelNode.deviceUpdatable = 1
            attrs = (LaunchAttr * 1)(attr)
            cfg.attrs = attrs
            cfg.numAttrs = 1
        rc = cuda.cuLaunchKernelEx(
            C.byref(cfg), b.fn, C.cast(b.arg_arrays[i], C.POINTER(C.c_void_p)), None
        )
        if rc != 0:
            cuda.cuStreamEndCapture(b.stream, C.byref(C.c_void_p()))
            raise RuntimeError(f"cuLaunchKernelEx rc={rc}")
    g = C.c_void_p()
    chk(cuda.cuStreamEndCapture(b.stream, C.byref(g)), "endCapture")
    return g


def case_devupdate(args, R, K):
    b = Bench(args.nodes, args.grid)
    try:
        g_du = _capture_chain(b, device_updatable=True)
    except RuntimeError as exc:
        print(f"device-updatable capture unsupported: {exc}")
        return
    g_norm = _capture_chain(b, device_updatable=False)
    e_du, e_norm = C.c_void_p(), C.c_void_p()
    rc = cuda.cuGraphInstantiateWithFlags(C.byref(e_du), g_du, C.c_ulonglong(0))
    if rc != 0:
        print(f"instantiate(device-updatable graph) rc={rc}")
        return
    chk(cuda.cuGraphInstantiateWithFlags(C.byref(e_norm), g_norm, C.c_ulonglong(0)), "inst")
    upd, frs = [], []
    for _ in range(K):
        upd.append(b.time_once(e_du, R))
        frs.append(b.time_once(e_norm, R))
    mu, mf = statistics.median(upd), statistics.median(frs)
    print(
        f"{'device-updatable pristine':<28s} devupd={mu:7.2f}us normal={mf:7.2f}us "
        f"penalty={100.0 * (mu - mf) / mf:+5.1f}%"
    )
    # Host-side whole-graph update against the device-updatable exec: allowed?
    cnt = C.c_size_t()
    chk(cuda.cuGraphGetNodes(g_du, None, C.byref(cnt)), "getNodes(count)")
    arr = (C.c_void_p * cnt.value)()
    chk(cuda.cuGraphGetNodes(g_du, arr, C.byref(cnt)), "getNodes")
    b.arg_val[0].value += 1
    p = b.node_params(0)
    rc = cuda.cuGraphKernelNodeSetParams_v2(C.c_void_p(arr[0]), C.byref(p))
    info = (C.c_void_p * 4)()
    rc2 = cuda.cuGraphExecUpdate_v2(e_du, g_du, info) if rc == 0 else -1
    print(f"host setParams on device-updatable graph node rc={rc}, execUpdate rc={rc2}")
    if rc2 == 0:
        upd2 = [b.time_once(e_du, R) for _ in range(K)]
        frs2 = [b.time_once(e_norm, R) for _ in range(K)]
        mu2, mf2 = statistics.median(upd2), statistics.median(frs2)
        print(
            f"{'device-updatable post-update':<28s} devupd={mu2:7.2f}us normal={mf2:7.2f}us "
            f"penalty={100.0 * (mu2 - mf2) / mf2:+5.1f}%"
        )
    cuda.cuGraphExecDestroy(e_du)
    cuda.cuGraphExecDestroy(e_norm)


def case_enable(args, R, K):
    """cuGraphNodeSetEnabled: (a) does an off->on round-trip demote like a
    param change? (b) what does a disabled node cost at replay?"""
    b = Bench(args.nodes, args.grid)
    e = b.instantiate()
    for node in b.nodes:
        chk(cuda.cuGraphNodeSetEnabled(e, node, 0), "setEnabled(0)")
    for node in b.nodes:
        chk(cuda.cuGraphNodeSetEnabled(e, node, 1), "setEnabled(1)")
    b.compare("enable toggle round-trip", e, R, K)
    for i, node in enumerate(b.nodes):
        if i % 2 == 1:
            chk(cuda.cuGraphNodeSetEnabled(e, node, 0), "setEnabled(0)")
    t_half = statistics.median([b.time_once(e, R) for _ in range(K)])
    fresh = b.instantiate()
    t_full = statistics.median([b.time_once(fresh, R) for _ in range(K)])
    n_off = b.N // 2
    print(
        f"half-disabled={t_half:.2f}us full={t_full:.2f}us "
        f"-> disabled-node cost ~{(t_half - t_full / 2) / n_off * 1000:+.0f}ns/node"
    )
    cuda.cuGraphExecDestroy(fresh)
    cuda.cuGraphExecDestroy(e)


CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
CU_GRAPH_COND_TYPE_IF = 0
CU_GRAPH_COND_TYPE_SWITCH = 2
CU_GRAPH_COND_ASSIGN_DEFAULT = 1


class CondNodeParams(C.Structure):
    """CUgraphNodeParams with the CUDA_CONDITIONAL_NODE_PARAMS union arm.
    Layout: 16B header + 232B union + 8B tail = 256B."""

    _fields_ = [
        ("type", C.c_int),
        ("reserved0", C.c_int * 3),
        ("handle", C.c_uint64),
        ("cond_type", C.c_int),
        ("size", C.c_uint),
        ("phGraph_out", C.POINTER(C.c_void_p)),
        ("ctx", C.c_void_p),
        ("tail", C.c_byte * (232 - 32)),
        ("reserved2", C.c_longlong),
    ]


def _build_chain_into(b, graph):
    prev = None
    for i in range(b.N):
        node = C.c_void_p()
        p = b.node_params(i)
        depa = (C.c_void_p * 1)(prev) if prev else None
        chk(
            cuda.cuGraphAddKernelNode(
                C.byref(node), graph, depa, C.c_size_t(1 if prev else 0), C.byref(p)
            ),
            "addKernelNode(body)",
        )
        prev = node


def case_conditional(args, R, K):
    """Per-replay overhead of running the same chain inside a conditional
    node body (IF, and SWITCH with 8 bodies) vs the flat graph. A no-update
    batch-size dispatcher needs this to be cheap."""
    b = Bench(args.nodes, args.grid)
    e_flat = b.instantiate()
    for tag, cond_type, n_bodies, default in (
        ("IF body (1)", CU_GRAPH_COND_TYPE_IF, 1, 1),
        ("SWITCH body[0] of 8", CU_GRAPH_COND_TYPE_SWITCH, 8, 0),
    ):
        g = C.c_void_p()
        chk(cuda.cuGraphCreate(C.byref(g), 0), "graphCreate")
        handle = C.c_uint64()
        rc = cuda.cuGraphConditionalHandleCreate(
            C.byref(handle), g, b.ctx, C.c_uint(default), C.c_uint(CU_GRAPH_COND_ASSIGN_DEFAULT)
        )
        if rc != 0:
            print(f"{tag}: conditionalHandleCreate rc={rc}; unsupported")
            continue
        params = CondNodeParams()
        params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.handle = handle.value
        params.cond_type = cond_type
        params.size = n_bodies
        params.ctx = b.ctx
        cnode = C.c_void_p()
        rc = cuda.cuGraphAddNode(C.byref(cnode), g, None, C.c_size_t(0), C.byref(params))
        if rc != 0:
            print(f"{tag}: cuGraphAddNode(conditional) rc={rc}; unsupported")
            continue
        body = C.c_void_p(params.phGraph_out[0])  # bodies populated by the driver
        _build_chain_into(b, body)
        e_cond = C.c_void_p()
        chk(cuda.cuGraphInstantiateWithFlags(C.byref(e_cond), g, C.c_ulonglong(0)), "instantiate")
        cond, flat = [], []
        for _ in range(K):
            cond.append(b.time_once(e_cond, R))
            flat.append(b.time_once(e_flat, R))
        mc, mf = statistics.median(cond), statistics.median(flat)
        print(
            f"{tag:<28s} cond={mc:7.2f}us flat={mf:7.2f}us "
            f"overhead={mc - mf:+6.2f}us ({100.0 * (mc - mf) / mf:+5.1f}%)"
        )
        cuda.cuGraphExecDestroy(e_cond)
    cuda.cuGraphExecDestroy(e_flat)


def case_instcost(args, R, K):
    b = Bench(args.nodes, args.grid)
    times = []
    for _ in range(12):
        t0 = time.perf_counter()
        e = b.instantiate()
        times.append((time.perf_counter() - t0) * 1e3)
        cuda.cuGraphExecDestroy(e)
    print(
        f"cuGraphInstantiate wall ms (n={b.N}): first={times[0]:.2f} "
        f"median-rest={statistics.median(times[1:]):.2f}"
    )
    # Upload deferral: cost of the FIRST launch after a no-upload instantiate.
    e = b.instantiate()
    chk(cuda.cuCtxSynchronize(), "ctxSync")
    t0 = time.perf_counter()
    chk(cuda.cuGraphLaunch(e, b.stream), "graphLaunch")
    t_enq = (time.perf_counter() - t0) * 1e3
    chk(cuda.cuStreamSynchronize(b.stream), "streamSync")
    t_tot = (time.perf_counter() - t0) * 1e3
    print(f"first launch after instantiate: enqueue={t_enq:.2f}ms launch+run={t_tot:.2f}ms")
    steady = b.time_once(e, R)
    print(f"steady replay: {steady:.2f}us")
    cuda.cuGraphExecDestroy(e)


def case_overlap(args, R, K):
    """Replay latency on thread A while thread-main instantiates. Quantifies
    whether cuGraphInstantiate can overlap concurrent graph replays."""
    b = Bench(args.nodes, args.grid)
    ex_a = b.instantiate()
    b2 = Bench(4096, args.grid)  # big separate graph: slower, production-like instantiate
    samples = []
    stop = threading.Event()
    batch = 20

    def replayer():
        chk(cuda.cuCtxSetCurrent(b.ctx), "ctxSetCurrent(replayer)")
        s = C.c_void_p()
        chk(cuda.cuStreamCreate(C.byref(s), 0), "streamCreate")
        e0, e1 = C.c_void_p(), C.c_void_p()
        chk(cuda.cuEventCreate(C.byref(e0), 0), "eventCreate")
        chk(cuda.cuEventCreate(C.byref(e1), 0), "eventCreate")
        for _ in range(50):
            chk(cuda.cuGraphLaunch(ex_a, s), "graphLaunch")
        chk(cuda.cuStreamSynchronize(s), "streamSync")
        while not stop.is_set():
            chk(cuda.cuEventRecord(e0, s), "eventRecord")
            for _ in range(batch):
                chk(cuda.cuGraphLaunch(ex_a, s), "graphLaunch")
            chk(cuda.cuEventRecord(e1, s), "eventRecord")
            chk(cuda.cuStreamSynchronize(s), "streamSync")
            ms = C.c_float()
            chk(cuda.cuEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
            samples.append((time.perf_counter(), ms.value * 1000.0 / batch))

    th = threading.Thread(target=replayer, daemon=True)
    th.start()
    time.sleep(2.0)

    windows = {}
    # Window 1: plain instantiate churn (upload deferred to first launch).
    t0 = time.perf_counter()
    inst_plain = []
    while time.perf_counter() - t0 < 3.0:
        ti = time.perf_counter()
        e2 = b2.instantiate()
        inst_plain.append((time.perf_counter() - ti) * 1e3)
        cuda.cuGraphExecDestroy(e2)
    windows["plain-instantiate"] = (t0, time.perf_counter())
    time.sleep(2.0)
    # Window 2: instantiate + upload on a dedicated stream.
    up_stream = C.c_void_p()
    chk(cuda.cuStreamCreate(C.byref(up_stream), 0), "streamCreate")
    t0 = time.perf_counter()
    inst_up = []
    while time.perf_counter() - t0 < 3.0:
        ti = time.perf_counter()
        e2 = b2.instantiate_upload(up_stream)
        inst_up.append((time.perf_counter() - ti) * 1e3)
        cuda.cuGraphExecDestroy(e2)
    chk(cuda.cuStreamSynchronize(up_stream), "streamSync")
    windows["instantiate+upload"] = (t0, time.perf_counter())
    time.sleep(2.0)
    stop.set()
    th.join()

    m = statistics.median
    quiet = [
        v for t, v in samples if all(not (w0 <= t <= w1 + 0.05) for w0, w1 in windows.values())
    ]
    print(f"replay us/graph quiet: median={m(quiet):.2f} max={max(quiet):.2f} n={len(quiet)}")
    for name, (w0, w1) in windows.items():
        dur = [v for t, v in samples if w0 <= t <= w1 + 0.05]
        if dur:
            print(
                f"replay us/graph during {name}: median={m(dur):.2f} "
                f"max={max(dur):.2f} n={len(dur)} (window {(w1 - w0) * 1e3:.0f}ms)"
            )
        else:
            print(f"replay during {name}: NO samples completed in window ({(w1 - w0) * 1e3:.0f}ms)")
    print(f"instantiate wall ms: plain={m(inst_plain):.2f} +upload={m(inst_up):.2f}")


def case_core(args, R, K):
    """The six primary update classes on one graph."""
    b = Bench(args.nodes, args.grid)

    # noop: whole-graph update with identical params.
    e = b.instantiate()
    b.set_all()
    b.whole_graph_update(e)
    b.compare("noop whole-graph", e, R, K)
    cuda.cuGraphExecDestroy(e)

    # args: change one scalar on every node.
    e = b.instantiate()
    b.set_all(bump_args=True)
    b.whole_graph_update(e)
    b.compare("args whole-graph", e, R, K)
    cuda.cuGraphExecDestroy(e)

    # grid: change grid dims on every node (control is fresh of same graph).
    e = b.instantiate()
    b.set_all(grid=args.grid * 2)
    b.whole_graph_update(e)
    b.compare("grid whole-graph", e, R, K)
    b.set_all(grid=args.grid)  # restore
    cuda.cuGraphExecDestroy(e)

    # func: swap every node to the identical twin kernel.
    e = b.instantiate()
    b.set_all(func=b.fn2)
    b.whole_graph_update(e)
    b.compare("func-swap whole-graph", e, R, K)
    b.set_all(func=b.fn)  # restore
    cuda.cuGraphExecDestroy(e)

    # roundtrip: 5 switches A->B->A->B->A (grid+args change per switch).
    e = b.instantiate()
    for k in range(5):
        b.set_all(grid=args.grid * 2 if k % 2 == 0 else args.grid, bump_args=True)
        b.whole_graph_update(e)
    b.compare("5x switch roundtrip", e, R, K)
    cuda.cuGraphExecDestroy(e)

    # exec-node: per-node exec update (args change), no whole-graph update.
    e = b.instantiate()
    for i, node in enumerate(b.nodes):
        b.arg_val[i].value += 1
        p = b.node_params(i)
        chk(cuda.cuGraphExecKernelNodeSetParams_v2(e, node, C.byref(p)), "execNodeSetParams")
    b.compare("args per-node EXEC update", e, R, K)
    cuda.cuGraphExecDestroy(e)


def case_upload(args, R, K):
    """Does cuGraphUpload after an update re-bake the fast path? (No.)"""
    b = Bench(args.nodes, args.grid)
    e = b.instantiate()
    b.set_all(bump_args=True)
    b.whole_graph_update(e)
    chk(cuda.cuGraphUpload(e, b.stream), "graphUpload")
    chk(cuda.cuStreamSynchronize(b.stream), "sync")
    b.compare("args update + cuGraphUpload", e, R, K)
    cuda.cuGraphExecDestroy(e)


def case_fractions(args, R, K):
    """Change args on k nodes only: penalty is linear in dirty count."""
    b = Bench(args.nodes, args.grid)
    for k_nodes in (1, 36, 180):
        e = b.instantiate()
        for i in range(k_nodes):
            b.arg_val[i].value += 1
            p = b.node_params(i)
            chk(cuda.cuGraphKernelNodeSetParams_v2(b.nodes[i], C.byref(p)), "nodeSetParams")
        b.whole_graph_update(e)
        b.compare(f"args on {k_nodes}/{b.N} nodes", e, R, K)
        cuda.cuGraphExecDestroy(e)


def case_renoop(args, R, K):
    """A no-op RE-update after a real one does not re-bake."""
    b = Bench(args.nodes, args.grid)
    e = b.instantiate()
    b.set_all(bump_args=True)
    b.whole_graph_update(e)
    b.whole_graph_update(e)
    b.compare("real update then noop update", e, R, K)
    cuda.cuGraphExecDestroy(e)


CASES = {
    "core": case_core,
    "upload": case_upload,
    "fractions": case_fractions,
    "renoop": case_renoop,
    "sanity": case_sanity,
    "threshold": case_threshold,
    "childsplit": case_childsplit,
    "attribution": case_attribution,
    "bigkernel": case_bigkernel,
    "devupdate": case_devupdate,
    "enable": case_enable,
    "conditional": case_conditional,
    "instcost": case_instcost,
    "overlap": case_overlap,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=365)
    ap.add_argument("--replays", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--cases", type=str, default="core", help=f"one or more of: {','.join(CASES)}")
    args = ap.parse_args()
    R, K = args.replays, args.rounds
    print(f"nodes={args.nodes} replays={R} rounds={K} grid={args.grid}")
    for name in args.cases.split(","):
        print(f"--- case: {name} ---")
        CASES[name.strip()](args, R, K)


if __name__ == "__main__":
    main()
