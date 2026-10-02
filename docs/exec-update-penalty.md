# Restored-graph replay performance

Restored (LOADed) graphs once replayed about 10% slower than the graphs SGLang
captured itself (Qwen3-1.7B, 1 GPU, H200: bs16 -19%, bs32 -12%, shrinking with
batch size). Three independent causes were behind it. This document records
each cause, how it was found and what fixed it, so the replay design in
[`graph-templates.md`](graph-templates.md) and the edge handling in
[`pdl-edge-batching.md`](pdl-edge-batching.md) can be read with their reasons.

| cause | where | fix |
|---|---|---|
| PDL edge data not serialized | archive format | edges carry their programmatic ports; restored as `CUgraphEdgeData` |
| every batch-size switch synchronized the device and rewrote the whole exec | replay path | update only on a member change, overlapped with the in-flight forward; same-member replays launch directly |
| `cuGraphExecUpdate` cleared the exec's constant-prefetch plan | CUDA driver's update path | the prefetch bits are restored on the updated descriptors after every update |

Bench for the last two: `tests/bench_exec_update.py`.

## 1. PDL edges were rebuilt as full dependencies

FA3 decode on Hopper uses Programmatic Dependent Launch: the
`prepare_varlen_num_blocks -> fwd -> combine` overlap is captured as
`CUgraphEdgeData` programmatic ports (28 edges per graph on Qwen3-1.7B). The
archive stored only `{from, to}` per edge, so every PDL edge was rebuilt as a
full-completion dependency and the overlap was lost.

Fix: capture keeps the ports, the graph JSON gained optional
`from_port / to_port / edge_type`, both rebuild sites pass reconstructed
`CUgraphEdgeData`, and `PROGRAMMATIC_STREAM_SERIALIZATION` is restored as a
node attribute. Archives written before this carry no edge data and replay at
the old speed until re-saved.

The same edges were lost a second time at insertion: libcuda 580-610 applies
the first edge's data to every edge of a bulk `cuGraphAddDependencies_v2`
call. Dependencies are now inserted per homogeneous edge record; see
[`pdl-edge-batching.md`](pdl-edge-batching.md).

## 2. Switching batch sizes paid a synchronize and a full rewrite

On-demand members share one `CUgraphExec` per template. The first replay path
handled a batch-size change with a device synchronize (draining the decode
pipeline), a ~374 us rewrite of every node's parameters and a
`cuGraphExecUpdate`; one bench sweep hit 606 such switches.

Fix: the switch is a host-only transaction with no GPU synchronization,
issued one step ahead so it overlaps the in-flight forward: rewrite the
members' parameters from the template, `cuGraphExecUpdate`, the descriptor
repair of section 3, a stream-ordered upload, then launch. A replay of the
same member as the previous step launches the exec directly with no prepare
or finish work. Byte-identical parameters cost nothing (the driver diffs
internally), so only nodes whose parameters really change are rewritten. With
the update off the GPU timeline, fixed-size bs1 decode replays at parity:
restored TPOT 2.03 / 2.04 / 2.04 ms against captured 2.02 / 2.05 / 2.02 ms.

## 3. An updated exec replayed slower for the rest of its life

With the switch cost gone, any exec that had been through `cuGraphExecUpdate`
still replayed slower than a fresh instantiate of the same graph. The
behavior was characterized with a 365-node microbench (interleaved medians
against a fresh exec of the same mutated graph, H200):

- Updates applied between instantiate and the exec's first upload demote
  every byte-changed node; after the first upload, kernel-argument changes are
  free but launch-config changes (grid, block, shared memory, function) still
  demote. Demotion is permanent: restoring the old values, no-op updates and
  re-upload do not recover it; only a fresh instantiate does.
- The cost is on the GPU timeline, about 0.1 us per demoted node per replay,
  linear in the dirty-node count (365/365 nodes: +13%) and independent of
  kernel duration. Execs of at most 128 nodes are exempt.
- Production decode graphs (365+ nodes, grids unique per batch size) sit in
  the full-penalty regime.

Root cause: graph instantiation adds a constant-prefetch plan to each
kernel's hardware work descriptor (QMD): on a plain kernel chain the node's
CB5 slot points at the successor's constant bank 0 with `VALID=1,
PREFETCH=POST`, so the next kernel's constants are fetched while the current
one runs; on a PDL chain the node's own CB0 carries `PREFETCH=PRE`. The
driver's generic descriptor builder, which rewrites a node's QMD on
`cuGraphExecUpdate`, keeps the addresses and sizes but clears exactly those
bits (CB5 `VALID` on the normal chain, CB0 `PREFETCH` on the PDL chain), and
nothing recomputes the plan after an update. The demoted nodes are the ones
replaying without prefetch; the 128-node exemption is where the plan is not
built in the first place.

Fix: restore the plan after every update. At LOAD each template's pristine
exec is inventoried once: for every kernel node, the QMD's prefetch state and
the identity of its prefetch target (the CB0 of the node it precedes). The
member-switch transaction of section 2 then runs a finish pass after
`cuGraphExecUpdate`, before the upload: it reads the updated descriptors,
checks that each node's successor and CB0 address/size still match the
inventory, re-sets the cleared CB5 `VALID` / CB0 `PRE` bits, and refreshes the
cached snapshot for the next transaction. Dynamic CB0 addresses change with
the parameters, so the snapshot is taken from the updated descriptors, not
frozen at LOAD. A node whose target is missing or ambiguous is rejected
rather than guessed, and an update, guard or upload failure retires that
exec. Validated on 4xH200 with Qwen3.5-122B and DeepSeek-V4-Flash EP4/DP4,
all decode sizes 1-256 without padding: updated execs replay within 0.5% of a
fresh exec, and 8,192 requests per configuration produce token-identical
output against the captured engine. The CPU side of a switch (prepare,
update, finish) stays on the host and is hidden behind the previous forward by
the one-step-ahead scheduler.

## Design

Foundry LOAD keeps one exec per template and switches members with
`cuGraphExecUpdate` plus the descriptor repair; a same-member replay launches
directly. Memory is one exec per template rather than one per member, and
every member replays at the speed of a captured graph.
