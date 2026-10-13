# Graph templates and member execs

How LOAD turns an archive of N captured graphs into N launchable graphs in a
fraction of the time it takes to build or capture them one by one, and the two
ways a member becomes a `CUgraphExec`.

## The cost being avoided

Rebuilding a captured graph from its node list costs about 37-39 ms per
~1000-node graph on H100: one `cuGraphAddKernelNode` per node, one
`cuGraphAddDependencies_v2` batch per edge record, then `cuGraphInstantiate`.
Node creation dominates; instantiate itself is ~0.9 ms and rewriting the
parameters of every node is ~0.4 ms (`SetParams` on 1024 nodes).

An SGLang engine capturing every decode batch size 1..256 has 256 graphs per
rank. Building each from parse costs 256 x 37 ms, about 10 s. Native capture of
the same set costs 40-80 s.

## Templates: build the structure once per topology

Graphs captured for different batch sizes of the same model almost always have
the same structure: the same kernels in the same order with the same
dependency DAG and the same cluster dimensions, differing only in kernel
parameters (grid sizes, pointers, scalars). SAVE groups graphs by a topology
key (node types + cluster dims, `save_graph_manifest`; prefill and decode
graphs are kept in separate groups) and records one **template** per group
plus the per-node parameter sets of every other **member**. A 256-graph
capture typically yields 12-37 groups.

A member never builds its own `CUgraph`. Its parameters are decoded on the CPU
and written onto its group's shared builder graph with `cuGraph*NodeSetParams`;
what happens next depends on the exec mode below.

## LOAD pipeline

`CUDAGraph::start_graph_builds` / `finish_graph_loads`, parallel path in
`csrc/CUDAGraphParallel.cpp`:

0. **Exec-pool prewarm** (SGLang: at setup, right after the recorded binaries
   are loaded; `start_exec_pool_prewarm`). On a background thread, instantiate
   a kernel-only copy of every archived graph (non-kernel nodes become empty
   nodes, nothing is launched), keep the execs until all exist, then destroy
   them. A `CUgraphExec` holds ~3 KB of device memory per node; while earlier
   execs stay alive every instantiate otherwise grows the driver's pool for its
   own exec, which is most of its cost. After the prewarm the Phase 2
   instantiates reuse that memory. Phase 2 never waits for it: a prewarm still
   running when Phase 2 starts is told to stop after its current graph (log:
   "abandoned, not waited for"). Cost in [Exec-pool prewarm: cost](#exec-pool-prewarm-cost).
1. **Phase 1**: parse all `.cugraph` binaries on a thread pool (~20 ms for 128).
2. **Phase 2**: a pipeline of three kinds of threads:
   - *prep* (pool): decode each graph's `.cugraph` into its on-demand data
     (params, function handles and attributes, events), templates first;
   - *build* (one thread): each template's `CUgraph` from its binary node
     table (`build_template_graph_binary`: node adds, attributes with
     `build_graph_from_parsed`'s precedence, dependencies), then each member's
     rewrite of its group's builder graph (`rewrite_shared_graph_for_member`);
   - *instantiate* (one thread): every `cuGraphInstantiate`, in the order the
     builds finish. Instantiation does not scale across threads, contexts or
     devices of one process, so one thread is the whole budget; a member's
     rewrite overlaps another group's instantiate.

   Archives without a complete binary (JSON only, or kernel attributes the
   node table cannot hold: `FLAG_COMPLETE_KERNEL_ATTRS` unset) build their
   templates from the JSON inside the instantiate job. The Phase 2 log line
   reports the pipeline (instantiate thread busy / idle, build thread waits,
   device memory the execs took beyond the pool).

Qwen3.5-122B-A10B-FP8 EP4, 128 graphs per rank (12 templates + 116 members),
4xH200, same archive: Phase 2 1.81-1.87 s (per-graph builds) -> 1.34-1.43 s
(binary templates + pipeline) -> 0.79-0.82 s (+ prewarm); Qwen3-30B-A3B EP4
(14 + 114): 1.48-1.59 -> 1.05-1.14 -> 0.61-0.67 s. The instantiate thread is
busy for 90% (122B) and 96% (30B) of Phase 2. Against native capture of the
same graphs (40-80 s) that is a 50-100x shorter graph phase; at 20-52 graphs
templating still saves 0.2-1.1 s.

## Exec modes

A member replays either from its own exec (eager, default) or from its
template's exec (share, `FOUNDRY_ENABLE_EXEC_SHARE=1`). Both modes share the
templates, the pipeline and the builder graphs. They are selected per LOAD and
can be switched freely between runs of the same archive.

**Eager** (default). In Phase 2 each member's parameters are written onto the
builder graph and a new exec is instantiated from it (~5 ms per ~1000-node
graph), plus the exec-pool prewarm. Execs are snapshots, so the template's
exec and earlier members' execs are untouched. A replay is one
`cudaGraphLaunch` of the member's own exec and runs at the speed of a natively
captured graph (TPOT within noise, PDL edges preserved per
[`pdl-edge-batching.md`](pdl-edge-batching.md)). Memory is one exec per graph,
as in native capture: 256 execs cost ~1.1 s of Phase 2 and ~1 GB of exec memory.

**Share** (`FOUNDRY_ENABLE_EXEC_SHARE=1`). Phase 2 instantiates and uploads
the template execs only: no prewarm, no member exec. Groups are first refined
by their full dependency table and, per kernel node, binary hash (an update
carries a switch to another function of the same binary, not to another binary
of the same function, e.g. a different Triton specialization), and a member
whose cluster dims differ from its
template is rejected. A replay of a different member than the previous one
rewrites the builder graph's parameters, `cuGraphExecUpdate`s the template's
exec and uploads it (host only, no synchronization); a replay of the same
member launches directly. Memory is one exec per template. An updated exec
replays slower unless the driver's update path is repaired
([`exec-update-penalty.md`](exec-update-penalty.md), section 3).

### Exec-pool prewarm: cost

Measured on Qwen3-30B-A3B EP4, 128 graphs (135k nodes) per rank, 4xH200, 2
on/off LOAD pairs with a 0.2 s sampler (per-process RSS, CPU, per-thread CPU,
nvidia-smi per-process memory) plus 3 earlier pairs:

| | prewarm on | prewarm off |
|---|---|---|
| prewarm thread, per rank | 1.3-2.4 s wall (read 25-50 ms, build 0.2-0.3 s, instantiate 0.9-1.9 s), 1.1-1.4 s CPU | - |
| device memory | the copies' execs take ~0.4-0.5 GB per rank; `cuGraphExecDestroy` does not return it to the device (the driver keeps it for the process) and Phase 2's execs reuse it | Phase 2's execs take the same ~0.4-0.5 GB |
| device memory at `/health` and after the bench | identical per process, KV cache identical | |
| scheduler RSS at Phase 2 start | 4.16-4.26 GB | 3.04-3.16 GB |
| scheduler RSS at `/health`, and after the bench | 5.36-5.37 GB, unchanged after | 4.96-4.99 GB, unchanged after |
| Phase 2 | 0.61-0.85 s | 1.05-1.26 s |
| TPOT bs 1 / 8 / 32 / 128 (ms) | 5.13 / 6.29 / 8.03 / 10.55 | 5.14 / 6.29 / 8.04 / 10.55 |

- **Memory:** host RAM only, +0.4 GB RSS per rank for the life of the process
  (the driver's host-side allocations for the copies). GPU memory: 0 net.
- **CPU:** one thread per rank for 1.1-1.4 CPU s, overlapping dist init,
  weight load and the KV pool; none of them moved measurably. With the engine
  pinned to 4 / 2 logical CPUs per rank the prewarm took 1.3-1.7 / 1.7-2.0 s
  and still ended 7-12 s before Phase 2.
- **Window:** it ended 7-13 s before Phase 2 on every run; a real checkpoint
  load only widens it, a model without DeepEP narrows it by ~3 s.
- **Abandoned** when Phase 2's instantiate thread starts before it finished:
  it stops after its current graph, destroys what it made, and Phase 2 runs as
  without it. A failure only costs speed.
- **No knob:** always on for an eager SGLang LOAD (decode and prefill graphs);
  graphs without a valid `.cugraph` are skipped.

## Cluster launches

A kernel node's cluster comes from the function's compiled `__cluster_dims__`
(SAVE records it as `func_attrs.required_cluster_*`) or from a
cluster-dimension launch attribute (`kernel_node_attrs.clusterDim*`); every
builder (JSON, binary template, member update, exec-pool prewarm) merges the
two the same way. A cluster wider than 8 blocks is non-portable: the driver
rejects the node (`cuGraphAddKernelNode`, member `SetParams`, or the cluster
attribute) with `CUDA_ERROR_INVALID_CLUSTER_SIZE` (912) unless the function
has `CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED=1`. The capturing
process sets that itself before its first launch (SGLang main's cluster-16
DSA top-k); the function LOAD resolves from the archive never sees that
launch. `include/ClusterOptIn.h` sets it (once per handle) whenever the merged
cluster exceeds 8 blocks; the attribute is not stored because the dims imply
it, so archives saved before this need no migration. If the recorded dims miss
a compiled cluster, the add retries once after reading the compiled dims from
the driver. Test: `tests/test_cluster_optin.py` (driver-level, no archive).

## Knobs

| Knob | Where | Effect |
|---|---|---|
| `graph_templates = true/false` | SAVE TOML | group graphs into templates (default) or store every graph in full |
| `FOUNDRY_ENABLE_EXEC_SHARE=1` | LOAD env | share mode: template execs only (no prewarm, no member exec), members switched with `cuGraphExecUpdate` at replay; unset/0 = eager |
| `FOUNDRY_EXEC_POOL_PREWARM=0` | LOAD env | ablation: skip the exec-pool prewarm (Phase 2 grows the driver's exec pool itself) |
| `FOUNDRY_PHASE2_PIPELINE=0` | LOAD env | ablation: no build/instantiate overlap, every instantiate runs inline on the build thread (the prep pool stays) |
| `TORCHDYNAMO_DISABLE=1` | SGLang SAVE / LOAD env | SAVE runs each shape's first forward inside the capture and torch >= 2.14's dynamo refuses to trace there; pinned by Foundry on SAVE / LOAD, exported by the recipes in every mode |
| `FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0` | SAVE env | group graphs that differ only in per-node cluster dimensions (deep_gemm picks the cluster size by M): Qwen3-30B-A3B EP2 goes from 26 to 10 templates |
| `FOUNDRY_MMAP_ARCHIVE` (default on; `0` disables) | LOAD env | mmap `fatbin_image_packed.img` instead of reading it into memory before `cuLibraryLoadData` |
| `FOUNDRY_DEBUG` build | compile flag | logs per-graph edge verification and template/member decisions |
