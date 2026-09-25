# Graph templates and eager per-member execs

How LOAD turns an archive of N captured decode graphs into N launchable
`CUgraphExec`s in a fraction of the time it takes to build (or capture) them
one by one, and why every member gets its own exec.

## The cost being avoided

Rebuilding a captured graph from its node list costs about 37–39 ms per
~1000-node graph on H100: one `cuGraphAddKernelNode` per node, one
`cuGraphAddDependencies_v2` batch per edge record, then `cuGraphInstantiate`.
Node creation dominates; instantiate itself is ~0.9 ms and rewriting the
parameters of every node is ~0.4 ms (`SetParams` on 1024 nodes).

An sglang engine capturing every decode batch size 1..256 has 256 graphs per
rank. Building each from parse costs 256 × 37 ms ≈ 10 s. Native capture of the
same set costs 40–80 s.

## Templates: build the structure once per topology

Graphs captured for different batch sizes of the same model almost always have
the same *structure*: the same kernels in the same order with the same
dependency DAG and the same cluster dimensions, differing only in kernel
parameters (grid sizes, pointers, scalars). SAVE groups graphs by a topology
key (node types + cluster dims, `save_graph_manifest`) and records one
**template** per group plus the per-node parameter sets of every other
**member**. A 256-graph capture typically yields 12–37 groups.

LOAD (`CUDAGraph::start_graph_builds` / `finish_graph_loads`, parallel path in
`csrc/CUDAGraphParallel.cpp`):

0. **Exec-pool prewarm** (sglang: at setup, right after the recorded binaries
   are loaded; `start_exec_pool_prewarm`) — on a background thread, instantiate a
   kernel-only copy of every archived graph (non-kernel nodes become empty nodes,
   nothing is launched), keep the execs until all exist, then destroy them. A
   `CUgraphExec` holds ~3 KB of device memory per node; while earlier execs stay
   alive every instantiate otherwise grows the driver's pool for its own exec,
   which is most of its cost. After the prewarm the Phase 2 instantiates reuse
   that memory (the execs take ~12-18 MB beyond it instead of ~400-600 MB). Phase 2
   never waits for it: a prewarm still running when Phase 2 starts is told to
   stop after its current graph (log: "abandoned, not waited for"). Details in
   [Exec-pool prewarm: cost](#exec-pool-prewarm-cost).
1. **Phase 1** — parse all `.cugraph` binaries on a thread pool (~20 ms for 128).
2. **Phase 2** — a pipeline of three kinds of threads:
   - *prep* (pool): decode each graph's `.cugraph` into its on-demand data
     (params, function handles and attributes, events), templates first;
   - *build* (one thread): each template's `CUgraph` from its binary node table
     (`build_template_graph_binary`: node adds, attributes with
     `build_graph_from_parsed`'s precedence, dependencies), then each member's
     rewrite of its group's builder graph (`rewrite_shared_graph_for_member`);
   - *instantiate* (one thread): every `cuGraphInstantiate`, in the order the
     builds finish. Instantiation does not scale across threads, contexts or
     devices of one process, so one thread is the whole budget; a member's
     rewrite overlaps another group's instantiate.

   Members still get a **dedicated** exec each (`instantiate_member_exec`):
   execs are snapshots, so the template's exec and earlier members' execs are
   untouched; the shared `CUgraph` is only a builder, rewritten in member index
   order per group. Archives without a complete binary (JSON only, or kernel
   attributes the node table cannot hold: `FLAG_COMPLETE_KERNEL_ATTRS` unset)
   build their templates from the JSON inside the instantiate job, as before.
   The Phase 2 log line reports the pipeline (instantiate thread busy / idle,
   build thread waits, device memory the execs took beyond the pool).

Qwen3.5-122B-A10B-FP8 EP4, 128 graphs per rank (12 templates + 116 members),
4xH200, same archive: Phase 2 1.81-1.87 s (before) -> 1.34-1.43 s (binary
templates + pipeline) -> 0.79-0.82 s (+ prewarm); Qwen3-30B-A3B EP4 (14 + 114):
1.48-1.59 -> 1.05-1.14 -> 0.61-0.67 s. The instantiate thread is busy for 90%
(122B) and 96% (30B) of Phase 2 (99% on Inkling-Small-21L EP4, 84 templates +
44 members, Phase 2 0.63-0.65 s). The prewarm's cost is host memory: each
scheduler keeps ~0.4 GB more RSS; device memory at `/health` is unchanged (next
section).

### Exec-pool prewarm: cost

Measured on Qwen3-30B-A3B EP4, 128 graphs (135k nodes) per rank, 4xH200, 2 on/off
LOAD pairs with a 0.2 s sampler (per-process RSS, CPU, per-thread CPU, nvidia-smi
per-process memory) plus 3 earlier pairs:

| | prewarm on | prewarm off |
|---|---|---|
| prewarm thread, per rank | 1.3-2.4 s wall (read 25-50 ms, build 0.2-0.3 s, instantiate 0.9-1.9 s), 1.1-1.4 s CPU (sampled every 0.2 s: up to 0.2 s more) | - |
| device memory | the copies' execs take ~0.4-0.5 GB per rank; `cuGraphExecDestroy` does not return it to the device (the driver keeps it for the process) and Phase 2's execs reuse it | Phase 2's execs take the same ~0.4 GB then |
| device memory at `/health` and after the bench | identical per process (127476 / 127780 / 127780 / 126100 MiB), KV cache identical | |
| scheduler RSS at Phase 2 start | 4.16-4.26 GB | 3.04-3.16 GB |
| scheduler RSS at `/health`, and after the bench | 5.36-5.37 GB, unchanged after | 4.96-4.99 GB, unchanged after |
| Phase 2 | 0.61-0.85 s | 1.05-1.26 s |
| TPOT bs 1 / 8 / 32 / 128 (ms) | 5.13 / 6.29 / 8.03 / 10.55 | 5.14 / 6.29 / 8.04 / 10.55 |

- **Memory:** host RAM only, **+0.4 GB RSS per rank for the life of the process**
  (the driver's host-side allocations for the copies: ~1.1 GB at Phase 2 start, of
  which Phase 2's own execs reuse ~0.7 GB). GPU memory: 0 net.
- **CPU:** one thread per rank for 1.1-1.4 CPU s. It starts after process start
  (launch -> dist init is 26-34 s either way) and overlaps dist init, weight load
  and the KV pool; none of them moved measurably (dist init 1.4-5.2 s on vs
  1.3-5.3 s off, weights 0.25-0.42 vs 0.21-2.55 s; the DeepEP bootstrap's 2.7-8.5 s
  spread dominates `/health`). Fewer cores: with the whole engine pinned to 16 /
  8 logical CPUs (4 / 2 per rank) the prewarm took 1.3-1.7 / 1.7-2.0 s and still
  ended 7.2 / 12.0 s before Phase 2; Phase 2 0.70 / 0.91-1.11 s vs 1.18-1.26 /
  1.64-1.71 s off, `/health` 54.8 vs 52.7 / 56.9 vs 56.9 s (one pair each).
- **Window:** it ended 7.0-12.6 s (30B) / 9.7-10.9 s (122B) before Phase 2 on every
  run; with dummy weights that window is dist init, the KV pool and the pre-capture
  bootstraps, so a real checkpoint load only widens it and a model without DeepEP
  narrows it by ~3 s.
- **Abandoned** when Phase 2's instantiate thread starts before it finished (e.g.
  `FOUNDRY_SGLANG_EARLY_GRAPH_BUILDS=1`): it stops after its current graph, destroys
  what it made, and Phase 2 runs as without it. A failure only costs speed.
- **No knob:** always on for an sglang LOAD (decode and prefill graphs); graphs
  without a valid `.cugraph` are skipped.

Earlier measurement (before the pipeline and the prewarm), Qwen3-30B-A3B EP=4,
256 graphs per rank (37 templates + 219 members):

| LOAD mode | Phase 2 build | sglang decode-graph phase | vs native capture (81 s) |
|---|---:|---:|---:|
| templates, eager execs (default) | 2.8 s | 4.8 s | 17× |
| templates, lazy execs (`FOUNDRY_LAZY_GRAPH_EXEC=1`) | 1.7 s | 3.7 s | 22× |
| no templates (`graph_templates = false` at SAVE) | 10.4 s | 12.5 s | 6.5× |

The gap grows linearly with graph count; at 20–52 graphs templating saves
0.2–1.1 s.

## Why a dedicated exec per member, not one shared exec

The first design kept one `CUgraphExec` per template and switched batch sizes
with `cuGraphExecUpdate`. Two problems, both measured
(`docs/exec-update-penalty.md`):

- After an ExecUpdate on a template with more than ~128 nodes the exec stays
  ~0.6 µs/node slower for every later replay, including replays of the
  template's own batch size: +11–13 % TPOT at bs 2–32.
- The update is on the replay path, so switching batch sizes costs latency at
  serving time.

With dedicated execs there is no ExecUpdate at all. Each member's exec is born
pristine from a graph whose params were set *before* instantiate, so it runs
exactly like a natively captured graph (TPOT within noise once PDL edges are
preserved, see `pdl-edge-batching.md`). `replay()` for a member is a single
`cudaGraphLaunch` of its own exec.

**Eager (default)** materializes every member exec during LOAD Phase 2, so the
first request at any batch size pays nothing extra: 256 execs cost ~1.1 s and
~1 GB of exec memory more than lazy. **Lazy** (`FOUNDRY_LAZY_GRAPH_EXEC=1`)
defers each member's SetParams+instantiate to its first replay (~6 ms once per
batch size; the first decode wave at a new size sees one ITL spike, e.g.
19 ms vs 9 ms max ITL at bs=8). Both modes share the code path; only the call
site of `materialize_on_demand_exec` differs.

## Knobs

| Knob | Where | Effect |
|---|---|---|
| `graph_templates = true/false` | SAVE TOML | group graphs into templates (default) or store every graph in full |
| `FOUNDRY_LAZY_GRAPH_EXEC=1` | LOAD env | defer member instantiation to first replay |
| `FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0` | SAVE env | group graphs that differ only in per-node cluster *dimensions* (deep_gemm picks the cluster size by M): Qwen3-30B-A3B EP2 goes from 26 to 10 templates, Phase 2 2.26 s -> 1.74 s. Members set their own cluster dims before instantiation (`apply_on_demand_updates` neutralises the template's dims first so the params update passes the driver's grid/cluster check). Default keeps the exact dims in the key. |
| `FOUNDRY_MMAP_ARCHIVE` (default on; `0`/`false` disables) | LOAD env | mmap `fatbin_image_packed.img` instead of reading it into memory before `cuLibraryLoadData` (images are loaded with `CU_LIBRARY_BINARY_IS_PRESERVED`, so the mapping stays alive). EP archives carry ~5 GB of sgl-kernel FlashAttention-3 fatbins (sm_80 + sm_86 + sm_90a, of which the driver parses only what it uses): `setup_graph_extension` drops from 3.1 s to 0.16 s on Qwen3-30B-A3B EP2. |
| `FOUNDRY_DEBUG` build | compile flag | logs per-graph edge verification and template/member decisions |
