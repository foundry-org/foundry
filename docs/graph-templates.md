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

1. **Phase 1** — parse all `.cugraph` binaries on a thread pool (~30 ms for 256).
2. **Phase 2a** — build each template's `CUgraph` node by node
   (`build_graph_from_parsed`) and instantiate it. This is the only place the
   37 ms/graph cost is paid. The template's full JSON (the `.cugraph` has no
   node list for `build_graph_from_parsed`) is parsed on an async thread one
   template ahead, and its own on-demand data is read from the `.cugraph`.
3. **Phase 2b/2c** — for each member: apply its parameter set to the template's
   `CUgraph` with `cuGraphKernelNodeSetParams` (`apply_on_demand_updates`) and
   `cudaGraphInstantiate` a **dedicated** exec (`materialize_on_demand_exec`).
   Execs are snapshots, so the template's exec and earlier members' execs are
   untouched; the shared `CUgraph` is only a builder. About 6 ms per member.

Measured on Qwen3-30B-A3B EP=4, 256 graphs per rank (37 templates + 219
members):

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
