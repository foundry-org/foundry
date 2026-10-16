# Restored graphs lose PDL edges: libcuda bulk edge-data broadcast (2026-09-05)

**Symptom.** Restored (LOADed) sglang decode graphs ran 1-7% slower per token
than the captured originals (worst on MoE at bs=1), even with no collectives.
Kernel time per step was *lower* on the restored graph while wall time was
higher: the natively captured kernels overlapped their predecessors through
programmatic dependent launch (PDL), the restored ones waited for full
completion (bs=1: summed inter-kernel gap 218 µs vs 473 µs per step).

**Where the data went.** The archive was correct: for the 966-node bs=1 graph
both the JSON and the v2 `.cugraph` hold 865 programmatic edges
(`from_port=1, type=1`). Both rebuild paths passed a correctly filled
`CUgraphEdgeData[]` to `cuGraphAddDependencies_v2`. Yet a `cuGraphGetEdges_v2`
query right after that call returned zero programmatic edges.

**Cause.** libcuda applies `edge_data[0]` to every edge of a bulk
`cuGraphAddDependencies_v2` call instead of reading one record per edge.
Verified with `tests/core/csrc/graph_dependencies.cu --raw` on the 580.126.20 host driver
and on the 595.91.07 and 610.57.04 forward-compat libraries: a mixed array
whose first record is default yields all-default edges; one whose first record
is PDL yields all-PDL edges; uniform arrays are preserved. A rebuilt graph
normally leads with a default edge, so the whole PDL structure was erased.

**Fix.** `include/GraphDependencies.h::add_graph_dependencies` groups the
edges by their full 8-byte record and issues one `cuGraphAddDependencies_v2`
per distinct record (typically two calls: default and programmatic). Each call
still passes a complete array, so the helper also follows the documented
contract on drivers without the bug. Both `CUDAGraph::load` and
`CUDAGraph::build_graph_from_parsed` use it; template members inherit the
template's edges.

**Restore-time cost.** The grouping itself is negligible (a map over ~1500
edges for each of the 12-26 templates, one extra driver call each). What does
cost something is instantiating graphs that now really carry their programmatic
edges: on a 1024-kernel chain `cuGraphInstantiate` takes 0.89 ms with PDL edges
vs 0.75 ms without, identical to a native capture, and `SetParams` on every node
is unchanged (0.36 ms). Over 256 graphs that is ~35 ms. Measured v2ep2 restore
of 256 graphs on the shared host: 1.93 s before the fix, 2.03 s and 2.20 s in
two runs after (run-to-run spread 0.17 s); single GPU, six graphs: 0.123 s
before, 0.126 s after.

**Verification.** With `FOUNDRY_DEBUG` the helper logs
`add_graph_dependencies: N edges, programmatic X in / Y out` after every
insertion and flags a mismatch. Results, restored vs unmodified sglang
(256 decode graphs, median TPOT):

| config | before | after (bs 1 / 8 / 32 / 128) |
|---|---|---|
| single GPU, 30B-A3B-FP8 | +4.0% | 3.99 vs 4.00 ms |
| tp2, Qwen3-32B symm-mem | +1..3% | -0.4 / -0.3 / +0.0 / -0.7% |
| ep2, DeepEP low-latency | +4% | -0.0 / -0.0 / +0.1 / +0.6% |
| v2ep2, DeepEP v2 (NCCL) | +6.4% | +0.0 / +0.0 / +0.2 / +0.2% |

**Checking a new driver.** `pytest tests/core/test_graph_dependencies.py -s` compiles
the reproducer; the second test prints whether the raw single call still
broadcasts. The grouped path is correct either way.
