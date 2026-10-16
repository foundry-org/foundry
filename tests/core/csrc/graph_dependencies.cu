// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foundry project
//
// Driver-level regression for foundry::add_graph_dependencies: mixed default /
// programmatic (PDL) edge records must survive insertion. No model, no archive.
//
//   nvcc -std=c++17 -arch=sm_90 -I include \
//     tests/core/csrc/graph_dependencies.cu -lcuda -o graph_dependencies
//   ./graph_dependencies          # grouped insertion (foundry); exit 0 = every record preserved
//   ./graph_dependencies --raw    # one bulk cuGraphAddDependencies_v2 call; exit 1 = driver
//   broadcasts edge_data[0]
#include "GraphDependencies.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void noop() {}

#define CHECK_RT(x)                                               \
  if (cudaError_t r_ = (x); r_ != cudaSuccess) {                  \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(r_)); \
    std::exit(2);                                                 \
  }
#define CHECK_CU(x)                                         \
  if (CUresult r_ = (x); r_ != CUDA_SUCCESS) {              \
    std::fprintf(stderr, "%s: CUresult %d\n", #x, (int)r_); \
    std::exit(2);                                           \
  }

int main(int argc, char** argv) {
  const bool raw = argc > 1 && std::strcmp(argv[1], "--raw") == 0;
  CHECK_RT(cudaSetDevice(0));
  constexpr size_t kNodes = 8, kEdges = kNodes - 1;
  // Edge patterns along a chain: mixed/default-first, mixed/PDL-first, all default, all PDL.
  const char* names[] = {"mixed default-first", "mixed PDL-first", "uniform default",
                         "uniform PDL"};
  int failures = 0;
  for (int pattern = 0; pattern < 4; ++pattern) {
    cudaGraph_t graph;
    CHECK_RT(cudaGraphCreate(&graph, 0));
    CUgraphNode nodes[kNodes];
    for (size_t i = 0; i < kNodes; ++i) {
      cudaKernelNodeParams params{};
      params.func = reinterpret_cast<void*>(noop);
      params.gridDim = params.blockDim = dim3(1);
      cudaGraphNode_t node;
      CHECK_RT(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
      nodes[i] = reinterpret_cast<CUgraphNode>(node);
    }
    CUgraphEdgeData expected[kEdges]{};
    for (size_t i = 0; i < kEdges; ++i) {
      const bool pdl = pattern == 0 ? i % 2 == 1 : pattern == 1 ? i % 2 == 0 : pattern == 3;
      expected[i].from_port = pdl ? CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC : 0;
      expected[i].type = pdl ? CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC : 0;
    }
    CUgraph cu_graph = reinterpret_cast<CUgraph>(graph);
    if (raw) {
      CHECK_CU(cuGraphAddDependencies_v2(cu_graph, nodes, nodes + 1, expected, kEdges));
    } else {
      CHECK_CU(foundry::add_graph_dependencies(cu_graph, nodes, nodes + 1, expected, kEdges));
    }
    CUgraphNode from[kEdges], to[kEdges];
    CUgraphEdgeData actual[kEdges]{};
    size_t num_edges = kEdges;
    CHECK_CU(cuGraphGetEdges_v2(cu_graph, from, to, actual, &num_edges));
    int mismatches = num_edges != kEdges;
    for (size_t i = 0; i < num_edges; ++i) {
      size_t src = 0;
      while (src < kEdges && nodes[src] != from[i])
        ++src;
      if (src == kEdges || to[i] != nodes[src + 1] ||
          std::memcmp(&actual[i], &expected[src], sizeof(CUgraphEdgeData)) != 0) {
        ++mismatches;
      }
    }
    std::printf("%-7s %-20s edges=%zu mismatches=%d\n", raw ? "raw" : "grouped", names[pattern],
                num_edges, mismatches);
    failures += mismatches;
    CHECK_RT(cudaGraphDestroy(graph));
  }
  return failures ? 1 : 0;
}
