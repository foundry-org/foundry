#pragma once

// Internal header shared between CUDAGraph.cpp and CUDAGraphParallel.cpp.
// Not part of the public API.

#include "CUDAGraph.h"
#include <future>

namespace foundry {

// Holds deferred metadata for the split start/finish graph loading flow.
// Returned by start_graph_builds_impl, consumed by finish_graph_loads_impl.
struct PendingGraphLoads {
  struct Entry {
    std::shared_ptr<CUDAGraph> graph;
    boost::json::value allocator_events;     // json::object extracted from root
    boost::json::value output_tensors_meta;  // json::object or null (default)
    boost::json::value generators_meta;      // json::array extracted from root (deferred)
  };
  std::vector<Entry> entries;
  MempoolId_t pool;
  c10::DeviceIndex dev;
  CUDAGeneratorStateRegistry* registry = nullptr;  // for deferred generator registration

  // Signaled when background graph building (Phase 2) completes.
  // finish_graph_loads_impl waits on this before allocator replay.
  std::shared_future<void> build_complete_;
};

// Split load: JSON parse + template build + on-demand prep (synchronous).
// Called by CUDAGraph::start_graph_builds.
std::shared_ptr<PendingGraphLoads> start_graph_builds_impl(
    const std::vector<std::string>& json_paths, MempoolId_t pool, int num_threads,
    CUDAGeneratorStateRegistry& registry);

// Split load: finish with allocator replay + output tensor reconstruction.
// Called by CUDAGraph::finish_graph_loads.
std::vector<GraphLoadResult> finish_graph_loads_impl(std::shared_ptr<PendingGraphLoads> pending,
                                                     ReconstructTensorFn reconstruct_fn);

// Per-entry variant: finish a single graph by index. Idempotent on the
// shared_future wait. Caller is responsible for visiting indices in the
// SAVE-time capture order so VMM cursor advances stay monotonic.
GraphLoadResult finish_one_graph_load_impl(std::shared_ptr<PendingGraphLoads> pending, size_t index,
                                           ReconstructTensorFn reconstruct_fn);

}  // namespace foundry
