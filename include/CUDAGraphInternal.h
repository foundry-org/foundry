#pragma once

// Internal header shared between CUDAGraph.cpp and CUDAGraphParallel.cpp.
// Not part of the public API.

#include "CUDAGraph.h"
#include <cuda.h>
#include <future>
#include <variant>

namespace foundry {

// Kernel function attributes across SAVE and LOAD.
//
// SAVE records, per kernel node, the function's MAX_DYNAMIC_SHARED_SIZE_BYTES
// and PREFERRED_SHARED_MEMORY_CARVEOUT attributes as they stand at capture
// (CUDAGraph::save). LOAD reapplies them with apply_saved_function_attributes
// before the node is added. Two driver details shape the LOAD side:
//   * cuGraphAddKernelNode / cuGraphKernelNodeSetParams validate the node's
//     sharedMemBytes against the attribute of the per-context CUfunction at
//     every size, and that function does not inherit a CUkernel attribute, so
//     a CUkernel handle gets the attribute on both.
//   * The attribute is process-wide state shared by every graph (template
//     builds and on-demand workers run concurrently) and is a cap, so it is
//     only ever raised, under a lock, from a per-handle high-water mark.
// ensure_dynamic_smem_optin runs right before a node add / params update and
// raises the cap to the node's own sharedMemBytes; it matters when a node is
// re-targeted to a member kernel whose recorded attribute was never applied.
void apply_saved_function_attributes(const std::variant<CUfunction, CUkernel>& handle, CUdevice dev,
                                     int max_dynamic_shared_size_bytes,
                                     int preferred_shared_memory_carveout, const char* where);
void ensure_dynamic_smem_optin(const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev,
                               const char* where);

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
