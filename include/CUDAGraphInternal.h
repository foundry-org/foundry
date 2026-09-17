#pragma once

// Internal header shared between CUDAGraph.cpp and CUDAGraphParallel.cpp.
// Not part of the public API.

#include "CUDAGraph.h"
#include <cuda.h>
#include <cstdio>
#include <future>

namespace foundry {

// Kernels that launch with more than 48 KB of dynamic shared memory need the
// MAX_DYNAMIC_SHARED_SIZE opt-in on the handle the graph API validates
// against. cuGraphAddKernelNode / cuGraphKernelNodeSetParams check the
// per-context CUfunction, which does not inherit a CUkernel attribute, and
// the recorded func_attrs may be smaller than the launch actually used
// (DeepGEMM raises the limit on its own CUfunction right before launching).
// Force the opt-in to at least the node's sharedMemBytes on every handle the
// node may resolve to, and say so when a driver call refuses.
inline void ensure_dynamic_smem_optin(const CUDA_KERNEL_NODE_PARAMS& p, CUdevice dev,
                                      const char* where) {
  if (p.sharedMemBytes <= 48 * 1024) {
    return;
  }
  const int need = static_cast<int>(p.sharedMemBytes);
  if (p.kern != nullptr) {
    CUresult r =
        cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need, p.kern, dev);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry %s] cuKernelSetAttribute(max dyn smem=%d) failed: %d\n", where,
              need, (int)r);
    }
    CUfunction ctx_func = nullptr;
    r = cuKernelGetFunction(&ctx_func, p.kern);
    if (r != CUDA_SUCCESS || ctx_func == nullptr) {
      fprintf(stderr,
              "[foundry %s] cuKernelGetFunction failed (%d): dynamic smem opt-in (%d B) not "
              "applied to the per-context function\n",
              where, (int)r, need);
    } else {
      r = cuFuncSetAttribute(ctx_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
      if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[foundry %s] cuFuncSetAttribute(max dyn smem=%d) failed: %d\n", where,
                need, (int)r);
      }
    }
  }
  if (p.func != nullptr) {
    CUresult r = cuFuncSetAttribute(p.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry %s] cuFuncSetAttribute(max dyn smem=%d) failed: %d\n", where, need,
              (int)r);
    }
  }
}

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
