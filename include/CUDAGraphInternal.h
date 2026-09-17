#pragma once

// Internal header shared between CUDAGraph.cpp and CUDAGraphParallel.cpp.
// Not part of the public API.

#include "CUDAGraph.h"
#include <cuda.h>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <future>

namespace foundry {

// Dynamic shared memory opt-in, kept monotonic. cuGraphAddKernelNode /
// cuGraphKernelNodeSetParams validate a node's sharedMemBytes against the
// MAX_DYNAMIC_SHARED_SIZE attribute of the per-context CUfunction (which does
// not inherit a CUkernel attribute) at any size, not only above 48 KB. SAVE
// records each node's own sharedMemBytes as that "attribute", and several
// kernels launch with batch-dependent dynamic smem (FlashMLA's
// get_mla_metadata_kernel: 20*bs+4 bytes; DeepGEMM fp8 GEMMs: 101-215 KB), so
// the template-build thread and the on-demand worker threads used to overwrite
// the shared attribute with *their* node's value and race each other:
// CUDA_ERROR_INVALID_VALUE when a smaller value landed between another
// thread's set and its add. The attribute is a cap, so only ever raise it, and
// serialize the read-modify-write.
inline void raise_dynamic_smem_optin(CUkernel kern, CUfunction func, CUdevice dev, int need,
                                     const char* where) {
  if (need <= 0) {
    return;
  }
  static std::mutex mu;
  static std::unordered_map<const void*, int> high_water;  // per CUkernel / CUfunction handle
  std::lock_guard<std::mutex> lock(mu);
  auto raise_key = [&](const void* key) {
    auto it = high_water.find(key);
    if (it != high_water.end() && it->second >= need) {
      return false;
    }
    high_water[key] = need;
    return true;
  };
  if (kern != nullptr && raise_key((const void*)kern)) {
    CUresult r =
        cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need, kern, dev);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry %s] cuKernelSetAttribute(max dyn smem=%d) failed: %d\n", where,
              need, (int)r);
    }
    CUfunction ctx_func = nullptr;
    r = cuKernelGetFunction(&ctx_func, kern);
    if (r != CUDA_SUCCESS || ctx_func == nullptr) {
      fprintf(stderr,
              "[foundry %s] cuKernelGetFunction failed (%d): dynamic smem opt-in (%d B) not "
              "applied to the per-context function\n",
              where, (int)r, need);
    } else {
      high_water[(const void*)ctx_func] = std::max(high_water[(const void*)ctx_func], need);
      r = cuFuncSetAttribute(ctx_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
      if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[foundry %s] cuFuncSetAttribute(max dyn smem=%d) failed: %d\n", where,
                need, (int)r);
      }
    }
  }
  if (func != nullptr && raise_key((const void*)func)) {
    CUresult r = cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry %s] cuFuncSetAttribute(max dyn smem=%d) failed: %d\n", where, need,
              (int)r);
    }
  }
}

inline void ensure_dynamic_smem_optin(const CUDA_KERNEL_NODE_PARAMS& p, CUdevice dev,
                                      const char* where) {
  raise_dynamic_smem_optin(p.kern, p.func, dev, static_cast<int>(p.sharedMemBytes), where);
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
