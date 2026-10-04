#pragma once

// Non-portable cluster sizes for rebuilt kernel nodes (every LOAD path).

#include <cuda.h>
#include <cstdio>
#include <mutex>
#include <unordered_set>

namespace foundry {

// A cluster wider than 8 blocks (a compiled __cluster_dims__ or a cluster-dimension launch
// attribute) needs CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED=1 on the function, or
// cuGraphAddKernelNode / cuGraphKernelNodeSetParams / setting the node's cluster dimension fails
// with CUDA_ERROR_INVALID_CLUSTER_SIZE (912). The capturing process set it (cudaFuncSetAttribute
// before its first launch, e.g. sglang's cluster-16 DSA top-k); the function LOAD resolves from
// the archive starts without it, and that launch never runs. SAVE does not record the attribute:
// it is implied by the recorded cluster dims (docs/graph-templates.md, "Cluster launches").
constexpr unsigned kPortableClusterSize = 8;

namespace cluster_optin_detail {
inline std::mutex& mutex() {
  static std::mutex m;
  return m;
}
// CUkernel / CUfunction handles this process set the attribute on.
inline std::unordered_set<const void*>& done() {
  static std::unordered_set<const void*> s;
  return s;
}
template <typename SetFn>
void set_once_locked(const void* key, SetFn&& set, const char* where) {
  if (done().count(key))
    return;
  CUresult r = set();
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "[foundry %s] setting NON_PORTABLE_CLUSTER_SIZE_ALLOWED=1 failed: %d\n",
                 where, (int)r);
    return;
  }
  done().insert(key);
}
inline unsigned dim(int v) {
  return v > 0 ? static_cast<unsigned>(v) : 0u;
}
}  // namespace cluster_optin_detail

// Sets the attribute on the node's function (once per handle; on a CUkernel's per-context
// function too, which does not inherit the CUkernel's value) when the cluster (cx, cy, cz;
// 0 = 1) exceeds the portable size. Returns whether the cluster is non-portable.
inline bool ensure_cluster_size_optin(const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev,
                                      unsigned cx, unsigned cy, unsigned cz, const char* where) {
  namespace d = cluster_optin_detail;
  const unsigned long long size =
      static_cast<unsigned long long>(cx > 0 ? cx : 1) * (cy > 0 ? cy : 1) * (cz > 0 ? cz : 1);
  if (size <= kPortableClusterSize)
    return false;
  std::lock_guard<std::mutex> lock(d::mutex());
  auto set_func = [where](CUfunction f) {
    d::set_once_locked(
        f,
        [f] {
          return cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
        },
        where);
  };
  if (params.kern != nullptr) {
    CUkernel kern = params.kern;
    d::set_once_locked(
        kern,
        [kern, dev] {
          return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1, kern,
                                      dev);
        },
        where);
    CUfunction ctx_func = nullptr;
    if (cuKernelGetFunction(&ctx_func, kern) == CUDA_SUCCESS && ctx_func != nullptr)
      set_func(ctx_func);
  }
  if (params.func != nullptr)
    set_func(params.func);
  return true;
}

// cuGraphAddKernelNode with the opt-in for the recorded cluster (cx, cy, cz: the node's cluster
// dims merged with the function's compiled ones; 0 = none). On INVALID_CLUSTER_SIZE (an archive
// whose recorded dims miss a compiled cluster) it opts in from the compiled cluster dims the
// driver reports for the function and retries once.
inline CUresult add_kernel_node_cluster_optin(CUgraphNode* node, CUgraph graph,
                                              const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev,
                                              unsigned cx, unsigned cy, unsigned cz,
                                              const char* where) {
  namespace d = cluster_optin_detail;
  ensure_cluster_size_optin(params, dev, cx, cy, cz, where);
  CUresult r = cuGraphAddKernelNode(node, graph, nullptr, 0, &params);
  if (r != CUDA_ERROR_INVALID_CLUSTER_SIZE)
    return r;
  int w = 0, h = 0, z = 0;
  if (params.kern != nullptr) {
    cuKernelGetAttribute(&w, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, params.kern, dev);
    cuKernelGetAttribute(&h, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, params.kern, dev);
    cuKernelGetAttribute(&z, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, params.kern, dev);
  } else if (params.func != nullptr) {
    cuFuncGetAttribute(&w, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, params.func);
    cuFuncGetAttribute(&h, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, params.func);
    cuFuncGetAttribute(&z, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, params.func);
  }
  if (!ensure_cluster_size_optin(params, dev, d::dim(w), d::dim(h), d::dim(z), where))
    return r;
  std::fprintf(stderr,
               "[foundry %s] cuGraphAddKernelNode: INVALID_CLUSTER_SIZE with recorded cluster "
               "(%u,%u,%u); compiled cluster (%d,%d,%d) is non-portable, opted in, retrying\n",
               where, cx, cy, cz, w, h, z);
  return cuGraphAddKernelNode(node, graph, nullptr, 0, &params);
}

}  // namespace foundry
