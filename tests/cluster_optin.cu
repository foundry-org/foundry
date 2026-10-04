// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foundry project
//
// Driver-level regression for foundry::add_kernel_node_cluster_optin: a restored node of a
// non-portable cluster kernel (16 blocks, as sglang's topk_small_batch_cluster_kernel) must be
// added, instantiated and launched from a function that never got the capture process's
// cudaFuncSetAttribute(NonPortableClusterSizeAllowed). No model, no archive.
//
//   nvcc -std=c++17 -arch=sm_90 -I include tests/cluster_optin.cu -lcuda -o cluster_optin
//   ./cluster_optin    # exit 0 = every case passes, 3 = device cannot run 16-block clusters
#include "ClusterOptIn.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void __cluster_dims__(1, 16, 1) compiled16(int* out) {
  if (threadIdx.x == 0)
    atomicAdd(out, 1);
}
__global__ void __cluster_dims__(1, 16, 1) compiled16_unrecorded(int* out) {
  if (threadIdx.x == 0)
    atomicAdd(out, 1);
}
__global__ void __cluster_dims__(1, 8, 1) compiled8(int* out) {
  if (threadIdx.x == 0)
    atomicAdd(out, 1);
}
__global__ void plain(int* out) {  // cluster from the launch attribute only
  if (threadIdx.x == 0)
    atomicAdd(out, 1);
}

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

static int* d_out;
static CUdevice dev;

static CUfunction func_of(const void* symbol) {
  cudaFunction_t f;
  CHECK_RT(cudaGetFuncBySymbol(&f, symbol));
  return reinterpret_cast<CUfunction>(f);
}

static int non_portable_attr(CUfunction f) {
  int v = -1;
  CHECK_CU(cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, f));
  return v;
}

// Adds one node of `f` (grid y = 16 or 8), optionally sets a launch-attribute cluster, then
// instantiates, launches and checks that every block ran. Returns the add result.
static CUresult run_case(const char* name, CUfunction f, unsigned grid_y, unsigned cx, unsigned cy,
                         unsigned cz, unsigned attr_cluster_y, bool* ok) {
  CUgraph g;
  CHECK_CU(cuGraphCreate(&g, 0));
  CHECK_RT(cudaMemset(d_out, 0, sizeof(int)));
  void* args[] = {&d_out};
  CUDA_KERNEL_NODE_PARAMS p{};
  p.func = f;
  p.gridDimX = 1;
  p.gridDimY = grid_y;
  p.gridDimZ = 1;
  p.blockDimX = 32;
  p.blockDimY = p.blockDimZ = 1;
  p.kernelParams = args;
  CUgraphNode n;
  CUresult r = foundry::add_kernel_node_cluster_optin(&n, g, p, dev, cx, cy, cz, "TEST");
  *ok = r == CUDA_SUCCESS;
  if (*ok && attr_cluster_y) {
    CUkernelNodeAttrValue v{};
    v.clusterDim.x = v.clusterDim.z = 1;
    v.clusterDim.y = attr_cluster_y;
    *ok = cuGraphKernelNodeSetAttribute(n, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &v) ==
          CUDA_SUCCESS;
  }
  int blocks = 0;
  if (*ok) {
    CUgraphExec e;
    *ok = cuGraphInstantiate(&e, g, 0) == CUDA_SUCCESS && cuGraphLaunch(e, 0) == CUDA_SUCCESS &&
          cuStreamSynchronize(0) == CUDA_SUCCESS;
    CHECK_RT(cudaMemcpy(&blocks, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    *ok = *ok && blocks == static_cast<int>(grid_y);
    cuGraphExecDestroy(e);
  }
  std::printf("%-44s add=%d blocks=%d/%u non_portable=%d %s\n", name, (int)r, blocks, grid_y,
              non_portable_attr(f), *ok ? "OK" : "FAIL");
  CHECK_CU(cuGraphDestroy(g));
  return r;
}

int main() {
  CHECK_RT(cudaSetDevice(0));
  CHECK_RT(cudaMalloc(&d_out, sizeof(int)));
  CHECK_CU(cuCtxGetDevice(&dev));
  int major = 0;
  CHECK_RT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  if (major < 9) {
    std::printf("thread-block clusters need sm_90+\n");
    return 3;
  }
  // Can this device run a 16-block cluster at all (with the opt-in)?
  {
    CHECK_RT(cudaFuncSetAttribute(plain, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1, 16, 1);
    cfg.blockDim = dim3(32);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 1;
    attr.val.clusterDim.y = 16;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, plain, &cfg) != cudaSuccess || clusters < 1) {
      std::printf("device cannot run 16-block clusters\n");
      return 3;
    }
    CHECK_RT(cudaFuncSetAttribute(plain, cudaFuncAttributeNonPortableClusterSizeAllowed, 0));
  }

  int failures = 0;
  bool ok = false;
  // 1. Informational: the bare driver call rejects the node, as on LOAD before the fix.
  {
    CUgraph g;
    CHECK_CU(cuGraphCreate(&g, 0));
    void* args[] = {&d_out};
    CUDA_KERNEL_NODE_PARAMS p{};
    p.func = func_of(reinterpret_cast<const void*>(compiled16_unrecorded));
    p.gridDimX = 1;
    p.gridDimY = 16;
    p.gridDimZ = 1;
    p.blockDimX = 32;
    p.blockDimY = p.blockDimZ = 1;
    p.kernelParams = args;
    CUgraphNode n;
    CUresult r = cuGraphAddKernelNode(&n, g, nullptr, 0, &p);
    std::printf("%-44s add=%d (912 = CUDA_ERROR_INVALID_CLUSTER_SIZE)\n",
                "bare cuGraphAddKernelNode, compiled 16", (int)r);
    CHECK_CU(cuGraphDestroy(g));
  }
  // 2. Compiled cluster 16, recorded dims (1,16,1): opted in up front.
  run_case("compiled 16, recorded (1,16,1)", func_of(reinterpret_cast<const void*>(compiled16)), 16,
           1, 16, 1, 0, &ok);
  failures += !ok;
  // 3. Compiled cluster 16, no recorded dims: the retry path.
  run_case("compiled 16, recorded none (retry)",
           func_of(reinterpret_cast<const void*>(compiled16_unrecorded)), 16, 0, 0, 0, 0, &ok);
  failures += !ok;
  // 4. Launch-attribute cluster 16 (no compiled dims): opted in before the attribute is set.
  run_case("launch attr 16, recorded (1,16,1)", func_of(reinterpret_cast<const void*>(plain)), 16,
           1, 16, 1, 16, &ok);
  failures += !ok;
  // 5. Portable cluster: untouched (attribute stays 0).
  CUfunction f8 = func_of(reinterpret_cast<const void*>(compiled8));
  run_case("compiled 8, recorded (1,8,1)", f8, 8, 1, 8, 1, 0, &ok);
  failures += !ok || non_portable_attr(f8) != 0;
  std::printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}
