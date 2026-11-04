// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foundry project
//
// Kernel for tests/core/test_cluster_restore.py: each block writes whether it was
// launched with an explicit cluster (%is_explicit_cluster) and its cluster's block count.
// launch_cluster() launches it with clusterDim = (cluster, 1, 1), or without the
// attribute when cluster is 0.

#include <cuda_runtime.h>

__global__ void explicit_cluster_kernel(int* out) {
  unsigned int is_explicit, nctarank;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mov.pred p, %%is_explicit_cluster;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(is_explicit));
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nctarank));
  out[2 * blockIdx.x] = static_cast<int>(is_explicit);
  out[2 * blockIdx.x + 1] = static_cast<int>(nctarank);
}

extern "C" int launch_cluster(int* out, unsigned int blocks, unsigned int cluster, void* stream) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(1);
  config.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = cluster ? 1 : 0;
  return static_cast<int>(cudaLaunchKernelEx(&config, explicit_cluster_kernel, out));
}
