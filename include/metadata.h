#pragma once

#include <vector>
#include <tuple>
#include <variant>
#include <cuda.h>

namespace foundry {

struct __attribute__((visibility("default"))) KernelNodeMetadata {
  struct KernelNodeAttrs {
    bool attr_query_available = false;

    bool has_cluster_dim = false;
    unsigned int clusterDimX = 0;
    unsigned int clusterDimY = 0;
    unsigned int clusterDimZ = 0;

    bool has_preferred_cluster_dim = false;
    unsigned int preferredClusterDimX = 0;
    unsigned int preferredClusterDimY = 0;
    unsigned int preferredClusterDimZ = 0;

    bool has_cluster_scheduling_policy = false;
    int clusterSchedulingPolicyPreference = 0;

    bool has_cooperative = false;
    int cooperative = 0;

    bool has_priority = false;
    int priority = 0;

    bool has_mem_sync_domain = false;
    int memSyncDomain = 0;

    bool has_mem_sync_domain_map = false;
    unsigned char memSyncDomainMapDefault = 0;
    unsigned char memSyncDomainMapRemote = 0;

    bool has_preferred_shared_mem_carveout = false;
    unsigned int preferredSharedMemCarveout = 0;

    bool has_access_policy_window = false;
    void* accessPolicyWindowBasePtr = nullptr;
    size_t accessPolicyWindowNumBytes = 0;
    float accessPolicyWindowHitRatio = 0.0f;
    int accessPolicyWindowHitProp = 0;
    int accessPolicyWindowMissProp = 0;

    bool has_device_updatable = false;
    int deviceUpdatable = 0;
    CUgraphDeviceNode deviceUpdatableNode = nullptr;
  };

  int num_params;
  std::vector<std::tuple<size_t, size_t>> offset_and_sizes;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int gridDimX, gridDimY, gridDimZ;
  CUfunction func;
  CUkernel kern;
  CUcontext ctx;
  unsigned int sharedMemBytes;
  KernelNodeAttrs node_attrs;
  void** kernelParams = nullptr;
  void* extra = nullptr;
  void* argBuffer = nullptr;
  size_t extraSize = 0;
  size_t* argBufferSize = nullptr;

  KernelNodeMetadata() = default;

  KernelNodeMetadata(const KernelNodeMetadata&) = delete;
  KernelNodeMetadata& operator=(const KernelNodeMetadata&) = delete;

  KernelNodeMetadata(KernelNodeMetadata&& other) noexcept
      : num_params(0),
        kernelParams(nullptr),
        extra(nullptr),
        argBuffer(nullptr),
        extraSize(0),
        argBufferSize(nullptr) {
    *this = std::move(other);
  }

  KernelNodeMetadata& operator=(KernelNodeMetadata&& other) noexcept {
    if (this != &other) {
      if (kernelParams) {
        for (int i = 0; i < num_params; ++i) {
          if (kernelParams[i]) {
            free(kernelParams[i]);
          }
        }
        free(kernelParams);
      }
      if (extra) {
        if (argBuffer) {
          free(argBuffer);
        }
        if (argBufferSize) {
          free(argBufferSize);
        }
        free(extra);
      }

      num_params = other.num_params;
      offset_and_sizes = std::move(other.offset_and_sizes);
      blockDimX = other.blockDimX;
      blockDimY = other.blockDimY;
      blockDimZ = other.blockDimZ;
      gridDimX = other.gridDimX;
      gridDimY = other.gridDimY;
      gridDimZ = other.gridDimZ;
      func = other.func;
      kern = other.kern;
      ctx = other.ctx;
      sharedMemBytes = other.sharedMemBytes;
      node_attrs = other.node_attrs;
      kernelParams = other.kernelParams;
      extra = other.extra;
      argBuffer = other.argBuffer;
      extraSize = other.extraSize;
      argBufferSize = other.argBufferSize;

      other.kernelParams = nullptr;
      other.extra = nullptr;
      other.argBuffer = nullptr;
      other.argBufferSize = nullptr;
      other.extraSize = 0;
      other.num_params = 0;
    }
    return *this;
  }

  ~KernelNodeMetadata() {
    if (kernelParams) {
      for (int i = 0; i < num_params; ++i) {
        if (kernelParams[i]) {
          free(kernelParams[i]);
        }
      }
      free(kernelParams);
    }
    if (extra) {
      if (argBuffer) {
        free(argBuffer);
      }
      if (argBufferSize) {
        free(argBufferSize);
      }
      free(extra);
    }
  }
};

struct MemsetNodeMetadata {
  CUdeviceptr dst;
  unsigned int elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
};

struct MemcpyNodeMetadata {
  size_t Depth;
  size_t Height;
  size_t WidthInBytes;
  CUarray dstArray;
  CUdeviceptr dstDevice;
  size_t dstHeight;
  void* dstHost;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  size_t dstPitch;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  void* reserved0;
  void* reserved1;
  CUarray srcArray;
  CUdeviceptr srcDevice;
  size_t srcHeight;
  const void* srcHost;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  size_t srcPitch;
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
};

struct EventRecordNodeMetadata {
  CUevent event;
};

struct EventWaitNodeMetadata {
  CUevent event;
};

struct EmptyNodeMetadata {};

struct GraphDependency {
  int from_index;
  int to_index;
};

using GraphNodeMetadata =
    std::variant<KernelNodeMetadata, MemsetNodeMetadata, MemcpyNodeMetadata,
                 EventRecordNodeMetadata, EventWaitNodeMetadata, EmptyNodeMetadata>;

struct GraphNode {
  size_t index;
  CUgraphNode node;
  GraphNodeMetadata metadata;
};

}  // namespace foundry
