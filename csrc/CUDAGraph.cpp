#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cuda.h>
#include <fstream>
#include <boost/json.hpp>
#include <iomanip>
#include <sstream>
#include <vector>
#include "hook.h"
#include "BinaryGraphFormat.h"
#include "GraphDependencies.h"
#include "ExecShare.h"

// #define FOUNDRY_DEBUG  // Verbose logging (see README "Debugging"); same flag as hook.cpp

namespace foundry {

static bool _cuda_graphs_debug = false;

void native_record_begin() {
  foundry::start_hook_record();
}

void native_record_abort() {
  foundry::end_hook_record();
  foundry::clear_hook_events();
}

// ---------------------------------------------------------------------------
// RNG state of an adopted torch capture.
// SAVE (foundry/graph.py): torch.cuda.use_mem_pool(<Foundry RNG pool>) around torch's
// capture puts torch's capture seed/offset tensors (CUDAGeneratorCaptureState) into a
// Foundry-owned caching-allocator pool whose segments are allocated inside record
// windows. The archived kernels read those addresses. LOAD allocates nothing: the
// allocator-event replay maps the segment at the same address (checked), and every
// replay writes (seed, offset) there with stream-ordered 64-bit stores, as torch's
// CUDAGeneratorState::replay_prologue + CUDAGeneratorCaptureState::setup_for_replay do
// (v2.14.1 aten/src/ATen/cuda/CUDAGeneratorImpl.cpp :153-170, :247-257). The values come
// from torch's default generator of the device (philox_state(increment) outside capture:
// current seed and offset, offset advanced by the increment), so loaded graphs and eager
// RNG share one counter as with torch's own replay.
// ---------------------------------------------------------------------------

namespace {
// Stream-ordered 64-bit store (exact for every uint64 value; replaces
// Tensor::fill_(int64), whose stable counterpart takes a double).
inline void write_u64(uint64_t addr, uint64_t value, cudaStream_t stream) {
  FOUNDRY_CU_CHECK(
      cuStreamWriteValue64(reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(addr),
                           static_cast<cuuint64_t>(value), CU_STREAM_WRITE_VALUE_DEFAULT));
}
}  // namespace

void CUDAGraph::register_native_rng(const boost::json::object& gen) {
  const uint64_t id = gen.at("id").to_number<uint64_t>();
  const uint64_t wholegraph_increment = gen.at("wholegraph_increment").to_number<uint64_t>();
  const auto placement = static_cast<RngPlacement>(gen.at("placement").to_number<uint64_t>());
  FOUNDRY_CHECK(placement == RngPlacement::Recorded,
                "graph load: unsupported RNG buffer placement ", static_cast<uint64_t>(placement),
                " (archive of an older build; re-capture it)");
  // A repeated id in one archive registers once (torch: get_capture_state(id, true)
  // returns the existing capture state).
  for (auto& r : native_rng_) {
    if (r.id == id) {
      r.wholegraph_increment = wholegraph_increment;
      return;
    }
  }
  LoadedRngBuffers r;
  r.id = id;
  r.wholegraph_increment = wholegraph_increment;
  // torch allocated the pair in Foundry's RNG pool inside this or an earlier graph's
  // record window; LOAD's allocator-event replay maps it (checked in verify_native_rng).
  r.seed_ptr = gen.at("seed_ptr").to_number<uint64_t>();
  r.offset_ptr = gen.at("offset_ptr").to_number<uint64_t>();
  native_rng_.push_back(std::move(r));
}

void CUDAGraph::verify_native_rng() const {
  for (const auto& r : native_rng_) {
    for (uint64_t addr : {r.seed_ptr, r.offset_ptr}) {
      char hex[32];
      snprintf(hex, sizeof(hex), "0x%lx", (unsigned long)addr);
      FOUNDRY_CHECK(foundry::recorded_allocation_contains(addr, sizeof(uint64_t)),
                    "graph load: the archived RNG ", addr == r.seed_ptr ? "seed" : "offset",
                    " buffer ", hex, " (generator id ", r.id,
                    ") is not inside an allocation Foundry replayed in this process; load the "
                    "graphs whose record windows allocated it (same archive, save order) first");
    }
  }
}

void CUDAGraph::set_native_rng_philox(size_t i, uint64_t seed, uint64_t offset) {
  auto& r = native_rng_.at(i);
  r.seed = seed;
  r.offset = offset;
  r.philox_set = true;
}

void CUDAGraph::native_rng_replay_prologue(cudaStream_t stream) {
  for (auto& r : native_rng_) {
    // = CUDAGeneratorState::replay_prologue + CUDAGeneratorCaptureState::setup_for_replay,
    // with the (seed, offset) the replay binding took from torch's generator.
    if (r.wholegraph_increment == 0) {
      continue;
    }
    FOUNDRY_CHECK(r.philox_set, "replay: no (seed, offset) taken from torch's generator for ",
                  "generator id ", r.id, " (replay a loaded graph through foundry.ops)");
    r.philox_set = false;
    write_u64(r.seed_ptr, r.seed, stream);
    write_u64(r.offset_ptr, r.offset, stream);
  }
}

void CUDAGraph::release_native_rng() {
  // The buffers are Foundry-mapped memory (replayed allocator events), not freed here.
  native_rng_.clear();
}

namespace {
// Every 8-byte-aligned word of the kernel parameters of one kernel node.
template <typename F>
void scan_kernel_node_words(CUgraphNode node, F&& check) {
  CUDA_KERNEL_NODE_PARAMS_v2 p{};
  FOUNDRY_CU_CHECK(cuGraphKernelNodeGetParams_v2(node, &p));
  CUfunction func = p.func;
  if (func == nullptr && p.kern != nullptr) {
    FOUNDRY_CU_CHECK(cuKernelGetFunction(&func, p.kern));
  }
  auto scan_bytes = [&](const void* data, size_t size) {
    const char* b = static_cast<const char*>(data);
    for (size_t w = 0; w + sizeof(uint64_t) <= size; w += sizeof(uint64_t)) {
      uint64_t v;
      std::memcpy(&v, b + w, sizeof(v));
      check(v);
    }
  };
  if (p.kernelParams != nullptr) {
    for (size_t i = 0;; ++i) {
      size_t off = 0, size = 0;
      if (cuFuncGetParamInfo(func, i, &off, &size) != CUDA_SUCCESS)
        break;
      scan_bytes(p.kernelParams[i], size);
    }
  } else if (p.extra != nullptr) {
    void* buf = nullptr;
    size_t* bufsize = nullptr;
    for (size_t i = 0; p.extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
      if (p.extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER)
        buf = p.extra[i + 1];
      if (p.extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE)
        bufsize = static_cast<size_t*>(p.extra[i + 1]);
    }
    if (buf != nullptr && bufsize != nullptr)
      scan_bytes(buf, *bufsize);
  }
}
}  // namespace

// Every 8-byte-aligned word of every kernel parameter, and the device addresses of every
// memcpy/memset node, of a raw graph and its child graphs (driver API only; no assumption
// about the PhiloxCudaState layout): which candidate values occur.
std::unordered_set<uint64_t> scan_raw_graph_words(CUgraph graph,
                                                  const std::unordered_set<uint64_t>& cand) {
  std::unordered_set<uint64_t> found;
  if (cand.empty())
    return found;
  auto check = [&](uint64_t v) {
    if (cand.count(v))
      found.insert(v);
  };
  std::vector<CUgraph> graphs{graph};
  while (!graphs.empty()) {
    CUgraph g = graphs.back();
    graphs.pop_back();
    size_t n = 0;
    FOUNDRY_CU_CHECK(cuGraphGetNodes(g, nullptr, &n));
    std::vector<CUgraphNode> nodes(n);
    FOUNDRY_CU_CHECK(cuGraphGetNodes(g, nodes.data(), &n));
    for (CUgraphNode node : nodes) {
      CUgraphNodeType t;
      FOUNDRY_CU_CHECK(cuGraphNodeGetType(node, &t));
      if (t == CU_GRAPH_NODE_TYPE_KERNEL) {
        scan_kernel_node_words(node, check);
      } else if (t == CU_GRAPH_NODE_TYPE_GRAPH) {
        CUgraph child = nullptr;
        FOUNDRY_CU_CHECK(cuGraphChildGraphNodeGetGraph(node, &child));
        graphs.push_back(child);
      } else if (t == CU_GRAPH_NODE_TYPE_MEMCPY) {
        CUDA_MEMCPY3D m{};
        FOUNDRY_CU_CHECK(cuGraphMemcpyNodeGetParams(node, &m));
        check(static_cast<uint64_t>(m.srcDevice));
        check(static_cast<uint64_t>(m.dstDevice));
      } else if (t == CU_GRAPH_NODE_TYPE_MEMSET) {
        CUDA_MEMSET_NODE_PARAMS m{};
        FOUNDRY_CU_CHECK(cuGraphMemsetNodeGetParams(node, &m));
        check(static_cast<uint64_t>(m.dst));
      }
    }
  }
  return found;
}

bool CUDAGraph::adopt_native_capture(uintptr_t raw_graph, MempoolId_t pool,
                                     const std::vector<NativeGenerator>& generators,
                                     const std::vector<uint64_t>& rng_candidates) {
  FOUNDRY_CHECK(!has_graph_ && !has_graph_exec_ && !capture_ended_ && !adopted_native_,
                "adopt_native_capture: this CUDAGraph already holds a graph");
  FOUNDRY_CHECK(raw_graph != 0, "adopt_native_capture: null cudaGraph_t");
  FOUNDRY_CHECK(generators.size() <= 1,
                "adopt_native_capture: at most one default-generator entry expected");
  // rng_candidates (foundry/graph.py) = blocks of Foundry's RNG pool that may hold this
  // capture's RNG state (created during it, or not held by Foundry and so possibly freed
  // and reused during it) and int64[1] blocks created during it outside the graph pool;
  // generators[0], if any, carries the default generator's pair (philox_state) and
  // increment. Any of them the graph reads other than that pair is RNG state Foundry
  // cannot replay.
  std::vector<NativeGenerator> gens = generators;
  std::unordered_set<uint64_t> cand(rng_candidates.begin(), rng_candidates.end());
  uint64_t dseed = 0, doff = 0;
  if (!gens.empty()) {
    dseed = std::get<4>(gens[0]);
    doff = std::get<5>(gens[0]);
    cand.insert(dseed);
    cand.insert(doff);
  }
  const std::unordered_set<uint64_t> found =
      scan_raw_graph_words(reinterpret_cast<CUgraph>(raw_graph), cand);
  for (uint64_t a : found) {
    if (gens.empty() || (a != dseed && a != doff)) {
      char hex[32];
      snprintf(hex, sizeof(hex), "0x%lx", (unsigned long)a);
      native_record_abort();
      FOUNDRY_CHECK(false, "foundry capture: unsupported: this graph reads ", hex,
                    ", RNG state of the capture (or memory allocated during it on a "
                    "non-capturing stream) that is not the default CUDA generator's seed/offset "
                    "pair: RNG from another torch.Generator (its seed and increment are not "
                    "obtainable with public API) or RNG state created on a thread other than "
                    "the capturing one");
    }
  }
  if (!gens.empty() && !found.count(dseed) && !found.count(doff)) {
    const uint64_t inc = std::get<2>(gens[0]);
    if (inc != 0) {
      native_record_abort();
      FOUNDRY_CHECK(false, "foundry capture: unsupported: the default generator advanced by ", inc,
                    " in this capture but no kernel parameter or copy node holds its seed/offset "
                    "buffer addresses");
    }
    gens.clear();  // no kernel reads the pair: nothing to archive, LOAD does nothing
  }
  // Same window end as torch's capture_end: after its endAllocateToPool.
  foundry::end_hook_record();
  allocator_events_ = foundry::save_hook_events_to_json();
  foundry::clear_hook_events();

  graph_ = reinterpret_cast<cudaGraph_t>(raw_graph);
  has_graph_ = true;
  adopted_native_ = true;
  capture_dev_ = current_device();
  mempool_id_ = pool;
  native_generators_ = gens;

  analyze_captured_graph();
  if (!keep_graph_) {
    // torch owns graph_; forget it.
    graph_ = nullptr;
    has_graph_ = false;
  }
  return !gens.empty();
}

namespace {
// c10::utils::check_env: "0" -> false, "1" -> true, anything else -> unset.
int torch_bool_env(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr)
    return -1;
  if (std::strcmp(v, "0") == 0)
    return 0;
  if (std::strcmp(v, "1") == 0)
    return 1;
  return -1;
}
}  // namespace

// Preallocates what getCurrentCUDABlasHandle(), getCurrentCUDABlasLtHandle() and
// getCUDABlasLtWorkspace() allocate on the current stream. On CUDA the Lt handle is the cuBLAS
// handle (CublasHandlePool.cpp getCurrentCUDABlasLtHandle), so the Lt calls allocate only
// when torch caches a separate Lt workspace per (handle, stream): workspace caching on
// (always on 2.14; TORCH_CUBLAS_WORKSPACE_CACHE=1 on 2.15+) and
// TORCH_CUBLASLT_UNIFIED_WORKSPACE=0. With a unified workspace getCUDABlasLtWorkspace()
// returns the entry getCurrentCUDABlasHandle() just made. Without caching (2.15+ default)
// torch's own Lt ops allocate per op and never read the cached Lt map.
void preallocate_cublas_workspaces(bool torch_has_workspace_cache_flag) {
  const bool caching =
      !torch_has_workspace_cache_flag || torch_bool_env("TORCH_CUBLAS_WORKSPACE_CACHE") == 1;
  const bool separate_lt = caching && torch_bool_env("TORCH_CUBLASLT_UNIFIED_WORKSPACE") == 0;
  FOUNDRY_CHECK(!separate_lt,
                "preallocate_cublas_workspaces: a separate cuBLASLt workspace "
                "(TORCH_CUBLASLT_UNIFIED_WORKSPACE=0 with torch's workspace caching on) is "
                "unsupported in the stable-ABI build of foundry.ops: torch has no stable "
                "shim that allocates it");
  // getCurrentCUDABlasHandle(): the handle, and with caching on the (handle, stream)
  // workspace, which a unified cuBLASLt workspace is.
  void* cublas_handle = nullptr;
  FOUNDRY_SHIM_CHECK(torch_get_current_cuda_blas_handle(&cublas_handle));
  FOUNDRY_CHECK(cublas_handle != nullptr, "Failed to get cuBLAS handle");
}

CUDAGraph::CUDAGraph(bool keep_graph) : keep_graph_(keep_graph) {}

void CUDAGraph::instantiate() {
  FOUNDRY_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    FOUNDRY_CHECK(keep_graph_,
                  "instantiate() is intended to be called by the user only when keep_graph=true");
    FOUNDRY_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
  graph_exec_ = instantiate_graph_exec(graph_);
  has_graph_exec_ = true;
  apply_event_bindings(reinterpret_cast<CUgraphExec>(graph_exec_));
}

cudaGraphExec_t CUDAGraph::instantiate_graph_exec(cudaGraph_t graph) {
  cudaGraphExec_t exec = nullptr;
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  int version = 0;
  FOUNDRY_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, 0);
    if (inst_err != cudaSuccess) {
      fprintf(stderr, "[foundry INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
              inst_err, cudaGetErrorString(inst_err));
      FOUNDRY_CUDA_CHECK(inst_err);
    }
#else
  cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
  if (inst_err != cudaSuccess) {
    fprintf(stderr, "[foundry INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
            inst_err, cudaGetErrorString(inst_err));
    FOUNDRY_CUDA_CHECK(inst_err);
  }
#endif
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  } else {
    cudaError_t inst_err =
        cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (inst_err != cudaSuccess) {
      fprintf(
          stderr,
          "[foundry INSTANTIATE ERROR] cudaGraphInstantiateWithFlags FAILED with error %d: %s\n",
          inst_err, cudaGetErrorString(inst_err));
      FOUNDRY_CUDA_CHECK(inst_err);
    }
  }
#endif
  return exec;
}

std::atomic<uint64_t> CUDAGraph::g_member_update_us{0};
std::atomic<uint64_t> CUDAGraph::g_member_inst_us{0};
std::atomic<uint64_t> CUDAGraph::g_member_attr_calls{0};

void CUDAGraph::apply_on_demand_updates() {
  auto& shared = on_demand_data_->shared_exec;
  for (size_t i = 0; i < on_demand_data_->updates.size(); ++i) {
    auto& u = on_demand_data_->updates[i];
    CUgraphNode node = shared->ordered_nodes[i];
    switch (u.type) {
      case OnDemandNodeUpdate::Kernel: {
        // Update kernel params on the graph node (not exec)
        // A member may launch a different cluster shape than its template (deep_gemm picks the
        // cluster size by M; FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0 puts such graphs in one group).
        // cuGraphKernelNodeSetParams validates the new grid against the node's *current* cluster
        // dims, so neutralise them to 1x1x1 first when they differ; the member's own dims are
        // applied right after the params update below.
        if (u.kernel_attrs.has_cluster_dim) {
          CUkernelNodeAttrValue cur;
          memset(&cur, 0, sizeof(cur));
          const unsigned wx = u.kernel_attrs.clusterDimX > 0 ? u.kernel_attrs.clusterDimX : 1;
          const unsigned wy = u.kernel_attrs.clusterDimY > 0 ? u.kernel_attrs.clusterDimY : 1;
          const unsigned wz = u.kernel_attrs.clusterDimZ > 0 ? u.kernel_attrs.clusterDimZ : 1;
          if (cuGraphKernelNodeGetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION,
                                            &cur) == CUDA_SUCCESS &&
              (cur.clusterDim.x != wx || cur.clusterDim.y != wy || cur.clusterDim.z != wz)) {
            CUkernelNodeAttrValue one;
            memset(&one, 0, sizeof(one));
            one.clusterDim.x = one.clusterDim.y = one.clusterDim.z = 1;
            g_member_attr_calls++;
            cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &one);
          }
        }
        ensure_dynamic_smem_optin(u.kernel_params, capture_dev_, "ON-DEMAND");
        if (u.kernel_attrs.has_cluster_dim)
          ensure_cluster_size_optin(u.kernel_params, capture_dev_, u.kernel_attrs.clusterDimX,
                                    u.kernel_attrs.clusterDimY, u.kernel_attrs.clusterDimZ,
                                    "ON-DEMAND");
        CUresult sp = cuGraphKernelNodeSetParams(node, &u.kernel_params);
        if (sp != CUDA_SUCCESS && u.kernel_params.kern && !u.kernel_params.func) {
          // Re-targeting a node to another CUkernel can be rejected; retry
          // with the kernel's per-context CUfunction instead.
          CUDA_KERNEL_NODE_PARAMS alt = u.kernel_params;
          alt.kern = nullptr;
          if (cuKernelGetFunction(&alt.func, u.kernel_params.kern) == CUDA_SUCCESS && alt.func) {
            ensure_dynamic_smem_optin(alt, capture_dev_, "ON-DEMAND");
            if (u.kernel_attrs.has_cluster_dim)
              ensure_cluster_size_optin(alt, capture_dev_, u.kernel_attrs.clusterDimX,
                                        u.kernel_attrs.clusterDimY, u.kernel_attrs.clusterDimZ,
                                        "ON-DEMAND");
            CUresult sp2 = cuGraphKernelNodeSetParams(node, &alt);
            fprintf(stderr,
                    "[foundry ON-DEMAND] graph %d node %zu: kern-based SetParams failed (%d), "
                    "func-based retry -> %d\n",
                    on_demand_data_->graph_id, i, (int)sp, (int)sp2);
            if (sp2 == CUDA_SUCCESS) {
              u.kernel_params = alt;
              sp = sp2;
            }
          }
        }
        if (sp != CUDA_SUCCESS) {
          const char* err = nullptr;
          cuGetErrorString(sp, &err);
          CUDA_KERNEL_NODE_PARAMS cur{};
          cuGraphKernelNodeGetParams(node, &cur);
          fprintf(stderr,
                  "[foundry ON-DEMAND ERROR] graph %d (%s) node %zu: cuGraphKernelNodeSetParams "
                  "failed: %d (%s). new: func=%p kern=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u "
                  "| template node: func=%p kern=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\n",
                  on_demand_data_->graph_id, on_demand_data_->graph_name.c_str(), i, (int)sp,
                  err ? err : "?", (void*)u.kernel_params.func, (void*)u.kernel_params.kern,
                  u.kernel_params.gridDimX, u.kernel_params.gridDimY, u.kernel_params.gridDimZ,
                  u.kernel_params.blockDimX, u.kernel_params.blockDimY, u.kernel_params.blockDimZ,
                  u.kernel_params.sharedMemBytes, (void*)cur.func, (void*)cur.kern, cur.gridDimX,
                  cur.gridDimY, cur.gridDimZ, cur.blockDimX, cur.blockDimY, cur.blockDimZ,
                  cur.sharedMemBytes);
        }
        FOUNDRY_CU_CHECK(sp);

        // Update kernel node attributes that may differ across batch sizes
        auto& a = u.kernel_attrs;
        if (a.has_cluster_dim) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.clusterDim.x = a.clusterDimX > 0 ? a.clusterDimX : 1;
          attr.clusterDim.y = a.clusterDimY > 0 ? a.clusterDimY : 1;
          attr.clusterDim.z = a.clusterDimZ > 0 ? a.clusterDimZ : 1;
          g_member_attr_calls++;
          FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &attr));
        }
        if (a.has_preferred_cluster_dim) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.preferredClusterDim.x = a.preferredClusterDimX;
          attr.preferredClusterDim.y = a.preferredClusterDimY;
          attr.preferredClusterDim.z = a.preferredClusterDimZ;
          g_member_attr_calls++;
          FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr));
        }
        if (a.clusterSchedulingPolicy >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.clusterSchedulingPolicyPreference =
              static_cast<CUclusterSchedulingPolicy>(a.clusterSchedulingPolicy);
          g_member_attr_calls++;
          FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr));
        }
        if (a.cooperative >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.cooperative = a.cooperative;
          FOUNDRY_CU_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &attr));
        }
        if (a.priority >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.priority = a.priority;
          FOUNDRY_CU_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr));
        }
        if (a.memSyncDomain >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(a.memSyncDomain);
          FOUNDRY_CU_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &attr));
        }
        if (a.has_mem_sync_domain_map) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.memSyncDomainMap.default_ = a.memSyncDomainMapDefault;
          attr.memSyncDomainMap.remote = a.memSyncDomainMapRemote;
          g_member_attr_calls++;
          FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr));
        }
        if (a.has_shared_mem_carveout) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.sharedMemCarveout = a.sharedMemCarveout;
          g_member_attr_calls++;
          FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr));
        }
        break;
      }
      case OnDemandNodeUpdate::Memset:
        FOUNDRY_CU_CHECK(cuGraphMemsetNodeSetParams(node, &u.memset_params));
        break;
      case OnDemandNodeUpdate::Memcpy:
        FOUNDRY_CU_CHECK(cuGraphMemcpyNodeSetParams(node, &u.memcpy_params));
        break;
      case OnDemandNodeUpdate::EventRecord:
        FOUNDRY_CU_CHECK(cuGraphEventRecordNodeSetEvent(node, u.event));
        break;
      case OnDemandNodeUpdate::EventWait:
        FOUNDRY_CU_CHECK(cuGraphEventWaitNodeSetEvent(node, u.event));
        break;
      case OnDemandNodeUpdate::Empty:
        break;
    }
  }
}

CUDAGraph::OnDemandData::~OnDemandData() {
  if (owns_exec && own_exec) {
    FOUNDRY_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(own_exec)));
  }
}

void CUDAGraph::bind_shared_exec() {
  FOUNDRY_CHECK(on_demand_data_ && on_demand_data_->shared_exec,
                "bind_shared_exec: graph is not linked to a shared exec");
  auto& shared = on_demand_data_->shared_exec;
  // Template (the shared exec was instantiated with these params), or share
  // mode: replay switches the shared exec to this graph before launch. An
  // eager member owns the exec Phase 2 instantiated for it instead.
  FOUNDRY_CHECK(shared->share || shared->current_params_id == on_demand_data_->graph_id,
                "bind_shared_exec: eager member ", on_demand_data_->graph_name,
                " cannot launch the template's exec");
  on_demand_data_->own_exec = shared->exec;
  on_demand_data_->owns_exec = false;
}

void CUDAGraph::rewrite_shared_graph_for_member() {
  auto t_upd = std::chrono::steady_clock::now();
  apply_on_demand_updates();
  on_demand_data_->shared_exec->current_params_id = on_demand_data_->graph_id;
  g_member_update_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t_upd)
                            .count();
}

void CUDAGraph::instantiate_member_exec() {
  auto& shared = on_demand_data_->shared_exec;
  auto t_inst = std::chrono::steady_clock::now();
  cudaGraphExec_t exec = nullptr;
  cudaError_t inst_err =
      cudaGraphInstantiate(&exec, reinterpret_cast<cudaGraph_t>(shared->graph), 0);
  auto t_end = std::chrono::steady_clock::now();
  g_member_inst_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_inst).count();
  if (inst_err != cudaSuccess) {
    fprintf(stderr,
            "[foundry ON-DEMAND ERROR] cudaGraphInstantiate FAILED for graph %d (%s): %d (%s)\n",
            on_demand_data_->graph_id, on_demand_data_->graph_name.c_str(), (int)inst_err,
            cudaGetErrorString(inst_err));
    FOUNDRY_CUDA_CHECK(inst_err);
  }
  on_demand_data_->own_exec = reinterpret_cast<CUgraphExec>(exec);
  on_demand_data_->owns_exec = true;
  apply_event_bindings(on_demand_data_->own_exec);
}

void CUDAGraph::replay() {
  // Eager: every graph launches its own exec (the template's is the shared one).
  // Share (FOUNDRY_ENABLE_EXEC_SHARE=1): a replay of another member than the
  // last one rewrites the builder graph to this member, updates the shared exec
  // from it (cuGraphExecUpdate) and uploads it; a repeat of the same member
  // launches directly. No stream or device synchronization.
  if (on_demand_data_) {
    DeviceGuard device_guard(capture_dev_);
    cudaStream_t stream = current_stream(-1);
    auto& shared = *on_demand_data_->shared_exec;
    std::unique_lock<std::mutex> replay_lock(shared.replay_mutex, std::defer_lock);
    if (shared.share) {
      // Held through the launch submission: one member's params on the exec at a time.
      replay_lock.lock();
      FOUNDRY_CHECK(shared.valid, "exec share: an earlier switch of this template's exec failed (",
                    on_demand_data_->graph_name, ")");
      if (shared.current_params_id != on_demand_data_->graph_id) {
        using clk = std::chrono::steady_clock;
        auto ns_since = [](clk::time_point t) {
          return static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count());
        };
        shared.valid = false;
        auto t = clk::now();
        apply_on_demand_updates();
        shared.rewrite_ns += ns_since(t);
        t = clk::now();
        CUgraphExecUpdateResultInfo info{};
        if (cgp_exec_update) {
          // cudagraph-patch: prepare + cuGraphExecUpdate_v2 + finish on a registered exec.
          // Until the patch passes unregistered / inert cases through to the real update,
          // any nonzero return means the exec was not (correctly) updated.
          const int rc = cgp_exec_update(shared.exec, shared.graph, &info);
          shared.update_ns += ns_since(t);
          FOUNDRY_CHECK(rc == 0 && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
                        "exec share: cgp_exec_update failed for ", on_demand_data_->graph_name,
                        " (rc ", rc, ", update result ", static_cast<int>(info.result), ")");
        } else {
          CUresult rc = cuGraphExecUpdate_v2(shared.exec, shared.graph, &info);
          shared.update_ns += ns_since(t);
          FOUNDRY_CHECK(rc == CUDA_SUCCESS && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
                        "exec share: cuGraphExecUpdate failed for ", on_demand_data_->graph_name,
                        " (CUresult ", static_cast<int>(rc), ", update result ",
                        static_cast<int>(info.result), ")");
        }
        t = clk::now();
        FOUNDRY_CU_CHECK(cuGraphUpload(shared.exec, reinterpret_cast<CUstream>(stream)));
        shared.upload_ns += ns_since(t);
        shared.current_params_id = on_demand_data_->graph_id;
        shared.updates++;
        shared.valid = true;
      }
    }
    FOUNDRY_CHECK(on_demand_data_->own_exec, "replay: ", on_demand_data_->graph_name,
                  " has no exec (LOAD Phase 2 did not finish)");
    native_rng_replay_prologue(stream);
    FOUNDRY_CUDA_CHECK(
        cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(on_demand_data_->own_exec), stream));
    return;
  }

#ifdef FOUNDRY_DEBUG
  fprintf(stderr, "[foundry DEBUG] replay: NO on_demand_data_, using direct exec path\n");
#endif

  FOUNDRY_CHECK(capture_ended_ || has_graph_exec_,
                "Called CUDAGraph::replay without a preceding successful capture or load.");

  if (!has_graph_exec_) {
    FOUNDRY_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  DeviceGuard device_guard(capture_dev_);
  cudaStream_t stream = current_stream(-1);

  native_rng_replay_prologue(stream);
  FOUNDRY_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));

  int version = 0;
  FOUNDRY_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    FOUNDRY_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if defined(CUDA_VERSION) || defined(USE_ROCM)
  if (_cuda_graphs_debug || keep_graph_) {
    FOUNDRY_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      FOUNDRY_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      FOUNDRY_CUDA_CHECK_WARN(
          cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose));
      if (!keep_graph_) {
        FOUNDRY_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    FOUNDRY_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
#else
  FOUNDRY_CHECK(false,
                "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  FOUNDRY_CHECK(keep_graph_,
                "You cannot access the raw cudaGraph_t instance unless CUDAGraph was initialized "
                "with keep_graph=true");
  FOUNDRY_CHECK(
      has_graph_,
      "You cannot access the raw cudaGraph_t instance until capture_end() has been called");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  FOUNDRY_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

CUDAGraph::SharedGraphExec::~SharedGraphExec() {
  if (exec) {
    if (share && cgp_exec_forget)
      cgp_exec_forget(exec);  // an exec it never registered returns CGP_E_UNKNOWN_EXEC
    FOUNDRY_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)));
  }
  if (graph) {
    FOUNDRY_CUDA_CHECK_WARN(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph)));
  }
}

void CUDAGraph::instantiate_shared_template(SharedGraphExec& shared) {
  if (!cgp_exec_register_begin || !cgp_exec_register_end) {
    shared.exec = reinterpret_cast<CUgraphExec>(
        instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(shared.graph)));
  } else {
    // cudagraph-patch records the exec's descriptors during this instantiate (same thread).
    // Census of the public graph for cgp_exec_register_expect: CUfunction per kernel
    // node, memcpy nodes, programmatic (PDL) edges.
    std::vector<void*> functions;
    unsigned memcpy_nodes = 0;
    for (CUgraphNode node : shared.ordered_nodes) {
      CUgraphNodeType type;
      FOUNDRY_CU_CHECK(cuGraphNodeGetType(node, &type));
      if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
        CUDA_KERNEL_NODE_PARAMS params{};
        FOUNDRY_CU_CHECK(cuGraphKernelNodeGetParams(node, &params));
        if (!params.func && params.kern)
          FOUNDRY_CU_CHECK(cuKernelGetFunction(&params.func, params.kern));
        FOUNDRY_CHECK(params.func, "exec share: cannot resolve a kernel node's CUfunction");
        functions.push_back(reinterpret_cast<void*>(params.func));
      } else {
        memcpy_nodes += type == CU_GRAPH_NODE_TYPE_MEMCPY;
      }
    }
    size_t count = 0;
    FOUNDRY_CU_CHECK(cuGraphGetEdges_v2(shared.graph, nullptr, nullptr, nullptr, &count));
    std::vector<CUgraphNode> from(count), to(count);
    std::vector<CUgraphEdgeData> edges(count);
    FOUNDRY_CU_CHECK(
        cuGraphGetEdges_v2(shared.graph, from.data(), to.data(), edges.data(), &count));
    unsigned pdl_edges = 0;
    for (const auto& edge : edges)
      pdl_edges += edge.type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
    const unsigned kernels = static_cast<unsigned>(functions.size());
    int rc = cgp_exec_register_begin();
    if (rc)
      fprintf(stderr, "[foundry] exec share: cgp_exec_register_begin failed (%d)\n", rc);
    try {
      if (cgp_exec_register_expect) {
        cgp_census_t census{};
        census.struct_size = sizeof(census);
        census.functions = functions.data();
        census.function_count = kernels;
        census.pdl_edges = pdl_edges;
        census.memcpy_nodes = memcpy_nodes;
        rc = cgp_exec_register_expect(&census);
        if (rc)
          fprintf(stderr, "[foundry] exec share: cgp_exec_register_expect failed (%d)\n", rc);
      }
      shared.exec = reinterpret_cast<CUgraphExec>(
          instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(shared.graph)));
    } catch (...) {
      cgp_exec_register_end(nullptr, 0);  // cancels the scope
      throw;
    }
    rc = cgp_exec_register_end(shared.exec, kernels);
    if (rc)  // the first switch of this exec then fails at cgp_exec_update
      fprintf(stderr, "[foundry] exec share: cgp_exec_register_end failed (%d)\n", rc);
  }
  // Upload before the first update: an update ahead of an exec's first upload
  // demotes every changed node (docs/exec-update-penalty.md).
  FOUNDRY_CU_CHECK(cuGraphUpload(shared.exec, nullptr));
}

std::string CUDAGraph::exec_share_stats() const {
  boost::json::object out;
  if (on_demand_data_ && on_demand_data_->shared_exec) {
    auto& shared = *on_demand_data_->shared_exec;
    std::lock_guard<std::mutex> lock(shared.replay_mutex);
    out["enabled"] = shared.share;
    out["route"] = exec_share::patched_route() ? "cgp" : "cuGraphExecUpdate";
    out["valid"] = shared.valid;
    out["exec"] = reinterpret_cast<uintptr_t>(shared.exec);
    out["graph_id"] = on_demand_data_->graph_id;
    out["current_graph_id"] = shared.current_params_id;
    out["updates"] = shared.updates;
    out["rewrite_ns"] = shared.rewrite_ns;
    out["update_ns"] = shared.update_ns;
    out["upload_ns"] = shared.upload_ns;
  } else {
    out["enabled"] = false;
  }
  return boost::json::serialize(out);
}

void CUDAGraph::transfer_to_shared_exec(std::shared_ptr<SharedGraphExec> shared,
                                        GraphTemplate&& tmpl) {
  shared->exec = reinterpret_cast<CUgraphExec>(graph_exec_);
  shared->graph = reinterpret_cast<CUgraph>(graph_);
  shared->ordered_nodes = std::move(tmpl.ordered_nodes);
  // This graph no longer owns the CUgraph/CUgraphExec
  graph_exec_ = nullptr;
  has_graph_exec_ = false;
  graph_ = nullptr;
  has_graph_ = false;
}

void CUDAGraph::OnDemandNodeUpdate::fixup_pointers() {
  if (type != Kernel)
    return;
  // Fix kernel param pointers
  if (!kernel_param_data.empty()) {
    kernel_param_ptrs.resize(kernel_param_data.size());
    for (size_t j = 0; j < kernel_param_data.size(); ++j)
      kernel_param_ptrs[j] = kernel_param_data[j].data();
    kernel_params.kernelParams = kernel_param_ptrs.data();
  }
  // Fix extra config — rebuild with stable pointers
  if (!arg_buffer.empty()) {
    extra_config.clear();
    extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
    extra_config.push_back(arg_buffer.data());
    extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
    extra_config.push_back(&arg_buffer_size);
    extra_config.push_back(CU_LAUNCH_PARAM_END);
    kernel_params.extra = extra_config.data();
  }
}

void CUDAGraph::reset() {
  // capture_ended_ is only ever set around a loaded graph's instantiate(); the
  // pool of a captured graph belongs to torch's CUDAGraph (adopt_native_capture).
  capture_ended_ = false;
  if (has_graph_ && adopted_native_) {
    // torch's CUDAGraph owns this cudaGraph_t.
    graph_ = nullptr;
    has_graph_ = false;
  }
  if (has_graph_) {
    FOUNDRY_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    FOUNDRY_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

MempoolId_t CUDAGraph::pool() {
  FOUNDRY_CHECK(capture_ended_, "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  release_native_rng();
  reset();

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  if (capture_dev_ != UNDEFINED_DEVICE) {
    FOUNDRY_CUDA_CHECK(cudaSetDevice(capture_dev_));
    FOUNDRY_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}

void CUDAGraph::analyze_captured_graph() {
  FOUNDRY_CHECK(has_graph_, "analyze_captured_graph() called before the graph is captured");

  graph_nodes.clear();
  graph_dependencies.clear();
  attr_query_stats_ = AttrQueryStats{};

  // A failed attribute query below is tolerated (the attribute is not recorded) and counted;
  // save() reports the counts in one line.
  auto note_attr_failure = [this](int attr_id, CUresult res) {
    auto it = attr_query_stats_.failures.find(attr_id);
    if (it == attr_query_stats_.failures.end()) {
      attr_query_stats_.failures.emplace(attr_id, std::make_pair(size_t{1}, res));
    } else {
      it->second.first++;
    }
  };

  CUgraph cuGraph = reinterpret_cast<CUgraph>(graph_);

  size_t numNodes = 0;
  FOUNDRY_CU_CHECK(cuGraphGetNodes(cuGraph, nullptr, &numNodes));

  FOUNDRY_CHECK(numNodes > 0, "Graph contains no nodes");

  std::vector<CUgraphNode> nodes(numNodes);
  FOUNDRY_CU_CHECK(cuGraphGetNodes(cuGraph, nodes.data(), &numNodes));

  for (size_t i = 0; i < numNodes; ++i) {
    CUgraphNodeType nodeType;
    FOUNDRY_CU_CHECK(cuGraphNodeGetType(nodes[i], &nodeType));

    GraphNode graphNode;
    graphNode.index = i;
    graphNode.node = nodes[i];

    if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      FOUNDRY_CU_CHECK(cuGraphKernelNodeGetParams(nodes[i], &params));

      KernelNodeMetadata metadata;
      metadata.blockDimX = params.blockDimX;
      metadata.blockDimY = params.blockDimY;
      metadata.blockDimZ = params.blockDimZ;
      metadata.gridDimX = params.gridDimX;
      metadata.gridDimY = params.gridDimY;
      metadata.gridDimZ = params.gridDimZ;
      metadata.func = params.func;
      metadata.kern = params.kern;
      metadata.ctx = params.ctx;
      metadata.sharedMemBytes = params.sharedMemBytes;

      attr_query_stats_.kernel_nodes++;
      {
        CUkernelNodeAttrValue attr_val;
        CUresult res = cuGraphKernelNodeGetAttribute(
            nodes[i], CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &attr_val);
        if (res == CUDA_SUCCESS) {
          metadata.node_attrs.attr_query_available = true;
          metadata.node_attrs.has_cluster_dim = true;
          metadata.node_attrs.clusterDimX = attr_val.clusterDim.x;
          metadata.node_attrs.clusterDimY = attr_val.clusterDim.y;
          metadata.node_attrs.clusterDimZ = attr_val.clusterDim.z;
        } else {
          // Without the cluster dims LOAD rebuilds the node as a non-cluster launch, and
          // every other attribute below is skipped for it.
          const char* env = std::getenv("FOUNDRY_ALLOW_ATTR_QUERY_FAILURE");
          if (!(env && std::string(env) == "1")) {
            const char* fname = nullptr;
            if (params.kern) {
              cuKernelGetName(&fname, params.kern);
            } else if (params.func) {
              cuFuncGetName(&fname, params.func);
            }
            const char* err = nullptr;
            cuGetErrorName(res, &err);
            FOUNDRY_CHECK(
                false, "[foundry] SAVE: cuGraphKernelNodeGetAttribute(CLUSTER_DIMENSION) ",
                "failed on kernel node ", i, " (", fname ? fname : "<unknown function>",
                "): ", err ? err : "unknown error", " (", static_cast<int>(res),
                "). A graph rebuilt without its cluster dims runs the wrong launch shape ",
                "silently. Set FOUNDRY_ALLOW_ATTR_QUERY_FAILURE=1 to save the node ",
                "without attribute data (attrQueryAvailable=false).");
          }
          metadata.node_attrs.attr_query_available = false;
          attr_query_stats_.unavailable_nodes++;
          note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, res);
        }

        if (metadata.node_attrs.attr_query_available) {
          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_cluster_dim = true;
            metadata.node_attrs.preferredClusterDimX = attr_val.preferredClusterDim.x;
            metadata.node_attrs.preferredClusterDimY = attr_val.preferredClusterDim.y;
            metadata.node_attrs.preferredClusterDimZ = attr_val.preferredClusterDim.z;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, res);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cluster_scheduling_policy = true;
            metadata.node_attrs.clusterSchedulingPolicyPreference =
                static_cast<int>(attr_val.clusterSchedulingPolicyPreference);
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, res);
          }

          res = cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE,
                                              &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cooperative = true;
            metadata.node_attrs.cooperative = attr_val.cooperative;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, res);
          }

          res =
              cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_priority = true;
            metadata.node_attrs.priority = attr_val.priority;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, res);
          }

          res = cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN,
                                              &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain = true;
            metadata.node_attrs.memSyncDomain = attr_val.memSyncDomain;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, res);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain_map = true;
            metadata.node_attrs.memSyncDomainMapDefault = attr_val.memSyncDomainMap.default_;
            metadata.node_attrs.memSyncDomainMapRemote = attr_val.memSyncDomainMap.remote;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, res);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_shared_mem_carveout = true;
            metadata.node_attrs.preferredSharedMemCarveout = attr_val.sharedMemCarveout;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, res);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_access_policy_window = true;
            metadata.node_attrs.accessPolicyWindowBasePtr = attr_val.accessPolicyWindow.base_ptr;
            metadata.node_attrs.accessPolicyWindowNumBytes = attr_val.accessPolicyWindow.num_bytes;
            metadata.node_attrs.accessPolicyWindowHitRatio = attr_val.accessPolicyWindow.hitRatio;
            metadata.node_attrs.accessPolicyWindowHitProp = attr_val.accessPolicyWindow.hitProp;
            metadata.node_attrs.accessPolicyWindowMissProp = attr_val.accessPolicyWindow.missProp;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, res);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_device_updatable = true;
            metadata.node_attrs.deviceUpdatable =
                attr_val.deviceUpdatableKernelNode.deviceUpdatable;
            metadata.node_attrs.deviceUpdatableNode = attr_val.deviceUpdatableKernelNode.devNode;
          } else {
            note_attr_failure(CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, res);
          }

          // PROGRAMMATIC_STREAM_SERIALIZATION is a launch-only attribute (CUlaunchAttributeID:
          // "Valid for launches"); a graph node never reports it. Programmatic dependent
          // launch is carried by the edge data (ports) below instead.
        }
      }

      metadata.num_params = 0;
      size_t offset, size;
      while (cuFuncGetParamInfo(params.func, metadata.num_params, &offset, &size) == CUDA_SUCCESS) {
        metadata.offset_and_sizes.emplace_back(offset, size);
        metadata.num_params++;
      }

      if (params.kernelParams) {
        size_t totalSize = 0;
        for (const auto& [offset, size] : metadata.offset_and_sizes) {
          totalSize += size;
        }

        if (totalSize > 0) {
          metadata.kernelParams = static_cast<void**>(malloc(metadata.num_params * sizeof(void*)));
          for (int j = 0; j < metadata.num_params; ++j) {
            metadata.kernelParams[j] = malloc(std::get<1>(metadata.offset_and_sizes[j]));
            memcpy(metadata.kernelParams[j], params.kernelParams[j],
                   std::get<1>(metadata.offset_and_sizes[j]));
          }
        }
      }

      if (params.extra) {
        void** config = params.extra;
        size_t argBufferSize = 0;
        void* argBufferPtr = nullptr;

        int idx;
        for (idx = 0; config[idx] != CU_LAUNCH_PARAM_END; idx++) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            argBufferPtr = config[idx + 1];
            idx++;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            idx++;
          }
        }
        int configItems = idx + 1;

        metadata.extraSize = configItems * sizeof(void*);
        metadata.extra = malloc(metadata.extraSize);
        void** newConfig = static_cast<void**>(metadata.extra);

        if (argBufferSize > 0) {
          metadata.argBufferSize = static_cast<size_t*>(malloc(sizeof(size_t)));
          *metadata.argBufferSize = argBufferSize;
          metadata.argBuffer = malloc(argBufferSize);
          FOUNDRY_INTERNAL_ASSERT(argBufferPtr != nullptr);
          memcpy(metadata.argBuffer, argBufferPtr, argBufferSize);
        }

        for (int idx = 0; idx < configItems; idx++) {
          if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBuffer;
            idx++;
          } else if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBufferSize;
            idx++;
          } else if (idx == configItems - 1) {
            newConfig[idx] = CU_LAUNCH_PARAM_END;
          } else {
            newConfig[idx] = config[idx];
          }
        }
      }

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      FOUNDRY_CU_CHECK(cuGraphMemsetNodeGetParams(nodes[i], &params));

      MemsetNodeMetadata metadata;
      metadata.dst = params.dst;
      metadata.elementSize = params.elementSize;
      metadata.height = params.height;
      metadata.pitch = params.pitch;
      metadata.value = params.value;
      metadata.width = params.width;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D params;
      memset(&params, 0, sizeof(params));
      FOUNDRY_CU_CHECK(cuGraphMemcpyNodeGetParams(nodes[i], &params));

      MemcpyNodeMetadata metadata;
      metadata.Depth = params.Depth;
      metadata.Height = params.Height;
      metadata.WidthInBytes = params.WidthInBytes;
      metadata.dstArray = params.dstArray;
      metadata.dstDevice = params.dstDevice;
      metadata.dstHeight = params.dstHeight;
      metadata.dstHost = params.dstHost;
      metadata.dstLOD = params.dstLOD;
      metadata.dstMemoryType = params.dstMemoryType;
      metadata.dstPitch = params.dstPitch;
      metadata.dstXInBytes = params.dstXInBytes;
      metadata.dstY = params.dstY;
      metadata.dstZ = params.dstZ;
      metadata.reserved0 = params.reserved0;
      metadata.reserved1 = params.reserved1;
      metadata.srcArray = params.srcArray;
      metadata.srcDevice = params.srcDevice;
      metadata.srcHeight = params.srcHeight;
      metadata.srcHost = params.srcHost;
      metadata.srcLOD = params.srcLOD;
      metadata.srcMemoryType = params.srcMemoryType;
      metadata.srcPitch = params.srcPitch;
      metadata.srcXInBytes = params.srcXInBytes;
      metadata.srcY = params.srcY;
      metadata.srcZ = params.srcZ;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
      CUevent event;
      FOUNDRY_CU_CHECK(cuGraphEventRecordNodeGetEvent(nodes[i], &event));

      EventRecordNodeMetadata metadata;
      metadata.event = event;
      metadata.has_flags = query_event_flags(event, &metadata.flags);

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      CUevent event;
      FOUNDRY_CU_CHECK(cuGraphEventWaitNodeGetEvent(nodes[i], &event));

      EventWaitNodeMetadata metadata;
      metadata.event = event;
      metadata.has_flags = query_event_flags(event, &metadata.flags);

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EMPTY) {
      EmptyNodeMetadata metadata;

      graphNode.metadata = std::move(metadata);
    } else {
      FOUNDRY_CHECK(false, "Graph contains unsupported node type!");
    }

    graph_nodes.push_back(std::move(graphNode));
  }

  size_t numEdges = 0;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  FOUNDRY_CU_CHECK(cuGraphGetEdges(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#else
  FOUNDRY_CU_CHECK(cuGraphGetEdges_v2(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#endif

  if (numEdges > 0) {
    std::vector<CUgraphNode> from_nodes(numEdges);
    std::vector<CUgraphNode> to_nodes(numEdges);
    std::vector<CUgraphEdgeData> edges(numEdges);
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
    FOUNDRY_CU_CHECK(
        cuGraphGetEdges(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
#else
    FOUNDRY_CU_CHECK(
        cuGraphGetEdges_v2(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
#endif
    std::unordered_map<CUgraphNode, int> node_to_index;
    for (size_t i = 0; i < nodes.size(); i++) {
      node_to_index[nodes[i]] = i;
    }

    for (size_t i = 0; i < numEdges; i++) {
      auto from_it = node_to_index.find(from_nodes[i]);
      auto to_it = node_to_index.find(to_nodes[i]);

      if (from_it != node_to_index.end() && to_it != node_to_index.end()) {
        GraphDependency dep;
        dep.from_index = from_it->second;
        dep.to_index = to_it->second;
        dep.from_port = edges[i].from_port;
        dep.to_port = edges[i].to_port;
        dep.edge_type = edges[i].type;
        graph_dependencies.push_back(dep);
      }
    }
  }
}

static boost::json::object serialize_tensor_metadata(const Tensor& tensor) {
  namespace json = boost::json;
  json::object obj;
  AtenTensorHandle h = tensor.handle->get();

  void* data_ptr = nullptr;
  int64_t dim = 0;
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  int64_t numel = 0;
  int32_t dtype = 0, device_type = 0, device_index = 0;
  FOUNDRY_SHIM_CHECK(aoti_torch_get_data_ptr(h, &data_ptr));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_dim(h, &dim));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_sizes(h, &sizes));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_strides(h, &strides));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_numel(h, &numel));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_dtype(h, &dtype));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_device_type(h, &device_type));
  FOUNDRY_SHIM_CHECK(aoti_torch_get_device_index(h, &device_index));

  obj["data_ptr"] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_ptr));

  json::array sizes_arr;
  for (int64_t i = 0; i < dim; ++i) {
    sizes_arr.push_back(sizes[i]);
  }
  obj["sizes"] = sizes_arr;

  json::array strides_arr;
  for (int64_t i = 0; i < dim; ++i) {
    strides_arr.push_back(strides[i]);
  }
  obj["strides"] = strides_arr;

  // The shim's dtype / device-type codes are c10's ScalarType / DeviceType
  // ordinals, the values earlier archives store.
  obj["dtype"] = static_cast<int>(dtype);
  obj["device_type"] = static_cast<int>(device_type);
  obj["device_index"] = static_cast<int>(device_index);
  obj["requires_grad"] = tensor.requires_grad;  // read by the binding from Python
  obj["numel"] = numel;
  obj["element_size"] = static_cast<int64_t>(aoti_torch_dtype_element_size(dtype));

  return obj;
}

static Tensor reconstruct_tensor_from_metadata(const boost::json::object& obj) {
  uintptr_t data_ptr = static_cast<uintptr_t>(obj.at("data_ptr").to_number<uint64_t>());

  const auto& sizes_arr = obj.at("sizes").as_array();
  std::vector<int64_t> sizes;
  for (const auto& s : sizes_arr) {
    sizes.push_back(s.to_number<int64_t>());
  }

  const auto& strides_arr = obj.at("strides").as_array();
  std::vector<int64_t> strides;
  for (const auto& s : strides_arr) {
    strides.push_back(s.to_number<int64_t>());
  }
  FOUNDRY_CHECK(sizes.size() == strides.size(), "output tensor metadata: ", sizes.size(),
                " sizes vs ", strides.size(), " strides");

  const auto dtype = obj.at("dtype").to_number<int32_t>();
  const auto device_type = obj.at("device_type").to_number<int32_t>();
  const auto device_index = obj.at("device_index").to_number<int32_t>();
  const bool requires_grad = obj.at("requires_grad").as_bool();

  // = at::from_blob(ptr, sizes, strides, options): no deleter, storage offset 0.
  AtenTensorHandle h = nullptr;
  FOUNDRY_SHIM_CHECK(aoti_torch_create_tensor_from_blob(
      reinterpret_cast<void*>(data_ptr), static_cast<int64_t>(sizes.size()), sizes.data(),
      strides.data(), 0, dtype, device_type, device_index, &h));
  Tensor t;
  t.handle = std::make_shared<TensorHandle>(h);
  if (requires_grad) {
    FOUNDRY_SHIM_CHECK(torch_set_requires_grad(h, true));
  }
  t.requires_grad = requires_grad;
  return t;
}

void CUDAGraph::save(const std::string& json_path, const OutputTensors& output_tensors,
                     OutputTensorType output_type) {
  FOUNDRY_CHECK(!graph_nodes.empty(), "Graph hasn't been captured yet or has no nodes");
  if (allocator_events_.contains("events")) {
    foundry::note_saved_hook_events(allocator_events_);
  }
  // LOAD maps exactly the driver allocations recorded in the windows of the graphs loaded so
  // far (= saved so far, in this order): the archived RNG pair must be in one of them.
  for (const auto& g : native_generators_) {
    for (uint64_t addr : {std::get<4>(g), std::get<5>(g)}) {
      char hex[32];
      snprintf(hex, sizeof(hex), "0x%lx", (unsigned long)addr);
      FOUNDRY_CHECK(foundry::recorded_allocation_contains(addr, sizeof(uint64_t)),
                    "foundry save: unsupported: the default generator's capture RNG buffer ", hex,
                    " is not inside a driver allocation recorded in the window of this graph or "
                    "of a graph saved before it (save the graph that allocated Foundry's RNG pool "
                    "segment first; LOAD loads in save order)");
    }
  }

  set_pack_fatbins_on_exit(true);

  namespace json = boost::json;
  json::object root;
  json::array nodes_array;

  std::unordered_map<CUevent, int> event_to_id;
  int next_event_id = 0;
  int event_record_nodes = 0, event_wait_nodes = 0, event_nodes_without_flags = 0;

  for (const auto& graphNode : graph_nodes) {
    json::object node_obj;
    node_obj["id"] = graphNode.index;

    if (std::holds_alternative<KernelNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<KernelNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "KernelNode";

      json::object params;
      params["blockDimX"] = metadata.blockDimX;
      params["blockDimY"] = metadata.blockDimY;
      params["blockDimZ"] = metadata.blockDimZ;
      params["gridDimX"] = metadata.gridDimX;
      params["gridDimY"] = metadata.gridDimY;
      params["gridDimZ"] = metadata.gridDimZ;
      params["sharedMemBytes"] = metadata.sharedMemBytes;

      // Only save non-default kernel node attributes to avoid unnecessary
      // cuGraphKernelNodeSetAttribute calls on load (each call acquires
      // the driver context lock).
      {
        json::object kernel_node_attrs;
        kernel_node_attrs["attrQueryAvailable"] = metadata.node_attrs.attr_query_available;
        // cluster_dim: the driver reports 0x0x0 for a launch without clusters. Save any
        // non-zero value, 1x1x1 included: that is an explicit cluster launch, and kernels
        // that use cluster-scoped instructions can trap when rebuilt without it.
        if (metadata.node_attrs.has_cluster_dim &&
            (metadata.node_attrs.clusterDimX > 0 || metadata.node_attrs.clusterDimY > 0 ||
             metadata.node_attrs.clusterDimZ > 0)) {
          kernel_node_attrs["clusterDimX"] = metadata.node_attrs.clusterDimX;
          kernel_node_attrs["clusterDimY"] = metadata.node_attrs.clusterDimY;
          kernel_node_attrs["clusterDimZ"] = metadata.node_attrs.clusterDimZ;
        }
        if (metadata.node_attrs.has_preferred_cluster_dim &&
            (metadata.node_attrs.preferredClusterDimX > 0 ||
             metadata.node_attrs.preferredClusterDimY > 0 ||
             metadata.node_attrs.preferredClusterDimZ > 0)) {
          kernel_node_attrs["preferredClusterDimX"] = metadata.node_attrs.preferredClusterDimX;
          kernel_node_attrs["preferredClusterDimY"] = metadata.node_attrs.preferredClusterDimY;
          kernel_node_attrs["preferredClusterDimZ"] = metadata.node_attrs.preferredClusterDimZ;
        }
        if (metadata.node_attrs.has_cluster_scheduling_policy &&
            metadata.node_attrs.clusterSchedulingPolicyPreference != 0) {
          kernel_node_attrs["clusterSchedulingPolicyPreference"] =
              metadata.node_attrs.clusterSchedulingPolicyPreference;
        }
        if (metadata.node_attrs.has_cooperative && metadata.node_attrs.cooperative != 0) {
          kernel_node_attrs["cooperative"] = metadata.node_attrs.cooperative;
        }
        if (metadata.node_attrs.has_priority && metadata.node_attrs.priority != 0) {
          kernel_node_attrs["priority"] = metadata.node_attrs.priority;
        }
        if (metadata.node_attrs.has_mem_sync_domain && metadata.node_attrs.memSyncDomain != 0) {
          kernel_node_attrs["memSyncDomain"] = metadata.node_attrs.memSyncDomain;
        }
        if (metadata.node_attrs.has_mem_sync_domain_map &&
            (metadata.node_attrs.memSyncDomainMapDefault != 0 ||
             metadata.node_attrs.memSyncDomainMapRemote != 0)) {
          kernel_node_attrs["memSyncDomainMapDefault"] =
              metadata.node_attrs.memSyncDomainMapDefault;
          kernel_node_attrs["memSyncDomainMapRemote"] = metadata.node_attrs.memSyncDomainMapRemote;
        }
        if (metadata.node_attrs.has_preferred_shared_mem_carveout &&
            metadata.node_attrs.preferredSharedMemCarveout != 0) {
          kernel_node_attrs["preferredSharedMemCarveout"] =
              metadata.node_attrs.preferredSharedMemCarveout;
        }
        if (metadata.node_attrs.has_access_policy_window &&
            metadata.node_attrs.accessPolicyWindowNumBytes != 0) {
          kernel_node_attrs["accessPolicyWindowBasePtr"] = static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(metadata.node_attrs.accessPolicyWindowBasePtr));
          kernel_node_attrs["accessPolicyWindowNumBytes"] =
              metadata.node_attrs.accessPolicyWindowNumBytes;
          kernel_node_attrs["accessPolicyWindowHitRatio"] =
              metadata.node_attrs.accessPolicyWindowHitRatio;
          kernel_node_attrs["accessPolicyWindowHitProp"] =
              metadata.node_attrs.accessPolicyWindowHitProp;
          kernel_node_attrs["accessPolicyWindowMissProp"] =
              metadata.node_attrs.accessPolicyWindowMissProp;
        }
        if (metadata.node_attrs.has_device_updatable && metadata.node_attrs.deviceUpdatable != 0) {
          kernel_node_attrs["deviceUpdatable"] = metadata.node_attrs.deviceUpdatable;
          kernel_node_attrs["deviceUpdatableNode"] = static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(metadata.node_attrs.deviceUpdatableNode));
        }
        params["kernel_node_attrs"] = kernel_node_attrs;
      }

      json::array kernel_params_array;
      if (metadata.kernelParams) {
        for (int i = 0; i < metadata.num_params; ++i) {
          FOUNDRY_CHECK(metadata.kernelParams[i], "kernelParams[", i, "] is null");
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);

          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.kernelParams[i]);
          for (size_t j = 0; j < std::get<1>(metadata.offset_and_sizes[i]); ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<int>(bytes[j]);
          }
          param_obj["value_hex"] = hex_stream.str();

          kernel_params_array.push_back(param_obj);
        }
      } else {
        for (int i = 0; i < metadata.num_params; ++i) {
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);
          kernel_params_array.push_back(param_obj);
        }
      }
      params["kernelParams"] = kernel_params_array;

      if (metadata.extra) {
        void** config = static_cast<void**>(metadata.extra);
        json::array extra_array;
        size_t argBufferSize = 0;

        int idx = 0;
        while (config[idx] != CU_LAUNCH_PARAM_END) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_POINTER");
            extra_array.push_back("null");
            idx += 2;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_SIZE");
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            extra_array.push_back(static_cast<uint64_t>(argBufferSize));
            idx += 2;
          } else {
            extra_array.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(config[idx])));
            idx++;
          }
        }
        extra_array.push_back("CU_LAUNCH_PARAM_END");

        params["extra"] = extra_array;

        if (metadata.argBuffer && argBufferSize > 0) {
          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.argBuffer);
          for (size_t j = 0; j < argBufferSize; ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<int>(bytes[j]);
          }
          params["extra_argBuffer_hex"] = hex_stream.str();
        } else {
          params["extra_argBuffer_hex"] = "";
        }
      } else {
        params["extra"] = json::array{};
        params["extra_argBuffer_hex"] = "";
      }

      CUkernel kern = metadata.kern;
      CUfunction func = metadata.func;
      std::string function_name;
      uint64_t binary_hash = 0;

      if (kern != nullptr) {
        const char* name = nullptr;
        if (cuKernelGetName(&name, kern) == CUDA_SUCCESS && name) {
          function_name = name;
        }

        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, kern) == CUDA_SUCCESS && lib != nullptr) {
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        }
      } else if (func != nullptr) {
        CUkernel temp_kern = reinterpret_cast<CUkernel>(func);
        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, temp_kern) == CUDA_SUCCESS && lib != nullptr) {
          const char* name = nullptr;
          if (cuKernelGetName(&name, temp_kern) == CUDA_SUCCESS && name) {
            function_name = name;
          }
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        } else {
          const char* name = nullptr;
          if (cuFuncGetName(&name, func) == CUDA_SUCCESS && name) {
            function_name = name;
          }

          CUmodule mod = nullptr;
          if (cuFuncGetModule(&mod, func) == CUDA_SUCCESS && mod != nullptr) {
            binary_hash = query_binary_hash(mod);
            mark_binary_used(binary_hash);
          }
        }
      }

      params["function_name"] = function_name;
      params["kernel_source_binary_hash"] = binary_hash;

      json::object func_attrs;

      // The function's dynamic-smem cap as it stands at capture. It is a
      // function attribute, not a node property: several kernels launch with
      // batch-dependent dynamic smem (FlashMLA's metadata kernel, DeepGEMM's
      // fp8 GEMMs), and recording this node's own sharedMemBytes here made
      // LOAD lower the shared cap under another graph's node.
      int max_dynamic_shared = 0;
      int preferred_carveout = 0;
      int cluster_scheduling = 0;
      int cluster_width = 0;
      int cluster_height = 0;
      int cluster_depth = 0;
      if (kern != nullptr) {
        cuKernelGetAttribute(&max_dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             kern, capture_dev_);
        cuKernelGetAttribute(&preferred_carveout,
                             CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_scheduling,
                             CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, kern,
                             capture_dev_);
      } else if (func != nullptr) {
        cuFuncGetAttribute(&max_dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           func);
        cuFuncGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                           func);
        cuFuncGetAttribute(&cluster_scheduling,
                           CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, func);
        cuFuncGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, func);
        cuFuncGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, func);
        cuFuncGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, func);
      }
      // The cap cannot be below what this node launched with; a driver that
      // reports 0 for a function without an explicit opt-in is floored there.
      func_attrs["max_dynamic_shared_size_bytes"] =
          std::max(max_dynamic_shared, static_cast<int>(metadata.sharedMemBytes));
      func_attrs["preferred_shared_memory_carveout"] = preferred_carveout;
      func_attrs["cluster_scheduling_policy_preference"] = cluster_scheduling;
      // Save cluster dimensions - these are required when reconstructing the graph for kernels that
      // use clusters
      func_attrs["required_cluster_width"] = cluster_width;
      func_attrs["required_cluster_height"] = cluster_height;
      func_attrs["required_cluster_depth"] = cluster_depth;
      params["func_attrs"] = func_attrs;

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemcpyNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemcpyNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemcpyNode";

      json::object params;
      params["Depth"] = metadata.Depth;
      params["Height"] = metadata.Height;
      params["WidthInBytes"] = metadata.WidthInBytes;

      FOUNDRY_CHECK(metadata.dstArray == nullptr, "dstArray must be null");
      FOUNDRY_CHECK(metadata.srcArray == nullptr, "srcArray must be null");
      FOUNDRY_CHECK(metadata.dstHost == nullptr, "dstHost must be null");
      FOUNDRY_CHECK(metadata.srcHost == nullptr, "srcHost must be null");
      FOUNDRY_CHECK(metadata.reserved0 == nullptr, "reserved0 must be null");
      FOUNDRY_CHECK(metadata.reserved1 == nullptr, "reserved1 must be null");

      params["dstDevice"] = static_cast<uint64_t>(metadata.dstDevice);
      params["dstHeight"] = metadata.dstHeight;
      params["dstLOD"] = metadata.dstLOD;
      params["dstMemoryType"] = static_cast<int>(metadata.dstMemoryType);
      params["dstPitch"] = metadata.dstPitch;
      params["dstXInBytes"] = metadata.dstXInBytes;
      params["dstY"] = metadata.dstY;
      params["dstZ"] = metadata.dstZ;

      params["srcDevice"] = static_cast<uint64_t>(metadata.srcDevice);
      params["srcHeight"] = metadata.srcHeight;
      params["srcLOD"] = metadata.srcLOD;
      params["srcMemoryType"] = static_cast<int>(metadata.srcMemoryType);
      params["srcPitch"] = metadata.srcPitch;
      params["srcXInBytes"] = metadata.srcXInBytes;
      params["srcY"] = metadata.srcY;
      params["srcZ"] = metadata.srcZ;

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemsetNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemsetNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemsetNode";

      json::object params;
      params["dst"] = static_cast<uint64_t>(metadata.dst);
      params["elementSize"] = metadata.elementSize;
      params["height"] = metadata.height;
      params["pitch"] = metadata.pitch;
      params["value"] = metadata.value;
      params["width"] = metadata.width;

      node_obj["params"] = params;

    } else if (std::holds_alternative<EventRecordNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventRecordNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventRecordNode";
      ++event_record_nodes;

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      if (metadata.has_flags) {
        params["event_flags"] = metadata.flags;
      } else {
        ++event_nodes_without_flags;
      }
      node_obj["params"] = params;

    } else if (std::holds_alternative<EventWaitNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventWaitNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventWaitNode";
      ++event_wait_nodes;

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      if (metadata.has_flags) {
        params["event_flags"] = metadata.flags;
      } else {
        ++event_nodes_without_flags;
      }
      node_obj["params"] = params;

    } else if (std::holds_alternative<EmptyNodeMetadata>(graphNode.metadata)) {
      node_obj["type"] = "EmptyNode";
      node_obj["params"] = json::object{};
    }

    nodes_array.push_back(node_obj);
  }

  if (event_record_nodes + event_wait_nodes > 0) {
    fprintf(stderr,
            "[foundry] %s: event nodes: %d record, %d wait (%zu events), %d without recorded "
            "flags (rebuilt as CU_EVENT_DEFAULT)\n",
            json_path.c_str(), event_record_nodes, event_wait_nodes, event_to_id.size(),
            event_nodes_without_flags);
  }

  // Compute topology key = node types + cluster dim values and cluster counts
  // per kernel node.
  // cuGraphExecUpdate does NOT propagate kernel node attribute changes
  // (set via cuGraphKernelNodeSetAttribute) to the CUgraphExec — only
  // CUDA_KERNEL_NODE_PARAMS fields are updated. This means cluster dim
  // changes are silently ignored, causing "cluster misconfiguration" crashes
  // at launch when grid dims no longer match the stale cluster dims.
  // To avoid this, we include full cluster dim values in the topology key
  // so all graphs in a group share identical cluster dims.
  // The cluster COUNT (grid / cluster dims, what %nclusterid reports) is in
  // the key as well: a grid-only exec update of a cluster node leaves the
  // device copy of that constant stale (driver 595.71.05; constant bank 0 is
  // refreshed only when the cluster attribute itself changes), and kernels
  // that read it (FlashInfer's Lamport all-reduce fusion uses it as the token
  // stride) then run with the template's value. Members of one template thus
  // launch every cluster kernel with the same number of clusters; a node
  // without clusters may still change its grid between members.
  // This is computed BEFORE common attr extraction (which removes per-node attrs).
  {
    std::string topology_key;
    topology_key.reserve(nodes_array.size() * 16);
    for (size_t n = 0; n < nodes_array.size(); ++n) {
      if (n > 0)
        topology_key += ',';
      const json::object& no = nodes_array[n].as_object();
      std::string type = no.at("type").as_string().c_str();
      topology_key += type;
      if (type == "KernelNode") {
        unsigned int cdx = 0, cdy = 0, cdz = 0;
        const json::object& p = no.at("params").as_object();
        if (p.contains("kernel_node_attrs")) {
          const json::object& kna = p.at("kernel_node_attrs").as_object();
          if (kna.contains("clusterDimX")) {
            cdx = kna.at("clusterDimX").to_number<unsigned int>();
            cdy = kna.at("clusterDimY").to_number<unsigned int>();
            cdz = kna.at("clusterDimZ").to_number<unsigned int>();
          }
        }
        if (cdx == 0 && p.contains("func_attrs")) {
          const json::object& fa = p.at("func_attrs").as_object();
          if (fa.contains("required_cluster_width"))
            cdx = fa.at("required_cluster_width").to_number<unsigned int>();
          if (fa.contains("required_cluster_height"))
            cdy = fa.at("required_cluster_height").to_number<unsigned int>();
          if (fa.contains("required_cluster_depth"))
            cdz = fa.at("required_cluster_depth").to_number<unsigned int>();
        }
        if (cdx > 0 || cdy > 0 || cdz > 0) {
          // FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0: key only records *whether* a node launches
          // clusters, not the dims or counts; it merges the groups that differ only by
          // deep_gemm's per-M cluster size (EP2: 26 -> ~3 templates). Safe in eager mode (every
          // member instantiates its own exec after apply_on_demand_updates() sets its cluster
          // dims per node); in share mode the members then switch cluster constants through the
          // exec update, which needs a driver-side refresh (cudagraph-patch cluster step).
          // Default keeps the exact dims and counts.
          static const bool cluster_values_in_key = [] {
            const char* e = std::getenv("FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES");
            return !(e && (std::string(e) == "0" || std::string(e) == "false"));
          }();
          if (cluster_values_in_key) {
            topology_key +=
                ":C" + std::to_string(cdx) + "_" + std::to_string(cdy) + "_" + std::to_string(cdz);
            // Cluster count per axis (0 stays 0: no grid recorded).
            const unsigned int gx =
                p.contains("gridDimX") ? p.at("gridDimX").to_number<unsigned int>() : 0;
            const unsigned int gy =
                p.contains("gridDimY") ? p.at("gridDimY").to_number<unsigned int>() : 0;
            const unsigned int gz =
                p.contains("gridDimZ") ? p.at("gridDimZ").to_number<unsigned int>() : 0;
            const unsigned int nx = gx / (cdx > 0 ? cdx : 1);
            const unsigned int ny = gy / (cdy > 0 ? cdy : 1);
            const unsigned int nz = gz / (cdz > 0 ? cdz : 1);
            topology_key +=
                "N" + std::to_string(nx) + "_" + std::to_string(ny) + "_" + std::to_string(nz);
          } else {
            // FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0 also leaves the cluster count out: both
            // the dims and the count then rely on a driver-side refresh of the node's
            // cluster constants after the update (cudagraph-patch cluster step).
            topology_key += ":C";
          }
        } else {
          topology_key += ":0";
        }
      }
    }
    root["topology_key"] = topology_key;
  }

  // Extract common kernel node attributes shared by ALL kernel nodes.
  // If all kernel nodes have the same non-default attrs, store them once at
  // the graph level and remove from individual nodes to avoid redundant
  // cuGraphKernelNodeSetAttribute calls on load.
  {
    bool first_kernel = true;
    json::object common_attrs;
    bool all_same = true;

    for (const auto& node_val : nodes_array) {
      const json::object& node_obj_ref = node_val.as_object();
      if (node_obj_ref.at("type").as_string() != "KernelNode")
        continue;
      const json::object& params_ref = node_obj_ref.at("params").as_object();
      if (!params_ref.contains("kernel_node_attrs"))
        continue;
      const json::object& attrs = params_ref.at("kernel_node_attrs").as_object();

      // Build a comparable version (exclude attrQueryAvailable which is metadata)
      json::object settable_attrs;
      for (const auto& kv : attrs) {
        if (kv.key() != "attrQueryAvailable") {
          settable_attrs[kv.key()] = kv.value();
        }
      }

      if (first_kernel) {
        common_attrs = settable_attrs;
        first_kernel = false;
      } else if (json::serialize(settable_attrs) != json::serialize(common_attrs)) {
        all_same = false;
        break;
      }
    }

    if (!first_kernel && all_same && !common_attrs.empty()) {
      root["common_kernel_node_attrs"] = common_attrs;
      // Remove per-node attrs that are now in the common section
      for (auto& node_val : nodes_array) {
        json::object& node_obj_ref = node_val.as_object();
        if (node_obj_ref.at("type").as_string() != "KernelNode")
          continue;
        json::object& params_ref = node_obj_ref.at("params").as_object();
        if (!params_ref.contains("kernel_node_attrs"))
          continue;
        // Keep only attrQueryAvailable in per-node attrs
        json::object trimmed;
        const json::object& attrs = params_ref.at("kernel_node_attrs").as_object();
        if (attrs.contains("attrQueryAvailable")) {
          trimmed["attrQueryAvailable"] = attrs.at("attrQueryAvailable");
        }
        params_ref["kernel_node_attrs"] = trimmed;
      }
    }
  }

  root["nodes"] = nodes_array;
  if (attr_query_stats_.unavailable_nodes > 0) {
    // LOAD warns from this count (JSON root / .cugraph header) without scanning the nodes.
    root["kernel_nodes_without_attrs"] = attr_query_stats_.unavailable_nodes;
  }

  json::array deps_array;
  for (const auto& dep : graph_dependencies) {
    json::object dep_obj;
    dep_obj["from"] = dep.from_index;
    dep_obj["to"] = dep.to_index;
    // Edge data (ports/type) only when non-default — default edges stay
    // compact and load exactly as before.
    if (dep.from_port != 0) {
      dep_obj["from_port"] = dep.from_port;
    }
    if (dep.to_port != 0) {
      dep_obj["to_port"] = dep.to_port;
    }
    if (dep.edge_type != 0) {
      dep_obj["edge_type"] = dep.edge_type;
    }
    deps_array.push_back(dep_obj);
  }
  root["dependencies"] = deps_array;

  // NativeGenerator entries from the public Python API (Generator.philox_state,
  // torch.cuda.memory_snapshot and the hook record; foundry/graph.py).
  std::vector<NativeGenerator> sorted_generators = native_generators_;

  std::sort(sorted_generators.begin(), sorted_generators.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

  json::array generators_array;
  for (const auto& [state_id, seed, wholegraph_increment, placement, seed_ptr, offset_ptr] :
       sorted_generators) {
    json::object gen_obj;
    gen_obj["id"] = state_id;
    gen_obj["seed"] = seed;
    gen_obj["wholegraph_increment"] = wholegraph_increment;
    gen_obj["placement"] = placement;
    gen_obj["seed_ptr"] = seed_ptr;
    gen_obj["offset_ptr"] = offset_ptr;
    generators_array.push_back(gen_obj);
  }
  root["generators"] = generators_array;

  root["allocator_events"] = allocator_events_;

  json::object output_tensors_obj;
  output_tensors_obj["type"] = static_cast<int>(output_type);

  json::array tensors_array;
  if (std::holds_alternative<Tensor>(output_tensors)) {
    tensors_array.push_back(serialize_tensor_metadata(std::get<Tensor>(output_tensors)));
  } else if (std::holds_alternative<std::vector<Tensor>>(output_tensors)) {
    for (const auto& t : std::get<std::vector<Tensor>>(output_tensors)) {
      tensors_array.push_back(serialize_tensor_metadata(t));
    }
  }
  output_tensors_obj["tensors"] = tensors_array;
  root["output_tensors"] = output_tensors_obj;

  std::ofstream file(json_path);
  FOUNDRY_CHECK(file.is_open(), "Failed to open file for writing: ", json_path);
  file << json::serialize(root);
  file.close();

  // Also write binary format for fast loading
  std::string bin_path = json_path;
  if (bin_path.size() >= 5 && bin_path.substr(bin_path.size() - 5) == ".json") {
    bin_path = bin_path.substr(0, bin_path.size() - 5) + ".cugraph";
  } else {
    bin_path += ".cugraph";
  }
  save_binary(bin_path, root);

  // Kernel-node attribute queries that analyze_captured_graph tolerated: those attributes are
  // not in the archive, so LOAD cannot restore them.
  const AttrQueryStats& aq = attr_query_stats_;
  if (aq.unavailable_nodes > 0 || !aq.failures.empty()) {
    std::ostringstream msg;
    msg << "[foundry] " << json_path << ": attribute queries on " << aq.kernel_nodes
        << " kernel node(s): " << aq.unavailable_nodes
        << " saved without attribute data (attrQueryAvailable=false); failed queries, not "
           "recorded:";
    for (const auto& [attr_id, entry] : aq.failures) {
      const char* err = nullptr;
      cuGetErrorName(entry.second, &err);
      msg << " attr " << attr_id << " x" << entry.first << " (" << (err ? err : "?") << ")";
    }
    fprintf(stderr, "%s\n", msg.str().c_str());
  }
}

size_t saved_attr_query_unavailable(const boost::json::object& root) {
  auto it = root.find("kernel_nodes_without_attrs");
  return it == root.end() ? 0 : static_cast<size_t>(it->value().to_number<uint64_t>());
}

void warn_attr_query_unavailable(const std::string& graph_path, size_t count) {
  if (count == 0)
    return;
  fprintf(stderr,
          "[foundry] %s: %zu kernel node(s) saved without attribute data "
          "(attrQueryAvailable=false): cluster dims and launch attributes were not restored\n",
          graph_path.c_str(), count);
}

// save_binary is implemented in BinaryGraphIO.cpp

GraphLoadResult CUDAGraph::load(const std::string& json_path, MempoolId_t pool) {
  std::ifstream file(json_path);
  FOUNDRY_CHECK(file.is_open(), "Failed to open file for reading: ", json_path);

  std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  namespace json = boost::json;
  json::value root_val = json::parse(json_str);
  const json::object& root = root_val.as_object();
  warn_attr_query_unavailable(json_path, saved_attr_query_unavailable(root));

  auto graph = std::make_shared<CUDAGraph>(true);
  graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();

  graph->capture_dev_ = current_device();
  // A loaded graph's mempool_id_ is only stored: pool() refuses a loaded graph
  // (capture_ended_ is false) and LOAD makes no allocator pool call.
  if (pool.first != 0 || pool.second != 0) {
    FOUNDRY_INTERNAL_ASSERT(!(pool.first && pool.second));
    graph->mempool_id_ = pool;
  }

  // No allocation: the RNG buffers are in memory the event replay below maps.
  for (const auto& gen_val : root.at("generators").as_array()) {
    graph->register_native_rng(gen_val.as_object());
  }

  const json::object& allocator_events = root.at("allocator_events").as_object();
  foundry::replay_hook_events_from_json(allocator_events);
  graph->verify_native_rng();

  CUgraph cuGraph;
  FOUNDRY_CU_CHECK(cuGraphCreate(&cuGraph, 0));
  graph->graph_ = reinterpret_cast<cudaGraph_t>(cuGraph);
  graph->has_graph_ = true;

  const json::array& nodes_array = root.at("nodes").as_array();
  // save() moves kernel node attrs shared by every kernel node to the graph level.
  const json::object* common_kernel_attrs = nullptr;
  if (root.contains("common_kernel_node_attrs"))
    common_kernel_attrs = &root.at("common_kernel_node_attrs").as_object();

  std::unordered_map<int, CUgraphNode> id_to_node;
  std::unordered_map<int, CUevent> event_id_to_event;

  std::vector<std::vector<std::vector<uint8_t>>> all_kernel_params;
  std::vector<std::vector<void*>> all_param_ptrs;
  std::vector<std::vector<void*>> all_extra_configs;
  std::vector<std::vector<uint8_t>> all_arg_buffers;
  std::vector<size_t> all_arg_buffer_sizes;

  CUcontext current_ctx;
  FOUNDRY_CU_CHECK(cuCtxGetCurrent(&current_ctx));

  for (const auto& node_val : nodes_array) {
    const json::object& node_obj = node_val.as_object();
    int node_id = node_obj.at("id").to_number<int>();
    std::string node_type = node_obj.at("type").as_string().c_str();
    const json::object& params = node_obj.at("params").as_object();

    CUgraphNode cuNode = nullptr;

    if (node_type == "KernelNode") {
      CUDA_KERNEL_NODE_PARAMS node_params;
      memset(&node_params, 0, sizeof(node_params));

      node_params.blockDimX = params.at("blockDimX").to_number<unsigned int>();
      node_params.blockDimY = params.at("blockDimY").to_number<unsigned int>();
      node_params.blockDimZ = params.at("blockDimZ").to_number<unsigned int>();
      node_params.gridDimX = params.at("gridDimX").to_number<unsigned int>();
      node_params.gridDimY = params.at("gridDimY").to_number<unsigned int>();
      node_params.gridDimZ = params.at("gridDimZ").to_number<unsigned int>();
      node_params.sharedMemBytes = params.at("sharedMemBytes").to_number<unsigned int>();
      node_params.ctx = current_ctx;

      std::string function_name = params.at("function_name").as_string().c_str();
      uint64_t binary_hash = params.at("kernel_source_binary_hash").to_number<uint64_t>();

      auto func_handle_variant = query_function_handle(binary_hash, function_name);
      if (std::holds_alternative<CUkernel>(func_handle_variant)) {
        CUkernel kern = std::get<CUkernel>(func_handle_variant);
        node_params.kern = kern;
      } else {
        CUfunction func = std::get<CUfunction>(func_handle_variant);
        node_params.func = func;
      }

      int cluster_width = 0, cluster_height = 0, cluster_depth = 0;
      bool has_preferred_cluster_dim = false;
      unsigned int preferred_cluster_x = 0;
      unsigned int preferred_cluster_y = 0;
      unsigned int preferred_cluster_z = 0;
      bool has_cluster_scheduling = false;
      int cluster_scheduling = 0;
      bool has_cooperative = false;
      int cooperative = 0;
      bool has_priority = false;
      int priority = 0;
      bool has_mem_sync_domain = false;
      int mem_sync_domain = 0;
      bool has_mem_sync_domain_map = false;
      unsigned char mem_sync_domain_default = 0;
      unsigned char mem_sync_domain_remote = 0;
      bool has_preferred_shared_mem_carveout = false;
      unsigned int preferred_shared_mem_carveout = 0;
      bool has_access_policy_window = false;
      uint64_t access_policy_base_ptr = 0;
      size_t access_policy_num_bytes = 0;
      float access_policy_hit_ratio = 0.0f;
      int access_policy_hit_prop = 0;
      int access_policy_miss_prop = 0;
      bool has_device_updatable = false;
      int device_updatable = 0;

      if (params.contains("kernel_node_attrs") || common_kernel_attrs) {
        // With common attrs, save() leaves only attrQueryAvailable on the node, so the
        // two sets never hold the same attr and the merge order does not matter.
        json::object merged_attrs;
        if (common_kernel_attrs) {
          merged_attrs = *common_kernel_attrs;
          if (params.contains("kernel_node_attrs")) {
            for (const auto& kv : params.at("kernel_node_attrs").as_object())
              merged_attrs[kv.key()] = kv.value();
          }
        }
        const json::object& node_attrs =
            common_kernel_attrs ? merged_attrs : params.at("kernel_node_attrs").as_object();
        if (node_attrs.contains("clusterDimX")) {
          cluster_width = node_attrs.at("clusterDimX").to_number<int>();
          cluster_height = node_attrs.at("clusterDimY").to_number<int>();
          cluster_depth = node_attrs.at("clusterDimZ").to_number<int>();
        }
        if (node_attrs.contains("preferredClusterDimX")) {
          has_preferred_cluster_dim = true;
          preferred_cluster_x = node_attrs.at("preferredClusterDimX").to_number<unsigned int>();
          preferred_cluster_y = node_attrs.at("preferredClusterDimY").to_number<unsigned int>();
          preferred_cluster_z = node_attrs.at("preferredClusterDimZ").to_number<unsigned int>();
        }
        if (node_attrs.contains("clusterSchedulingPolicyPreference")) {
          has_cluster_scheduling = true;
          cluster_scheduling = node_attrs.at("clusterSchedulingPolicyPreference").to_number<int>();
        }
        if (node_attrs.contains("cooperative")) {
          has_cooperative = true;
          cooperative = node_attrs.at("cooperative").to_number<int>();
        }
        if (node_attrs.contains("priority")) {
          has_priority = true;
          priority = node_attrs.at("priority").to_number<int>();
        }
        if (node_attrs.contains("memSyncDomain")) {
          has_mem_sync_domain = true;
          mem_sync_domain = node_attrs.at("memSyncDomain").to_number<int>();
        }
        if (node_attrs.contains("memSyncDomainMapDefault")) {
          has_mem_sync_domain_map = true;
          mem_sync_domain_default =
              static_cast<unsigned char>(node_attrs.at("memSyncDomainMapDefault").to_number<int>());
          mem_sync_domain_remote =
              static_cast<unsigned char>(node_attrs.at("memSyncDomainMapRemote").to_number<int>());
        }
        if (node_attrs.contains("preferredSharedMemCarveout")) {
          has_preferred_shared_mem_carveout = true;
          preferred_shared_mem_carveout =
              node_attrs.at("preferredSharedMemCarveout").to_number<unsigned int>();
        }
        if (node_attrs.contains("accessPolicyWindowNumBytes")) {
          has_access_policy_window = true;
          access_policy_base_ptr = node_attrs.at("accessPolicyWindowBasePtr").to_number<uint64_t>();
          access_policy_num_bytes = node_attrs.at("accessPolicyWindowNumBytes").to_number<size_t>();
          access_policy_hit_ratio = node_attrs.at("accessPolicyWindowHitRatio").to_number<float>();
          access_policy_hit_prop = node_attrs.at("accessPolicyWindowHitProp").to_number<int>();
          access_policy_miss_prop = node_attrs.at("accessPolicyWindowMissProp").to_number<int>();
        }
        if (node_attrs.contains("deviceUpdatable")) {
          has_device_updatable = true;
          device_updatable = node_attrs.at("deviceUpdatable").to_number<int>();
        }
      }

      if (params.contains("func_attrs")) {
        const json::object& func_attrs = params.at("func_attrs").as_object();
        int max_shared = func_attrs.at("max_dynamic_shared_size_bytes").to_number<int>();
        int preferred_carveout = func_attrs.at("preferred_shared_memory_carveout").to_number<int>();
        // int cluster_scheduling =
        // func_attrs.at("cluster_scheduling_policy_preference").to_number<int>();

        // Read cluster dimensions from func_attrs - override name-based detection if non-zero
        if (func_attrs.contains("required_cluster_width")) {
          int attr_width = func_attrs.at("required_cluster_width").to_number<int>();
          if (attr_width > 0)
            cluster_width = attr_width;
        }
        if (func_attrs.contains("required_cluster_height")) {
          int attr_height = func_attrs.at("required_cluster_height").to_number<int>();
          if (attr_height > 0)
            cluster_height = attr_height;
        }
        if (func_attrs.contains("required_cluster_depth")) {
          int attr_depth = func_attrs.at("required_cluster_depth").to_number<int>();
          if (attr_depth > 0)
            cluster_depth = attr_depth;
        }

        apply_saved_function_attributes(func_handle_variant, graph->capture_dev_, max_shared,
                                        preferred_carveout, "LOAD");
        // NOTE: CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_{WIDTH,HEIGHT,DEPTH} are not set: a compiled
        // cluster cannot be changed at runtime. NON_PORTABLE_CLUSTER_SIZE_ALLOWED can, and is set
        // by add_restored_kernel_node when the cluster exceeds the portable size.
      }

      const json::array& kernel_params_array = params.at("kernelParams").as_array();
      int num_params = kernel_params_array.size();

      bool has_kernel_params = false;
      if (num_params > 0) {
        const json::object& first_param = kernel_params_array[0].as_object();
        has_kernel_params = first_param.contains("value_hex");
      }

      if (has_kernel_params) {
        all_kernel_params.emplace_back(num_params);
        all_param_ptrs.emplace_back(num_params);
        auto& param_data = all_kernel_params.back();
        auto& param_ptrs = all_param_ptrs.back();

        for (size_t i = 0; i < kernel_params_array.size(); ++i) {
          const json::object& param_obj = kernel_params_array[i].as_object();
          size_t param_size = param_obj.at("size").to_number<size_t>();
          std::string value_hex = param_obj.at("value_hex").as_string().c_str();

          param_data[i].resize(param_size);
          for (size_t j = 0; j < param_size; ++j) {
            std::string byte_str = value_hex.substr(j * 2, 2);
            param_data[i][j] = std::stoul(byte_str, nullptr, 16);
          }
          param_ptrs[i] = param_data[i].data();
        }
        node_params.kernelParams = param_ptrs.data();
      }

      const json::array& extra_array = params.at("extra").as_array();

      if (!extra_array.empty()) {
        all_extra_configs.emplace_back();
        auto& extra_config = all_extra_configs.back();

        size_t arg_buffer_idx = all_arg_buffers.size();
        bool has_arg_buffer = false;

        for (size_t i = 0; i < extra_array.size(); ++i) {
          if (extra_array[i].is_string()) {
            std::string str_val = extra_array[i].as_string().c_str();
            if (str_val == "CU_LAUNCH_PARAM_BUFFER_POINTER") {
              extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
            } else if (str_val == "CU_LAUNCH_PARAM_BUFFER_SIZE") {
              extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
            } else if (str_val == "CU_LAUNCH_PARAM_END") {
              extra_config.push_back(CU_LAUNCH_PARAM_END);
            } else if (str_val == "null") {
              FOUNDRY_CHECK(!has_arg_buffer,
                            "Encountered multiple 'null' entries in extra field, but only one is "
                            "expected for CU_LAUNCH_PARAM_BUFFER_POINTER");
              std::string argBuffer_hex = params.at("extra_argBuffer_hex").as_string().c_str();
              if (!argBuffer_hex.empty()) {
                all_arg_buffers.emplace_back();
                auto& arg_buffer = all_arg_buffers.back();
                arg_buffer.resize(argBuffer_hex.length() / 2);
                for (size_t j = 0; j < arg_buffer.size(); ++j) {
                  std::string byte_str = argBuffer_hex.substr(j * 2, 2);
                  arg_buffer[j] = std::stoul(byte_str, nullptr, 16);
                }
                has_arg_buffer = true;
                extra_config.push_back(all_arg_buffers[arg_buffer_idx].data());
              } else {
                extra_config.push_back(nullptr);
              }
            }
          } else if (extra_array[i].is_uint64() || extra_array[i].is_int64()) {
            uint64_t val = extra_array[i].to_number<uint64_t>();
            if (i > 0 && extra_config.back() == CU_LAUNCH_PARAM_BUFFER_SIZE) {
              all_arg_buffer_sizes.push_back(val);
              extra_config.push_back(&all_arg_buffer_sizes.back());
            } else {
              extra_config.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(val)));
            }
          }
        }

        node_params.extra = extra_config.data();
      }

      CUresult kernel_result =
          add_restored_kernel_node(&cuNode, cuGraph, node_params, graph->capture_dev_,
                                   cluster_width, cluster_height, cluster_depth, "LOAD");
      if (kernel_result != CUDA_SUCCESS) {
        std::string function_name = params.at("function_name").as_string().c_str();
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddKernelNode FAILED for node %d with error %d\n",
                node_id, kernel_result);
        fprintf(stderr, "[foundry LOAD ERROR]   function: %s\n", function_name.c_str());
        fprintf(stderr, "[foundry LOAD ERROR]   grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                node_params.gridDimX, node_params.gridDimY, node_params.gridDimZ,
                node_params.blockDimX, node_params.blockDimY, node_params.blockDimZ,
                node_params.sharedMemBytes);
        FOUNDRY_CU_CHECK(kernel_result);
      }

      // Set kernel node attributes — only non-default attributes are
      // present in JSON (filtered at save time), so load blindly calls APIs.
      if (cluster_width > 0 || cluster_height > 0 || cluster_depth > 0) {
        CUkernelNodeAttrValue clusterAttr;
        memset(&clusterAttr, 0, sizeof(clusterAttr));
        clusterAttr.clusterDim.x = cluster_width > 0 ? cluster_width : 1;
        clusterAttr.clusterDim.y = cluster_height > 0 ? cluster_height : 1;
        clusterAttr.clusterDim.z = cluster_depth > 0 ? cluster_depth : 1;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &clusterAttr));
      }
      if (has_preferred_cluster_dim) {
        CUkernelNodeAttrValue pref_attr;
        memset(&pref_attr, 0, sizeof(pref_attr));
        pref_attr.preferredClusterDim.x = preferred_cluster_x;
        pref_attr.preferredClusterDim.y = preferred_cluster_y;
        pref_attr.preferredClusterDim.z = preferred_cluster_z;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &pref_attr));
      }
      if (has_cluster_scheduling) {
        CUkernelNodeAttrValue sched_attr;
        memset(&sched_attr, 0, sizeof(sched_attr));
        sched_attr.clusterSchedulingPolicyPreference =
            static_cast<CUclusterSchedulingPolicy>(cluster_scheduling);
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &sched_attr));
      }
      if (has_cooperative) {
        CUkernelNodeAttrValue coop_attr;
        memset(&coop_attr, 0, sizeof(coop_attr));
        coop_attr.cooperative = cooperative;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(cuNode, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE,
                                                       &coop_attr));
      }
      if (has_priority) {
        CUkernelNodeAttrValue priority_attr;
        memset(&priority_attr, 0, sizeof(priority_attr));
        priority_attr.priority = priority;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(cuNode, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY,
                                                       &priority_attr));
      }
      if (has_mem_sync_domain) {
        CUkernelNodeAttrValue mem_domain_attr;
        memset(&mem_domain_attr, 0, sizeof(mem_domain_attr));
        mem_domain_attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(mem_sync_domain);
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &mem_domain_attr));
      }
      if (has_mem_sync_domain_map) {
        CUkernelNodeAttrValue mem_map_attr;
        memset(&mem_map_attr, 0, sizeof(mem_map_attr));
        mem_map_attr.memSyncDomainMap.default_ = mem_sync_domain_default;
        mem_map_attr.memSyncDomainMap.remote = mem_sync_domain_remote;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &mem_map_attr));
      }
      if (has_preferred_shared_mem_carveout) {
        CUkernelNodeAttrValue carveout_attr;
        memset(&carveout_attr, 0, sizeof(carveout_attr));
        carveout_attr.sharedMemCarveout = preferred_shared_mem_carveout;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &carveout_attr));
      }
      if (has_access_policy_window) {
        CUkernelNodeAttrValue apw_attr;
        memset(&apw_attr, 0, sizeof(apw_attr));
        apw_attr.accessPolicyWindow.base_ptr =
            reinterpret_cast<void*>(static_cast<uintptr_t>(access_policy_base_ptr));
        apw_attr.accessPolicyWindow.num_bytes = access_policy_num_bytes;
        apw_attr.accessPolicyWindow.hitRatio = access_policy_hit_ratio;
        apw_attr.accessPolicyWindow.hitProp = static_cast<CUaccessProperty>(access_policy_hit_prop);
        apw_attr.accessPolicyWindow.missProp =
            static_cast<CUaccessProperty>(access_policy_miss_prop);
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &apw_attr));
      }
      if (has_device_updatable) {
        CUkernelNodeAttrValue updatable_attr;
        memset(&updatable_attr, 0, sizeof(updatable_attr));
        updatable_attr.deviceUpdatableKernelNode.deviceUpdatable = device_updatable;
        FOUNDRY_CU_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &updatable_attr));
      }
    } else if (node_type == "MemcpyNode") {
      CUDA_MEMCPY3D copy_params;
      memset(&copy_params, 0, sizeof(copy_params));

      copy_params.Depth = params.at("Depth").to_number<size_t>();
      copy_params.Height = params.at("Height").to_number<size_t>();
      copy_params.WidthInBytes = params.at("WidthInBytes").to_number<size_t>();
      copy_params.dstDevice = params.at("dstDevice").to_number<CUdeviceptr>();
      copy_params.dstHeight = params.at("dstHeight").to_number<size_t>();
      copy_params.dstLOD = params.at("dstLOD").to_number<size_t>();
      copy_params.dstMemoryType =
          static_cast<CUmemorytype>(params.at("dstMemoryType").to_number<int>());
      copy_params.dstPitch = params.at("dstPitch").to_number<size_t>();
      copy_params.dstXInBytes = params.at("dstXInBytes").to_number<size_t>();
      copy_params.dstY = params.at("dstY").to_number<size_t>();
      copy_params.dstZ = params.at("dstZ").to_number<size_t>();
      copy_params.srcDevice = params.at("srcDevice").to_number<CUdeviceptr>();
      copy_params.srcHeight = params.at("srcHeight").to_number<size_t>();
      copy_params.srcLOD = params.at("srcLOD").to_number<size_t>();
      copy_params.srcMemoryType =
          static_cast<CUmemorytype>(params.at("srcMemoryType").to_number<int>());
      copy_params.srcPitch = params.at("srcPitch").to_number<size_t>();
      copy_params.srcXInBytes = params.at("srcXInBytes").to_number<size_t>();
      copy_params.srcY = params.at("srcY").to_number<size_t>();
      copy_params.srcZ = params.at("srcZ").to_number<size_t>();

      CUresult memcpy_result =
          cuGraphAddMemcpyNode(&cuNode, cuGraph, nullptr, 0, &copy_params, current_ctx);
      if (memcpy_result != CUDA_SUCCESS) {
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddMemcpyNode FAILED for node %d with error %d\n",
                node_id, memcpy_result);
        fprintf(stderr, "[foundry LOAD ERROR]   srcDevice=0x%llx srcMemoryType=%d srcPitch=%zu\n",
                (unsigned long long)copy_params.srcDevice, copy_params.srcMemoryType,
                copy_params.srcPitch);
        fprintf(stderr, "[foundry LOAD ERROR]   dstDevice=0x%llx dstMemoryType=%d dstPitch=%zu\n",
                (unsigned long long)copy_params.dstDevice, copy_params.dstMemoryType,
                copy_params.dstPitch);
        fprintf(stderr, "[foundry LOAD ERROR]   WidthInBytes=%zu Height=%zu Depth=%zu\n",
                copy_params.WidthInBytes, copy_params.Height, copy_params.Depth);
        fprintf(stderr, "[foundry LOAD ERROR]   srcXInBytes=%zu srcY=%zu srcZ=%zu\n",
                copy_params.srcXInBytes, copy_params.srcY, copy_params.srcZ);
        fprintf(stderr, "[foundry LOAD ERROR]   dstXInBytes=%zu dstY=%zu dstZ=%zu\n",
                copy_params.dstXInBytes, copy_params.dstY, copy_params.dstZ);
        FOUNDRY_CU_CHECK(memcpy_result);
      }
    } else if (node_type == "MemsetNode") {
      CUDA_MEMSET_NODE_PARAMS memset_params;
      memset(&memset_params, 0, sizeof(memset_params));

      memset_params.dst = params.at("dst").to_number<CUdeviceptr>();
      memset_params.elementSize = params.at("elementSize").to_number<unsigned int>();
      memset_params.height = params.at("height").to_number<size_t>();
      memset_params.pitch = params.at("pitch").to_number<size_t>();
      memset_params.value = params.at("value").to_number<unsigned int>();
      memset_params.width = params.at("width").to_number<size_t>();

      CUresult memset_result =
          cuGraphAddMemsetNode(&cuNode, cuGraph, nullptr, 0, &memset_params, current_ctx);
      if (memset_result != CUDA_SUCCESS) {
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddMemsetNode FAILED for node %d with error %d\n",
                node_id, memset_result);
        fprintf(stderr, "[foundry LOAD ERROR]   dst=0x%llx width=%zu height=%zu pitch=%zu\n",
                (unsigned long long)memset_params.dst, memset_params.width, memset_params.height,
                memset_params.pitch);
        FOUNDRY_CU_CHECK(memset_result);
      }
    } else if (node_type == "EventRecordNode") {
      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        FOUNDRY_CU_CHECK(cuEventCreate(&event, saved_event_flags(params)));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      FOUNDRY_CU_CHECK(cuGraphAddEventRecordNode(&cuNode, cuGraph, nullptr, 0, event));
      {
        const size_t node_idx = static_cast<size_t>(&node_val - nodes_array.data());
        auto& res = *graph->loaded_graph_resources_;
        res.note_event(event_id, event, saved_event_flags(params), node_idx, true);
        res.event_nodes[node_idx] = cuNode;
      }

    } else if (node_type == "EventWaitNode") {
      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        FOUNDRY_CU_CHECK(cuEventCreate(&event, saved_event_flags(params)));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      FOUNDRY_CU_CHECK(cuGraphAddEventWaitNode(&cuNode, cuGraph, nullptr, 0, event));
      {
        const size_t node_idx = static_cast<size_t>(&node_val - nodes_array.data());
        auto& res = *graph->loaded_graph_resources_;
        res.note_event(event_id, event, saved_event_flags(params), node_idx, false);
        res.event_nodes[node_idx] = cuNode;
      }

    } else if (node_type == "EmptyNode") {
      FOUNDRY_CU_CHECK(cuGraphAddEmptyNode(&cuNode, cuGraph, nullptr, 0));
    }

    if (cuNode) {
      id_to_node[node_id] = cuNode;
    }
  }

  const json::array& deps_array = root.at("dependencies").as_array();
  if (!deps_array.empty()) {
    std::vector<CUgraphNode> from_nodes;
    std::vector<CUgraphNode> to_nodes;
    from_nodes.reserve(deps_array.size());
    to_nodes.reserve(deps_array.size());

    std::vector<CUgraphEdgeData> edge_data(deps_array.size());
    memset(edge_data.data(), 0, edge_data.size() * sizeof(CUgraphEdgeData));
    size_t di = 0;
    for (const auto& dep_val : deps_array) {
      const json::object& dep_obj = dep_val.as_object();
      int from_id = dep_obj.at("from").to_number<int>();
      int to_id = dep_obj.at("to").to_number<int>();

      from_nodes.push_back(id_to_node[from_id]);
      to_nodes.push_back(id_to_node[to_id]);
      // Restore edge data (PDL programmatic ports); absent fields = default.
      if (auto* v = dep_obj.if_contains("from_port")) {
        edge_data[di].from_port = (unsigned char)v->to_number<int>();
      }
      if (auto* v = dep_obj.if_contains("to_port")) {
        edge_data[di].to_port = (unsigned char)v->to_number<int>();
      }
      if (auto* v = dep_obj.if_contains("edge_type")) {
        edge_data[di].type = (unsigned char)v->to_number<int>();
      }
      di++;
    }

    CUresult dep_result = add_graph_dependencies(cuGraph, from_nodes.data(), to_nodes.data(),
                                                 edge_data.data(), deps_array.size());
    if (dep_result != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry LOAD ERROR] cuGraphAddDependencies FAILED with error %d\n",
              dep_result);
      FOUNDRY_CU_CHECK(dep_result);
    }
  }
  // No allocator pool registration around instantiate: Foundry's instantiate is
  // cudaGraphInstantiate[WithFlags] and never calls the torch allocator.
  graph->capture_ended_ = true;
  graph->instantiate();

  // NOTE(yongji): bypass destructor's release memory pool call
  graph->capture_ended_ = false;
  GraphLoadResult result;
  result.graph = graph;
  result.output_type = OutputTensorType::None;
  result.outputs = std::monostate{};

  if (root.contains("output_tensors")) {
    const json::object& output_tensors_obj = root.at("output_tensors").as_object();
    result.output_type =
        static_cast<OutputTensorType>(output_tensors_obj.at("type").to_number<int>());

    const json::array& tensors_array = output_tensors_obj.at("tensors").as_array();

    if (result.output_type == OutputTensorType::Single && tensors_array.size() == 1) {
      result.outputs = reconstruct_tensor_from_metadata(tensors_array[0].as_object());
    } else if (result.output_type == OutputTensorType::List ||
               result.output_type == OutputTensorType::Tuple) {
      std::vector<Tensor> tensors;
      tensors.reserve(tensors_array.size());
      for (const auto& t_val : tensors_array) {
        tensors.push_back(reconstruct_tensor_from_metadata(t_val.as_object()));
      }
      result.outputs = std::move(tensors);
    }
  }

  return result;
}

void CUDAGraph::LoadedGraphResources::note_event(int event_id, CUevent event, unsigned int flags,
                                                 size_t node_idx, bool record) {
  auto it = std::find_if(events.begin(), events.end(),
                         [&](const LoadedEvent& e) { return e.event_id == event_id; });
  if (it == events.end()) {
    events.push_back(LoadedEvent{event_id, event, flags, {}, {}});
    it = std::prev(events.end());
  }
  (record ? it->record_nodes : it->wait_nodes).push_back(node_idx);
}

const std::vector<CUDAGraph::LoadedEvent>& CUDAGraph::loaded_events() const {
  static const std::vector<LoadedEvent> none;
  return loaded_graph_resources_ ? loaded_graph_resources_->events : none;
}

void CUDAGraph::apply_event_bindings(CUgraphExec exec) {
  if (!loaded_graph_resources_ || loaded_graph_resources_->bound_events.empty() || !exec)
    return;
  auto& res = *loaded_graph_resources_;
  const std::vector<CUgraphNode>* ordered = on_demand_data_ && on_demand_data_->shared_exec
                                                ? &on_demand_data_->shared_exec->ordered_nodes
                                                : nullptr;
  auto node_at = [&](size_t idx) -> CUgraphNode {
    if (ordered) {
      FOUNDRY_CHECK(idx < ordered->size(), "bind_events: node ", idx, " out of range (",
                    ordered->size(), " nodes)");
      return (*ordered)[idx];
    }
    auto it = res.event_nodes.find(idx);
    FOUNDRY_CHECK(it != res.event_nodes.end(), "bind_events: no node handle for node ", idx);
    return it->second;
  };
  for (const auto& e : res.events) {
    auto b = res.bound_events.find(e.event_id);
    if (b == res.bound_events.end())
      continue;
    for (size_t idx : e.record_nodes)
      FOUNDRY_CU_CHECK(cuGraphExecEventRecordNodeSetEvent(exec, node_at(idx), b->second));
    for (size_t idx : e.wait_nodes)
      FOUNDRY_CU_CHECK(cuGraphExecEventWaitNodeSetEvent(exec, node_at(idx), b->second));
  }
}

void CUDAGraph::bind_events(const std::vector<std::pair<int, CUevent>>& events) {
  FOUNDRY_CHECK(loaded_graph_resources_ && !loaded_graph_resources_->events.empty(),
                "bind_events: graph has no events rebuilt by LOAD");
  auto& res = *loaded_graph_resources_;
  for (const auto& [event_id, event] : events) {
    FOUNDRY_CHECK(event, "bind_events: null event for event_id ", event_id);
    FOUNDRY_CHECK(std::any_of(res.events.begin(), res.events.end(),
                              [&](const LoadedEvent& e) { return e.event_id == event_id; }),
                  "bind_events: graph has no event_id ", event_id);
  }
  CUgraphExec exec = nullptr;
  if (on_demand_data_) {
    FOUNDRY_CHECK(on_demand_data_->shared_exec, "bind_events: ", on_demand_data_->graph_name,
                  " is not linked to its template (LOAD Phase 2 did not finish)");
    if (on_demand_data_->shared_exec->share) {
      bind_shared_exec_events(events);
      return;
    }
    exec = on_demand_data_->own_exec;
    FOUNDRY_CHECK(exec, "bind_events: ", on_demand_data_->graph_name,
                  " has no exec (LOAD Phase 2 did not finish)");
  } else {
    FOUNDRY_CHECK(has_graph_exec_, "bind_events: graph has no exec");
    exec = reinterpret_cast<CUgraphExec>(graph_exec_);
  }
  auto previous = res.bound_events;
  for (const auto& [event_id, event] : events)
    res.bound_events[event_id] = event;
  try {
    apply_event_bindings(exec);
  } catch (...) {
    res.bound_events = std::move(previous);
    throw;
  }
}

void CUDAGraph::bind_shared_exec_events(const std::vector<std::pair<int, CUevent>>& events) {
  // A switch to this graph (replay) writes its updates, events included, onto the builder
  // graph and cuGraphExecUpdate carries them into the shared exec, so the binding goes into
  // the updates. When this graph is the one on the exec (current_params_id), the exec and the
  // builder graph get the events directly. replay_mutex keeps a switch out meanwhile.
  auto& res = *loaded_graph_resources_;
  auto& shared = *on_demand_data_->shared_exec;
  auto& updates = on_demand_data_->updates;
  std::lock_guard<std::mutex> lock(shared.replay_mutex);
  const bool resident = shared.current_params_id == on_demand_data_->graph_id;
  auto previous_bound = res.bound_events;
  std::vector<std::pair<size_t, CUevent>> previous_events;  // (update index, event)
  auto set_event = [&](size_t idx, CUevent event, bool record) {
    FOUNDRY_CHECK(idx < updates.size() && idx < shared.ordered_nodes.size(), "bind_events: node ",
                  idx, " out of range (", updates.size(), " updates)");
    auto& u = updates[idx];
    const auto want = record ? OnDemandNodeUpdate::EventRecord : OnDemandNodeUpdate::EventWait;
    FOUNDRY_CHECK(u.type == want, "bind_events: ", on_demand_data_->graph_name, " node ", idx,
                  " is not an event ", record ? "record" : "wait", " node");
    previous_events.emplace_back(idx, u.event);
    u.event = event;
    if (!resident)
      return;
    CUgraphNode node = shared.ordered_nodes[idx];
    if (record) {
      FOUNDRY_CU_CHECK(cuGraphEventRecordNodeSetEvent(node, event));
      FOUNDRY_CU_CHECK(cuGraphExecEventRecordNodeSetEvent(shared.exec, node, event));
    } else {
      FOUNDRY_CU_CHECK(cuGraphEventWaitNodeSetEvent(node, event));
      FOUNDRY_CU_CHECK(cuGraphExecEventWaitNodeSetEvent(shared.exec, node, event));
    }
  };
  try {
    for (const auto& [event_id, event] : events) {
      const auto& e = *std::find_if(res.events.begin(), res.events.end(),
                                    [&](const LoadedEvent& le) { return le.event_id == event_id; });
      for (size_t idx : e.record_nodes)
        set_event(idx, event, true);
      for (size_t idx : e.wait_nodes)
        set_event(idx, event, false);
      res.bound_events[event_id] = event;
    }
  } catch (...) {
    for (auto it = previous_events.rbegin(); it != previous_events.rend(); ++it)
      updates[it->first].event = it->second;
    res.bound_events = std::move(previous_bound);
    // The exec may hold part of the binding: the next replay switches it from the updates.
    if (resident)
      shared.current_params_id = -1;
    throw;
  }
}

std::shared_ptr<PendingGraphLoads> CUDAGraph::start_graph_builds(
    const std::vector<std::string>& json_paths, MempoolId_t pool, int num_threads) {
  return start_graph_builds_impl(json_paths, pool, num_threads);
}

std::vector<GraphLoadResult> CUDAGraph::finish_graph_loads(
    std::shared_ptr<PendingGraphLoads> pending) {
  return finish_graph_loads_impl(std::move(pending), reconstruct_tensor_from_metadata);
}

GraphLoadResult CUDAGraph::finish_one_graph_load(std::shared_ptr<PendingGraphLoads> pending,
                                                 size_t index) {
  return finish_one_graph_load_impl(std::move(pending), index, reconstruct_tensor_from_metadata);
}

// ---- Kernel function attributes (contract in CUDAGraphInternal.h) ----------

namespace {

std::mutex g_smem_cap_mutex;
// CUkernel / CUfunction handle -> MAX_DYNAMIC_SHARED_SIZE_BYTES this process set on it.
std::unordered_map<const void*, int> g_smem_cap_high_water;

// Raise one handle's cap to `need` unless it is already there. Caller holds
// g_smem_cap_mutex. A failed set is reported and leaves the mark unchanged so
// the next node with the same need retries.
template <typename SetFn>
void raise_smem_cap_locked(const void* key, int need, SetFn&& set, const char* where) {
  int& cap = g_smem_cap_high_water[key];
  if (cap >= need) {
    return;
  }
  CUresult r = set();
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "[foundry %s] setting MAX_DYNAMIC_SHARED_SIZE_BYTES=%d failed: %d\n", where,
            need, (int)r);
    return;
  }
  cap = need;
}

void raise_dynamic_smem_cap(CUkernel kern, CUfunction func, CUdevice dev, int need,
                            const char* where) {
  if (need <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_smem_cap_mutex);
  if (kern != nullptr) {
    raise_smem_cap_locked(
        kern, need,
        [&] {
          return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need, kern,
                                      dev);
        },
        where);
    CUfunction ctx_func = nullptr;
    CUresult r = cuKernelGetFunction(&ctx_func, kern);
    if (r != CUDA_SUCCESS || ctx_func == nullptr) {
      fprintf(stderr,
              "[foundry %s] cuKernelGetFunction failed (%d): dynamic smem cap %d not applied to "
              "the per-context function\n",
              where, (int)r, need);
    } else {
      raise_smem_cap_locked(
          ctx_func, need,
          [&] {
            return cuFuncSetAttribute(ctx_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      need);
          },
          where);
    }
  }
  if (func != nullptr) {
    raise_smem_cap_locked(
        func, need,
        [&] {
          return cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
        },
        where);
  }
}

}  // namespace

void apply_saved_function_attributes(const std::variant<CUfunction, CUkernel>& handle, CUdevice dev,
                                     int max_dynamic_shared_size_bytes,
                                     int preferred_shared_memory_carveout, const char* where) {
  // CLUSTER_SCHEDULING_POLICY_PREFERENCE is recorded but not reapplied: it is
  // only a hint and fails with cudaErrorInvalidClusterSize when the kernel's
  // compiled cluster requirements differ from the saved preference.
  if (std::holds_alternative<CUkernel>(handle)) {
    CUkernel kern = std::get<CUkernel>(handle);
    raise_dynamic_smem_cap(kern, nullptr, dev, max_dynamic_shared_size_bytes, where);
    if (preferred_shared_memory_carveout >= 0) {
      FOUNDRY_CU_CHECK(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                            preferred_shared_memory_carveout, kern, dev));
    }
  } else {
    CUfunction func = std::get<CUfunction>(handle);
    raise_dynamic_smem_cap(nullptr, func, dev, max_dynamic_shared_size_bytes, where);
    if (preferred_shared_memory_carveout >= 0) {
      FOUNDRY_CU_CHECK(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                          preferred_shared_memory_carveout));
    }
  }
}

void ensure_dynamic_smem_optin(const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev,
                               const char* where) {
  raise_dynamic_smem_cap(params.kern, params.func, dev, static_cast<int>(params.sharedMemBytes),
                         where);
}

CUresult add_restored_kernel_node(CUgraphNode* node, CUgraph graph,
                                  const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev, unsigned cx,
                                  unsigned cy, unsigned cz, const char* where) {
  ensure_dynamic_smem_optin(params, dev, where);
  return add_kernel_node_cluster_optin(node, graph, params, dev, cx, cy, cz, where);
}

}  // namespace foundry
