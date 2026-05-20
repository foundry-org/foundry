#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Functions.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#if __has_include(<ATen/cuda/MemPool.h>)
#include <ATen/cuda/MemPool.h>
#define FOUNDRY_HAS_ATEN_CUDA_MEMPOOL 1
#endif
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/driver_api.h>
#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <cuda.h>
#include <fstream>
#include <boost/json.hpp>
#include <iomanip>
#include <sstream>
#include <vector>
#include "hook.h"
#include "BinaryGraphFormat.h"

// #define CGE_DEBUG_REPLAY  // Uncomment for verbose on-demand replay logging

namespace at {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::register_graph(cuda::CUDAGraph* graph) {
    at::cuda::assertNotCapturing(
        "Cannot register the state during capturing stage.");
  
    if (registered_graphs_.empty()) {
      auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
      seed_extragraph_ = at::empty({1}, options);
      offset_extragraph_ = at::empty({1}, options);
    }
  
    if (registered_graphs_.find(graph) == registered_graphs_.end()) {
      registered_graphs_.insert(graph);
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::unregister_graph(cuda::CUDAGraph* graph) {
    TORCH_CHECK(
        registered_graphs_.find(graph) != registered_graphs_.end(),
        "The graph should be registered to the state");
  
    registered_graphs_.erase(graph);
  
    if (registered_graphs_.empty()) {
      seed_extragraph_.reset();
      offset_extragraph_.reset();
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::increase(uint64_t increment) {
    increment = ((increment + 3) / 4) * 4;
    if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
      TORCH_CHECK(
          capturing_,
          "Attempt to increase offset for a CUDA generator not in capture mode.");
      TORCH_INTERNAL_ASSERT(
          offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
      TORCH_INTERNAL_ASSERT(
          offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
          "Increment causes overflow in the offset value.");
      offset_intragraph_ += increment;
    } else {
      TORCH_CHECK(
          !capturing_,
          "Offset increment outside graph capture encountered unexpectedly.");
      TORCH_INTERNAL_ASSERT(
          philox_offset_per_thread_ % 4 == 0,
          "RNG offset must be a multiple of 4.");
      philox_offset_per_thread_ += increment;
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::capture_prologue() {
    capturing_ = true;
    offset_intragraph_ = 0;
    seed_extragraph_.fill_(int64_t(seed_));
    offset_extragraph_.fill_(int64_t(0));
  }

  __attribute__((visibility("hidden")))
  uint64_t CUDAGeneratorState::capture_epilogue() {
    capturing_ = false;
    return offset_intragraph_;
  }
  
  __attribute__((visibility("hidden")))
  void CUDAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
    at::cuda::assertNotCapturing(
        "Cannot prepare for replay during capturing stage.");
    if (wholegraph_increment) {
        seed_extragraph_.fill_(int64_t(seed_));
        offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
        increase(wholegraph_increment);
    }
  }

  __attribute__((visibility("hidden")))
  void CUDAGeneratorImpl::register_graph(cuda::CUDAGraph* graph) {
    auto cge_graph = reinterpret_cast<foundry::CUDAGraph*>(graph);
    cge_graph->register_generator_state(state_);
    state_->register_graph(graph);
  }
  
  __attribute__((visibility("hidden")))
  void CUDAGeneratorImpl::unregister_graph(cuda::CUDAGraph* graph) {
    state_->unregister_graph(graph);
  }
  
#pragma GCC diagnostic pop
}

namespace foundry {

namespace {

MempoolId_t torch_graph_pool_handle(bool is_user_created = true) {
#if FOUNDRY_HAS_ATEN_CUDA_MEMPOOL
  return at::cuda::MemPool::graph_pool_handle(is_user_created);
#else
  return c10::cuda::MemPool::graph_pool_handle(is_user_created);
#endif
}

} // namespace

static bool _cuda_graphs_debug = false;

static CUDAGeneratorStateRegistry global_generator_state_registry;

uint64_t CUDAGeneratorStateRegistry::query_state_id(at::CUDAGeneratorState* state) {
  uint64_t result_id = 0;

  auto visit_fn = [&](const auto& value) {
    result_id = value.second;
  };

  if (!id_map_.cvisit(state, visit_fn)) {
    result_id = id_counter.fetch_add(1, std::memory_order_relaxed);
    id_map_.emplace(state, result_id);
  }

  return result_id;
}

c10::intrusive_ptr<at::CUDAGeneratorState> CUDAGeneratorStateRegistry::get_state_from_id(uint64_t id, uint64_t seed) {
  c10::intrusive_ptr<at::CUDAGeneratorState> result_state;

  auto visit_fn = [&](const auto& value) {
    result_state = value.second;
  };

  if (!state_pool_.cvisit(id, visit_fn)) {
    result_state = c10::make_intrusive<at::CUDAGeneratorState>(seed, 0, 0);
    state_pool_.emplace(id, result_state);
  }

  return result_state;
}

MempoolId_t graph_pool_handle() {
  return torch_graph_pool_handle();
}

void preallocate_cublas_workspaces() {
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublas_handle != nullptr, "Failed to get cuBLAS handle");

  cublasLtHandle_t cublaslt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CHECK(cublaslt_handle != nullptr, "Failed to get cuBLASLt handle");

  void* cublaslt_workspace = at::cuda::getCUDABlasLtWorkspace();
  TORCH_CHECK(cublaslt_workspace != nullptr, "Failed to allocate cuBLASLt workspace");
}

CUDAGraph::CUDAGraph(bool keep_graph)
  : capture_stream_(at::cuda::getCurrentCUDAStream()),
    keep_graph_(keep_graph) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  global_generator_state_registry.query_state_id(state.get());
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<at::CUDAGeneratorImpl> cuda_gen =
      c10::dynamic_intrusive_pointer_cast<at::CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  cuda_gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state,
    uint64_t wholegraph_increment) {
  state->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
  captured_generator_states_[state] = wholegraph_increment;
}

void CUDAGraph::capture_begin(MempoolId_t pool, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    mempool_id_ = torch_graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](cudaStream_t stream) {
      cudaStreamCaptureStatus status{};
      CaptureId_t stream_capture_id = 0;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });
  // foundry::resume_allocation_region();
  foundry::start_hook_record();

  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
}

void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  foundry::end_hook_record();
  // foundry::stop_allocation_region();
  allocator_events_ = foundry::save_hook_events_to_json();
  foundry::clear_hook_events();

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;

  analyze_captured_graph();

  if (!keep_graph_) {
    instantiate();
    if (!_cuda_graphs_debug) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
    has_graph_ = false;
  }
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (inst_err != cudaSuccess) {
      fprintf(stderr, "[CGE INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
              inst_err, cudaGetErrorString(inst_err));
      AT_CUDA_CHECK(inst_err);
    }
#else
    cudaError_t inst_err = cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);
    if (inst_err != cudaSuccess) {
      fprintf(stderr, "[CGE INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
              inst_err, cudaGetErrorString(inst_err));
      AT_CUDA_CHECK(inst_err);
    }
#endif
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  } else {
    cudaError_t inst_err = cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (inst_err != cudaSuccess) {
      fprintf(stderr, "[CGE INSTANTIATE ERROR] cudaGraphInstantiateWithFlags FAILED with error %d: %s\n",
              inst_err, cudaGetErrorString(inst_err));
      AT_CUDA_CHECK(inst_err);
    }
  }
#endif
  has_graph_exec_ = true;
}

void CUDAGraph::replay() {

  // On-demand replay: update shared graph nodes + cuGraphExecUpdate, then launch.
  if (on_demand_data_) {
    auto& shared = on_demand_data_->shared_exec;
#ifdef CGE_DEBUG_REPLAY
    fprintf(stderr, "[CGE REPLAY] graph %d (%s): current_params_id=%d, shared_exec=%p, updates=%zu\n",
            on_demand_data_->graph_id,
            on_demand_data_->graph_name.c_str(),
            shared->current_params_id,
            (void*)shared.get(),
            on_demand_data_->updates.size());
#endif
    if (shared->current_params_id != on_demand_data_->graph_id) {
      int prev_id = shared->current_params_id;
#ifdef CGE_DEBUG_REPLAY
      fprintf(stderr, "[CGE DEBUG] graph %d: syncing before on-demand update (prev %d)...\n",
              on_demand_data_->graph_id, prev_id);
#endif
      AT_CUDA_CHECK(cudaDeviceSynchronize());
      auto t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < on_demand_data_->updates.size(); ++i) {
        auto& u = on_demand_data_->updates[i];
        CUgraphNode node = shared->ordered_nodes[i];
        switch (u.type) {
        case OnDemandNodeUpdate::Kernel: {
          // Update kernel params on the graph node (not exec)
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetParams(node, &u.kernel_params));

          // Update kernel node attributes that may differ across batch sizes
          auto& a = u.kernel_attrs;
          if (a.has_cluster_dim) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.clusterDim.x = a.clusterDimX > 0 ? a.clusterDimX : 1;
            attr.clusterDim.y = a.clusterDimY > 0 ? a.clusterDimY : 1;
            attr.clusterDim.z = a.clusterDimZ > 0 ? a.clusterDimZ : 1;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &attr));
          }
          if (a.has_preferred_cluster_dim) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.preferredClusterDim.x = a.preferredClusterDimX;
            attr.preferredClusterDim.y = a.preferredClusterDimY;
            attr.preferredClusterDim.z = a.preferredClusterDimZ;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr));
          }
          if (a.clusterSchedulingPolicy >= 0) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.clusterSchedulingPolicyPreference =
                static_cast<CUclusterSchedulingPolicy>(a.clusterSchedulingPolicy);
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr));
          }
          if (a.cooperative >= 0) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.cooperative = a.cooperative;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &attr));
          }
          if (a.priority >= 0) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.priority = a.priority;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr));
          }
          if (a.memSyncDomain >= 0) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(a.memSyncDomain);
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &attr));
          }
          if (a.has_mem_sync_domain_map) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.memSyncDomainMap.default_ = a.memSyncDomainMapDefault;
            attr.memSyncDomainMap.remote = a.memSyncDomainMapRemote;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr));
          }
          if (a.has_shared_mem_carveout) {
            CUkernelNodeAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.sharedMemCarveout = a.sharedMemCarveout;
            C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
                node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr));
          }
          break;
        }
        case OnDemandNodeUpdate::Memset:
          C10_CUDA_DRIVER_CHECK(cuGraphMemsetNodeSetParams(node, &u.memset_params));
          break;
        case OnDemandNodeUpdate::Memcpy:
          C10_CUDA_DRIVER_CHECK(cuGraphMemcpyNodeSetParams(node, &u.memcpy_params));
          break;
        case OnDemandNodeUpdate::EventRecord:
          C10_CUDA_DRIVER_CHECK(cuGraphEventRecordNodeSetEvent(node, u.event));
          break;
        case OnDemandNodeUpdate::EventWait:
          C10_CUDA_DRIVER_CHECK(cuGraphEventWaitNodeSetEvent(node, u.event));
          break;
        case OnDemandNodeUpdate::Empty:
          break;
        }
      }
      // Apply all graph node changes to the exec in bulk
      CUgraphExecUpdateResultInfo resultInfo = {};
      CUresult update_result = cuGraphExecUpdate(shared->exec, shared->graph, &resultInfo);
      if (update_result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(update_result, &err_str);
        fprintf(stderr, "[CGE ON-DEMAND ERROR] cuGraphExecUpdate FAILED for graph %d (%s): %d (%s)\n",
                on_demand_data_->graph_id,
                on_demand_data_->graph_name.c_str(),
                (int)update_result, err_str ? err_str : "unknown");
        fprintf(stderr, "[CGE ON-DEMAND ERROR]   resultInfo: result=%d, errorNode=%p, errorFromNode=%p\n",
                (int)resultInfo.result, (void*)resultInfo.errorNode, (void*)resultInfo.errorFromNode);
        C10_CUDA_DRIVER_CHECK(update_result);
      }

      shared->current_params_id = on_demand_data_->graph_id;
      double update_us = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - t0).count();
      fprintf(stderr, "[CGE ON-DEMAND] graph %d (%s): updated %zu nodes in %.1f us (prev graph %d)\n",
              on_demand_data_->graph_id,
              on_demand_data_->graph_name.c_str(),
              on_demand_data_->updates.size(),
              update_us,
              prev_id);
    } else {
#ifdef CGE_DEBUG_REPLAY
      fprintf(stderr, "[CGE REPLAY] graph %d (%s): SKIPPED update (already current)\n",
              on_demand_data_->graph_id,
              on_demand_data_->graph_name.c_str());
#endif
    }

    c10::OptionalDeviceGuard device_guard{
        c10::Device(c10::kCUDA, capture_dev_)};
    for (auto& [generator_state, wholegraph_increments] :
         captured_generator_states_) {
      generator_state->replay_prologue(wholegraph_increments);
    }
#ifdef CGE_DEBUG_REPLAY
    fprintf(stderr, "[CGE DEBUG] graph %d: launching on stream %p...\n",
            on_demand_data_->graph_id,
            (void*)at::cuda::getCurrentCUDAStream().stream());
#endif
    AT_CUDA_CHECK(cudaGraphLaunch(
        reinterpret_cast<cudaGraphExec_t>(shared->exec),
        at::cuda::getCurrentCUDAStream()));
#ifdef CGE_DEBUG_REPLAY
    fprintf(stderr, "[CGE DEBUG] graph %d: launched (async)\n", on_demand_data_->graph_id);
#endif
    return;
  }

#ifdef CGE_DEBUG_REPLAY
  fprintf(stderr, "[CGE DEBUG] replay: NO on_demand_data_, using direct exec path\n");
#endif

  TORCH_CHECK(capture_ended_ || has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture or load.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if defined(CUDA_VERSION) || defined(USE_ROCM)
  if (_cuda_graphs_debug || keep_graph_) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose));
      if (!keep_graph_) {
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(keep_graph_, "You cannot access the raw cudaGraph_t instance unless CUDAGraph was initialized with keep_graph=true");
  TORCH_CHECK(has_graph_, "You cannot access the raw cudaGraph_t instance until capture_end() has been called");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

CUDAGraph::SharedGraphExec::~SharedGraphExec() {
  if (exec) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)));
  }
  if (graph) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph)));
  }
}

void CUDAGraph::transfer_to_shared_exec(
    std::shared_ptr<SharedGraphExec> shared,
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
  if (type != Kernel) return;
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
  if (capture_ended_) {
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

MempoolId_t CUDAGraph::pool() {
TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
  }
  reset();

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  if (capture_dev_ != UNDEFINED_DEVICE)
  {
    AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}


void CUDAGraph::analyze_captured_graph() {
  TORCH_CHECK(has_graph_, "analyze_captured_graph() called before the graph is captured");

  graph_nodes.clear();
  graph_dependencies.clear();

  CUgraph cuGraph = reinterpret_cast<CUgraph>(graph_);

  size_t numNodes = 0;
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nullptr, &numNodes));

  TORCH_CHECK(numNodes > 0, "Graph contains no nodes");

  std::vector<CUgraphNode> nodes(numNodes);
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nodes.data(), &numNodes));

  for (size_t i = 0; i < numNodes; ++i) {
    CUgraphNodeType nodeType;
    C10_CUDA_DRIVER_CHECK(cuGraphNodeGetType(nodes[i], &nodeType));

    GraphNode graphNode;
    graphNode.index = i;
    graphNode.node = nodes[i];

    if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(nodes[i], &params));

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
          metadata.node_attrs.attr_query_available = false;
        }

        if (metadata.node_attrs.attr_query_available) {
          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_cluster_dim = true;
            metadata.node_attrs.preferredClusterDimX = attr_val.preferredClusterDim.x;
            metadata.node_attrs.preferredClusterDimY = attr_val.preferredClusterDim.y;
            metadata.node_attrs.preferredClusterDimZ = attr_val.preferredClusterDim.z;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cluster_scheduling_policy = true;
            metadata.node_attrs.clusterSchedulingPolicyPreference =
                static_cast<int>(attr_val.clusterSchedulingPolicyPreference);
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cooperative = true;
            metadata.node_attrs.cooperative = attr_val.cooperative;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_priority = true;
            metadata.node_attrs.priority = attr_val.priority;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain = true;
            metadata.node_attrs.memSyncDomain = attr_val.memSyncDomain;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain_map = true;
            metadata.node_attrs.memSyncDomainMapDefault = attr_val.memSyncDomainMap.default_;
            metadata.node_attrs.memSyncDomainMapRemote = attr_val.memSyncDomainMap.remote;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_shared_mem_carveout = true;
            metadata.node_attrs.preferredSharedMemCarveout = attr_val.sharedMemCarveout;
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
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_device_updatable = true;
            metadata.node_attrs.deviceUpdatable = attr_val.deviceUpdatableKernelNode.deviceUpdatable;
            metadata.node_attrs.deviceUpdatableNode = attr_val.deviceUpdatableKernelNode.devNode;
          }
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
            memcpy(metadata.kernelParams[j], params.kernelParams[j], std::get<1>(metadata.offset_and_sizes[j]));
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
          TORCH_INTERNAL_ASSERT(argBufferPtr != nullptr);
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
      C10_CUDA_DRIVER_CHECK(cuGraphMemsetNodeGetParams(nodes[i], &params));

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
      C10_CUDA_DRIVER_CHECK(cuGraphMemcpyNodeGetParams(nodes[i], &params));

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
      C10_CUDA_DRIVER_CHECK(cuGraphEventRecordNodeGetEvent(nodes[i], &event));

      EventRecordNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      CUevent event;
      C10_CUDA_DRIVER_CHECK(cuGraphEventWaitNodeGetEvent(nodes[i], &event));

      EventWaitNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EMPTY) {
      EmptyNodeMetadata metadata;

      graphNode.metadata = std::move(metadata);
    } else {
      TORCH_CHECK(false, "Graph contains unsupported node type!");
    }

    graph_nodes.push_back(std::move(graphNode));
  }
  
  size_t numEdges = 0;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)  
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#else
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#endif
  
  if (numEdges > 0) {
    std::vector<CUgraphNode> from_nodes(numEdges);
    std::vector<CUgraphNode> to_nodes(numEdges);
    std::vector<CUgraphEdgeData> edges(numEdges);
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)  
    C10_CUDA_DRIVER_CHECK(cuGraphGetEdges(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
#else
    C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
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
        graph_dependencies.push_back(dep);
      }
    }
  }
}


static boost::json::object serialize_tensor_metadata(const at::Tensor& tensor) {
  namespace json = boost::json;
  json::object obj;

  obj["data_ptr"] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()));

  json::array sizes_arr;
  for (auto s : tensor.sizes()) {
    sizes_arr.push_back(static_cast<int64_t>(s));
  }
  obj["sizes"] = sizes_arr;

  json::array strides_arr;
  for (auto s : tensor.strides()) {
    strides_arr.push_back(static_cast<int64_t>(s));
  }
  obj["strides"] = strides_arr;

  obj["dtype"] = static_cast<int>(tensor.scalar_type());
  obj["device_type"] = static_cast<int>(tensor.device().type());
  obj["device_index"] = tensor.device().index();
  obj["requires_grad"] = tensor.requires_grad();
  obj["numel"] = static_cast<int64_t>(tensor.numel());
  obj["element_size"] = static_cast<int64_t>(tensor.element_size());

  return obj;
}

static at::Tensor reconstruct_tensor_from_metadata(const boost::json::object& obj) {
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

  auto dtype = static_cast<c10::ScalarType>(obj.at("dtype").to_number<int>());
  auto device_type = static_cast<c10::DeviceType>(obj.at("device_type").to_number<int>());
  auto device_index = static_cast<c10::DeviceIndex>(obj.at("device_index").to_number<int>());
  bool requires_grad = obj.at("requires_grad").as_bool();

  auto options = at::TensorOptions()
      .dtype(dtype)
      .device(c10::Device(device_type, device_index))
      .requires_grad(requires_grad);

  auto tensor = at::from_blob(
      reinterpret_cast<void*>(data_ptr),
      sizes,
      strides,
      options
  );

  return tensor;
}

void CUDAGraph::save(const std::string& json_path,
                     const OutputTensors& output_tensors,
                     OutputTensorType output_type) {
  TORCH_CHECK(!graph_nodes.empty(), "Graph hasn't been captured yet or has no nodes");

  set_pack_fatbins_on_exit(true);

  namespace json = boost::json;
  json::object root;
  json::array nodes_array;

  std::unordered_map<CUevent, int> event_to_id;
  int next_event_id = 0;

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
        // cluster_dim: default is 1x1x1, only save if any dimension > 1
        if (metadata.node_attrs.has_cluster_dim &&
            (metadata.node_attrs.clusterDimX > 1 ||
             metadata.node_attrs.clusterDimY > 1 ||
             metadata.node_attrs.clusterDimZ > 1)) {
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
          kernel_node_attrs["memSyncDomainMapDefault"] = metadata.node_attrs.memSyncDomainMapDefault;
          kernel_node_attrs["memSyncDomainMapRemote"] = metadata.node_attrs.memSyncDomainMapRemote;
        }
        if (metadata.node_attrs.has_preferred_shared_mem_carveout &&
            metadata.node_attrs.preferredSharedMemCarveout != 0) {
          kernel_node_attrs["preferredSharedMemCarveout"] = metadata.node_attrs.preferredSharedMemCarveout;
        }
        if (metadata.node_attrs.has_access_policy_window &&
            metadata.node_attrs.accessPolicyWindowNumBytes != 0) {
          kernel_node_attrs["accessPolicyWindowBasePtr"] =
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(metadata.node_attrs.accessPolicyWindowBasePtr));
          kernel_node_attrs["accessPolicyWindowNumBytes"] = metadata.node_attrs.accessPolicyWindowNumBytes;
          kernel_node_attrs["accessPolicyWindowHitRatio"] = metadata.node_attrs.accessPolicyWindowHitRatio;
          kernel_node_attrs["accessPolicyWindowHitProp"] = metadata.node_attrs.accessPolicyWindowHitProp;
          kernel_node_attrs["accessPolicyWindowMissProp"] = metadata.node_attrs.accessPolicyWindowMissProp;
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
          TORCH_CHECK(metadata.kernelParams[i], "kernelParams[", i, "] is null");
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);

          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.kernelParams[i]);
          for (size_t j = 0; j < std::get<1>(metadata.offset_and_sizes[i]); ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[j]);
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
            hex_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[j]);
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
      func_attrs["max_dynamic_shared_size_bytes"] = static_cast<int>(metadata.sharedMemBytes);

      int preferred_carveout = 0;
      int cluster_scheduling = 0;
      int cluster_width = 0;
      int cluster_height = 0;
      int cluster_depth = 0;
      if (kern != nullptr) {
        cuKernelGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kern, capture_dev_);
        cuKernelGetAttribute(&cluster_scheduling, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kern, capture_dev_);
        cuKernelGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, kern, capture_dev_);
        cuKernelGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, kern, capture_dev_);
        cuKernelGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, kern, capture_dev_);
      } else if (func != nullptr) {
        cuFuncGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, func);
        cuFuncGetAttribute(&cluster_scheduling, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, func);
        cuFuncGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, func);
        cuFuncGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, func);
        cuFuncGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, func);
      }
      func_attrs["preferred_shared_memory_carveout"] = preferred_carveout;
      func_attrs["cluster_scheduling_policy_preference"] = cluster_scheduling;
      // Save cluster dimensions - these are required when reconstructing the graph for kernels that use clusters
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

      TORCH_CHECK(metadata.dstArray == nullptr, "dstArray must be null");
      TORCH_CHECK(metadata.srcArray == nullptr, "srcArray must be null");
      TORCH_CHECK(metadata.dstHost == nullptr, "dstHost must be null");
      TORCH_CHECK(metadata.srcHost == nullptr, "srcHost must be null");
      TORCH_CHECK(metadata.reserved0 == nullptr, "reserved0 must be null");
      TORCH_CHECK(metadata.reserved1 == nullptr, "reserved1 must be null");

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

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EventWaitNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventWaitNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventWaitNode";

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EmptyNodeMetadata>(graphNode.metadata)) {
      node_obj["type"] = "EmptyNode";
      node_obj["params"] = json::object{};
    }

    nodes_array.push_back(node_obj);
  }

  // Compute topology key = node types + cluster dim values per kernel node.
  // cuGraphExecUpdate does NOT propagate kernel node attribute changes
  // (set via cuGraphKernelNodeSetAttribute) to the CUgraphExec — only
  // CUDA_KERNEL_NODE_PARAMS fields are updated. This means cluster dim
  // changes are silently ignored, causing "cluster misconfiguration" crashes
  // at launch when grid dims no longer match the stale cluster dims.
  // To avoid this, we include full cluster dim values in the topology key
  // so all graphs in a group share identical cluster dims.
  // This is computed BEFORE common attr extraction (which removes per-node attrs).
  {
    std::string topology_key;
    topology_key.reserve(nodes_array.size() * 16);
    for (size_t n = 0; n < nodes_array.size(); ++n) {
      if (n > 0) topology_key += ',';
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
          topology_key += ":C" + std::to_string(cdx) + "_" +
                          std::to_string(cdy) + "_" + std::to_string(cdz);
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
      if (node_obj_ref.at("type").as_string() != "KernelNode") continue;
      const json::object& params_ref = node_obj_ref.at("params").as_object();
      if (!params_ref.contains("kernel_node_attrs")) continue;
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
        if (node_obj_ref.at("type").as_string() != "KernelNode") continue;
        json::object& params_ref = node_obj_ref.at("params").as_object();
        if (!params_ref.contains("kernel_node_attrs")) continue;
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

  json::array deps_array;
  for (const auto& dep : graph_dependencies) {
    json::object dep_obj;
    dep_obj["from"] = dep.from_index;
    dep_obj["to"] = dep.to_index;
    deps_array.push_back(dep_obj);
  }
  root["dependencies"] = deps_array;

  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sorted_generators;
  for (const auto& [state, wholegraph_increment] : captured_generator_states_) {
    uint64_t state_id = global_generator_state_registry.query_state_id(state.get());
    sorted_generators.emplace_back(state_id, state->seed_, wholegraph_increment);
  }

  std::sort(sorted_generators.begin(), sorted_generators.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

  json::array generators_array;
  for (const auto& [state_id, seed, wholegraph_increment] : sorted_generators) {
    json::object gen_obj;
    gen_obj["id"] = state_id;
    gen_obj["seed"] = seed;
    gen_obj["wholegraph_increment"] = wholegraph_increment;
    generators_array.push_back(gen_obj);
  }
  root["generators"] = generators_array;

  root["allocator_events"] = allocator_events_;

  json::object output_tensors_obj;
  output_tensors_obj["type"] = static_cast<int>(output_type);

  json::array tensors_array;
  if (std::holds_alternative<at::Tensor>(output_tensors)) {
    tensors_array.push_back(serialize_tensor_metadata(std::get<at::Tensor>(output_tensors)));
  } else if (std::holds_alternative<std::vector<at::Tensor>>(output_tensors)) {
    for (const auto& t : std::get<std::vector<at::Tensor>>(output_tensors)) {
      tensors_array.push_back(serialize_tensor_metadata(t));
    }
  }
  output_tensors_obj["tensors"] = tensors_array;
  root["output_tensors"] = output_tensors_obj;

  std::ofstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for writing: ", json_path);
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
}

// save_binary is implemented in BinaryGraphIO.cpp

GraphLoadResult CUDAGraph::load(const std::string& json_path, MempoolId_t pool) {
  std::ifstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for reading: ", json_path);

  std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  namespace json = boost::json;
  json::value root_val = json::parse(json_str);
  const json::object& root = root_val.as_object();

  auto graph = std::make_shared<CUDAGraph>(true);
  graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();

  graph->capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    graph->mempool_id_ = pool;
  } else {
    graph->mempool_id_ = torch_graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(graph->mempool_id_.first > 0);
  }

  const json::array& generators_array = root.at("generators").as_array();
  for (const auto& gen_val : generators_array) {
    const json::object& gen_obj = gen_val.as_object();
    uint64_t state_id = gen_obj.at("id").to_number<uint64_t>();
    uint64_t seed = gen_obj.at("seed").to_number<uint64_t>();
    uint64_t wholegraph_increment = gen_obj.at("wholegraph_increment").to_number<uint64_t>();

    auto state = global_generator_state_registry.get_state_from_id(state_id, seed);
    state->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(graph.get()));
    graph->captured_generator_states_[state] = wholegraph_increment;
  }

  const json::object& allocator_events = root.at("allocator_events").as_object();
  foundry::replay_hook_events_from_json(allocator_events);

  CUgraph cuGraph;
  C10_CUDA_DRIVER_CHECK(cuGraphCreate(&cuGraph, 0));
  graph->graph_ = reinterpret_cast<cudaGraph_t>(cuGraph);
  graph->has_graph_ = true;

  const json::array& nodes_array = root.at("nodes").as_array();

  std::unordered_map<int, CUgraphNode> id_to_node;
  std::unordered_map<int, CUevent> event_id_to_event;

  std::vector<std::vector<std::vector<uint8_t>>> all_kernel_params;
  std::vector<std::vector<void*>> all_param_ptrs;
  std::vector<std::vector<void*>> all_extra_configs;
  std::vector<std::vector<uint8_t>> all_arg_buffers;
  std::vector<size_t> all_arg_buffer_sizes;

  CUcontext current_ctx;
  C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current_ctx));

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

      if (params.contains("kernel_node_attrs")) {
        const json::object& node_attrs = params.at("kernel_node_attrs").as_object();
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
          mem_sync_domain_default = static_cast<unsigned char>(
              node_attrs.at("memSyncDomainMapDefault").to_number<int>());
          mem_sync_domain_remote = static_cast<unsigned char>(
              node_attrs.at("memSyncDomainMapRemote").to_number<int>());
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
        // int cluster_scheduling = func_attrs.at("cluster_scheduling_policy_preference").to_number<int>();

        // Read cluster dimensions from func_attrs - override name-based detection if non-zero
        if (func_attrs.contains("required_cluster_width")) {
          int attr_width = func_attrs.at("required_cluster_width").to_number<int>();
          if (attr_width > 0) cluster_width = attr_width;
        }
        if (func_attrs.contains("required_cluster_height")) {
          int attr_height = func_attrs.at("required_cluster_height").to_number<int>();
          if (attr_height > 0) cluster_height = attr_height;
        }
        if (func_attrs.contains("required_cluster_depth")) {
          int attr_depth = func_attrs.at("required_cluster_depth").to_number<int>();
          if (attr_depth > 0) cluster_depth = attr_depth;
        }

        if (std::holds_alternative<CUkernel>(func_handle_variant)) {
          CUkernel kern = std::get<CUkernel>(func_handle_variant);
          if (max_shared > 0) {
            C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_shared, kern, graph->capture_dev_));
          }
          if (preferred_carveout >= 0) {
            C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
                CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, preferred_carveout, kern, graph->capture_dev_));
          }
          // NOTE: Skip cluster_scheduling on load - it can cause cudaErrorInvalidClusterSize
          // if the kernel's compiled cluster requirements don't match the saved preference.
          // The preference is just a hint and the kernel will still work without it.
          // if (cluster_scheduling > 0) {
          //   C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(
          //       CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cluster_scheduling, kern, graph->capture_dev_));
          // }
        } else {
          CUfunction func = std::get<CUfunction>(func_handle_variant);
          if (max_shared > 0) {
            C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_shared));
          }
          if (preferred_carveout >= 0) {
            C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, preferred_carveout));
          }
          // NOTE: Skip cluster_scheduling on load - see comment above
          // if (cluster_scheduling > 0) {
          //   C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(
          //       func, CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cluster_scheduling));
          // }
        }
        // NOTE: We do not set CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
        // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
        // and CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED because these are
        // kernel properties may not be changed at runtime.
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
              TORCH_CHECK(!has_arg_buffer, "Encountered multiple 'null' entries in extra field, but only one is expected for CU_LAUNCH_PARAM_BUFFER_POINTER");
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

      CUresult kernel_result = cuGraphAddKernelNode(&cuNode, cuGraph, nullptr, 0, &node_params);
      if (kernel_result != CUDA_SUCCESS) {
        std::string function_name = params.at("function_name").as_string().c_str();
        fprintf(stderr, "[CGE LOAD ERROR] cuGraphAddKernelNode FAILED for node %d with error %d\n", node_id, kernel_result);
        fprintf(stderr, "[CGE LOAD ERROR]   function: %s\n", function_name.c_str());
        fprintf(stderr, "[CGE LOAD ERROR]   grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                node_params.gridDimX, node_params.gridDimY, node_params.gridDimZ,
                node_params.blockDimX, node_params.blockDimY, node_params.blockDimZ,
                node_params.sharedMemBytes);
        C10_CUDA_DRIVER_CHECK(kernel_result);
      }

      // Set kernel node attributes — only non-default attributes are
      // present in JSON (filtered at save time), so load blindly calls APIs.
      if (cluster_width > 0 || cluster_height > 0 || cluster_depth > 0) {
        CUkernelNodeAttrValue clusterAttr;
        memset(&clusterAttr, 0, sizeof(clusterAttr));
        clusterAttr.clusterDim.x = cluster_width > 0 ? cluster_width : 1;
        clusterAttr.clusterDim.y = cluster_height > 0 ? cluster_height : 1;
        clusterAttr.clusterDim.z = cluster_depth > 0 ? cluster_depth : 1;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &clusterAttr));
      }
      if (has_preferred_cluster_dim) {
        CUkernelNodeAttrValue pref_attr;
        memset(&pref_attr, 0, sizeof(pref_attr));
        pref_attr.preferredClusterDim.x = preferred_cluster_x;
        pref_attr.preferredClusterDim.y = preferred_cluster_y;
        pref_attr.preferredClusterDim.z = preferred_cluster_z;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &pref_attr));
      }
      if (has_cluster_scheduling) {
        CUkernelNodeAttrValue sched_attr;
        memset(&sched_attr, 0, sizeof(sched_attr));
        sched_attr.clusterSchedulingPolicyPreference =
            static_cast<CUclusterSchedulingPolicy>(cluster_scheduling);
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &sched_attr));
      }
      if (has_cooperative) {
        CUkernelNodeAttrValue coop_attr;
        memset(&coop_attr, 0, sizeof(coop_attr));
        coop_attr.cooperative = cooperative;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &coop_attr));
      }
      if (has_priority) {
        CUkernelNodeAttrValue priority_attr;
        memset(&priority_attr, 0, sizeof(priority_attr));
        priority_attr.priority = priority;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &priority_attr));
      }
      if (has_mem_sync_domain) {
        CUkernelNodeAttrValue mem_domain_attr;
        memset(&mem_domain_attr, 0, sizeof(mem_domain_attr));
        mem_domain_attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(mem_sync_domain);
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &mem_domain_attr));
      }
      if (has_mem_sync_domain_map) {
        CUkernelNodeAttrValue mem_map_attr;
        memset(&mem_map_attr, 0, sizeof(mem_map_attr));
        mem_map_attr.memSyncDomainMap.default_ = mem_sync_domain_default;
        mem_map_attr.memSyncDomainMap.remote = mem_sync_domain_remote;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &mem_map_attr));
      }
      if (has_preferred_shared_mem_carveout) {
        CUkernelNodeAttrValue carveout_attr;
        memset(&carveout_attr, 0, sizeof(carveout_attr));
        carveout_attr.sharedMemCarveout = preferred_shared_mem_carveout;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
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
        apw_attr.accessPolicyWindow.missProp = static_cast<CUaccessProperty>(access_policy_miss_prop);
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &apw_attr));
      }
      if (has_device_updatable) {
        CUkernelNodeAttrValue updatable_attr;
        memset(&updatable_attr, 0, sizeof(updatable_attr));
        updatable_attr.deviceUpdatableKernelNode.deviceUpdatable = device_updatable;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
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
      copy_params.dstMemoryType = static_cast<CUmemorytype>(params.at("dstMemoryType").to_number<int>());
      copy_params.dstPitch = params.at("dstPitch").to_number<size_t>();
      copy_params.dstXInBytes = params.at("dstXInBytes").to_number<size_t>();
      copy_params.dstY = params.at("dstY").to_number<size_t>();
      copy_params.dstZ = params.at("dstZ").to_number<size_t>();
      copy_params.srcDevice = params.at("srcDevice").to_number<CUdeviceptr>();
      copy_params.srcHeight = params.at("srcHeight").to_number<size_t>();
      copy_params.srcLOD = params.at("srcLOD").to_number<size_t>();
      copy_params.srcMemoryType = static_cast<CUmemorytype>(params.at("srcMemoryType").to_number<int>());
      copy_params.srcPitch = params.at("srcPitch").to_number<size_t>();
      copy_params.srcXInBytes = params.at("srcXInBytes").to_number<size_t>();
      copy_params.srcY = params.at("srcY").to_number<size_t>();
      copy_params.srcZ = params.at("srcZ").to_number<size_t>();

      CUresult memcpy_result = cuGraphAddMemcpyNode(&cuNode, cuGraph, nullptr, 0, &copy_params, current_ctx);
      if (memcpy_result != CUDA_SUCCESS) {
        fprintf(stderr, "[CGE LOAD ERROR] cuGraphAddMemcpyNode FAILED for node %d with error %d\n", node_id, memcpy_result);
        fprintf(stderr, "[CGE LOAD ERROR]   srcDevice=0x%llx srcMemoryType=%d srcPitch=%zu\n",
                (unsigned long long)copy_params.srcDevice, copy_params.srcMemoryType, copy_params.srcPitch);
        fprintf(stderr, "[CGE LOAD ERROR]   dstDevice=0x%llx dstMemoryType=%d dstPitch=%zu\n",
                (unsigned long long)copy_params.dstDevice, copy_params.dstMemoryType, copy_params.dstPitch);
        fprintf(stderr, "[CGE LOAD ERROR]   WidthInBytes=%zu Height=%zu Depth=%zu\n",
                copy_params.WidthInBytes, copy_params.Height, copy_params.Depth);
        fprintf(stderr, "[CGE LOAD ERROR]   srcXInBytes=%zu srcY=%zu srcZ=%zu\n",
                copy_params.srcXInBytes, copy_params.srcY, copy_params.srcZ);
        fprintf(stderr, "[CGE LOAD ERROR]   dstXInBytes=%zu dstY=%zu dstZ=%zu\n",
                copy_params.dstXInBytes, copy_params.dstY, copy_params.dstZ);
        C10_CUDA_DRIVER_CHECK(memcpy_result);
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

      CUresult memset_result = cuGraphAddMemsetNode(&cuNode, cuGraph, nullptr, 0, &memset_params, current_ctx);
      if (memset_result != CUDA_SUCCESS) {
        fprintf(stderr, "[CGE LOAD ERROR] cuGraphAddMemsetNode FAILED for node %d with error %d\n", node_id, memset_result);
        fprintf(stderr, "[CGE LOAD ERROR]   dst=0x%llx width=%zu height=%zu pitch=%zu\n",
                (unsigned long long)memset_params.dst, memset_params.width, memset_params.height, memset_params.pitch);
        C10_CUDA_DRIVER_CHECK(memset_result);
      }
    } else if (node_type == "EventRecordNode") {

      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      C10_CUDA_DRIVER_CHECK(cuGraphAddEventRecordNode(&cuNode, cuGraph, nullptr, 0, event));

    } else if (node_type == "EventWaitNode") {

      int event_id = params.at("event_id").to_number<int>();

      CUevent event;
      if (event_id_to_event.find(event_id) == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = event_id_to_event[event_id];
      }

      C10_CUDA_DRIVER_CHECK(cuGraphAddEventWaitNode(&cuNode, cuGraph, nullptr, 0, event));

    } else if (node_type == "EmptyNode") {

      C10_CUDA_DRIVER_CHECK(cuGraphAddEmptyNode(&cuNode, cuGraph, nullptr, 0));
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

    for (const auto& dep_val : deps_array) {
      const json::object& dep_obj = dep_val.as_object();
      int from_id = dep_obj.at("from").to_number<int>();
      int to_id = dep_obj.at("to").to_number<int>();

      from_nodes.push_back(id_to_node[from_id]);
      to_nodes.push_back(id_to_node[to_id]);
    }

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
    CUresult dep_result = cuGraphAddDependencies(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, deps_array.size());
#else
    CUresult dep_result = cuGraphAddDependencies_v2(cuGraph, from_nodes.data(), to_nodes.data(), nullptr, deps_array.size());
#endif
    if (dep_result != CUDA_SUCCESS) {
      fprintf(stderr, "[CGE LOAD ERROR] cuGraphAddDependencies FAILED with error %d\n", dep_result);
      C10_CUDA_DRIVER_CHECK(dep_result);
    }
  }
  // Register the memory pool with PyTorch's caching allocator before instantiation.
  // This is required because PyTorch's allocator checks for registered graph pools
  // during graph instantiation. During normal capture, this is done by capture_begin(),
  // but during load we need to register it manually.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      graph->capture_dev_,
      graph->mempool_id_,
      [](cudaStream_t) { return false; }
  );

  graph->capture_ended_ = true;
  graph->instantiate();
  c10::cuda::CUDACachingAllocator::endAllocateToPool(graph->capture_dev_, graph->mempool_id_);

  // NOTE(yongji): bypass destructor's release memory pool call
  graph->capture_ended_ = false;
  GraphLoadResult result;
  result.graph = graph;
  result.output_type = OutputTensorType::None;
  result.outputs = std::monostate{};

  if (root.contains("output_tensors")) {
    const json::object& output_tensors_obj = root.at("output_tensors").as_object();
    result.output_type = static_cast<OutputTensorType>(output_tensors_obj.at("type").to_number<int>());

    const json::array& tensors_array = output_tensors_obj.at("tensors").as_array();

    if (result.output_type == OutputTensorType::Single && tensors_array.size() == 1) {
      result.outputs = reconstruct_tensor_from_metadata(tensors_array[0].as_object());
    } else if (result.output_type == OutputTensorType::List || result.output_type == OutputTensorType::Tuple) {
      std::vector<at::Tensor> tensors;
      tensors.reserve(tensors_array.size());
      for (const auto& t_val : tensors_array) {
        tensors.push_back(reconstruct_tensor_from_metadata(t_val.as_object()));
      }
      result.outputs = std::move(tensors);
    }
  }

  return result;
}

std::shared_ptr<PendingGraphLoads> CUDAGraph::start_graph_builds(
    const std::vector<std::string>& json_paths,
    MempoolId_t pool,
    int num_threads) {
  return start_graph_builds_impl(
      json_paths, pool, num_threads,
      global_generator_state_registry);
}

std::vector<GraphLoadResult> CUDAGraph::finish_graph_loads(
    std::shared_ptr<PendingGraphLoads> pending) {
  return finish_graph_loads_impl(
      std::move(pending),
      reconstruct_tensor_from_metadata);
}

} // namespace foundry
