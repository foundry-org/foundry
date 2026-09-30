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
#include <thread>
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <chrono>
#include "hook.h"
#include "BinaryGraphFormat.h"
#include "GraphDependencies.h"

namespace {

class SimpleThreadPool {
 public:
  explicit SimpleThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty())
              return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template <class F>
  std::future<typename std::invoke_result<F>::type> submit(F&& f) {
    using return_type = typename std::invoke_result<F>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    std::future<return_type> result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return result;
  }

  ~SimpleThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      w.join();
    }
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

// One thread that runs jobs in submission order with a CUDA context bound (LOAD
// Phase 2's instantiate thread). The first exception is kept, later jobs are
// dropped, and on_error runs so that a producer waiting on a job's side effect
// can stop waiting. busy_ms / idle_ms: time running jobs / waiting for one.
class SerialWorker {
 public:
  SerialWorker(CUcontext ctx, std::function<void()> on_error)
      : on_error_(std::move(on_error)), thread_([this, ctx] { run(ctx); }) {}
  ~SerialWorker() { finish(); }
  void push(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push(std::move(job));
    }
    cv_.notify_one();
  }
  // Runs the queued jobs to the end and joins (rethrow() then reports a job failure).
  void finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable())
      thread_.join();
  }
  void rethrow() {
    if (error_)
      std::rethrow_exception(error_);
  }
  double busy_ms = 0, idle_ms = 0;

 private:
  void run(CUcontext ctx) {
    using clk = std::chrono::steady_clock;
    bool failed = cuCtxSetCurrent(ctx) != CUDA_SUCCESS;
    if (failed) {
      error_ = std::make_exception_ptr(std::runtime_error("SerialWorker: cuCtxSetCurrent failed"));
      on_error_();
    }
    while (true) {
      std::function<void()> job;
      {
        auto t_wait = clk::now();
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return closed_ || !jobs_.empty(); });
        idle_ms += std::chrono::duration<double, std::milli>(clk::now() - t_wait).count();
        if (jobs_.empty())
          return;
        job = std::move(jobs_.front());
        jobs_.pop();
      }
      if (failed)
        continue;
      auto t_job = clk::now();
      try {
        job();
      } catch (...) {
        error_ = std::current_exception();
        failed = true;
        on_error_();
      }
      busy_ms += std::chrono::duration<double, std::milli>(clk::now() - t_job).count();
    }
  }
  std::function<void()> on_error_;
  std::queue<std::function<void()>> jobs_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool closed_ = false;
  std::exception_ptr error_;
  std::thread thread_;  // last: starts after the members above are constructed
};

// File I/O + JSON parse - no shared state, safe to run on worker threads.
boost::json::value read_and_parse_graph_json(const std::string& json_path) {
  std::ifstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for reading: ", json_path);

  std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  return boost::json::parse(json_str);
}

}  // anonymous namespace

namespace foundry {

namespace {

MempoolId_t torch_graph_pool_handle(bool is_user_created = true) {
#if FOUNDRY_HAS_ATEN_CUDA_MEMPOOL
  return at::cuda::MemPool::graph_pool_handle(is_user_created);
#else
  return c10::cuda::MemPool::graph_pool_handle(is_user_created);
#endif
}

}  // namespace

// ============================================================================
// Phase 1b: graph shell creation + generator registration
// ============================================================================

// Create graph shell and register generators, but do NOT replay allocator events.
// Used by start_graph_builds to create shells early before weight loading.
ParsedGraphData CUDAGraph::prepare_graph_shell(boost::json::value&& root_val, MempoolId_t pool,
                                               CUDAGeneratorStateRegistry& registry) {
  const boost::json::object& root = root_val.as_object();

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

  // Register generator states (not thread-safe, modifies global state).
  // Generators may have been extracted by start_graph_builds_impl to defer
  // registration until after weight loading (avoids 2MB VMM pointer advance).
  auto gen_it = root.find("generators");
  if (gen_it != root.end()) {
    const boost::json::array& generators_array = gen_it->value().as_array();
    for (const auto& gen_val : generators_array) {
      const boost::json::object& gen_obj = gen_val.as_object();
      uint64_t state_id = gen_obj.at("id").to_number<uint64_t>();
      uint64_t seed = gen_obj.at("seed").to_number<uint64_t>();
      uint64_t wholegraph_increment = gen_obj.at("wholegraph_increment").to_number<uint64_t>();

      auto state = registry.get_state_from_id(state_id, seed);
      graph->register_generator_state(state, wholegraph_increment);
    }
  }

  ParsedGraphData parsed;
  parsed.graph = std::move(graph);
  parsed.root_val = std::move(root_val);
  parsed.pool = pool;
  return parsed;
}

// ============================================================================
// build_graph_from_parsed: single-pass graph build from pre-parsed data.
// Does JSON traversal, function lookup, CUDA graph API calls, and instantiation.
// ============================================================================

// LOAD-time breakdown of the last template build (ms), printed with the [foundry BUILD] line.
// Categories: json field extraction + kernel-param decode, function lookup, function attributes
// (smem cap / carveout), cuGraphAdd*Node, per-node kernel attributes, dependencies, instantiate.
struct BuildTiming {
  double json_ms = 0, func_ms = 0, func_attr_ms = 0, add_ms = 0, node_attr_ms = 0, deps_ms = 0,
         inst_ms = 0;
  size_t func_attr_calls = 0, node_attr_calls = 0;
};
static thread_local BuildTiming g_last_build_timing;
BuildTiming last_build_timing() {
  return g_last_build_timing;
}
namespace {
using bt_clock = std::chrono::steady_clock;
inline double bt_ms(bt_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(bt_clock::now() - a).count();
}
}  // namespace

GraphLoadResult CUDAGraph::build_graph_from_parsed(ParsedGraphData&& parsed, CUcontext ctx,
                                                   ReconstructTensorFn reconstruct_fn,
                                                   GraphTemplate* out_template) {
  namespace json = boost::json;

  auto& graph = parsed.graph;
  const json::object& root = parsed.root_val.as_object();

  CUgraph cuGraph;
  C10_CUDA_DRIVER_CHECK(cuGraphCreate(&cuGraph, 0));
  graph->graph_ = reinterpret_cast<cudaGraph_t>(cuGraph);
  graph->has_graph_ = true;
  g_last_build_timing = BuildTiming{};
  BuildTiming& bt = g_last_build_timing;

  if (out_template) {
    out_template->cuGraph = cuGraph;
  }

  const json::array& nodes_array = root.at("nodes").as_array();

  // Parse common kernel node attributes (shared by all kernels, set once at graph level)
  bool has_common_cluster_scheduling = false;
  int common_cluster_scheduling = 0;
  bool has_common_mem_sync_domain_map = false;
  unsigned char common_mem_sync_domain_default = 0, common_mem_sync_domain_remote = 0;
  bool has_common_mem_sync_domain = false;
  int common_mem_sync_domain = 0;
  bool has_common_cooperative = false;
  int common_cooperative = 0;
  bool has_common_priority = false;
  int common_priority = 0;
  bool has_common_cluster_dim = false;
  int common_cluster_width = 0, common_cluster_height = 0, common_cluster_depth = 0;
  bool has_common_preferred_cluster_dim = false;
  unsigned int common_preferred_cluster_x = 0, common_preferred_cluster_y = 0,
               common_preferred_cluster_z = 0;
  bool has_common_preferred_shared_mem_carveout = false;
  unsigned int common_preferred_shared_mem_carveout = 0;
  bool has_common_programmatic_stream_serialization = false;
  int common_programmatic_stream_serialization = 0;
  bool has_common_attrs = false;

  if (root.contains("common_kernel_node_attrs")) {
    has_common_attrs = true;
    const json::object& common = root.at("common_kernel_node_attrs").as_object();
    if (common.contains("clusterSchedulingPolicyPreference")) {
      has_common_cluster_scheduling = true;
      common_cluster_scheduling = common.at("clusterSchedulingPolicyPreference").to_number<int>();
    }
    if (common.contains("memSyncDomainMapDefault")) {
      has_common_mem_sync_domain_map = true;
      common_mem_sync_domain_default =
          static_cast<unsigned char>(common.at("memSyncDomainMapDefault").to_number<int>());
      common_mem_sync_domain_remote =
          static_cast<unsigned char>(common.at("memSyncDomainMapRemote").to_number<int>());
    }
    if (common.contains("memSyncDomain")) {
      has_common_mem_sync_domain = true;
      common_mem_sync_domain = common.at("memSyncDomain").to_number<int>();
    }
    if (common.contains("cooperative")) {
      has_common_cooperative = true;
      common_cooperative = common.at("cooperative").to_number<int>();
    }
    if (common.contains("priority")) {
      has_common_priority = true;
      common_priority = common.at("priority").to_number<int>();
    }
    if (common.contains("clusterDimX")) {
      has_common_cluster_dim = true;
      common_cluster_width = common.at("clusterDimX").to_number<int>();
      common_cluster_height = common.at("clusterDimY").to_number<int>();
      common_cluster_depth = common.at("clusterDimZ").to_number<int>();
    }
    if (common.contains("preferredClusterDimX")) {
      has_common_preferred_cluster_dim = true;
      common_preferred_cluster_x = common.at("preferredClusterDimX").to_number<unsigned int>();
      common_preferred_cluster_y = common.at("preferredClusterDimY").to_number<unsigned int>();
      common_preferred_cluster_z = common.at("preferredClusterDimZ").to_number<unsigned int>();
    }
    if (common.contains("preferredSharedMemCarveout")) {
      has_common_preferred_shared_mem_carveout = true;
      common_preferred_shared_mem_carveout =
          common.at("preferredSharedMemCarveout").to_number<unsigned int>();
    }
    if (common.contains("programmaticStreamSerialization")) {
      has_common_programmatic_stream_serialization = true;
      common_programmatic_stream_serialization =
          common.at("programmaticStreamSerialization").to_number<int>();
    }
  }

  // Collect kernel nodes for deferred common-attr application
  std::vector<CUgraphNode> kernel_nodes_for_common_attrs;

  std::unordered_map<int, CUgraphNode> id_to_node;
  std::unordered_map<int, CUevent> event_id_to_event;

  // Owning storage for kernel params (must outlive cuGraphAddKernelNode)
  std::vector<std::vector<std::vector<uint8_t>>> all_kernel_params;
  std::vector<std::vector<void*>> all_param_ptrs;
  std::vector<std::vector<void*>> all_extra_configs;
  std::vector<std::vector<uint8_t>> all_arg_buffers;
  std::vector<size_t> all_arg_buffer_sizes;

  for (const auto& node_val : nodes_array) {
    auto t_nattr = bt_clock::now();
    const json::object& node_obj = node_val.as_object();
    int node_id = node_obj.at("id").to_number<int>();
    std::string node_type = node_obj.at("type").as_string().c_str();
    const json::object& params = node_obj.at("params").as_object();

    CUgraphNode cuNode = nullptr;

    if (node_type == "KernelNode") {
      auto t_json = bt_clock::now();
      CUDA_KERNEL_NODE_PARAMS node_params;
      memset(&node_params, 0, sizeof(node_params));

      node_params.blockDimX = params.at("blockDimX").to_number<unsigned int>();
      node_params.blockDimY = params.at("blockDimY").to_number<unsigned int>();
      node_params.blockDimZ = params.at("blockDimZ").to_number<unsigned int>();
      node_params.gridDimX = params.at("gridDimX").to_number<unsigned int>();
      node_params.gridDimY = params.at("gridDimY").to_number<unsigned int>();
      node_params.gridDimZ = params.at("gridDimZ").to_number<unsigned int>();
      node_params.sharedMemBytes = params.at("sharedMemBytes").to_number<unsigned int>();
      node_params.ctx = ctx;

      std::string function_name = params.at("function_name").as_string().c_str();
      uint64_t binary_hash = params.at("kernel_source_binary_hash").to_number<uint64_t>();

      bt.json_ms += bt_ms(t_json);
      auto t_func = bt_clock::now();
      auto func_handle_variant = query_function_handle(binary_hash, function_name);
      bt.func_ms += bt_ms(t_func);
      t_json = bt_clock::now();
      if (std::holds_alternative<CUkernel>(func_handle_variant)) {
        node_params.kern = std::get<CUkernel>(func_handle_variant);
      } else {
        node_params.func = std::get<CUfunction>(func_handle_variant);
      }

      // Parse kernel node attributes
      int cluster_width = 0, cluster_height = 0, cluster_depth = 0;
      bool has_preferred_cluster_dim = false;
      unsigned int preferred_cluster_x = 0, preferred_cluster_y = 0, preferred_cluster_z = 0;
      bool has_cluster_scheduling = false;
      int cluster_scheduling = 0;
      bool has_cooperative = false;
      int cooperative = 0;
      bool has_priority = false;
      int priority = 0;
      bool has_mem_sync_domain = false;
      int mem_sync_domain = 0;
      bool has_mem_sync_domain_map = false;
      unsigned char mem_sync_domain_default = 0, mem_sync_domain_remote = 0;
      bool has_preferred_shared_mem_carveout = false;
      unsigned int preferred_shared_mem_carveout = 0;
      bool has_access_policy_window = false;
      uint64_t access_policy_base_ptr = 0;
      size_t access_policy_num_bytes = 0;
      float access_policy_hit_ratio = 0.0f;
      int access_policy_hit_prop = 0, access_policy_miss_prop = 0;
      bool has_device_updatable = false;
      int device_updatable = 0;
      bool has_programmatic_stream_serialization = false;
      int programmatic_stream_serialization = 0;

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
        if (node_attrs.contains("programmaticStreamSerialization")) {
          has_programmatic_stream_serialization = true;
          programmatic_stream_serialization =
              node_attrs.at("programmaticStreamSerialization").to_number<int>();
        }
      }

      // Set function attributes
      if (params.contains("func_attrs")) {
        const json::object& func_attrs = params.at("func_attrs").as_object();
        int max_shared = func_attrs.at("max_dynamic_shared_size_bytes").to_number<int>();
        int preferred_carveout = func_attrs.at("preferred_shared_memory_carveout").to_number<int>();

        if (func_attrs.contains("required_cluster_width")) {
          int v = func_attrs.at("required_cluster_width").to_number<int>();
          if (v > 0)
            cluster_width = v;
        }
        if (func_attrs.contains("required_cluster_height")) {
          int v = func_attrs.at("required_cluster_height").to_number<int>();
          if (v > 0)
            cluster_height = v;
        }
        if (func_attrs.contains("required_cluster_depth")) {
          int v = func_attrs.at("required_cluster_depth").to_number<int>();
          if (v > 0)
            cluster_depth = v;
        }

        bt.json_ms += bt_ms(t_json);
        auto t_fa = bt_clock::now();
        apply_saved_function_attributes(func_handle_variant, graph->capture_dev_, max_shared,
                                        preferred_carveout, "LOAD");
        bt.func_attr_ms += bt_ms(t_fa);
        bt.func_attr_calls++;
        t_json = bt_clock::now();
      }

      // Decode kernelParams
      const json::array& kernel_params_array = params.at("kernelParams").as_array();
      int num_params = kernel_params_array.size();
      bool has_kernel_params = false;
      if (num_params > 0) {
        has_kernel_params = kernel_params_array[0].as_object().contains("value_hex");
      }

      if (has_kernel_params) {
        all_kernel_params.emplace_back(num_params);
        all_param_ptrs.emplace_back(num_params);
        auto& param_data = all_kernel_params.back();
        auto& param_ptrs = all_param_ptrs.back();

        for (int i = 0; i < num_params; ++i) {
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

      // Decode extra config
      const json::array& extra_array = params.at("extra").as_array();
      if (!extra_array.empty()) {
        all_extra_configs.emplace_back();
        auto& extra_config = all_extra_configs.back();

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
              std::string argBuffer_hex = params.at("extra_argBuffer_hex").as_string().c_str();
              if (!argBuffer_hex.empty()) {
                all_arg_buffers.emplace_back();
                auto& arg_buffer = all_arg_buffers.back();
                arg_buffer.resize(argBuffer_hex.length() / 2);
                for (size_t j = 0; j < arg_buffer.size(); ++j) {
                  std::string byte_str = argBuffer_hex.substr(j * 2, 2);
                  arg_buffer[j] = std::stoul(byte_str, nullptr, 16);
                }
                extra_config.push_back(arg_buffer.data());
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

      // Add kernel node
      bt.json_ms += bt_ms(t_json);
      auto t_add = bt_clock::now();
      CUresult kernel_result =
          add_restored_kernel_node(&cuNode, cuGraph, node_params, graph->capture_dev_,
                                   cluster_width, cluster_height, cluster_depth, "LOAD");
      bt.add_ms += bt_ms(t_add);
      t_nattr = bt_clock::now();
      if (kernel_result != CUDA_SUCCESS) {
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddKernelNode FAILED for node %d with error %d\n",
                node_id, kernel_result);
        fprintf(stderr, "[foundry LOAD ERROR]   function: %s\n", function_name.c_str());
        fprintf(stderr, "[foundry LOAD ERROR]   grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                node_params.gridDimX, node_params.gridDimY, node_params.gridDimZ,
                node_params.blockDimX, node_params.blockDimY, node_params.blockDimZ,
                node_params.sharedMemBytes);
        C10_CUDA_DRIVER_CHECK(kernel_result);
      }

      // Set kernel node attributes.
      // Common attrs (shared by all kernels) are applied in batch after the node loop.
      // Per-node attrs (if any remain after common extraction) are applied here.
      if (has_common_attrs) {
        kernel_nodes_for_common_attrs.push_back(cuNode);
      }

      // Per-node attributes (only present if they differ from common or no common extraction)
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
        apw_attr.accessPolicyWindow.missProp =
            static_cast<CUaccessProperty>(access_policy_miss_prop);
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
      if (has_programmatic_stream_serialization) {
        CUkernelNodeAttrValue pss_attr;
        memset(&pss_attr, 0, sizeof(pss_attr));
        pss_attr.programmaticStreamSerializationAllowed = programmatic_stream_serialization;
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            cuNode,
            static_cast<CUkernelNodeAttrID>(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION),
            &pss_attr));
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

      CUresult result = cuGraphAddMemcpyNode(&cuNode, cuGraph, nullptr, 0, &copy_params, ctx);
      if (result != CUDA_SUCCESS) {
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddMemcpyNode FAILED for node %d with error %d\n",
                node_id, result);
        C10_CUDA_DRIVER_CHECK(result);
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

      CUresult result = cuGraphAddMemsetNode(&cuNode, cuGraph, nullptr, 0, &memset_params, ctx);
      if (result != CUDA_SUCCESS) {
        fprintf(stderr,
                "[foundry LOAD ERROR] cuGraphAddMemsetNode FAILED for node %d with error %d\n",
                node_id, result);
        C10_CUDA_DRIVER_CHECK(result);
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
      if (node_type == "KernelNode") {
        bt.node_attr_ms += bt_ms(t_nattr);
      }
      id_to_node[node_id] = cuNode;
      if (out_template) {
        out_template->ordered_nodes.push_back(cuNode);
        out_template->node_types.push_back(node_type);
      }
    }
  }

  // Apply common kernel node attributes in batch (one set of attr values for all kernel nodes)
  auto t_cattr = bt_clock::now();
  if (has_common_attrs && !kernel_nodes_for_common_attrs.empty()) {
    if (has_common_cluster_dim) {
      CUkernelNodeAttrValue clusterAttr;
      memset(&clusterAttr, 0, sizeof(clusterAttr));
      clusterAttr.clusterDim.x = common_cluster_width > 0 ? common_cluster_width : 1;
      clusterAttr.clusterDim.y = common_cluster_height > 0 ? common_cluster_height : 1;
      clusterAttr.clusterDim.z = common_cluster_depth > 0 ? common_cluster_depth : 1;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &clusterAttr));
      }
    }
    if (has_common_preferred_cluster_dim) {
      CUkernelNodeAttrValue pref_attr;
      memset(&pref_attr, 0, sizeof(pref_attr));
      pref_attr.preferredClusterDim.x = common_preferred_cluster_x;
      pref_attr.preferredClusterDim.y = common_preferred_cluster_y;
      pref_attr.preferredClusterDim.z = common_preferred_cluster_z;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &pref_attr));
      }
    }
    if (has_common_cluster_scheduling) {
      CUkernelNodeAttrValue sched_attr;
      memset(&sched_attr, 0, sizeof(sched_attr));
      sched_attr.clusterSchedulingPolicyPreference =
          static_cast<CUclusterSchedulingPolicy>(common_cluster_scheduling);
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &sched_attr));
      }
    }
    if (has_common_cooperative) {
      CUkernelNodeAttrValue coop_attr;
      memset(&coop_attr, 0, sizeof(coop_attr));
      coop_attr.cooperative = common_cooperative;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(
            cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &coop_attr));
      }
    }
    if (has_common_priority) {
      CUkernelNodeAttrValue priority_attr;
      memset(&priority_attr, 0, sizeof(priority_attr));
      priority_attr.priority = common_priority;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(
            cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &priority_attr));
      }
    }
    if (has_common_mem_sync_domain) {
      CUkernelNodeAttrValue mem_domain_attr;
      memset(&mem_domain_attr, 0, sizeof(mem_domain_attr));
      mem_domain_attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(common_mem_sync_domain);
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &mem_domain_attr));
      }
    }
    if (has_common_mem_sync_domain_map) {
      CUkernelNodeAttrValue mem_map_attr;
      memset(&mem_map_attr, 0, sizeof(mem_map_attr));
      mem_map_attr.memSyncDomainMap.default_ = common_mem_sync_domain_default;
      mem_map_attr.memSyncDomainMap.remote = common_mem_sync_domain_remote;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &mem_map_attr));
      }
    }
    if (has_common_preferred_shared_mem_carveout) {
      CUkernelNodeAttrValue carveout_attr;
      memset(&carveout_attr, 0, sizeof(carveout_attr));
      carveout_attr.sharedMemCarveout = common_preferred_shared_mem_carveout;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &carveout_attr));
      }
    }
    if (has_common_programmatic_stream_serialization) {
      CUkernelNodeAttrValue pss_attr;
      memset(&pss_attr, 0, sizeof(pss_attr));
      pss_attr.programmaticStreamSerializationAllowed = common_programmatic_stream_serialization;
      for (auto node : kernel_nodes_for_common_attrs) {
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
            node,
            static_cast<CUkernelNodeAttrID>(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION),
            &pss_attr));
      }
    }
  }

  // Add dependencies
  bt.node_attr_ms += bt_ms(t_cattr);
  auto t_deps = bt_clock::now();
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
      from_nodes.push_back(id_to_node[dep_obj.at("from").to_number<int>()]);
      to_nodes.push_back(id_to_node[dep_obj.at("to").to_number<int>()]);
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
      C10_CUDA_DRIVER_CHECK(dep_result);
    }
  }

  bt.deps_ms += bt_ms(t_deps);
  // Instantiate
  auto t_inst = bt_clock::now();
  graph->capture_ended_ = true;
  graph->instantiate();
  bt.inst_ms += bt_ms(t_inst);
  // NOTE(yongji): bypass destructor's release memory pool call
  graph->capture_ended_ = false;

  // Reconstruct output tensors
  GraphLoadResult load_result;
  load_result.graph = graph;
  load_result.output_type = OutputTensorType::None;
  load_result.outputs = std::monostate{};

  if (root.contains("output_tensors")) {
    const json::object& output_tensors_obj = root.at("output_tensors").as_object();
    load_result.output_type =
        static_cast<OutputTensorType>(output_tensors_obj.at("type").to_number<int>());

    const json::array& tensors_array = output_tensors_obj.at("tensors").as_array();
    if (load_result.output_type == OutputTensorType::Single && tensors_array.size() == 1) {
      load_result.outputs = reconstruct_fn(tensors_array[0].as_object());
    } else if (load_result.output_type == OutputTensorType::List ||
               load_result.output_type == OutputTensorType::Tuple) {
      std::vector<at::Tensor> tensors;
      tensors.reserve(tensors_array.size());
      for (const auto& t_val : tensors_array) {
        tensors.push_back(reconstruct_fn(t_val.as_object()));
      }
      load_result.outputs = std::move(tensors);
    }
  }

  return load_result;
}

// ============================================================================
// prepare_on_demand_graph: parse node params without building a CUgraph.
//
// A member of a topology group never builds its own CUgraph. Its node params
// are pre-decoded here (CPU only, runs in parallel with the template builds)
// and later, in materialize_on_demand_exec(), written onto the group's shared
// builder graph with cuGraph*NodeSetParams, from which the member instantiates
// a DEDICATED exec (~5 ms per ~1000-node graph). Every member is instantiated
// at load by default (Phase 2c); FOUNDRY_LAZY_GRAPH_EXEC=1 defers it to the
// first replay. Nothing calls cuGraphExecUpdate: an exec updated in place
// launches every node slower for the rest of its life.
//
// Savings vs full build_graph_from_parsed:
//   - No cuGraphCreate / cuGraphAdd*Node / cuGraphAddDependencies per member
//   - Parsing is pure CPU-side JSON/binary decoding
// ============================================================================

void CUDAGraph::prepare_on_demand_graph(ParsedGraphData& parsed, CUcontext ctx,
                                        std::shared_ptr<SharedGraphExec> shared_exec,
                                        int graph_id) {
  namespace json = boost::json;

  auto& graph = parsed.graph;
  const json::object& root = parsed.root_val.as_object();
  const json::array& nodes_array = root.at("nodes").as_array();

  auto data = std::make_unique<OnDemandData>();
  data->graph_id = graph_id;
  data->updates.resize(nodes_array.size());

  // shared_exec may be nullptr when on-demand prep runs in parallel with
  // template builds.  link_on_demand_shared_exec() sets it later.
  if (shared_exec) {
    TORCH_CHECK(nodes_array.size() == shared_exec->ordered_nodes.size(),
                "On-demand mismatch: JSON has ", nodes_array.size(), " nodes but template has ",
                shared_exec->ordered_nodes.size());
    data->shared_exec = shared_exec;
  }

  // Parse common kernel node attributes (shared by all kernels in this graph)
  OnDemandNodeUpdate::KernelNodeAttrs common_attrs;
  if (root.contains("common_kernel_node_attrs")) {
    const json::object& common = root.at("common_kernel_node_attrs").as_object();
    if (common.contains("clusterDimX")) {
      common_attrs.has_cluster_dim = true;
      common_attrs.clusterDimX = common.at("clusterDimX").to_number<unsigned int>();
      common_attrs.clusterDimY = common.at("clusterDimY").to_number<unsigned int>();
      common_attrs.clusterDimZ = common.at("clusterDimZ").to_number<unsigned int>();
    }
    if (common.contains("preferredClusterDimX")) {
      common_attrs.has_preferred_cluster_dim = true;
      common_attrs.preferredClusterDimX =
          common.at("preferredClusterDimX").to_number<unsigned int>();
      common_attrs.preferredClusterDimY =
          common.at("preferredClusterDimY").to_number<unsigned int>();
      common_attrs.preferredClusterDimZ =
          common.at("preferredClusterDimZ").to_number<unsigned int>();
    }
    if (common.contains("clusterSchedulingPolicyPreference"))
      common_attrs.clusterSchedulingPolicy =
          common.at("clusterSchedulingPolicyPreference").to_number<int>();
    if (common.contains("cooperative"))
      common_attrs.cooperative = common.at("cooperative").to_number<int>();
    if (common.contains("priority"))
      common_attrs.priority = common.at("priority").to_number<int>();
    if (common.contains("memSyncDomain"))
      common_attrs.memSyncDomain = common.at("memSyncDomain").to_number<int>();
    if (common.contains("memSyncDomainMapDefault")) {
      common_attrs.has_mem_sync_domain_map = true;
      common_attrs.memSyncDomainMapDefault =
          static_cast<unsigned char>(common.at("memSyncDomainMapDefault").to_number<int>());
      common_attrs.memSyncDomainMapRemote =
          static_cast<unsigned char>(common.at("memSyncDomainMapRemote").to_number<int>());
    }
    if (common.contains("preferredSharedMemCarveout")) {
      common_attrs.has_shared_mem_carveout = true;
      common_attrs.sharedMemCarveout =
          common.at("preferredSharedMemCarveout").to_number<unsigned int>();
    }
  }

  // Track events by ID so record/wait pairs share the same CUevent
  std::unordered_map<int, CUevent> event_id_to_event;

  for (size_t idx = 0; idx < nodes_array.size(); ++idx) {
    const json::object& node_obj = nodes_array[idx].as_object();
    std::string node_type = node_obj.at("type").as_string().c_str();
    const json::object& params = node_obj.at("params").as_object();

    auto& u = data->updates[idx];

    if (node_type == "KernelNode") {
      u.type = OnDemandNodeUpdate::Kernel;
      memset(&u.kernel_params, 0, sizeof(u.kernel_params));

      u.kernel_params.blockDimX = params.at("blockDimX").to_number<unsigned int>();
      u.kernel_params.blockDimY = params.at("blockDimY").to_number<unsigned int>();
      u.kernel_params.blockDimZ = params.at("blockDimZ").to_number<unsigned int>();
      u.kernel_params.gridDimX = params.at("gridDimX").to_number<unsigned int>();
      u.kernel_params.gridDimY = params.at("gridDimY").to_number<unsigned int>();
      u.kernel_params.gridDimZ = params.at("gridDimZ").to_number<unsigned int>();
      u.kernel_params.sharedMemBytes = params.at("sharedMemBytes").to_number<unsigned int>();
      u.kernel_params.ctx = ctx;

      std::string function_name = params.at("function_name").as_string().c_str();
      uint64_t binary_hash = params.at("kernel_source_binary_hash").to_number<uint64_t>();

      auto func_handle_variant = query_function_handle(binary_hash, function_name);
      if (std::holds_alternative<CUkernel>(func_handle_variant)) {
        u.kernel_params.kern = std::get<CUkernel>(func_handle_variant);
      } else {
        u.kernel_params.func = std::get<CUfunction>(func_handle_variant);
      }

      // Set function attributes (max_dynamic_shared_size_bytes, etc.)
      // With type-only topology grouping, the kernel function may differ
      // from the template's, so we must set attrs on each kernel handle.
      int cluster_width = 0, cluster_height = 0, cluster_depth = 0;
      if (params.contains("func_attrs")) {
        const json::object& func_attrs = params.at("func_attrs").as_object();
        int max_shared = func_attrs.at("max_dynamic_shared_size_bytes").to_number<int>();
        int preferred_carveout = func_attrs.at("preferred_shared_memory_carveout").to_number<int>();

        if (func_attrs.contains("required_cluster_width")) {
          int v = func_attrs.at("required_cluster_width").to_number<int>();
          if (v > 0)
            cluster_width = v;
        }
        if (func_attrs.contains("required_cluster_height")) {
          int v = func_attrs.at("required_cluster_height").to_number<int>();
          if (v > 0)
            cluster_height = v;
        }
        if (func_attrs.contains("required_cluster_depth")) {
          int v = func_attrs.at("required_cluster_depth").to_number<int>();
          if (v > 0)
            cluster_depth = v;
        }

        apply_saved_function_attributes(func_handle_variant, parsed.graph->capture_dev_, max_shared,
                                        preferred_carveout, "LOAD");
      }

      // Resolve kernel node attributes: start from common, override with per-node
      u.kernel_attrs = common_attrs;

      if (params.contains("kernel_node_attrs")) {
        const json::object& node_attrs = params.at("kernel_node_attrs").as_object();
        if (node_attrs.contains("clusterDimX")) {
          u.kernel_attrs.has_cluster_dim = true;
          u.kernel_attrs.clusterDimX = node_attrs.at("clusterDimX").to_number<unsigned int>();
          u.kernel_attrs.clusterDimY = node_attrs.at("clusterDimY").to_number<unsigned int>();
          u.kernel_attrs.clusterDimZ = node_attrs.at("clusterDimZ").to_number<unsigned int>();
        }
        if (node_attrs.contains("preferredClusterDimX")) {
          u.kernel_attrs.has_preferred_cluster_dim = true;
          u.kernel_attrs.preferredClusterDimX =
              node_attrs.at("preferredClusterDimX").to_number<unsigned int>();
          u.kernel_attrs.preferredClusterDimY =
              node_attrs.at("preferredClusterDimY").to_number<unsigned int>();
          u.kernel_attrs.preferredClusterDimZ =
              node_attrs.at("preferredClusterDimZ").to_number<unsigned int>();
        }
        if (node_attrs.contains("clusterSchedulingPolicyPreference"))
          u.kernel_attrs.clusterSchedulingPolicy =
              node_attrs.at("clusterSchedulingPolicyPreference").to_number<int>();
        if (node_attrs.contains("cooperative"))
          u.kernel_attrs.cooperative = node_attrs.at("cooperative").to_number<int>();
        if (node_attrs.contains("priority"))
          u.kernel_attrs.priority = node_attrs.at("priority").to_number<int>();
        if (node_attrs.contains("memSyncDomain"))
          u.kernel_attrs.memSyncDomain = node_attrs.at("memSyncDomain").to_number<int>();
        if (node_attrs.contains("memSyncDomainMapDefault")) {
          u.kernel_attrs.has_mem_sync_domain_map = true;
          u.kernel_attrs.memSyncDomainMapDefault =
              static_cast<unsigned char>(node_attrs.at("memSyncDomainMapDefault").to_number<int>());
          u.kernel_attrs.memSyncDomainMapRemote =
              static_cast<unsigned char>(node_attrs.at("memSyncDomainMapRemote").to_number<int>());
        }
        if (node_attrs.contains("preferredSharedMemCarveout")) {
          u.kernel_attrs.has_shared_mem_carveout = true;
          u.kernel_attrs.sharedMemCarveout =
              node_attrs.at("preferredSharedMemCarveout").to_number<unsigned int>();
        }
      }

      // func_attrs required_cluster overrides per-node cluster dims
      if (cluster_width > 0 || cluster_height > 0 || cluster_depth > 0) {
        u.kernel_attrs.has_cluster_dim = true;
        u.kernel_attrs.clusterDimX = cluster_width > 0 ? cluster_width : 1;
        u.kernel_attrs.clusterDimY = cluster_height > 0 ? cluster_height : 1;
        u.kernel_attrs.clusterDimZ = cluster_depth > 0 ? cluster_depth : 1;
      }

      // NOTE: With cluster dim presence in the topology key, graphs with
      // different cluster dim presence are never in the same group.
      // No need to force has_cluster_dim=true — that would cause
      // "cluster misconfiguration" errors on non-clustered kernels.

      // Decode kernelParams
      const json::array& kernel_params_array = params.at("kernelParams").as_array();
      int num_params = kernel_params_array.size();
      bool has_kernel_params = false;
      if (num_params > 0) {
        has_kernel_params = kernel_params_array[0].as_object().contains("value_hex");
      }

      if (has_kernel_params) {
        u.kernel_param_data.resize(num_params);
        for (int i = 0; i < num_params; ++i) {
          const json::object& param_obj = kernel_params_array[i].as_object();
          size_t param_size = param_obj.at("size").to_number<size_t>();
          std::string value_hex = param_obj.at("value_hex").as_string().c_str();

          u.kernel_param_data[i].resize(param_size);
          for (size_t j = 0; j < param_size; ++j) {
            std::string byte_str = value_hex.substr(j * 2, 2);
            u.kernel_param_data[i][j] = std::stoul(byte_str, nullptr, 16);
          }
        }
        // kernel_param_ptrs set in fixup_pointers()
      }

      // Decode extra config — only store the arg buffer and size
      const json::array& extra_array = params.at("extra").as_array();
      if (!extra_array.empty()) {
        for (size_t i = 0; i < extra_array.size(); ++i) {
          if (extra_array[i].is_string()) {
            std::string str_val = extra_array[i].as_string().c_str();
            if (str_val == "null") {
              std::string argBuffer_hex = params.at("extra_argBuffer_hex").as_string().c_str();
              if (!argBuffer_hex.empty()) {
                u.arg_buffer.resize(argBuffer_hex.length() / 2);
                for (size_t j = 0; j < u.arg_buffer.size(); ++j) {
                  std::string byte_str = argBuffer_hex.substr(j * 2, 2);
                  u.arg_buffer[j] = std::stoul(byte_str, nullptr, 16);
                }
              }
            }
          } else if (extra_array[i].is_uint64() || extra_array[i].is_int64()) {
            uint64_t val = extra_array[i].to_number<uint64_t>();
            // Check if this is the buffer size (follows CU_LAUNCH_PARAM_BUFFER_SIZE)
            if (i > 0 && extra_array[i - 1].is_string() &&
                extra_array[i - 1].as_string() == "CU_LAUNCH_PARAM_BUFFER_SIZE") {
              u.arg_buffer_size = val;
            }
          }
        }
        // extra_config rebuilt in fixup_pointers()
      }

    } else if (node_type == "MemsetNode") {
      u.type = OnDemandNodeUpdate::Memset;
      memset(&u.memset_params, 0, sizeof(u.memset_params));

      u.memset_params.dst = params.at("dst").to_number<CUdeviceptr>();
      u.memset_params.elementSize = params.at("elementSize").to_number<unsigned int>();
      u.memset_params.height = params.at("height").to_number<size_t>();
      u.memset_params.pitch = params.at("pitch").to_number<size_t>();
      u.memset_params.value = params.at("value").to_number<unsigned int>();
      u.memset_params.width = params.at("width").to_number<size_t>();

    } else if (node_type == "MemcpyNode") {
      u.type = OnDemandNodeUpdate::Memcpy;
      memset(&u.memcpy_params, 0, sizeof(u.memcpy_params));

      u.memcpy_params.Depth = params.at("Depth").to_number<size_t>();
      u.memcpy_params.Height = params.at("Height").to_number<size_t>();
      u.memcpy_params.WidthInBytes = params.at("WidthInBytes").to_number<size_t>();
      u.memcpy_params.dstDevice = params.at("dstDevice").to_number<CUdeviceptr>();
      u.memcpy_params.dstHeight = params.at("dstHeight").to_number<size_t>();
      u.memcpy_params.dstLOD = params.at("dstLOD").to_number<size_t>();
      u.memcpy_params.dstMemoryType =
          static_cast<CUmemorytype>(params.at("dstMemoryType").to_number<int>());
      u.memcpy_params.dstPitch = params.at("dstPitch").to_number<size_t>();
      u.memcpy_params.dstXInBytes = params.at("dstXInBytes").to_number<size_t>();
      u.memcpy_params.dstY = params.at("dstY").to_number<size_t>();
      u.memcpy_params.dstZ = params.at("dstZ").to_number<size_t>();
      u.memcpy_params.srcDevice = params.at("srcDevice").to_number<CUdeviceptr>();
      u.memcpy_params.srcHeight = params.at("srcHeight").to_number<size_t>();
      u.memcpy_params.srcLOD = params.at("srcLOD").to_number<size_t>();
      u.memcpy_params.srcMemoryType =
          static_cast<CUmemorytype>(params.at("srcMemoryType").to_number<int>());
      u.memcpy_params.srcPitch = params.at("srcPitch").to_number<size_t>();
      u.memcpy_params.srcXInBytes = params.at("srcXInBytes").to_number<size_t>();
      u.memcpy_params.srcY = params.at("srcY").to_number<size_t>();
      u.memcpy_params.srcZ = params.at("srcZ").to_number<size_t>();

    } else if (node_type == "EventRecordNode") {
      u.type = OnDemandNodeUpdate::EventRecord;
      int event_id = params.at("event_id").to_number<int>();
      CUevent event;
      auto eit = event_id_to_event.find(event_id);
      if (eit == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        if (!graph->loaded_graph_resources_) {
          graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();
        }
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = eit->second;
      }
      u.event = event;

    } else if (node_type == "EventWaitNode") {
      u.type = OnDemandNodeUpdate::EventWait;
      int event_id = params.at("event_id").to_number<int>();
      CUevent event;
      auto eit = event_id_to_event.find(event_id);
      if (eit == event_id_to_event.end()) {
        C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        event_id_to_event[event_id] = event;
        if (!graph->loaded_graph_resources_) {
          graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();
        }
        graph->loaded_graph_resources_->created_events.push_back(event);
      } else {
        event = eit->second;
      }
      u.event = event;

    } else if (node_type == "EmptyNode") {
      u.type = OnDemandNodeUpdate::Empty;
    }
  }

  // Fix up internal pointers now that updates vector is finalized
  for (auto& u : data->updates) {
    u.fixup_pointers();
  }

  graph->on_demand_data_ = std::move(data);
}

// ============================================================================
// Binary-native prepare_on_demand_graph: reads directly from BinaryGraphFile.
// No JSON parsing, no hex decoding. Raw memcpy for kernel param data.
// ============================================================================

void CUDAGraph::prepare_on_demand_graph_binary(const BinaryGraphFile& bin_file,
                                               std::shared_ptr<CUDAGraph> graph, CUcontext ctx,
                                               std::shared_ptr<SharedGraphExec> shared_exec,
                                               int graph_id) {
  namespace bf = binary_format;

  const bf::BinNodeEntry* nodes = bin_file.node_table();
  uint32_t num_nodes = bin_file.header.num_nodes;

  auto data = std::make_unique<OnDemandData>();
  data->graph_id = graph_id;
  data->updates.resize(num_nodes);

  if (shared_exec) {
    TORCH_CHECK(num_nodes == shared_exec->ordered_nodes.size(), "On-demand mismatch: binary has ",
                num_nodes, " nodes but template has ", shared_exec->ordered_nodes.size());
    data->shared_exec = shared_exec;
  }

  // Parse common kernel attrs from binary
  OnDemandNodeUpdate::KernelNodeAttrs common_attrs;
  if (bin_file.header.flags & bf::FLAG_HAS_COMMON_KERNEL_ATTRS) {
    const bf::BinCommonKernelAttrs* ca =
        bin_file.section_ptr<bf::BinCommonKernelAttrs>(bf::SECTION_COMMON_KERNEL_ATTRS);
    if (ca->flags & bf::KNA_CLUSTER_DIM) {
      common_attrs.has_cluster_dim = true;
      common_attrs.clusterDimX = ca->clusterDimX;
      common_attrs.clusterDimY = ca->clusterDimY;
      common_attrs.clusterDimZ = ca->clusterDimZ;
    }
    if (ca->flags & bf::KNA_PREFERRED_CLUSTER_DIM) {
      common_attrs.has_preferred_cluster_dim = true;
      common_attrs.preferredClusterDimX = ca->preferredClusterDimX;
      common_attrs.preferredClusterDimY = ca->preferredClusterDimY;
      common_attrs.preferredClusterDimZ = ca->preferredClusterDimZ;
    }
    if (ca->flags & bf::KNA_CLUSTER_SCHEDULING)
      common_attrs.clusterSchedulingPolicy = ca->clusterSchedulingPolicy;
    if (ca->flags & bf::KNA_COOPERATIVE)
      common_attrs.cooperative = ca->cooperative;
    if (ca->flags & bf::KNA_PRIORITY)
      common_attrs.priority = ca->priority;
    if (ca->flags & bf::KNA_MEM_SYNC_DOMAIN)
      common_attrs.memSyncDomain = ca->memSyncDomain;
    if (ca->flags & bf::KNA_MEM_SYNC_DOMAIN_MAP) {
      common_attrs.has_mem_sync_domain_map = true;
      common_attrs.memSyncDomainMapDefault = ca->memSyncDomainMapDefault;
      common_attrs.memSyncDomainMapRemote = ca->memSyncDomainMapRemote;
    }
    if (ca->flags & bf::KNA_SHARED_MEM_CARVEOUT) {
      common_attrs.has_shared_mem_carveout = true;
      common_attrs.sharedMemCarveout = ca->preferredSharedMemCarveout;
    }
  }

  // Helper to convert binary kna flags to KernelNodeAttrs
  auto kna_from_bin = [](const bf::BinKernelNode& k) -> OnDemandNodeUpdate::KernelNodeAttrs {
    OnDemandNodeUpdate::KernelNodeAttrs a;
    if (k.kna_flags & bf::KNA_CLUSTER_DIM) {
      a.has_cluster_dim = true;
      a.clusterDimX = k.kna_clusterDimX;
      a.clusterDimY = k.kna_clusterDimY;
      a.clusterDimZ = k.kna_clusterDimZ;
    }
    if (k.kna_flags & bf::KNA_PREFERRED_CLUSTER_DIM) {
      a.has_preferred_cluster_dim = true;
      a.preferredClusterDimX = k.kna_preferredClusterDimX;
      a.preferredClusterDimY = k.kna_preferredClusterDimY;
      a.preferredClusterDimZ = k.kna_preferredClusterDimZ;
    }
    if (k.kna_flags & bf::KNA_CLUSTER_SCHEDULING)
      a.clusterSchedulingPolicy = k.kna_clusterSchedulingPolicy;
    if (k.kna_flags & bf::KNA_COOPERATIVE)
      a.cooperative = k.kna_cooperative;
    if (k.kna_flags & bf::KNA_PRIORITY)
      a.priority = k.kna_priority;
    if (k.kna_flags & bf::KNA_MEM_SYNC_DOMAIN)
      a.memSyncDomain = k.kna_memSyncDomain;
    if (k.kna_flags & bf::KNA_MEM_SYNC_DOMAIN_MAP) {
      a.has_mem_sync_domain_map = true;
      a.memSyncDomainMapDefault = k.kna_memSyncDomainMapDefault;
      a.memSyncDomainMapRemote = k.kna_memSyncDomainMapRemote;
    }
    if (k.kna_flags & bf::KNA_SHARED_MEM_CARVEOUT) {
      a.has_shared_mem_carveout = true;
      a.sharedMemCarveout = k.kna_preferredSharedMemCarveout;
    }
    return a;
  };

  const bf::BinParamEntry* param_idx = bin_file.param_index();
  const uint8_t* pd = bin_file.param_data();
  const uint8_t* abd = bin_file.arg_buffer_data();

  std::unordered_map<int, CUevent> event_id_to_event;

  for (uint32_t idx = 0; idx < num_nodes; idx++) {
    const bf::BinNodeEntry& entry = nodes[idx];
    auto& u = data->updates[idx];

    switch (entry.type) {
      case bf::NODE_KERNEL: {
        u.type = OnDemandNodeUpdate::Kernel;
        const auto& k = entry.kernel;
        memset(&u.kernel_params, 0, sizeof(u.kernel_params));

        u.kernel_params.blockDimX = k.blockDimX;
        u.kernel_params.blockDimY = k.blockDimY;
        u.kernel_params.blockDimZ = k.blockDimZ;
        u.kernel_params.gridDimX = k.gridDimX;
        u.kernel_params.gridDimY = k.gridDimY;
        u.kernel_params.gridDimZ = k.gridDimZ;
        u.kernel_params.sharedMemBytes = k.sharedMemBytes;
        u.kernel_params.ctx = ctx;

        // Resolve function handle
        std::string function_name =
            bin_file.get_string(k.function_name_offset, k.function_name_length);
        auto func_handle_variant = query_function_handle(k.binary_hash, function_name);
        if (std::holds_alternative<CUkernel>(func_handle_variant)) {
          u.kernel_params.kern = std::get<CUkernel>(func_handle_variant);
        } else {
          u.kernel_params.func = std::get<CUfunction>(func_handle_variant);
        }

        apply_saved_function_attributes(func_handle_variant, graph->capture_dev_,
                                        k.max_dynamic_shared_size_bytes,
                                        k.preferred_shared_memory_carveout, "LOAD");

        // Kernel node attrs: start from common, override with per-node
        u.kernel_attrs = common_attrs;
        if (k.kna_flags != 0) {
          auto per_node = kna_from_bin(k);
          // Merge: per-node overrides common
          if (per_node.has_cluster_dim) {
            u.kernel_attrs.has_cluster_dim = true;
            u.kernel_attrs.clusterDimX = per_node.clusterDimX;
            u.kernel_attrs.clusterDimY = per_node.clusterDimY;
            u.kernel_attrs.clusterDimZ = per_node.clusterDimZ;
          }
          if (per_node.has_preferred_cluster_dim) {
            u.kernel_attrs.has_preferred_cluster_dim = true;
            u.kernel_attrs.preferredClusterDimX = per_node.preferredClusterDimX;
            u.kernel_attrs.preferredClusterDimY = per_node.preferredClusterDimY;
            u.kernel_attrs.preferredClusterDimZ = per_node.preferredClusterDimZ;
          }
          if (per_node.clusterSchedulingPolicy >= 0)
            u.kernel_attrs.clusterSchedulingPolicy = per_node.clusterSchedulingPolicy;
          if (per_node.cooperative >= 0)
            u.kernel_attrs.cooperative = per_node.cooperative;
          if (per_node.priority >= 0)
            u.kernel_attrs.priority = per_node.priority;
          if (per_node.memSyncDomain >= 0)
            u.kernel_attrs.memSyncDomain = per_node.memSyncDomain;
          if (per_node.has_mem_sync_domain_map) {
            u.kernel_attrs.has_mem_sync_domain_map = true;
            u.kernel_attrs.memSyncDomainMapDefault = per_node.memSyncDomainMapDefault;
            u.kernel_attrs.memSyncDomainMapRemote = per_node.memSyncDomainMapRemote;
          }
          if (per_node.has_shared_mem_carveout) {
            u.kernel_attrs.has_shared_mem_carveout = true;
            u.kernel_attrs.sharedMemCarveout = per_node.sharedMemCarveout;
          }
        }

        // func_attrs required_cluster overrides
        if (k.required_cluster_width > 0 || k.required_cluster_height > 0 ||
            k.required_cluster_depth > 0) {
          u.kernel_attrs.has_cluster_dim = true;
          u.kernel_attrs.clusterDimX = k.required_cluster_width > 0 ? k.required_cluster_width : 1;
          u.kernel_attrs.clusterDimY =
              k.required_cluster_height > 0 ? k.required_cluster_height : 1;
          u.kernel_attrs.clusterDimZ = k.required_cluster_depth > 0 ? k.required_cluster_depth : 1;
        }

        // Decode kernel params — DIRECT MEMCPY from binary, no hex decode!
        const bf::BinParamEntry* pe = reinterpret_cast<const bf::BinParamEntry*>(
            reinterpret_cast<const uint8_t*>(param_idx) + k.param_index_offset);

        if (k.arg_buffer_offset != 0xFFFFFFFF) {
          // Extra-style: copy arg buffer directly
          u.arg_buffer.resize(k.arg_buffer_size);
          memcpy(u.arg_buffer.data(), abd + k.arg_buffer_offset, k.arg_buffer_size);
          u.arg_buffer_size = k.arg_buffer_size;
          // extra_config rebuilt in fixup_pointers()
        } else {
          // Per-param style: copy each param's raw bytes
          u.kernel_param_data.resize(k.num_params);
          for (uint32_t j = 0; j < k.num_params; j++) {
            u.kernel_param_data[j].resize(pe[j].size);
            memcpy(u.kernel_param_data[j].data(), pd + pe[j].data_offset, pe[j].size);
          }
          // kernel_param_ptrs set in fixup_pointers()
        }
        break;
      }
      case bf::NODE_MEMSET: {
        u.type = OnDemandNodeUpdate::Memset;
        const auto& m = entry.memset;
        memset(&u.memset_params, 0, sizeof(u.memset_params));
        u.memset_params.dst = m.dst;
        u.memset_params.elementSize = m.elementSize;
        u.memset_params.height = m.height;
        u.memset_params.pitch = m.pitch;
        u.memset_params.value = m.value;
        u.memset_params.width = m.width;
        break;
      }
      case bf::NODE_MEMCPY: {
        u.type = OnDemandNodeUpdate::Memcpy;
        const auto& m = entry.memcpy;
        memset(&u.memcpy_params, 0, sizeof(u.memcpy_params));
        u.memcpy_params.Depth = m.Depth;
        u.memcpy_params.Height = m.Height;
        u.memcpy_params.WidthInBytes = m.WidthInBytes;
        u.memcpy_params.dstDevice = m.dstDevice;
        u.memcpy_params.dstHeight = m.dstHeight;
        u.memcpy_params.dstLOD = m.dstLOD;
        u.memcpy_params.dstMemoryType = static_cast<CUmemorytype>(m.dstMemoryType);
        u.memcpy_params.dstPitch = m.dstPitch;
        u.memcpy_params.dstXInBytes = m.dstXInBytes;
        u.memcpy_params.dstY = m.dstY;
        u.memcpy_params.dstZ = m.dstZ;
        u.memcpy_params.srcDevice = m.srcDevice;
        u.memcpy_params.srcHeight = m.srcHeight;
        u.memcpy_params.srcLOD = m.srcLOD;
        u.memcpy_params.srcMemoryType = static_cast<CUmemorytype>(m.srcMemoryType);
        u.memcpy_params.srcPitch = m.srcPitch;
        u.memcpy_params.srcXInBytes = m.srcXInBytes;
        u.memcpy_params.srcY = m.srcY;
        u.memcpy_params.srcZ = m.srcZ;
        break;
      }
      case bf::NODE_EVENT_RECORD:
      case bf::NODE_EVENT_WAIT: {
        u.type = (entry.type == bf::NODE_EVENT_RECORD) ? OnDemandNodeUpdate::EventRecord
                                                       : OnDemandNodeUpdate::EventWait;
        int event_id = entry.event.event_id;
        auto eit = event_id_to_event.find(event_id);
        CUevent event;
        if (eit == event_id_to_event.end()) {
          C10_CUDA_DRIVER_CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
          event_id_to_event[event_id] = event;
          if (!graph->loaded_graph_resources_) {
            graph->loaded_graph_resources_ = std::make_unique<LoadedGraphResources>();
          }
          graph->loaded_graph_resources_->created_events.push_back(event);
        } else {
          event = eit->second;
        }
        u.event = event;
        break;
      }
      case bf::NODE_EMPTY:
        u.type = OnDemandNodeUpdate::Empty;
        break;
    }
  }

  for (auto& u : data->updates) {
    u.fixup_pointers();
  }

  graph->on_demand_data_ = std::move(data);
}

// ============================================================================
// build_template_graph_binary: the template's CUgraph from the .cugraph data.
//
// Params, function handles, function attributes and events come from the
// template's prepared on-demand data (prepare_on_demand_graph_binary, run on a
// pool worker). Node attributes are read from the node table and applied with
// the same precedence as build_graph_from_parsed: per-node attributes (with
// the function's required cluster overriding the node's cluster dims) right
// after each node add, then the common attributes on every kernel node.
// Requires FLAG_COMPLETE_KERNEL_ATTRS (every attribute the JSON records is in
// the binary). Does not instantiate.
// ============================================================================

namespace {
// Kernel-node attributes of a .cugraph node (kna flags + extension flags) on a graph node, with
// build_graph_from_parsed's semantics. per_node: the node's own attributes, where the function's
// required cluster dims override the node's cluster dims and the extension attributes apply;
// otherwise the graph's common attributes (flags/values in `k`).
void apply_binary_kernel_attrs(CUgraphNode n, uint32_t flags, const binary_format::BinKernelNode& k,
                               bool per_node) {
  namespace bf = binary_format;
  auto set_attr = [](CUgraphNode n, CUkernelNodeAttrID id, const CUkernelNodeAttrValue& v) {
    C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(n, id, &v));
  };
  CUkernelNodeAttrValue v;
  unsigned cx = 0, cy = 0, cz = 0;
  if (flags & bf::KNA_CLUSTER_DIM) {
    cx = k.kna_clusterDimX;
    cy = k.kna_clusterDimY;
    cz = k.kna_clusterDimZ;
  }
  if (per_node) {
    if (k.required_cluster_width > 0)
      cx = k.required_cluster_width;
    if (k.required_cluster_height > 0)
      cy = k.required_cluster_height;
    if (k.required_cluster_depth > 0)
      cz = k.required_cluster_depth;
  }
  if ((flags & bf::KNA_CLUSTER_DIM) || cx > 0 || cy > 0 || cz > 0) {
    memset(&v, 0, sizeof(v));
    v.clusterDim.x = cx > 0 ? cx : 1;
    v.clusterDim.y = cy > 0 ? cy : 1;
    v.clusterDim.z = cz > 0 ? cz : 1;
    if (per_node ? (cx > 0 || cy > 0 || cz > 0) : true)
      set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, v);
  }
  if (flags & bf::KNA_PREFERRED_CLUSTER_DIM) {
    memset(&v, 0, sizeof(v));
    v.preferredClusterDim.x = k.kna_preferredClusterDimX;
    v.preferredClusterDim.y = k.kna_preferredClusterDimY;
    v.preferredClusterDim.z = k.kna_preferredClusterDimZ;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, v);
  }
  if (flags & bf::KNA_CLUSTER_SCHEDULING) {
    memset(&v, 0, sizeof(v));
    v.clusterSchedulingPolicyPreference =
        static_cast<CUclusterSchedulingPolicy>(k.kna_clusterSchedulingPolicy);
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, v);
  }
  if (flags & bf::KNA_COOPERATIVE) {
    memset(&v, 0, sizeof(v));
    v.cooperative = k.kna_cooperative;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, v);
  }
  if (flags & bf::KNA_PRIORITY) {
    memset(&v, 0, sizeof(v));
    v.priority = k.kna_priority;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, v);
  }
  if (flags & bf::KNA_MEM_SYNC_DOMAIN) {
    memset(&v, 0, sizeof(v));
    v.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(k.kna_memSyncDomain);
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, v);
  }
  if (flags & bf::KNA_MEM_SYNC_DOMAIN_MAP) {
    memset(&v, 0, sizeof(v));
    v.memSyncDomainMap.default_ = k.kna_memSyncDomainMapDefault;
    v.memSyncDomainMap.remote = k.kna_memSyncDomainMapRemote;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, v);
  }
  if (flags & bf::KNA_SHARED_MEM_CARVEOUT) {
    memset(&v, 0, sizeof(v));
    v.sharedMemCarveout = k.kna_preferredSharedMemCarveout;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, v);
  }
  if (per_node && (k.ext_flags & bf::KNX_DEVICE_UPDATABLE)) {
    memset(&v, 0, sizeof(v));
    v.deviceUpdatableKernelNode.deviceUpdatable = k.ext_device_updatable;
    set_attr(n, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, v);
  }
  if (per_node && (k.ext_flags & bf::KNX_PROGRAMMATIC_STREAM_SERIALIZATION)) {
    memset(&v, 0, sizeof(v));
    v.programmaticStreamSerializationAllowed = k.ext_programmatic_stream_serialization;
    set_attr(n,
             static_cast<CUkernelNodeAttrID>(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION),
             v);
  }
}

// The graph's common attributes in a node record, for apply_binary_kernel_attrs(per_node=false).
binary_format::BinKernelNode common_attrs_as_node(const binary_format::BinCommonKernelAttrs& ca) {
  binary_format::BinKernelNode k{};
  k.kna_clusterDimX = ca.clusterDimX;
  k.kna_clusterDimY = ca.clusterDimY;
  k.kna_clusterDimZ = ca.clusterDimZ;
  k.kna_preferredClusterDimX = ca.preferredClusterDimX;
  k.kna_preferredClusterDimY = ca.preferredClusterDimY;
  k.kna_preferredClusterDimZ = ca.preferredClusterDimZ;
  k.kna_clusterSchedulingPolicy = ca.clusterSchedulingPolicy;
  k.kna_cooperative = ca.cooperative;
  k.kna_priority = ca.priority;
  k.kna_memSyncDomain = ca.memSyncDomain;
  k.kna_memSyncDomainMapDefault = ca.memSyncDomainMapDefault;
  k.kna_memSyncDomainMapRemote = ca.memSyncDomainMapRemote;
  k.kna_preferredSharedMemCarveout = ca.preferredSharedMemCarveout;
  return k;
}
}  // namespace

void CUDAGraph::build_template_graph_binary(const BinaryGraphFile& bin_file, CUDAGraph& graph,
                                            CUcontext ctx, GraphTemplate* out_template,
                                            BuildTiming* timing) {
  namespace bf = binary_format;
  TORCH_CHECK(graph.on_demand_data_ != nullptr,
              "build_template_graph_binary: the template's on-demand data is not prepared");
  TORCH_CHECK(bin_file.header.flags & bf::FLAG_COMPLETE_KERNEL_ATTRS,
              "build_template_graph_binary: binary does not hold every kernel attribute");
  BuildTiming& bt = *timing;
  const bf::BinNodeEntry* nodes = bin_file.node_table();
  const uint32_t num_nodes = bin_file.header.num_nodes;
  auto& updates = graph.on_demand_data_->updates;
  TORCH_CHECK(updates.size() == num_nodes, "build_template_graph_binary: ", updates.size(),
              " prepared nodes vs ", num_nodes, " in the binary");

  CUgraph cuGraph;
  C10_CUDA_DRIVER_CHECK(cuGraphCreate(&cuGraph, 0));
  out_template->cuGraph = cuGraph;
  out_template->ordered_nodes.reserve(num_nodes);

  std::vector<CUgraphNode> kernel_nodes;
  std::unordered_map<uint32_t, CUgraphNode> id_to_node;
  id_to_node.reserve(num_nodes);
  for (uint32_t idx = 0; idx < num_nodes; ++idx) {
    const bf::BinNodeEntry& entry = nodes[idx];
    auto& u = updates[idx];
    CUgraphNode cuNode = nullptr;
    auto t_add = bt_clock::now();
    switch (entry.type) {
      case bf::NODE_KERNEL: {
        // u.kernel_attrs: the node's cluster dims merged with the function's compiled ones.
        const auto& ka = u.kernel_attrs;
        CUresult r = add_restored_kernel_node(&cuNode, cuGraph, u.kernel_params, graph.capture_dev_,
                                              ka.has_cluster_dim ? ka.clusterDimX : 0,
                                              ka.has_cluster_dim ? ka.clusterDimY : 0,
                                              ka.has_cluster_dim ? ka.clusterDimZ : 0, "LOAD");
        if (r != CUDA_SUCCESS) {
          fprintf(stderr,
                  "[foundry LOAD ERROR] cuGraphAddKernelNode FAILED for node %u with error %d\n",
                  entry.node_id, r);
          fprintf(
              stderr, "[foundry LOAD ERROR]   function: %s\n",
              bin_file
                  .get_string(entry.kernel.function_name_offset, entry.kernel.function_name_length)
                  .c_str());
          fprintf(stderr, "[foundry LOAD ERROR]   grid=(%u,%u,%u) block=(%u,%u,%u) sharedMem=%u\n",
                  u.kernel_params.gridDimX, u.kernel_params.gridDimY, u.kernel_params.gridDimZ,
                  u.kernel_params.blockDimX, u.kernel_params.blockDimY, u.kernel_params.blockDimZ,
                  u.kernel_params.sharedMemBytes);
          C10_CUDA_DRIVER_CHECK(r);
        }
        bt.add_ms += bt_ms(t_add);
        auto t_attr = bt_clock::now();
        apply_binary_kernel_attrs(cuNode, entry.kernel.kna_flags, entry.kernel, /*per_node=*/true);
        kernel_nodes.push_back(cuNode);
        bt.node_attr_ms += bt_ms(t_attr);
        break;
      }
      case bf::NODE_MEMCPY: {
        CUresult r = cuGraphAddMemcpyNode(&cuNode, cuGraph, nullptr, 0, &u.memcpy_params, ctx);
        if (r != CUDA_SUCCESS) {
          fprintf(stderr,
                  "[foundry LOAD ERROR] cuGraphAddMemcpyNode FAILED for node %u with error %d\n",
                  entry.node_id, r);
          C10_CUDA_DRIVER_CHECK(r);
        }
        bt.add_ms += bt_ms(t_add);
        break;
      }
      case bf::NODE_MEMSET: {
        CUresult r = cuGraphAddMemsetNode(&cuNode, cuGraph, nullptr, 0, &u.memset_params, ctx);
        if (r != CUDA_SUCCESS) {
          fprintf(stderr,
                  "[foundry LOAD ERROR] cuGraphAddMemsetNode FAILED for node %u with error %d\n",
                  entry.node_id, r);
          C10_CUDA_DRIVER_CHECK(r);
        }
        bt.add_ms += bt_ms(t_add);
        break;
      }
      case bf::NODE_EVENT_RECORD:
        C10_CUDA_DRIVER_CHECK(cuGraphAddEventRecordNode(&cuNode, cuGraph, nullptr, 0, u.event));
        bt.add_ms += bt_ms(t_add);
        break;
      case bf::NODE_EVENT_WAIT:
        C10_CUDA_DRIVER_CHECK(cuGraphAddEventWaitNode(&cuNode, cuGraph, nullptr, 0, u.event));
        bt.add_ms += bt_ms(t_add);
        break;
      case bf::NODE_EMPTY:
        C10_CUDA_DRIVER_CHECK(cuGraphAddEmptyNode(&cuNode, cuGraph, nullptr, 0));
        bt.add_ms += bt_ms(t_add);
        break;
    }
    TORCH_CHECK(cuNode != nullptr, "build_template_graph_binary: unknown node type ",
                (int)entry.type, " at node ", idx);
    id_to_node[entry.node_id] = cuNode;
    out_template->ordered_nodes.push_back(cuNode);
  }

  // Common attributes last, on every kernel node (build_graph_from_parsed's order).
  auto t_cattr = bt_clock::now();
  if ((bin_file.header.flags & bf::FLAG_HAS_COMMON_KERNEL_ATTRS) && !kernel_nodes.empty()) {
    const bf::BinCommonKernelAttrs* ca =
        bin_file.section_ptr<bf::BinCommonKernelAttrs>(bf::SECTION_COMMON_KERNEL_ATTRS);
    const bf::BinKernelNode k = common_attrs_as_node(*ca);
    for (auto n : kernel_nodes)
      apply_binary_kernel_attrs(n, ca->flags, k, /*per_node=*/false);
  }
  bt.node_attr_ms += bt_ms(t_cattr);

  // Dependencies (v1 records carry default edges only).
  auto t_deps = bt_clock::now();
  const uint32_t num_deps = bin_file.header.num_dependencies;
  if ((bin_file.header.flags & bf::FLAG_HAS_DEPENDENCIES) && num_deps > 0) {
    std::vector<CUgraphNode> from_nodes(num_deps), to_nodes(num_deps);
    std::vector<CUgraphEdgeData> edge_data(num_deps);
    memset(edge_data.data(), 0, edge_data.size() * sizeof(CUgraphEdgeData));
    auto lookup = [&](uint32_t id) {
      auto it = id_to_node.find(id);
      TORCH_CHECK(it != id_to_node.end(),
                  "build_template_graph_binary: dependency on unknown node ", id);
      return it->second;
    };
    if (bin_file.header.version >= 2) {
      const bf::BinDependency* deps =
          bin_file.section_ptr<bf::BinDependency>(bf::SECTION_DEPENDENCY_TABLE);
      for (uint32_t i = 0; i < num_deps; ++i) {
        from_nodes[i] = lookup(deps[i].from_id);
        to_nodes[i] = lookup(deps[i].to_id);
        edge_data[i].from_port = deps[i].from_port;
        edge_data[i].to_port = deps[i].to_port;
        edge_data[i].type = deps[i].edge_type;
      }
    } else {
      const bf::BinDependencyV1* deps =
          bin_file.section_ptr<bf::BinDependencyV1>(bf::SECTION_DEPENDENCY_TABLE);
      for (uint32_t i = 0; i < num_deps; ++i) {
        from_nodes[i] = lookup(deps[i].from_id);
        to_nodes[i] = lookup(deps[i].to_id);
      }
    }
    CUresult dep_result = add_graph_dependencies(cuGraph, from_nodes.data(), to_nodes.data(),
                                                 edge_data.data(), num_deps);
    if (dep_result != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry LOAD ERROR] cuGraphAddDependencies FAILED with error %d\n",
              dep_result);
      C10_CUDA_DRIVER_CHECK(dep_result);
    }
  }
  bt.deps_ms += bt_ms(t_deps);
}

void CUDAGraph::link_on_demand_shared_exec(CUDAGraph& graph,
                                           std::shared_ptr<SharedGraphExec> shared_exec) {
  TORCH_CHECK(graph.on_demand_data_ != nullptr,
              "link_on_demand_shared_exec: graph has no on_demand_data_");
  TORCH_CHECK(shared_exec != nullptr, "link_on_demand_shared_exec: shared_exec is nullptr");
  TORCH_CHECK(graph.on_demand_data_->updates.size() == shared_exec->ordered_nodes.size(),
              "On-demand mismatch: graph has ", graph.on_demand_data_->updates.size(),
              " updates but template has ", shared_exec->ordered_nodes.size());

  graph.on_demand_data_->shared_exec = shared_exec;
}

// Reconstruct output tensors from an extracted json::value (may be null).
static GraphLoadResult make_load_result_from_extracted(
    std::shared_ptr<CUDAGraph> graph, const boost::json::value& output_tensors_meta,
    ReconstructTensorFn reconstruct_fn) {
  GraphLoadResult result;
  result.graph = std::move(graph);
  result.output_type = OutputTensorType::None;
  result.outputs = std::monostate{};

  if (!output_tensors_meta.is_null()) {
    const boost::json::object& output_tensors_obj = output_tensors_meta.as_object();
    result.output_type =
        static_cast<OutputTensorType>(output_tensors_obj.at("type").to_number<int>());

    const boost::json::array& tensors_array = output_tensors_obj.at("tensors").as_array();

    if (result.output_type == OutputTensorType::Single && tensors_array.size() == 1) {
      result.outputs = reconstruct_fn(tensors_array[0].as_object());
    } else if (result.output_type == OutputTensorType::List ||
               result.output_type == OutputTensorType::Tuple) {
      std::vector<at::Tensor> tensors;
      tensors.reserve(tensors_array.size());
      for (const auto& t_val : tensors_array) {
        tensors.push_back(reconstruct_fn(t_val.as_object()));
      }
      result.outputs = std::move(tensors);
    }
  }

  return result;
}

// ============================================================================
// Exec-pool prewarm (contract in CUDAGraph.h, start_exec_pool_prewarm).
// ============================================================================

namespace {
std::mutex g_prewarm_mutex;
std::shared_future<void> g_prewarm_done;
// Set by Phase 2 when it starts before the prewarm finished: the prewarm stops after its
// current graph (Phase 2 never waits for it).
std::atomic<bool> g_prewarm_abandon{false};

double epoch_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Resident set of this process in MB (/proc/self/statm), 0 if unavailable.
double rss_mb() {
  std::ifstream f("/proc/self/statm");
  size_t pages_total = 0, pages_rss = 0;
  if (!(f >> pages_total >> pages_rss))
    return 0.0;
  return pages_rss * 4096.0 / (1024.0 * 1024.0);
}

std::string cugraph_path_for(const std::string& json_path) {
  if (json_path.size() >= 5 && json_path.substr(json_path.size() - 5) == ".json")
    return json_path.substr(0, json_path.size() - 5) + ".cugraph";
  return json_path;
}

// Kernel-only copy of one archive graph (non-kernel nodes -> empty nodes, same edges),
// built from its .cugraph. Returns nullptr (after a warning) if a node cannot be added.
// Node attributes are not copied: with them the Phase 2 instantiates were no faster and the
// prewarm build took ~0.5 s longer (Qwen3-30B ep4, 128 graphs).
CUgraph build_prewarm_graph(const BinaryGraphFile& b, CUcontext ctx, CUdevice dev,
                            size_t* num_nodes) {
  namespace bf = binary_format;
  const bf::BinNodeEntry* nodes = b.node_table();
  const bf::BinParamEntry* param_idx = b.param_index();
  const uint8_t* pd = b.param_data();
  const uint8_t* abd = b.arg_buffer_data();
  CUgraph g;
  C10_CUDA_DRIVER_CHECK(cuGraphCreate(&g, 0));
  std::unordered_map<uint32_t, CUgraphNode> id_to_node;
  std::vector<void*> ptrs;
  for (uint32_t idx = 0; idx < b.header.num_nodes; ++idx) {
    const bf::BinNodeEntry& e = nodes[idx];
    CUgraphNode n = nullptr;
    CUresult r = CUDA_SUCCESS;
    if (e.type == bf::NODE_KERNEL) {
      const auto& k = e.kernel;
      CUDA_KERNEL_NODE_PARAMS p;
      memset(&p, 0, sizeof(p));
      p.gridDimX = k.gridDimX;
      p.gridDimY = k.gridDimY;
      p.gridDimZ = k.gridDimZ;
      p.blockDimX = k.blockDimX;
      p.blockDimY = k.blockDimY;
      p.blockDimZ = k.blockDimZ;
      p.sharedMemBytes = k.sharedMemBytes;
      p.ctx = ctx;
      auto h = query_function_handle(k.binary_hash,
                                     b.get_string(k.function_name_offset, k.function_name_length));
      if (std::holds_alternative<CUkernel>(h))
        p.kern = std::get<CUkernel>(h);
      else
        p.func = std::get<CUfunction>(h);
      size_t arg_size = k.arg_buffer_size;
      void* extra[5];
      if (k.arg_buffer_offset != 0xFFFFFFFF) {
        extra[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        extra[1] = const_cast<uint8_t*>(abd + k.arg_buffer_offset);
        extra[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        extra[3] = &arg_size;
        extra[4] = CU_LAUNCH_PARAM_END;
        p.extra = extra;
      } else {
        const bf::BinParamEntry* pe = reinterpret_cast<const bf::BinParamEntry*>(
            reinterpret_cast<const uint8_t*>(param_idx) + k.param_index_offset);
        ptrs.resize(k.num_params);
        for (uint32_t j = 0; j < k.num_params; ++j)
          ptrs[j] = const_cast<uint8_t*>(pd + pe[j].data_offset);
        p.kernelParams = ptrs.data();
      }
      // No node attributes here, so only a compiled cluster (func_attrs) reaches the driver.
      auto dim = [](int32_t v) { return v > 0 ? static_cast<unsigned>(v) : 0u; };
      r = add_restored_kernel_node(&n, g, p, dev, dim(k.required_cluster_width),
                                   dim(k.required_cluster_height), dim(k.required_cluster_depth),
                                   "PREWARM");
    } else {
      r = cuGraphAddEmptyNode(&n, g, nullptr, 0);
    }
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry] exec pool prewarm: node %u not added (%d); graph skipped\n",
              e.node_id, (int)r);
      cuGraphDestroy(g);
      return nullptr;
    }
    id_to_node[e.node_id] = n;
  }
  const uint32_t nd = b.header.num_dependencies;
  if ((b.header.flags & bf::FLAG_HAS_DEPENDENCIES) && nd > 0) {
    std::vector<CUgraphNode> from(nd), to(nd);
    std::vector<CUgraphEdgeData> ed(nd);
    memset(ed.data(), 0, ed.size() * sizeof(CUgraphEdgeData));
    for (uint32_t i = 0; i < nd; ++i) {
      uint32_t f, t;
      if (b.header.version >= 2) {
        const auto* d = b.section_ptr<bf::BinDependency>(bf::SECTION_DEPENDENCY_TABLE) + i;
        f = d->from_id;
        t = d->to_id;
        ed[i].from_port = d->from_port;
        ed[i].to_port = d->to_port;
        ed[i].type = d->edge_type;
      } else {
        const auto* d = b.section_ptr<bf::BinDependencyV1>(bf::SECTION_DEPENDENCY_TABLE) + i;
        f = d->from_id;
        t = d->to_id;
      }
      from[i] = id_to_node.at(f);
      to[i] = id_to_node.at(t);
    }
    CUresult r = add_graph_dependencies(g, from.data(), to.data(), ed.data(), nd);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[foundry] exec pool prewarm: dependencies not added (%d); graph skipped\n",
              (int)r);
      cuGraphDestroy(g);
      return nullptr;
    }
  }
  *num_nodes = b.header.num_nodes;
  return g;
}

void run_exec_pool_prewarm(const std::vector<std::string>& json_paths, CUcontext ctx,
                           CUdevice dev) {
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
  };
  C10_CUDA_DRIVER_CHECK(cuCtxSetCurrent(ctx));
  auto t0 = clk::now();
  const double start_epoch = epoch_s();
  size_t free0 = 0, free_peak = 0, free_end = 0, total = 0;
  C10_CUDA_DRIVER_CHECK(cuMemGetInfo(&free0, &total));
  const double rss0 = rss_mb();
  std::vector<cudaGraphExec_t> execs;
  size_t nodes = 0;
  double read_ms = 0, build_ms = 0, inst_ms = 0;
  bool abandoned = false;
  for (const auto& path : json_paths) {
    if (g_prewarm_abandon.load()) {
      abandoned = true;
      break;
    }
    auto t_r = clk::now();
    BinaryGraphFile b = read_binary_graph_file(cugraph_path_for(path));
    read_ms += ms_since(t_r);
    if (!b.valid())
      continue;
    auto t_b = clk::now();
    size_t n = 0;
    CUgraph g = build_prewarm_graph(b, ctx, dev, &n);
    build_ms += ms_since(t_b);
    if (!g)
      continue;
    auto t_i = clk::now();
    execs.push_back(CUDAGraph::instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(g)));
    inst_ms += ms_since(t_i);
    C10_CUDA_DRIVER_CHECK(cuGraphDestroy(g));
    nodes += n;
  }
  C10_CUDA_DRIVER_CHECK(cuMemGetInfo(&free_peak, &total));
  const double rss_peak = rss_mb();
  for (auto e : execs)
    C10_CUDA_CHECK(cudaGraphExecDestroy(e));
  C10_CUDA_DRIVER_CHECK(cuMemGetInfo(&free_end, &total));
  const double rss_end = rss_mb();
  // Device-memory and RSS deltas include whatever the engine allocated concurrently.
  fprintf(stderr,
          "[foundry] exec pool prewarm: %zu graphs, %zu nodes, %.1f ms (read %.1f, build %.1f, "
          "instantiate %.1f)%s; start %.3f end %.3f (epoch s); device free %.0f -> %.0f at peak "
          "-> %.0f MB after destroy; RSS %.0f -> %.0f -> %.0f MB\n",
          execs.size(), nodes, ms_since(t0), read_ms, build_ms, inst_ms,
          abandoned ? ", abandoned: Phase 2 started first" : "", start_epoch, epoch_s(),
          free0 / 1048576.0, free_peak / 1048576.0, free_end / 1048576.0, rss0, rss_peak, rss_end);
}

// Phase 2 (instantiate thread, before its first instantiate): never waits for the prewarm.
// Finished: collect it (a failure only costs speed). Still running: tell it to stop after its
// current graph and go on without it. Returns whether it had finished.
bool join_exec_pool_prewarm() {
  std::shared_future<void> f;
  {
    std::lock_guard<std::mutex> lock(g_prewarm_mutex);
    f = g_prewarm_done;
  }
  if (!f.valid())
    return false;
  if (f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    g_prewarm_abandon = true;
    fprintf(stderr,
            "[foundry] exec pool prewarm still running at Phase 2 start (%.3f epoch s): "
            "abandoned, not waited for\n",
            epoch_s());
    return false;
  }
  try {
    f.get();
  } catch (const std::exception& e) {
    fprintf(stderr, "[foundry] exec pool prewarm failed (%s); instantiating without it\n",
            e.what());
    return false;
  }
  return true;
}
}  // namespace

void CUDAGraph::start_exec_pool_prewarm(const std::vector<std::string>& json_paths) {
  CUcontext ctx = nullptr;
  C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&ctx));
  TORCH_CHECK(ctx != nullptr, "start_exec_pool_prewarm: no CUDA context on this thread");
  CUdevice dev;
  C10_CUDA_DRIVER_CHECK(cuCtxGetDevice(&dev));
  auto promise = std::make_shared<std::promise<void>>();
  {
    std::lock_guard<std::mutex> lock(g_prewarm_mutex);
    g_prewarm_abandon = false;
    g_prewarm_done = promise->get_future().share();
  }
  std::thread([json_paths, ctx, dev, promise]() {
    try {
      run_exec_pool_prewarm(json_paths, ctx, dev);
      promise->set_value();
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  }).detach();
}

// ============================================================================
// start_graph_builds_impl: JSON parse + template build + on-demand prep
//
// Phase 1:
//   1a. Parallel JSON file I/O + parse (worker threads)
//   1b. Sequential graph shell creation + extract deferred metadata
//   1c. Extract allocator_events + output_tensors
//   1d. Compute topology groups
//
// Phase 2 (synchronous):
//   2a. Build template graphs (one per topology group)
//   2b. Prepare on-demand data for remaining graphs (CPU-only, parallelized)
//
// Returns PendingGraphLoads with all graphs fully built.
// Does NOT replay allocator events — that happens in finish_graph_loads_impl.
// ============================================================================

std::shared_ptr<PendingGraphLoads> start_graph_builds_impl(
    const std::vector<std::string>& json_paths, MempoolId_t pool, int num_threads,
    CUDAGeneratorStateRegistry& registry) {
  auto pending = std::make_shared<PendingGraphLoads>();
  size_t num_graphs = json_paths.size();

  if (num_graphs == 0) {
    return pending;
  }

  int actual_threads = std::min(num_threads, static_cast<int>(num_graphs));

  CUcontext main_ctx;
  C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&main_ctx));
  TORCH_CHECK(main_ctx != nullptr, "No CUDA context on main thread");

  c10::DeviceIndex dev = c10::cuda::current_device();

  MempoolId_t resolved_pool = pool;
  if (resolved_pool.first == 0 && resolved_pool.second == 0) {
    resolved_pool = torch_graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(resolved_pool.first > 0);
  }

  pending->pool = resolved_pool;
  pending->dev = dev;
  pending->registry = &registry;
  pending->entries.resize(num_graphs);

  using Clock = std::chrono::steady_clock;
  auto t_start = Clock::now();

  // ---- Phase 1a: parallel parse ----
  // Read JSON for metadata (generators, allocator_events, output_tensors, topology_key)
  // ---- Phase 1a: parallel file reads ----
  // Binary: read file + validate header (lightweight, ~300KB per graph).
  // JSON: only for graphs without binary (fallback).
  std::vector<BinaryGraphFile> bin_files(num_graphs);
  std::vector<boost::json::value> parsed_jsons(num_graphs);  // only used for non-binary graphs
  size_t num_binary = 0, num_json = 0;
  {
    SimpleThreadPool thread_pool(actual_threads);

    // Read binary files in parallel
    std::vector<std::future<BinaryGraphFile>> bin_futures;
    bin_futures.reserve(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
      bin_futures.push_back(thread_pool.submit([&json_paths, i]() {
        std::string bin_path = json_paths[i];
        if (bin_path.size() >= 5 && bin_path.substr(bin_path.size() - 5) == ".json") {
          bin_path = bin_path.substr(0, bin_path.size() - 5) + ".cugraph";
        }
        return read_binary_graph_file(bin_path);
      }));
    }
    for (size_t i = 0; i < num_graphs; ++i) {
      bin_files[i] = bin_futures[i].get();
      if (bin_files[i].valid())
        num_binary++;
      else
        num_json++;
    }

    // Parse JSON only for graphs without binary
    if (num_json > 0) {
      std::vector<std::pair<size_t, std::future<boost::json::value>>> json_work;
      for (size_t i = 0; i < num_graphs; ++i) {
        if (!bin_files[i].valid()) {
          json_work.emplace_back(i, thread_pool.submit([&json_paths, i]() {
            return read_and_parse_graph_json(json_paths[i]);
          }));
        }
      }
      for (auto& [idx, fut] : json_work) {
        parsed_jsons[idx] = fut.get();
      }
    }
  }

  double phase1a_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

  // ---- Phase 1b: sequential metadata extraction + graph shell creation ----
  // Binary graphs: extract generators/allocator_events/output_tensors directly
  // from binary sections. No JSON construction for node data.
  // JSON graphs: extract from parsed JSON (original flow).
  // Generator registration is deferred to finish_graph_loads_impl.
  auto t_1b = Clock::now();

  std::vector<ParsedGraphData> all_parsed;
  all_parsed.reserve(num_graphs);

  for (size_t i = 0; i < num_graphs; ++i) {
    if (bin_files[i].valid()) {
      // ---- Binary path: extract metadata directly from binary sections ----
      namespace bf = binary_format;
      const auto& bf_data = bin_files[i];

      // Build minimal JSON for prepare_graph_shell (it only reads generators
      // and sets up the graph shell — doesn't touch nodes).
      boost::json::object shell_root;
      shell_root["topology_key"] = bf_data.topology_key();
      shell_root["nodes"] = boost::json::array{};

      // Generators → extract to pending (deferred registration),
      // but also include in shell_root so prepare_graph_shell can
      // set up captured_generator_states_ if needed.
      // Actually, generators are extracted BEFORE prepare_graph_shell
      // and deferred — so we just extract them here.
      if (bf_data.header.flags & bf::FLAG_HAS_GENERATORS) {
        const bf::BinGenerator* gens =
            bf_data.section_ptr<bf::BinGenerator>(bf::SECTION_GENERATORS_TABLE);
        boost::json::array gens_array;
        for (uint32_t g = 0; g < bf_data.header.num_generators; g++) {
          boost::json::object go;
          go["id"] = gens[g].id;
          go["seed"] = gens[g].seed;
          go["wholegraph_increment"] = gens[g].wholegraph_increment;
          gens_array.push_back(go);
        }
        pending->entries[i].generators_meta = boost::json::value(std::move(gens_array));
      }

      // Create graph shell via prepare_graph_shell (handles mempool, capture_dev, etc.)
      // Generators already extracted above, so shell_root has no generators key.
      ParsedGraphData parsed = CUDAGraph::prepare_graph_shell(
          boost::json::value(std::move(shell_root)), resolved_pool, registry);

      // Allocator events (embedded JSON, ~0.5KB)
      if (bf_data.header.flags & bf::FLAG_HAS_ALLOCATOR_EVENTS) {
        auto ptr = bf_data.section_ptr<char>(bf::SECTION_ALLOCATOR_EVENTS);
        auto sz = bf_data.section_size(bf::SECTION_ALLOCATOR_EVENTS);
        pending->entries[i].allocator_events = boost::json::parse(std::string_view(ptr, sz));
      }

      // Output tensors (embedded JSON, ~0.2KB)
      if (bf_data.header.flags & bf::FLAG_HAS_OUTPUT_TENSORS) {
        auto ptr = bf_data.section_ptr<char>(bf::SECTION_OUTPUT_TENSORS);
        auto sz = bf_data.section_size(bf::SECTION_OUTPUT_TENSORS);
        pending->entries[i].output_tensors_meta = boost::json::parse(std::string_view(ptr, sz));
      }

      pending->entries[i].graph = parsed.graph;
      all_parsed.push_back(std::move(parsed));

    } else {
      // ---- JSON path: original flow ----
      boost::json::object& pre_root = parsed_jsons[i].as_object();

      auto gen_it = pre_root.find("generators");
      if (gen_it != pre_root.end()) {
        pending->entries[i].generators_meta = std::move(gen_it->value());
        pre_root.erase(gen_it);
      }

      ParsedGraphData parsed =
          CUDAGraph::prepare_graph_shell(std::move(parsed_jsons[i]), resolved_pool, registry);

      boost::json::object& root = parsed.root_val.as_object();

      auto alloc_it = root.find("allocator_events");
      TORCH_CHECK(alloc_it != root.end(), "Missing allocator_events in graph JSON");
      pending->entries[i].allocator_events = std::move(alloc_it->value());
      root.erase(alloc_it);

      auto output_it = root.find("output_tensors");
      if (output_it != root.end()) {
        pending->entries[i].output_tensors_meta = std::move(output_it->value());
        root.erase(output_it);
      }

      pending->entries[i].graph = parsed.graph;
      all_parsed.push_back(std::move(parsed));
    }
  }

  double phase1b_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_1b).count();

  // ---- Phase 1b2: compute topology groups ----
  // Try to read graph_manifest.json for pre-computed template assignments.
  // Falls back to computing topology groups from node data if no manifest.
  std::unordered_map<std::string, std::vector<size_t>> topology_groups;
  std::vector<size_t> template_for(num_graphs, SIZE_MAX);
  bool used_manifest = false;

  // Build filename → index map for manifest lookup
  std::unordered_map<std::string, size_t> filename_to_idx;
  for (size_t i = 0; i < num_graphs; ++i) {
    auto pos = json_paths[i].rfind('/');
    std::string fname = pos != std::string::npos ? json_paths[i].substr(pos + 1) : json_paths[i];
    filename_to_idx[fname] = i;
  }

  // Try manifest
  if (!json_paths.empty()) {
    auto slash_pos = json_paths[0].rfind('/');
    std::string dir = slash_pos != std::string::npos ? json_paths[0].substr(0, slash_pos) : ".";
    std::string manifest_path = dir + "/graph_manifest.json";
    std::ifstream manifest_file(manifest_path);
    if (manifest_file.is_open()) {
      try {
        std::string manifest_str((std::istreambuf_iterator<char>(manifest_file)),
                                 std::istreambuf_iterator<char>());
        manifest_file.close();
        auto manifest_val = boost::json::parse(manifest_str);
        const auto& groups = manifest_val.as_object().at("topology_groups").as_array();

        for (const auto& group_val : groups) {
          const auto& group = group_val.as_object();
          std::string tmpl_name = group.at("template").as_string().c_str();

          auto tmpl_it = filename_to_idx.find(tmpl_name);
          if (tmpl_it == filename_to_idx.end())
            continue;
          size_t tmpl_idx = tmpl_it->second;

          std::string key = std::to_string(tmpl_idx);  // use template index as key
          topology_groups[key].push_back(tmpl_idx);

          const auto& members = group.at("members").as_array();
          for (const auto& member_val : members) {
            std::string member_name = member_val.as_string().c_str();
            if (member_name == tmpl_name)
              continue;
            auto member_it = filename_to_idx.find(member_name);
            if (member_it == filename_to_idx.end())
              continue;
            size_t member_idx = member_it->second;
            topology_groups[key].push_back(member_idx);
            template_for[member_idx] = tmpl_idx;
          }
        }
        used_manifest = true;
        fprintf(stderr, "[foundry] Using graph_manifest.json (%zu topology groups)\n",
                topology_groups.size());
      } catch (const std::exception& e) {
        fprintf(stderr,
                "[foundry] Warning: failed to parse graph_manifest.json: %s, "
                "falling back to topology computation\n",
                e.what());
        topology_groups.clear();
        std::fill(template_for.begin(), template_for.end(), SIZE_MAX);
      }
    }
  }

  // Fallback: compute topology groups from node data (or saved topology_key)
  if (!used_manifest) {
    for (size_t i = 0; i < num_graphs; ++i) {
      const boost::json::object& root = all_parsed[i].root_val.as_object();
      std::string key;

      // When binary-loaded, nodes array is empty but topology_key is saved
      if (root.contains("topology_key") && !root.at("topology_key").as_string().empty()) {
        key = std::string(root.at("topology_key").as_string());
      } else {
        const boost::json::array& nodes = root.at("nodes").as_array();
        key.reserve(nodes.size() * 14);
        for (size_t n = 0; n < nodes.size(); ++n) {
          if (n > 0)
            key += ',';
          const auto& no = nodes[n].as_object();
          std::string type = no.at("type").as_string().c_str();
          key += type;
          if (type == "KernelNode") {
            unsigned int cdx = 0, cdy = 0, cdz = 0;
            const auto& p = no.at("params").as_object();
            if (p.contains("kernel_node_attrs")) {
              const auto& kna = p.at("kernel_node_attrs").as_object();
              if (kna.contains("clusterDimX")) {
                cdx = kna.at("clusterDimX").to_number<unsigned int>();
                cdy = kna.at("clusterDimY").to_number<unsigned int>();
                cdz = kna.at("clusterDimZ").to_number<unsigned int>();
              }
            }
            if (cdx == 0 && root.contains("common_kernel_node_attrs")) {
              const auto& ckna = root.at("common_kernel_node_attrs").as_object();
              if (ckna.contains("clusterDimX")) {
                cdx = ckna.at("clusterDimX").to_number<unsigned int>();
                cdy = ckna.at("clusterDimY").to_number<unsigned int>();
                cdz = ckna.at("clusterDimZ").to_number<unsigned int>();
              }
            }
            if (cdx == 0 && p.contains("func_attrs")) {
              const auto& fa = p.at("func_attrs").as_object();
              if (fa.contains("required_cluster_width"))
                cdx = fa.at("required_cluster_width").to_number<unsigned int>();
              if (fa.contains("required_cluster_height"))
                cdy = fa.at("required_cluster_height").to_number<unsigned int>();
              if (fa.contains("required_cluster_depth"))
                cdz = fa.at("required_cluster_depth").to_number<unsigned int>();
            }
            if (cdx > 0 || cdy > 0 || cdz > 0) {
              key += ":C" + std::to_string(cdx) + "_" + std::to_string(cdy) + "_" +
                     std::to_string(cdz);
            } else {
              key += ":0";
            }
          }
        }
      }  // end else (compute from nodes)
      topology_groups[key].push_back(i);
    }

    for (const auto& [key, indices] : topology_groups) {
      for (size_t j = 1; j < indices.size(); ++j) {
        template_for[indices[j]] = indices[0];
      }
    }
  }

  double phase1_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
  fprintf(stderr,
          "[foundry] Phase 1: %zu graphs parsed in %.1f ms (parse: %.1f, shell: %.1f), "
          "%zu topologies, %d threads, %zu binary + %zu json\n",
          num_graphs, phase1_ms, phase1a_ms, phase1b_ms, topology_groups.size(), actual_threads,
          num_binary, num_json);

  // ---- Phase 2: background graph build with on-demand optimization ----
  // Runs on a detached thread (the pipeline is described at its start below).
  // Template identification is done at save time (graph_manifest.json), so the
  // members' on-demand data is decoded in parallel with the template builds.
  // finish_graph_loads_impl waits on build_complete_ before allocator replay.
  std::vector<std::string> graph_names;
  graph_names.reserve(num_graphs);
  for (size_t i = 0; i < num_graphs; ++i) {
    auto pos = json_paths[i].rfind('/');
    graph_names.push_back(pos != std::string::npos ? json_paths[i].substr(pos + 1) : json_paths[i]);
  }

  auto build_promise = std::make_shared<std::promise<void>>();
  pending->build_complete_ = build_promise->get_future().share();

  // Copy json_paths for the JSON template builds in Phase 2
  std::vector<std::string> json_path_list(json_paths.begin(), json_paths.end());

  std::thread bg_thread([all_parsed = std::move(all_parsed), bin_files = std::move(bin_files),
                         json_path_list = std::move(json_path_list), build_promise, main_ctx,
                         actual_threads, template_for = std::move(template_for),
                         topology_groups = std::move(topology_groups),
                         graph_names = std::move(graph_names)]() mutable {
    auto t_phase2 = std::chrono::steady_clock::now();
    size_t num = all_parsed.size();

    try {
      C10_CUDA_DRIVER_CHECK(cuCtxSetCurrent(main_ctx));

      // Shared execs, indexed by template graph index (written by this thread only).
      std::unordered_map<size_t, std::shared_ptr<CUDAGraph::SharedGraphExec>> shared_execs;

      // Phase 2 pipeline. Three kinds of work:
      //   prep (pool workers): decode each graph's .cugraph into its on-demand data
      //     (params, function handles + attributes, events); templates are queued first.
      //   build (this thread): each template's CUgraph from its binary node table (node adds,
      //     attributes, dependencies), then each member's rewrite of its group's builder graph.
      //   instantiate (one SerialWorker): every cuGraphInstantiate, in the order the builds
      //     finish. The driver serializes instantiation (it does not scale across threads or
      //     contexts), so one instantiate thread is the whole budget; graph-side param rewrites
      //     of one group overlap the instantiation of another group's graph.
      // Member order within a group is unchanged (index order on the group's single builder
      // graph: a member inherits the attributes it does not record from the previous
      // rewrite, as before); members of different groups are interleaved round-robin so a
      // rewrite always has another group's instantiate to overlap. Templates without a
      // complete binary (JSON-only archives, or attributes the node table cannot hold) are
      // built from the JSON inside their instantiate job, as before.
      using clk = std::chrono::steady_clock;
      auto ms_since = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
      };
      const char* lazy_env = std::getenv("FOUNDRY_LAZY_GRAPH_EXEC");
      const bool lazy_graph_exec = lazy_env && lazy_env[0] == '1';
      auto binary_template = [&](size_t idx) {
        return bin_files[idx].valid() &&
               (bin_files[idx].header.flags & binary_format::FLAG_COMPLETE_KERNEL_ATTRS);
      };

      std::vector<size_t> tmpl_order;
      tmpl_order.reserve(topology_groups.size());
      for (const auto& [key, indices] : topology_groups)
        tmpl_order.push_back(indices[0]);

      std::vector<double> prep_ms(num, 0.0);
      auto prep_task = [&all_parsed, &graph_names, &bin_files, &prep_ms, main_ctx](size_t i) {
        return [&all_parsed, &graph_names, &bin_files, &prep_ms, main_ctx, i]() {
          // Bind the captured context on this pool worker thread: prep creates events
          // (cross-stream graphs) and sets function attributes.
          C10_CUDA_DRIVER_CHECK(cuCtxSetCurrent(main_ctx));
          auto t0 = std::chrono::steady_clock::now();
          if (bin_files[i].valid()) {
            CUDAGraph::prepare_on_demand_graph_binary(bin_files[i], all_parsed[i].graph, main_ctx,
                                                      nullptr, static_cast<int>(i));
          } else {
            CUDAGraph::prepare_on_demand_graph(all_parsed[i], main_ctx, nullptr,
                                               static_cast<int>(i));
          }
          all_parsed[i].graph->on_demand_data_->graph_name = graph_names[i];
          prep_ms[i] =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                  .count();
        };
      };
      SimpleThreadPool pool(actual_threads);
      std::unordered_map<size_t, std::future<void>> tmpl_prep;
      for (size_t idx : tmpl_order)
        if (binary_template(idx))
          tmpl_prep[idx] = pool.submit(prep_task(idx));
      std::vector<std::future<void>> on_demand_futures;
      for (size_t i = 0; i < num; ++i)
        if (template_for[i] != SIZE_MAX)
          on_demand_futures.push_back(pool.submit(prep_task(i)));

      // A group's builder graph is busy from the enqueue of an instantiate of it until that
      // instantiate returns; the build thread rewrites it only when it is free.
      std::mutex group_mu;
      std::condition_variable group_cv;
      std::unordered_map<size_t, bool> group_busy;
      bool inst_failed = false;
      auto set_free = [&](size_t g) {
        {
          std::lock_guard<std::mutex> lock(group_mu);
          group_busy[g] = false;
        }
        group_cv.notify_all();
      };
      auto wait_free = [&](size_t g) {
        std::unique_lock<std::mutex> lock(group_mu);
        group_cv.wait(lock, [&] { return inst_failed || !group_busy[g]; });
        return !inst_failed;
      };
      // Declared before the instantiate worker: its jobs use them until it is joined.
      double tmpl_build_ms = 0, tmpl_inst_ms = 0, member_wait_ms = 0;
      bool prewarm_done = false;
      const double phase2_start_epoch = epoch_s();
      std::mutex stats_mu;  // tmpl_inst_ms is written by the instantiate thread
      SerialWorker inst(main_ctx, [&] {
        {
          std::lock_guard<std::mutex> lock(group_mu);
          inst_failed = true;
        }
        group_cv.notify_all();
      });

      // Free device memory around Phase 2's instantiates: what the execs took beyond the driver's
      // pool (the exec pool prewarm is meant to make this ~0).
      size_t free_before = 0, free_after = 0, mem_total = 0;
      inst.push([&prewarm_done, &free_before, &mem_total] {
        prewarm_done = join_exec_pool_prewarm();
        C10_CUDA_DRIVER_CHECK(cuMemGetInfo(&free_before, &mem_total));
      });
      for (size_t tmpl_idx : tmpl_order) {
        auto shared = std::make_shared<CUDAGraph::SharedGraphExec>();
        shared->ctx = main_ctx;
        shared->current_params_id = static_cast<int>(tmpl_idx);
        shared_execs[tmpl_idx] = shared;
        {
          std::lock_guard<std::mutex> lock(group_mu);
          group_busy[tmpl_idx] = true;
        }
        auto graph = all_parsed[tmpl_idx].graph;
        const std::string& name = graph_names[tmpl_idx];

        if (binary_template(tmpl_idx)) {
          auto t_tmpl = clk::now();
          tmpl_prep[tmpl_idx].get();
          double prep_wait = ms_since(t_tmpl);
          CUDAGraph::GraphTemplate tmpl;
          BuildTiming bt;
          CUDAGraph::build_template_graph_binary(bin_files[tmpl_idx], *graph, main_ctx, &tmpl, &bt);
          shared->graph = tmpl.cuGraph;
          shared->ordered_nodes = std::move(tmpl.ordered_nodes);
          graph->on_demand_data_->shared_exec = shared;
          double build_ms = ms_since(t_tmpl);
          tmpl_build_ms += build_ms;
          double prep = prep_ms[tmpl_idx];
          auto t_enq = clk::now();
          inst.push([=, &set_free, &stats_mu, &tmpl_inst_ms, &ms_since]() {
            double queued = ms_since(t_enq);
            auto t_i = clk::now();
            shared->exec = reinterpret_cast<CUgraphExec>(
                CUDAGraph::instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(shared->graph)));
            double inst_ms = ms_since(t_i);
            // Bind the template to the shared exec before any member rewrites the graph.
            graph->materialize_on_demand_exec();
            {
              std::lock_guard<std::mutex> lock(stats_mu);
              tmpl_inst_ms += inst_ms;
            }
            fprintf(stderr,
                    "[foundry BUILD] Template %zu (%s): %zu nodes, done in %.1f ms (binary: prep "
                    "%.1f on a worker, prep wait %.1f, add %.1f, node-attrs %.1f, deps %.1f, "
                    "queued %.1f, instantiate %.1f)\n",
                    tmpl_idx, name.c_str(), shared->ordered_nodes.size(), ms_since(t_tmpl), prep,
                    prep_wait, bt.add_ms, bt.node_attr_ms, bt.deps_ms, queued, inst_ms);
            set_free(tmpl_idx);
          });
        } else {
          // JSON template (archive without a complete binary): the whole build runs in its
          // instantiate job, as it did before the pipeline.
          inst.push([=, &all_parsed, &json_path_list, &bin_files, &set_free, &stats_mu,
                     &tmpl_inst_ms, &ms_since]() {
            fprintf(stderr, "[foundry BUILD] Template %zu (%s): building...\n", tmpl_idx,
                    name.c_str());
            auto t_tmpl = clk::now();
            const bool from_binary = bin_files[tmpl_idx].valid();
            if (from_binary) {
              // Binary without every attribute: the full tree comes from the JSON copy.
              boost::json::value v = read_and_parse_graph_json(json_path_list[tmpl_idx]);
              auto& re_root = v.as_object();
              re_root.erase("generators");
              re_root.erase("allocator_events");
              re_root.erase("output_tensors");
              all_parsed[tmpl_idx].root_val = std::move(v);
            }
            boost::json::value tmpl_json_copy;
            if (!from_binary)
              tmpl_json_copy = all_parsed[tmpl_idx].root_val;
            double reread_ms = ms_since(t_tmpl);
            CUDAGraph::GraphTemplate tmpl;
            auto t_build = clk::now();
            auto result = CUDAGraph::build_graph_from_parsed(std::move(all_parsed[tmpl_idx]),
                                                             main_ctx, nullptr, &tmpl);
            double build_ms = ms_since(t_build);
            result.graph->transfer_to_shared_exec(shared, std::move(tmpl));
            if (from_binary) {
              CUDAGraph::prepare_on_demand_graph_binary(bin_files[tmpl_idx], result.graph, main_ctx,
                                                        shared, static_cast<int>(tmpl_idx));
            } else {
              ParsedGraphData tmpl_parsed;
              tmpl_parsed.graph = result.graph;
              tmpl_parsed.root_val = std::move(tmpl_json_copy);
              CUDAGraph::prepare_on_demand_graph(tmpl_parsed, main_ctx, shared,
                                                 static_cast<int>(tmpl_idx));
            }
            result.graph->on_demand_data_->graph_name = name;
            result.graph->materialize_on_demand_exec();
            double tmpl_ms = ms_since(t_tmpl);
            BuildTiming b = last_build_timing();
            {
              std::lock_guard<std::mutex> lock(stats_mu);
              tmpl_inst_ms += b.inst_ms;
            }
            fprintf(stderr,
                    "[foundry BUILD] Template %zu (%s): %zu nodes, done in %.1f ms (reread %.1f, "
                    "build %.1f: json %.1f, func %.1f, func-attrs %.1f/%zu calls, add %.1f, "
                    "node-attrs %.1f, deps %.1f, instantiate %.1f; on-demand prep %.1f)\n",
                    tmpl_idx, name.c_str(), shared->ordered_nodes.size(), tmpl_ms, reread_ms,
                    build_ms, b.json_ms, b.func_ms, b.func_attr_ms, b.func_attr_calls, b.add_ms,
                    b.node_attr_ms, b.deps_ms, b.inst_ms, tmpl_ms - reread_ms - build_ms);
            set_free(tmpl_idx);
          });
        }
      }

      // Wait for all on-demand prep to finish
      for (auto& f : on_demand_futures) {
        f.get();
      }

      // Phase 2c: members, round-robin across groups (index order within a group). Each is
      // linked to its group's shared exec, rewrites the builder graph once the previous
      // instantiate of it has returned, and is instantiated on the instantiate thread
      // (~5-10 ms per ~1000-1600-node graph) so no replay ever stalls.
      // FOUNDRY_LAZY_GRAPH_EXEC=1 defers the rewrite + instantiate to the first replay.
      std::vector<std::vector<size_t>> group_members;
      std::unordered_map<size_t, size_t> group_pos;
      for (size_t idx : tmpl_order) {
        group_pos[idx] = group_members.size();
        group_members.emplace_back();
      }
      for (size_t i = 0; i < num; ++i)
        if (template_for[i] != SIZE_MAX)
          group_members[group_pos.at(template_for[i])].push_back(i);
      size_t max_members = 0;
      for (auto& gm : group_members)
        max_members = std::max(max_members, gm.size());
      std::vector<size_t> member_order;
      for (size_t round = 0; round < max_members; ++round)
        for (auto& gm : group_members)
          if (round < gm.size())
            member_order.push_back(gm[round]);

      for (size_t i : member_order) {
        size_t g = template_for[i];
        auto t_w = clk::now();
        if (!wait_free(g))
          break;  // the instantiate thread failed; finish() rethrows below
        member_wait_ms += ms_since(t_w);
        CUDAGraph* member = all_parsed[i].graph.get();
        CUDAGraph::link_on_demand_shared_exec(*member, shared_execs.at(g));
        if (lazy_graph_exec)
          continue;
        member->rewrite_shared_graph_for_member();
        {
          std::lock_guard<std::mutex> lock(group_mu);
          group_busy[g] = true;
        }
        inst.push([member, g, &set_free]() {
          member->instantiate_member_exec();
          set_free(g);
        });
      }
      inst.push([&free_after, &mem_total] {
        C10_CUDA_DRIVER_CHECK(cuMemGetInfo(&free_after, &mem_total));
      });
      inst.finish();
      inst.rethrow();

      double phase2_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_phase2)
              .count();
      fprintf(stderr,
              "[foundry] Phase 2: %zu templates + %zu on-demand = %zu graphs built in %.1f ms "
              "(members: param/attr rewrite %.1f ms, instantiate %.1f ms, %llu attribute calls), "
              "pipeline: template builds %.1f ms, template instantiates %.1f ms, build thread "
              "waited %.1f ms for a free builder graph, instantiate thread busy %.1f ms / idle "
              "%.1f ms, exec pool prewarm %s, device memory taken during the instantiates %.0f "
              "MB; Phase 2 start %.3f (epoch s)\n",
              topology_groups.size(), num - topology_groups.size(), num, phase2_ms,
              CUDAGraph::g_member_update_us.load() / 1000.0,
              CUDAGraph::g_member_inst_us.load() / 1000.0,
              (unsigned long long)CUDAGraph::g_member_attr_calls.load(), tmpl_build_ms,
              tmpl_inst_ms, member_wait_ms, inst.busy_ms, inst.idle_ms,
              prewarm_done ? "finished before Phase 2" : "not used",
              ((double)free_before - (double)free_after) / (1024.0 * 1024.0), phase2_start_epoch);

      build_promise->set_value();
    } catch (...) {
      build_promise->set_exception(std::current_exception());
    }
  });
  bg_thread.detach();

  return pending;
}

// ============================================================================
// finish_graph_loads_impl: generator registration + allocator replay +
//                          output tensor reconstruction
//
// Called after model initialization. Generator registration is done here
// (not in start_graph_builds) because CUDAGeneratorState creation
// allocates a GPU tensor via cuMemAlloc_v2. Doing it before weight
// loading would shift the VMM pointer, causing address mismatch vs
// SAVE mode.
// ============================================================================

// Per-entry finish. Idempotent on build_complete_ wait (shared_future).
GraphLoadResult finish_one_graph_load_impl(std::shared_ptr<PendingGraphLoads> pending, size_t index,
                                           ReconstructTensorFn reconstruct_fn) {
  if (pending->build_complete_.valid()) {
    pending->build_complete_.wait();
  }

  if (index >= pending->entries.size()) {
    throw std::out_of_range("finish_one_graph_load: index " + std::to_string(index) +
                            " out of range (num_graphs=" + std::to_string(pending->entries.size()) +
                            ")");
  }

  auto& entry = pending->entries[index];

  // Register deferred generators (before allocator replay to match
  // SAVE mode timing where generators are created before graph capture).
  //
  // The first registration lazily creates the generator state's
  // seed/offset_extragraph_ tensors (two at::empty in
  // CUDAGeneratorState::register_graph). Whether that at::empty grows a
  // fresh caching-allocator segment depends on the free-pool state, which
  // is NOT mirrored between SAVE and LOAD (SAVE's lazy init fires inside
  // capture_begin, before the start_base_addr snapshot). If the growth
  // happened inside the tracked region here, the VMM cursor would move
  // past the recorded start_base_addr and replay_hook_events_from_json
  // below would abort ("Memory offset mismatch during replay", observed
  // as a 2 MB drift on A100). Suspend the region so any segment growth
  // lands outside the tracked cursor.
  if (pending->registry && !entry.generators_meta.is_null()) {
    foundry::SuspendAllocationRegion suspend_region;
    const boost::json::array& gen_array = entry.generators_meta.as_array();
    for (const auto& gen_val : gen_array) {
      const boost::json::object& gen_obj = gen_val.as_object();
      uint64_t state_id = gen_obj.at("id").to_number<uint64_t>();
      uint64_t seed = gen_obj.at("seed").to_number<uint64_t>();
      uint64_t wholegraph_increment = gen_obj.at("wholegraph_increment").to_number<uint64_t>();

      auto state = pending->registry->get_state_from_id(state_id, seed);
      entry.graph->register_generator_state(state, wholegraph_increment);
    }
  }

  // Replay allocator events (advances VMM cursor)
  const boost::json::object& alloc_events = entry.allocator_events.as_object();
  foundry::replay_hook_events_from_json(alloc_events);

  // Reconstruct output tensors from extracted metadata
  return make_load_result_from_extracted(entry.graph, entry.output_tensors_meta, reconstruct_fn);
}

std::vector<GraphLoadResult> finish_graph_loads_impl(std::shared_ptr<PendingGraphLoads> pending,
                                                     ReconstructTensorFn reconstruct_fn) {
  size_t num_graphs = pending->entries.size();
  std::vector<GraphLoadResult> results;
  results.reserve(num_graphs);

  using Clock = std::chrono::steady_clock;
  auto t_start = Clock::now();

  // Wait for background graph building to complete
  if (pending->build_complete_.valid()) {
    auto t_wait = Clock::now();
    pending->build_complete_.get();
    double wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_wait).count();
    if (wait_ms > 1.0) {
      fprintf(stderr, "[foundry] finish_graph_loads: waited %.1f ms for background build\n",
              wait_ms);
    }
  }

  for (size_t i = 0; i < num_graphs; ++i) {
    results.push_back(finish_one_graph_load_impl(pending, i, reconstruct_fn));
  }

  double ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
  fprintf(stderr, "[foundry] finish_graph_loads: %zu graphs, %.1f ms\n", num_graphs, ms);

  return results;
}

}  // namespace foundry
