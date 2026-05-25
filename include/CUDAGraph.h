#pragma once

#include <ATen/Tensor.h>
#include <boost/unordered/concurrent_flat_map_fwd.hpp>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/json.hpp>
#include <c10/core/Device.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/intrusive_ptr.h>
#include "metadata.h"

#include <vector>
#include <memory>
#include <atomic>
#include <variant>

namespace at {

struct Generator;
struct CUDAGeneratorImpl;
struct CUDAGeneratorState;

}  // namespace at

namespace foundry {

struct BinaryGraphFile;  // forward declaration, defined in BinaryGraphFormat.h

using MempoolId_t = c10::MempoolId_t;
using CaptureId_t = c10::CaptureId_t;

enum class OutputTensorType { None, Single, List, Tuple };

using OutputTensors = std::variant<std::monostate, at::Tensor, std::vector<at::Tensor>>;

struct CUDAGraph;
struct PendingGraphLoads;

using ReconstructTensorFn = at::Tensor (*)(const boost::json::object&);

struct GraphLoadResult {
  std::shared_ptr<CUDAGraph> graph;
  OutputTensors outputs;
  OutputTensorType output_type;
};

struct ParsedGraphData {
  std::shared_ptr<CUDAGraph> graph;
  boost::json::value root_val;  // owns the parsed JSON
  MempoolId_t pool;
};

MempoolId_t graph_pool_handle();

void preallocate_cublas_workspaces();

struct CUDAGeneratorStateRegistry {
  uint64_t query_state_id(at::CUDAGeneratorState* state);
  c10::intrusive_ptr<at::CUDAGeneratorState> get_state_from_id(uint64_t id, uint64_t seed);

 protected:
  std::atomic<uint64_t> id_counter{0};
  boost::concurrent_flat_map<at::CUDAGeneratorState*, uint64_t> id_map_;
  boost::concurrent_flat_map<uint64_t, c10::intrusive_ptr<at::CUDAGeneratorState>> state_pool_;
};

struct CUDAGraph {
  CUDAGraph(bool keep_graph = false);
  ~CUDAGraph();

  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state,
                                uint64_t wholegraph_increment);
  void capture_begin(MempoolId_t pool = {0, 0},
                     cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal);
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);
  cudaGraph_t raw_cuda_graph();
  cudaGraphExec_t raw_cuda_graph_exec();

  void analyze_captured_graph();
  void save(const std::string& json_path, const OutputTensors& output_tensors = std::monostate{},
            OutputTensorType output_type = OutputTensorType::None);
  void save_binary(const std::string& bin_path, const boost::json::object& json_root);
  static GraphLoadResult load(const std::string& json_path, MempoolId_t pool = {0, 0});

  // Create graph shell and register generators, but do NOT replay allocator events.
  // Used by start_graph_builds to create shells early before weight loading.
  static ParsedGraphData prepare_graph_shell(boost::json::value&& root_val, MempoolId_t pool,
                                             CUDAGeneratorStateRegistry& registry);
  // Template for topology-grouped graph building
  struct GraphTemplate {
    CUgraph cuGraph;                         // the template CUgraph
    std::vector<CUgraphNode> ordered_nodes;  // nodes in creation order
    std::vector<std::string> node_types;     // type of each node
  };

  // Pre-decoded node params for on-demand exec update at replay time.
  // Instead of building a separate CUgraph+CUgraphExec per batch size,
  // graphs in the same topology group share one exec and update its
  // node params before each launch via cuGraphExecKernelNodeSetParams.
  struct OnDemandNodeUpdate {
    enum NodeType { Kernel, Memset, Memcpy, EventRecord, EventWait, Empty };
    NodeType type = Empty;

    // KernelNode
    CUDA_KERNEL_NODE_PARAMS kernel_params{};
    std::vector<std::vector<uint8_t>> kernel_param_data;
    std::vector<void*> kernel_param_ptrs;
    std::vector<void*> extra_config;
    std::vector<uint8_t> arg_buffer;
    size_t arg_buffer_size = 0;

    // Kernel node attributes (set via cuGraphKernelNodeSetAttribute).
    // Needed because cuGraphExecKernelNodeSetParams cannot update these.
    struct KernelNodeAttrs {
      unsigned int clusterDimX = 0, clusterDimY = 0, clusterDimZ = 0;
      bool has_cluster_dim = false;
      unsigned int preferredClusterDimX = 0, preferredClusterDimY = 0, preferredClusterDimZ = 0;
      bool has_preferred_cluster_dim = false;
      int clusterSchedulingPolicy = -1;
      int cooperative = -1;
      int priority = -1;
      int memSyncDomain = -1;
      unsigned char memSyncDomainMapDefault = 0, memSyncDomainMapRemote = 0;
      bool has_mem_sync_domain_map = false;
      unsigned int sharedMemCarveout = 0;
      bool has_shared_mem_carveout = false;
    };
    KernelNodeAttrs kernel_attrs;

    // MemsetNode
    CUDA_MEMSET_NODE_PARAMS memset_params{};

    // MemcpyNode
    CUDA_MEMCPY3D memcpy_params{};

    // EventRecord/EventWait
    CUevent event = nullptr;

    // Rebuild internal pointers after struct is in its final location.
    void fixup_pointers();
  };

  // Shared exec state for a topology group. Owned jointly by all graphs
  // in the group; the template's CUgraph/CUgraphExec are transferred here.
  struct SharedGraphExec {
    CUgraphExec exec = nullptr;
    CUgraph graph = nullptr;
    std::vector<CUgraphNode> ordered_nodes;
    CUcontext ctx = nullptr;
    int current_params_id = -1;

    ~SharedGraphExec();
  };

  struct OnDemandData {
    std::shared_ptr<SharedGraphExec> shared_exec;
    int graph_id = -1;
    std::string graph_name;  // e.g. filename for debug messages
    std::vector<OnDemandNodeUpdate> updates;
  };

  static GraphLoadResult build_graph_from_parsed(ParsedGraphData&& parsed, CUcontext ctx,
                                                 ReconstructTensorFn reconstruct_fn,
                                                 GraphTemplate* out_template = nullptr);

  // Prepare on-demand replay data: parse node params from JSON without
  // building a CUgraph. At replay time, the shared exec is updated.
  // shared_exec can be nullptr — call link_on_demand_shared_exec() later.
  static void prepare_on_demand_graph(ParsedGraphData& parsed, CUcontext ctx,
                                      std::shared_ptr<SharedGraphExec> shared_exec, int graph_id);

  // Binary-native overload: reads directly from BinaryGraphFile structs.
  // No JSON parsing, no hex decoding. ~10x faster than JSON path.
  static void prepare_on_demand_graph_binary(const struct BinaryGraphFile& bin_file,
                                             std::shared_ptr<CUDAGraph> graph, CUcontext ctx,
                                             std::shared_ptr<SharedGraphExec> shared_exec,
                                             int graph_id);

  // Link a SharedGraphExec to an already-prepared on-demand graph.
  // Called after template builds complete to associate on-demand data
  // with the template's shared exec.
  static void link_on_demand_shared_exec(CUDAGraph& graph,
                                         std::shared_ptr<SharedGraphExec> shared_exec);

  // Transfer graph/exec ownership to a SharedGraphExec.
  // After this call, the CUDAGraph no longer owns graph_ or graph_exec_.
  void transfer_to_shared_exec(std::shared_ptr<SharedGraphExec> shared, GraphTemplate&& tmpl);

  // Split load API: start builds early to overlap with weight loading.
  static std::shared_ptr<PendingGraphLoads> start_graph_builds(
      const std::vector<std::string>& json_paths, MempoolId_t pool = {0, 0}, int num_threads = 4);
  // Split load API: finish with allocator replay + output tensor reconstruction.
  static std::vector<GraphLoadResult> finish_graph_loads(
      std::shared_ptr<PendingGraphLoads> pending);
  // Per-graph variant: finish one entry by index. Waits on background build
  // on first call (idempotent for shared_future). Caller must walk indices in
  // the same order SAVE captured them so VMM cursor advances stay monotonic.
  static GraphLoadResult finish_one_graph_load(std::shared_ptr<PendingGraphLoads> pending,
                                               size_t index);

 protected:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;

  bool has_graph_ = false;
  bool capture_ended_ = false;
  bool has_graph_exec_ = false;

  CaptureId_t capture_id_ = -1;

  MempoolId_t mempool_id_;

  at::cuda::CUDAStream capture_stream_;

  ska::flat_hash_map<c10::intrusive_ptr<at::CUDAGeneratorState>, uint64_t>
      captured_generator_states_;

  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};

  bool keep_graph_ = false;

  std::vector<GraphNode> graph_nodes;
  std::vector<GraphDependency> graph_dependencies;

  boost::json::object allocator_events_;

  struct LoadedGraphResources {
    std::vector<CUevent> created_events;
  };
  std::unique_ptr<LoadedGraphResources> loaded_graph_resources_;

 public:
  // On-demand replay: this graph shares a template exec and updates
  // node params before each launch instead of owning its own exec.
  std::unique_ptr<OnDemandData> on_demand_data_;
};

}  // namespace foundry
