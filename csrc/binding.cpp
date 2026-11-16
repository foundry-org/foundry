// foundry.ops native part: Foundry's entry points as torch custom ops in the "foundry"
// namespace, registered through the LibTorch Stable ABI (STABLE_TORCH_LIBRARY + TORCH_BOX
// over StableIValue, target torch 2.14). No Python C API: libfoundry_ops.so is a plain
// shared library that foundry/ops.py loads with torch.ops.load_library and calls through
// torch.ops.foundry.<op>, so one build serves every CPython and every torch >= 2.14.
//
// Argument types are those the stable ABI converts (int, float, bool, str, Tensor,
// Optional, List): objects (CUDAGraph, PendingGraphLoads) cross as int64 handles that
// box a std::shared_ptr; pairs and tuples cross as flat int lists; uint64 values cross
// as their int64 bit pattern. foundry/ops.py owns the Python shapes (classes, tuples,
// output containers, events). torch releases the GIL around every op.
// Errors are std::runtime_error only (FOUNDRY_CHECK): torch maps other C++ exception
// types to Python types that differ between torch versions.

#include <dlfcn.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"
#include "hook.h"

namespace {

using StableTensor = torch::stable::Tensor;
using GraphPtr = std::shared_ptr<::foundry::CUDAGraph>;
using PendingPtr = std::shared_ptr<::foundry::PendingGraphLoads>;
using IntList = std::vector<int64_t>;

// Live boxed handles (tests: every handle foundry/ops.py creates is destroyed once).
std::atomic<int64_t> g_live_graph_handles{0};
std::atomic<int64_t> g_live_pending_handles{0};

template <typename T>
T as_ptr(int64_t v) {
  return reinterpret_cast<T>(static_cast<uintptr_t>(v));
}

template <typename T>
int64_t as_int(T p) {
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p));
}

size_t as_size(int64_t v, const char* what) {
  FOUNDRY_CHECK(v >= 0, what, " must be >= 0, got ", v);
  return static_cast<size_t>(v);
}

::foundry::MempoolId_t as_pool(const std::optional<IntList>& pool) {
  if (!pool)
    return {0, 0};
  FOUNDRY_CHECK(pool->size() == 2, "pool must be a (int, int) pair");
  return {static_cast<unsigned long long>((*pool)[0]), static_cast<unsigned long long>((*pool)[1])};
}

// ------------------------------------------------------------------ handles

int64_t box_graph(GraphPtr graph) {
  FOUNDRY_CHECK(graph, "foundry.ops: no graph to box");
  auto* boxed = new GraphPtr(std::move(graph));
  g_live_graph_handles.fetch_add(1, std::memory_order_relaxed);
  return as_int(boxed);
}

::foundry::CUDAGraph& graph_of(int64_t h) {
  FOUNDRY_CHECK(h != 0, "foundry.ops: null graph handle");
  return **as_ptr<GraphPtr*>(h);
}

int64_t box_pending(PendingPtr pending) {
  FOUNDRY_CHECK(pending, "foundry.ops: no pending loads to box");
  auto* boxed = new PendingPtr(std::move(pending));
  g_live_pending_handles.fetch_add(1, std::memory_order_relaxed);
  return as_int(boxed);
}

const PendingPtr& pending_of(int64_t h) {
  FOUNDRY_CHECK(h != 0, "foundry.ops: null PendingGraphLoads handle");
  return *as_ptr<PendingPtr*>(h);
}

int64_t graph_create(bool keep_graph) {
  return box_graph(std::make_shared<::foundry::CUDAGraph>(keep_graph));
}

// The Python owner calls this exactly once (weakref.finalize). Drops this handle's
// reference; the CUDAGraph dies with its last owner (other handles, pending loads).
void graph_destroy(int64_t h) {
  FOUNDRY_CHECK(h != 0, "foundry.ops: null graph handle");
  delete as_ptr<GraphPtr*>(h);
  g_live_graph_handles.fetch_sub(1, std::memory_order_relaxed);
}

void pending_destroy(int64_t h) {
  FOUNDRY_CHECK(h != 0, "foundry.ops: null PendingGraphLoads handle");
  delete as_ptr<PendingPtr*>(h);
  g_live_pending_handles.fetch_sub(1, std::memory_order_relaxed);
}

std::tuple<int64_t, int64_t> live_handles() {
  return {g_live_graph_handles.load(), g_live_pending_handles.load()};
}

// Path of this library as the loader mapped it (foundry/ops.py refuses a second copy).
std::string library_path() {
  Dl_info info{};
  FOUNDRY_CHECK(dladdr(reinterpret_cast<void*>(&library_path), &info) != 0 && info.dli_fname,
                "foundry.ops: dladdr failed");
  return info.dli_fname;
}

// ------------------------------------------------------------------ tensors

// A new owning handle on the same TensorImpl (the stable Tensor keeps its own).
::foundry::Tensor from_stable(const StableTensor& t, bool requires_grad) {
  AtenTensorHandle h = nullptr;
  FOUNDRY_SHIM_CHECK(aoti_torch_new_tensor_handle(t.get(), &h));
  ::foundry::Tensor out;
  out.handle = std::make_shared<::foundry::TensorHandle>(h);
  out.requires_grad = requires_grad;
  return out;
}

StableTensor to_stable(const ::foundry::Tensor& t) {
  AtenTensorHandle h = nullptr;
  FOUNDRY_SHIM_CHECK(aoti_torch_new_tensor_handle(t.handle->get(), &h));
  return StableTensor(h);
}

// Output kinds (foundry/ops.py _OUT_*): what the pybind binding built as Python objects.
constexpr int64_t kOutNone = 0, kOutSingle = 1, kOutList = 2, kOutTuple = 3;

using LoadTuple = std::tuple<int64_t, std::vector<StableTensor>, int64_t>;

void pack_outputs(const ::foundry::GraphLoadResult& r, std::vector<StableTensor>& tensors,
                  int64_t& kind) {
  using ::foundry::OutputTensorType;
  kind = kOutNone;
  if (r.output_type == OutputTensorType::Single) {
    if (std::holds_alternative<::foundry::Tensor>(r.outputs)) {
      tensors.push_back(to_stable(std::get<::foundry::Tensor>(r.outputs)));
      kind = kOutSingle;
    }
  } else if (r.output_type == OutputTensorType::List || r.output_type == OutputTensorType::Tuple) {
    if (std::holds_alternative<std::vector<::foundry::Tensor>>(r.outputs)) {
      for (const auto& t : std::get<std::vector<::foundry::Tensor>>(r.outputs))
        tensors.push_back(to_stable(t));
      kind = r.output_type == OutputTensorType::List ? kOutList : kOutTuple;
    }
  }
}

LoadTuple pack_load_result(const ::foundry::GraphLoadResult& r) {
  std::vector<StableTensor> tensors;
  int64_t kind = kOutNone;
  pack_outputs(r, tensors, kind);
  return {box_graph(r.graph), std::move(tensors), kind};
}

// ------------------------------------------------------------------ driver helpers

void check_driver_call(CUresult rc, const char* call) {
  if (rc == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  if (cuGetErrorName(rc, &name) != CUDA_SUCCESS || !name)
    name = "unknown CUresult";
  FOUNDRY_CHECK(false, call, " failed: ", name, " (", static_cast<int>(rc), ")");
}

// Driver calls need a current context on the calling thread; cudaSetDevice makes the
// primary context of the runtime's current device current (CUDA 12+).
void ensure_current_context() {
  CUcontext ctx = nullptr;
  check_driver_call(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  if (!ctx) {
    int device = 0;
    FOUNDRY_CUDA_CHECK(cudaGetDevice(&device));
    FOUNDRY_CUDA_CHECK(cudaSetDevice(device));
  }
}

// ------------------------------------------------------------------ module functions

void native_record_begin() {
  ::foundry::native_record_begin();
}
void native_record_abort() {
  ::foundry::native_record_abort();
}
bool recorded_allocation_contains(int64_t addr, int64_t len) {
  return ::foundry::recorded_allocation_contains(static_cast<uint64_t>(addr), as_size(len, "len"));
}
void set_allocation_region(int64_t base, int64_t size) {
  ::foundry::set_allocation_region(as_ptr<void*>(base), as_size(size, "size"));
}
void stop_allocation_region() {
  ::foundry::stop_allocation_region();
}
void resume_allocation_region() {
  ::foundry::resume_allocation_region();
}
bool preallocate_region(int64_t size) {
  return ::foundry::preallocate_region(as_size(size, "size"));
}
// Flat [offset0, size0, offset1, size1, ...].
IntList get_live_region_ranges() {
  IntList flat;
  for (const auto& [offset, size] : ::foundry::get_live_region_ranges()) {
    flat.push_back(static_cast<int64_t>(offset));
    flat.push_back(static_cast<int64_t>(size));
  }
  return flat;
}
bool preallocate_ranges(IntList ranges_flat, int64_t end_offset) {
  FOUNDRY_CHECK(ranges_flat.size() % 2 == 0, "ranges must be (offset, size) pairs");
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(ranges_flat.size() / 2);
  for (size_t i = 0; i < ranges_flat.size(); i += 2)
    ranges.emplace_back(as_size(ranges_flat[i], "range offset"),
                        as_size(ranges_flat[i + 1], "range size"));
  return ::foundry::preallocate_ranges(ranges, as_size(end_offset, "end_offset"));
}
void free_preallocated_region() {
  ::foundry::free_preallocated_region();
}
int64_t get_current_alloc_offset() {
  return static_cast<int64_t>(::foundry::get_current_alloc_offset());
}
void set_current_alloc_offset(int64_t offset) {
  ::foundry::set_current_alloc_offset(as_size(offset, "offset"));
}
void set_pack_fatbins_on_exit(bool enabled) {
  ::foundry::set_pack_fatbins_on_exit(enabled);
}
void set_skip_fatbin_processing(bool enabled) {
  ::foundry::set_skip_fatbin_processing(enabled);
}
void set_nvshmem_auto_init(bool enabled) {
  ::foundry::set_nvshmem_auto_init(enabled);
}
// Process-wide binary steps (once per process, from the integration's main thread). The
// pybind11 binding ran them under the GIL, i.e. never two at once; ops run without it, so
// this lock keeps them mutually exclusive.
std::mutex g_binary_steps_mutex;
int64_t init_nvshmem_for_loaded_modules() {
  std::lock_guard<std::mutex> lock(g_binary_steps_mutex);
  return ::foundry::init_nvshmem_for_loaded_modules();
}
void pack_fatbins_to_folder(std::string folder_path) {
  std::lock_guard<std::mutex> lock(g_binary_steps_mutex);
  ::foundry::pack_fatbins_to_folder(folder_path);
}
void load_cuda_modules_and_libraries(std::string archive_dir) {
  std::lock_guard<std::mutex> lock(g_binary_steps_mutex);
  ::foundry::load_cuda_modules_and_libraries(archive_dir);
}
// torch >= 2.15 (foundry/ops.py reads torch.__version__).
void preallocate_cublas_workspaces(bool torch_has_workspace_cache_flag) {
  ::foundry::preallocate_cublas_workspaces(torch_has_workspace_cache_flag);
}

// CUevent / CUstream handles as integers (foundry.graph.RestoredEvent). Driver errors
// raise RuntimeError with the error name; event_query maps CUDA_ERROR_NOT_READY to False.
bool event_query(int64_t handle) {
  ensure_current_context();
  const CUresult rc = cuEventQuery(as_ptr<CUevent>(handle));
  if (rc == CUDA_ERROR_NOT_READY)
    return false;
  check_driver_call(rc, "cuEventQuery");
  return true;
}
void event_synchronize(int64_t handle) {
  ensure_current_context();
  check_driver_call(cuEventSynchronize(as_ptr<CUevent>(handle)), "cuEventSynchronize");
}
double event_elapsed_time(int64_t start, int64_t end) {
  ensure_current_context();
  float ms = 0.0f;
  check_driver_call(cuEventElapsedTime(&ms, as_ptr<CUevent>(start), as_ptr<CUevent>(end)),
                    "cuEventElapsedTime");
  return ms;
}
void event_record(int64_t handle, int64_t stream) {
  ensure_current_context();
  check_driver_call(cuEventRecord(as_ptr<CUevent>(handle), as_ptr<CUstream>(stream)),
                    "cuEventRecord");
}
void stream_wait_event(int64_t stream, int64_t handle) {
  ensure_current_context();
  check_driver_call(cuStreamWaitEvent(as_ptr<CUstream>(stream), as_ptr<CUevent>(handle), 0),
                    "cuStreamWaitEvent");
}

// ------------------------------------------------------------------ CUDAGraph

int64_t graph_ptr(int64_t h) {
  return as_int(&graph_of(h));
}

// generators: flattened NativeGenerator 6-tuples; uint64 values as int64 bit patterns.
void graph_adopt_native_capture(int64_t h, int64_t raw_graph, IntList pool, IntList generators,
                                IntList rng_candidates) {
  FOUNDRY_CHECK(generators.size() % 6 == 0, "generators must be 6-tuples");
  std::vector<::foundry::NativeGenerator> gens;
  for (size_t i = 0; i < generators.size(); i += 6) {
    const auto u = [&](size_t k) { return static_cast<uint64_t>(generators[i + k]); };
    gens.emplace_back(u(0), u(1), u(2), u(3), u(4), u(5));
  }
  std::vector<uint64_t> candidates(rng_candidates.begin(), rng_candidates.end());
  graph_of(h).adopt_native_capture(static_cast<uintptr_t>(raw_graph), as_pool(pool), gens,
                                   candidates);
}

void graph_replay(int64_t h) {
  graph_of(h).replay();
}

std::string graph_exec_share_stats(int64_t h) {
  return graph_of(h).exec_share_stats();
}

// Per event: id, handle, flags, and its record / wait node indices as counts + flat lists.
std::tuple<IntList, IntList, IntList, IntList, IntList, IntList, IntList> graph_loaded_events(
    int64_t h) {
  IntList ids, handles, flags, record_counts, record_nodes, wait_counts, wait_nodes;
  for (const auto& e : graph_of(h).loaded_events()) {
    ids.push_back(e.event_id);
    handles.push_back(as_int(e.event));
    flags.push_back(static_cast<int64_t>(e.flags));
    record_counts.push_back(static_cast<int64_t>(e.record_nodes.size()));
    for (size_t n : e.record_nodes)
      record_nodes.push_back(static_cast<int64_t>(n));
    wait_counts.push_back(static_cast<int64_t>(e.wait_nodes.size()));
    for (size_t n : e.wait_nodes)
      wait_nodes.push_back(static_cast<int64_t>(n));
  }
  return {ids, handles, flags, record_counts, record_nodes, wait_counts, wait_nodes};
}

void graph_bind_events(int64_t h, IntList event_ids, IntList handles) {
  FOUNDRY_CHECK(event_ids.size() == handles.size(),
                "bind_events: ids and handles differ in length");
  std::vector<std::pair<int, CUevent>> events;
  events.reserve(event_ids.size());
  for (size_t i = 0; i < event_ids.size(); ++i)
    events.emplace_back(static_cast<int>(event_ids[i]), as_ptr<CUevent>(handles[i]));
  graph_of(h).bind_events(events);
}

void graph_reset(int64_t h) {
  graph_of(h).reset();
}

std::tuple<int64_t, int64_t> graph_pool(int64_t h) {
  const auto pool = graph_of(h).pool();
  return {static_cast<int64_t>(pool.first), static_cast<int64_t>(pool.second)};
}

void graph_enable_debug_mode(int64_t h) {
  graph_of(h).enable_debug_mode();
}

void graph_debug_dump(int64_t h, std::string debug_path) {
  graph_of(h).debug_dump(debug_path);
}

int64_t graph_raw_cuda_graph(int64_t h) {
  return as_int(graph_of(h).raw_cuda_graph());
}

// outputs + requires_grad (no stable getter in 2.14; foundry/ops.py passes it) + kind.
void graph_save(int64_t h, std::string json_path, std::vector<StableTensor> outputs,
                IntList requires_grad, int64_t output_kind) {
  using ::foundry::OutputTensorType;
  FOUNDRY_CHECK(requires_grad.size() == outputs.size(), "save: requires_grad per output");
  ::foundry::OutputTensors outs = std::monostate{};
  OutputTensorType type = OutputTensorType::None;
  if (output_kind == kOutSingle) {
    FOUNDRY_CHECK(outputs.size() == 1, "save: a single output needs one tensor");
    outs = from_stable(outputs[0], requires_grad[0] != 0);
    type = OutputTensorType::Single;
  } else if (output_kind == kOutList || output_kind == kOutTuple) {
    std::vector<::foundry::Tensor> tensors;
    tensors.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i)
      tensors.push_back(from_stable(outputs[i], requires_grad[i] != 0));
    outs = std::move(tensors);
    type = output_kind == kOutList ? OutputTensorType::List : OutputTensorType::Tuple;
  } else {
    FOUNDRY_CHECK(output_kind == kOutNone && outputs.empty(), "save: bad output kind ",
                  output_kind);
  }
  graph_of(h).save(json_path, outs, type);
}

LoadTuple graph_load(std::string json_path, std::optional<IntList> pool) {
  return pack_load_result(::foundry::CUDAGraph::load(json_path, as_pool(pool)));
}

void start_exec_pool_prewarm(std::vector<std::string> json_paths) {
  ::foundry::CUDAGraph::start_exec_pool_prewarm(json_paths);
}

int64_t start_graph_builds(std::vector<std::string> json_paths, std::optional<IntList> pool,
                           int64_t num_threads) {
  return box_pending(::foundry::CUDAGraph::start_graph_builds(json_paths, as_pool(pool),
                                                              static_cast<int>(num_threads)));
}

int64_t pending_size(int64_t p) {
  return static_cast<int64_t>(pending_of(p)->entries.size());
}

// Seconds from start_graph_builds' entry to the end of Phase 2; waits for the
// background builds first (0 for an empty or failed build).
double pending_build_seconds(int64_t p) {
  auto& pending = *pending_of(p);
  if (pending.build_complete_.valid())
    pending.build_complete_.wait();
  return *pending.build_seconds_;
}

// Per graph: handle, output kind and output count; outputs concatenated.
std::tuple<IntList, std::vector<StableTensor>, IntList, IntList> finish_graph_loads(int64_t p) {
  auto results = ::foundry::CUDAGraph::finish_graph_loads(pending_of(p));
  IntList handles, counts, kinds;
  std::vector<StableTensor> tensors;
  for (const auto& r : results) {
    const size_t before = tensors.size();
    int64_t kind = kOutNone;
    pack_outputs(r, tensors, kind);
    handles.push_back(box_graph(r.graph));
    counts.push_back(static_cast<int64_t>(tensors.size() - before));
    kinds.push_back(kind);
  }
  return {handles, std::move(tensors), counts, kinds};
}

LoadTuple finish_one_graph_load(int64_t p, int64_t index) {
  return pack_load_result(
      ::foundry::CUDAGraph::finish_one_graph_load(pending_of(p), as_size(index, "index")));
}

}  // namespace

STABLE_TORCH_LIBRARY(foundry, m) {
  m.def("library_path() -> str");
  m.def("live_handles() -> (int, int)");
  m.def("native_record_begin() -> ()");
  m.def("native_record_abort() -> ()");
  m.def("recorded_allocation_contains(int addr, int len) -> bool");
  m.def("set_allocation_region(int base, int size) -> ()");
  m.def("stop_allocation_region() -> ()");
  m.def("resume_allocation_region() -> ()");
  m.def("preallocate_region(int size) -> bool");
  m.def("get_live_region_ranges() -> int[]");
  m.def("preallocate_ranges(int[] ranges, int end_offset) -> bool");
  m.def("free_preallocated_region() -> ()");
  m.def("get_current_alloc_offset() -> int");
  m.def("set_current_alloc_offset(int offset) -> ()");
  m.def("set_pack_fatbins_on_exit(bool enabled) -> ()");
  m.def("set_skip_fatbin_processing(bool enabled) -> ()");
  m.def("set_nvshmem_auto_init(bool enabled) -> ()");
  m.def("init_nvshmem_for_loaded_modules() -> int");
  m.def("pack_fatbins_to_folder(str folder_path) -> ()");
  m.def("load_cuda_modules_and_libraries(str archive_dir) -> ()");
  m.def("preallocate_cublas_workspaces(bool torch_has_workspace_cache_flag) -> ()");
  m.def("event_query(int handle) -> bool");
  m.def("event_synchronize(int handle) -> ()");
  m.def("event_elapsed_time(int start, int end) -> float");
  m.def("event_record(int handle, int stream) -> ()");
  m.def("stream_wait_event(int stream, int handle) -> ()");
  m.def("graph_create(bool keep_graph) -> int");
  m.def("graph_destroy(int h) -> ()");
  m.def("graph_ptr(int h) -> int");
  m.def(
      "graph_adopt_native_capture(int h, int raw_graph, int[] pool, int[] generators, "
      "int[] rng_candidates) -> ()");
  m.def("graph_replay(int h) -> ()");
  m.def("graph_exec_share_stats(int h) -> str");
  m.def("graph_loaded_events(int h) -> (int[], int[], int[], int[], int[], int[], int[])");
  m.def("graph_bind_events(int h, int[] event_ids, int[] handles) -> ()");
  m.def("graph_reset(int h) -> ()");
  m.def("graph_pool(int h) -> (int, int)");
  m.def("graph_enable_debug_mode(int h) -> ()");
  m.def("graph_debug_dump(int h, str debug_path) -> ()");
  m.def("graph_raw_cuda_graph(int h) -> int");
  m.def(
      "graph_save(int h, str json_path, Tensor[] outputs, int[] requires_grad, "
      "int output_kind) -> ()");
  m.def("graph_load(str json_path, int[]? pool=None) -> (int, Tensor[], int)");
  m.def("start_exec_pool_prewarm(str[] json_paths) -> ()");
  m.def("start_graph_builds(str[] json_paths, int[]? pool, int num_threads) -> int");
  m.def("pending_destroy(int p) -> ()");
  m.def("pending_size(int p) -> int");
  m.def("pending_build_seconds(int p) -> float");
  m.def("finish_graph_loads(int p) -> (int[], Tensor[], int[], int[])");
  m.def("finish_one_graph_load(int p, int index) -> (int, Tensor[], int)");
}

STABLE_TORCH_LIBRARY_IMPL(foundry, CompositeExplicitAutograd, m) {
  m.impl("library_path", TORCH_BOX(&library_path));
  m.impl("live_handles", TORCH_BOX(&live_handles));
  m.impl("native_record_begin", TORCH_BOX(&native_record_begin));
  m.impl("native_record_abort", TORCH_BOX(&native_record_abort));
  m.impl("recorded_allocation_contains", TORCH_BOX(&recorded_allocation_contains));
  m.impl("set_allocation_region", TORCH_BOX(&set_allocation_region));
  m.impl("stop_allocation_region", TORCH_BOX(&stop_allocation_region));
  m.impl("resume_allocation_region", TORCH_BOX(&resume_allocation_region));
  m.impl("preallocate_region", TORCH_BOX(&preallocate_region));
  m.impl("get_live_region_ranges", TORCH_BOX(&get_live_region_ranges));
  m.impl("preallocate_ranges", TORCH_BOX(&preallocate_ranges));
  m.impl("free_preallocated_region", TORCH_BOX(&free_preallocated_region));
  m.impl("get_current_alloc_offset", TORCH_BOX(&get_current_alloc_offset));
  m.impl("set_current_alloc_offset", TORCH_BOX(&set_current_alloc_offset));
  m.impl("set_pack_fatbins_on_exit", TORCH_BOX(&set_pack_fatbins_on_exit));
  m.impl("set_skip_fatbin_processing", TORCH_BOX(&set_skip_fatbin_processing));
  m.impl("set_nvshmem_auto_init", TORCH_BOX(&set_nvshmem_auto_init));
  m.impl("init_nvshmem_for_loaded_modules", TORCH_BOX(&init_nvshmem_for_loaded_modules));
  m.impl("pack_fatbins_to_folder", TORCH_BOX(&pack_fatbins_to_folder));
  m.impl("load_cuda_modules_and_libraries", TORCH_BOX(&load_cuda_modules_and_libraries));
  m.impl("preallocate_cublas_workspaces", TORCH_BOX(&preallocate_cublas_workspaces));
  m.impl("event_query", TORCH_BOX(&event_query));
  m.impl("event_synchronize", TORCH_BOX(&event_synchronize));
  m.impl("event_elapsed_time", TORCH_BOX(&event_elapsed_time));
  m.impl("event_record", TORCH_BOX(&event_record));
  m.impl("stream_wait_event", TORCH_BOX(&stream_wait_event));
  m.impl("graph_create", TORCH_BOX(&graph_create));
  m.impl("graph_destroy", TORCH_BOX(&graph_destroy));
  m.impl("graph_ptr", TORCH_BOX(&graph_ptr));
  m.impl("graph_adopt_native_capture", TORCH_BOX(&graph_adopt_native_capture));
  m.impl("graph_replay", TORCH_BOX(&graph_replay));
  m.impl("graph_exec_share_stats", TORCH_BOX(&graph_exec_share_stats));
  m.impl("graph_loaded_events", TORCH_BOX(&graph_loaded_events));
  m.impl("graph_bind_events", TORCH_BOX(&graph_bind_events));
  m.impl("graph_reset", TORCH_BOX(&graph_reset));
  m.impl("graph_pool", TORCH_BOX(&graph_pool));
  m.impl("graph_enable_debug_mode", TORCH_BOX(&graph_enable_debug_mode));
  m.impl("graph_debug_dump", TORCH_BOX(&graph_debug_dump));
  m.impl("graph_raw_cuda_graph", TORCH_BOX(&graph_raw_cuda_graph));
  m.impl("graph_save", TORCH_BOX(&graph_save));
  m.impl("graph_load", TORCH_BOX(&graph_load));
  m.impl("start_exec_pool_prewarm", TORCH_BOX(&start_exec_pool_prewarm));
  m.impl("start_graph_builds", TORCH_BOX(&start_graph_builds));
  m.impl("pending_destroy", TORCH_BOX(&pending_destroy));
  m.impl("pending_size", TORCH_BOX(&pending_size));
  m.impl("pending_build_seconds", TORCH_BOX(&pending_build_seconds));
  m.impl("finish_graph_loads", TORCH_BOX(&finish_graph_loads));
  m.impl("finish_one_graph_load", TORCH_BOX(&finish_one_graph_load));
}
