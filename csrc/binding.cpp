#include <torch/extension.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <cstring>
#include <tuple>
#include "BinaryGraphFormat.h"
#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"
#include "metadata.h"
#include "hook.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace {

void check_driver_call(CUresult rc, const char* call) {
  if (rc == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  if (cuGetErrorName(rc, &name) != CUDA_SUCCESS || !name)
    name = "unknown CUresult";
  TORCH_CHECK(false, call, " failed: ", name, " (", static_cast<int>(rc), ")");
}

// Driver calls need a current context on the calling thread; cudaSetDevice makes the
// primary context of the runtime's current device current (CUDA 12+).
void ensure_current_context() {
  CUcontext ctx = nullptr;
  check_driver_call(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  if (!ctx) {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaSetDevice(device));
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graph_pool_handle", &::foundry::graph_pool_handle);

  m.def(
      "get_graph_ptr",
      [](::foundry::CUDAGraph& graph) { return reinterpret_cast<uintptr_t>(&graph); },
      py::arg("graph"), "Get pointer address of CUDAGraph object for hook integration");

  m.def(
      "set_allocation_region",
      [](uintptr_t base, size_t size) {
        ::foundry::set_allocation_region(reinterpret_cast<void*>(base), size);
      },
      py::arg("base"), py::arg("size"), "Set allocation region for CUDA VMM hooks");

  m.def(
      "stop_allocation_region", []() { ::foundry::stop_allocation_region(); },
      "Stop allocation region for CUDA VMM hooks");

  m.def(
      "resume_allocation_region", []() { ::foundry::resume_allocation_region(); },
      "Resume previously set allocation region for CUDA VMM hooks");

  m.def(
      "preallocate_region", [](size_t size) { return ::foundry::preallocate_region(size); },
      py::arg("size"),
      "Preallocate memory in the allocation region for fast subsequent allocations");

  m.def(
      "set_param_patches",
      [](const std::vector<
          std::tuple<std::string, std::string, uint32_t, uint32_t, uint32_t, uint64_t>>& patches) {
        std::vector<::foundry::ParamPatch> v;
        for (const auto& [g, k, pi, off, w, val] : patches)
          v.push_back({g, k, pi, off, w, val});
        ::foundry::set_param_patches(std::move(v));
      },
      py::arg("patches"),
      "Archive share: (graph_file, kernel_substr, param_index, byte_offset, width, value) "
      "patches applied "
      "in memory to every .cugraph read afterwards");
  m.def(
      "param_patch_counts", []() { return ::foundry::param_patch_counts(); },
      "(patched params, unappliable matches) since set_param_patches");
  m.def(
      "get_live_region_ranges", []() { return ::foundry::get_live_region_ranges(); },
      "Live (offset, size) ranges inside the allocation region, relative to its base");
  m.def(
      "preallocate_ranges",
      [](const std::vector<std::pair<size_t, size_t>>& ranges, size_t end_offset) {
        return ::foundry::preallocate_ranges(ranges, end_offset);
      },
      py::arg("ranges"), py::arg("end_offset"),
      "Map only the given live ranges and move the cursor to end_offset (sparse LOAD "
      "preallocation)");
  m.def(
      "free_preallocated_region", []() { ::foundry::free_preallocated_region(); },
      "Free the preallocated memory region");

  m.def(
      "get_current_alloc_offset", []() { return ::foundry::get_current_alloc_offset(); },
      "Get the current allocation offset from the region base address");

  m.def(
      "set_current_alloc_offset",
      [](size_t offset) { ::foundry::set_current_alloc_offset(offset); }, py::arg("offset"),
      "Set the current allocation offset from the region base address");

  m.def(
      "set_pack_fatbins_on_exit",
      [](bool enabled) { ::foundry::set_pack_fatbins_on_exit(enabled); }, py::arg("enabled"),
      "Enable or disable packing fatbins on exit");

  m.def(
      "set_skip_fatbin_processing",
      [](bool enabled) { ::foundry::set_skip_fatbin_processing(enabled); }, py::arg("enabled"),
      "Skip heavy fatbin processing in LOAD mode for faster startup");

  m.def(
      "set_nvshmem_auto_init", [](bool enabled) { ::foundry::set_nvshmem_auto_init(enabled); },
      py::arg("enabled"),
      "Enable/disable automatic NVSHMEM initialization for DeepEP kernels on load (default: true)");

  m.def(
      "init_nvshmem_for_loaded_modules",
      []() { return ::foundry::init_nvshmem_for_loaded_modules(); },
      "Initialize NVSHMEM for all already-loaded modules that require it. Call AFTER NVSHMEM "
      "runtime is initialized (e.g., after DeepEP Buffer creation). Returns count of modules "
      "initialized.");

  m.def(
      "pack_fatbins_to_folder",
      [](const std::string& folder_path) { ::foundry::pack_fatbins_to_folder(folder_path); },
      py::arg("folder_path"), "Pack all loaded fatbins/cubins to the specified folder");

  m.def(
      "load_cuda_modules_and_libraries",
      [](const std::string& archive_dir) {
        ::foundry::load_cuda_modules_and_libraries(archive_dir);
      },
      py::arg("archive_dir"), "Load CUDA modules and libraries from archive directory");

  m.def(
      "preallocate_cublas_workspaces", []() { ::foundry::preallocate_cublas_workspaces(); },
      "Pre-allocate cuBLAS and cuBLASLt handles and workspaces for the current stream");

  // CUevent handles as integers (foundry.graph.RestoredEvent). Driver errors raise
  // RuntimeError with the error name; event_query maps CUDA_ERROR_NOT_READY to False.
  m.def(
      "event_query",
      [](uintptr_t handle) {
        ensure_current_context();
        const CUresult rc = cuEventQuery(reinterpret_cast<CUevent>(handle));
        if (rc == CUDA_ERROR_NOT_READY)
          return false;
        check_driver_call(rc, "cuEventQuery");
        return true;
      },
      py::arg("handle"));
  m.def(
      "event_synchronize",
      [](uintptr_t handle) {
        ensure_current_context();
        check_driver_call(cuEventSynchronize(reinterpret_cast<CUevent>(handle)),
                          "cuEventSynchronize");
      },
      py::arg("handle"), py::call_guard<py::gil_scoped_release>());
  m.def(
      "event_elapsed_time",
      [](uintptr_t start, uintptr_t end) {
        ensure_current_context();
        float ms = 0.0f;
        check_driver_call(cuEventElapsedTime(&ms, reinterpret_cast<CUevent>(start),
                                             reinterpret_cast<CUevent>(end)),
                          "cuEventElapsedTime");
        return ms;
      },
      py::arg("start"), py::arg("end"));
  m.def(
      "event_record",
      [](uintptr_t handle, uintptr_t stream) {
        ensure_current_context();
        check_driver_call(
            cuEventRecord(reinterpret_cast<CUevent>(handle), reinterpret_cast<CUstream>(stream)),
            "cuEventRecord");
      },
      py::arg("handle"), py::arg("stream"));
  m.def(
      "stream_wait_event",
      [](uintptr_t stream, uintptr_t handle) {
        ensure_current_context();
        check_driver_call(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),
                                            reinterpret_cast<CUevent>(handle), 0),
                          "cuStreamWaitEvent");
      },
      py::arg("stream"), py::arg("handle"));

  py::class_<::foundry::KernelNodeMetadata>(m, "KernelNodeMetadata")
      .def_property_readonly("func",
                             [](const ::foundry::KernelNodeMetadata& self) {
                               return reinterpret_cast<uintptr_t>(self.func);
                             })
      .def_property_readonly("kern",
                             [](const ::foundry::KernelNodeMetadata& self) {
                               return reinterpret_cast<uintptr_t>(self.kern);
                             })
      .def_property_readonly("ctx",
                             [](const ::foundry::KernelNodeMetadata& self) {
                               return reinterpret_cast<uintptr_t>(self.ctx);
                             })
      .def_readonly("gridDimX", &::foundry::KernelNodeMetadata::gridDimX)
      .def_readonly("gridDimY", &::foundry::KernelNodeMetadata::gridDimY)
      .def_readonly("gridDimZ", &::foundry::KernelNodeMetadata::gridDimZ)
      .def_readonly("blockDimX", &::foundry::KernelNodeMetadata::blockDimX)
      .def_readonly("blockDimY", &::foundry::KernelNodeMetadata::blockDimY)
      .def_readonly("blockDimZ", &::foundry::KernelNodeMetadata::blockDimZ)
      .def_readonly("sharedMemBytes", &::foundry::KernelNodeMetadata::sharedMemBytes)
      .def_readonly("num_params", &::foundry::KernelNodeMetadata::num_params)
      .def_readonly("offset_and_sizes", &::foundry::KernelNodeMetadata::offset_and_sizes);

  py::class_<::foundry::PendingGraphLoads, std::shared_ptr<::foundry::PendingGraphLoads>>(
      m, "PendingGraphLoads")
      // Seconds from start_graph_builds' entry to the end of Phase 2; waits
      // for the background builds first (0 for an empty or failed build).
      .def_property_readonly("build_seconds", [](::foundry::PendingGraphLoads& self) {
        if (self.build_complete_.valid()) {
          py::gil_scoped_release release;
          self.build_complete_.wait();
        }
        return *self.build_seconds_;
      });

  shared_ptr_class_<::foundry::CUDAGraph>(m, "CUDAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::foundry::CUDAGraph& self, std::optional<c10::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            cudaStreamCaptureMode capture_mode{};
            c10::MempoolId_t pool =
                pool_opt.has_value() ? pool_opt.value() : c10::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(false,
                          "Unknown capture error mode. Expected `global`, `thread_local`, or "
                          "`relaxed`, got ",
                          capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"), py::arg("capture_error_mode"), py::call_guard<py::gil_scoped_release>())
      .def("capture_end", &foundry::CUDAGraph::capture_end,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "register_generator_state",
          [](::foundry::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def("replay", &foundry::CUDAGraph::replay, py::call_guard<py::gil_scoped_release>())
      .def("exec_share_stats", &foundry::CUDAGraph::exec_share_stats)
      .def("loaded_events",
           [](py::object self_obj) {
             // foundry.graph.LoadedEvent records (loaded graphs are ops.CUDAGraph objects,
             // so the Python-side shape lives here, not on foundry.CUDAGraph). Each
             // RestoredEvent holds a reference to the graph that owns its CUevent.
             const auto& self = self_obj.cast<const ::foundry::CUDAGraph&>();
             py::module_ graph_mod = py::module_::import("foundry.graph");
             py::object record = graph_mod.attr("LoadedEvent");
             py::object restored = graph_mod.attr("RestoredEvent");
             py::list out;
             for (const auto& e : self.loaded_events()) {
               const auto handle = reinterpret_cast<uintptr_t>(e.event);
               out.append(record(e.event_id, handle, e.flags, e.record_nodes, e.wait_nodes,
                                 restored(handle, e.flags, self_obj)));
             }
             return out;
           })
      .def(
          "bind_events",
          [](::foundry::CUDAGraph& self, const py::dict& events) {
            // {event_id: torch.cuda.Event | CUevent handle}
            std::vector<std::pair<int, CUevent>> handles;
            for (const auto& [key, value] : events) {
              uintptr_t handle = 0;
              if (py::isinstance<py::int_>(value)) {
                handle = value.cast<uintptr_t>();
              } else {
                TORCH_CHECK(py::hasattr(value, "cuda_event"),
                            "bind_events: expected a torch.cuda.Event or an int handle");
                handle = value.attr("cuda_event").cast<uintptr_t>();
                if (!handle) {  // torch.cuda.Event creates its CUDA event on first record
                  value.attr("record")();
                  handle = value.attr("cuda_event").cast<uintptr_t>();
                }
              }
              handles.emplace_back(key.cast<int>(), reinterpret_cast<CUevent>(handle));
            }
            py::gil_scoped_release release;
            self.bind_events(handles);
          },
          py::arg("events"))
      .def("reset", &foundry::CUDAGraph::reset, py::call_guard<py::gil_scoped_release>())
      .def("pool", &foundry::CUDAGraph::pool, py::call_guard<py::gil_scoped_release>())
      .def("debug_dump", &foundry::CUDAGraph::debug_dump, py::call_guard<py::gil_scoped_release>())
      .def("enable_debug_mode", &foundry::CUDAGraph::enable_debug_mode,
           py::call_guard<py::gil_scoped_release>())
      .def("debug_dump", &foundry::CUDAGraph::debug_dump, py::arg("debug_path"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_cuda_graph",
          [](::foundry::CUDAGraph& self) {
            cudaGraph_t graph = self.raw_cuda_graph();
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "save",
          [](::foundry::CUDAGraph& self, const std::string& json_path,
             std::optional<py::object> output_tensors_opt) {
            ::foundry::OutputTensors outputs = std::monostate{};
            ::foundry::OutputTensorType output_type = ::foundry::OutputTensorType::None;

            if (output_tensors_opt.has_value()) {
              py::object output_tensors = output_tensors_opt.value();
              if (py::isinstance<py::list>(output_tensors)) {
                py::list tensor_list = output_tensors.cast<py::list>();
                std::vector<at::Tensor> tensors;
                tensors.reserve(tensor_list.size());
                for (auto item : tensor_list) {
                  tensors.push_back(item.cast<torch::Tensor>());
                }
                outputs = std::move(tensors);
                output_type = ::foundry::OutputTensorType::List;
              } else if (py::isinstance<py::tuple>(output_tensors)) {
                py::tuple tensor_tuple = output_tensors.cast<py::tuple>();
                std::vector<at::Tensor> tensors;
                tensors.reserve(tensor_tuple.size());
                for (auto item : tensor_tuple) {
                  tensors.push_back(item.cast<torch::Tensor>());
                }
                outputs = std::move(tensors);
                output_type = ::foundry::OutputTensorType::Tuple;
              } else {
                try {
                  torch::Tensor t = output_tensors.cast<torch::Tensor>();
                  outputs = t;
                  output_type = ::foundry::OutputTensorType::Single;
                } catch (const py::cast_error&) {
                  throw std::runtime_error(
                      "output_tensors must be a Tensor, list[Tensor], or tuple[Tensor]");
                }
              }
            }

            py::gil_scoped_release release;
            self.save(json_path, outputs, output_type);
          },
          py::arg("json_path"), py::arg("output_tensors") = py::none())
      .def_static(
          "load",
          [](const std::string& json_path, std::optional<c10::MempoolId_t> pool_opt) {
            c10::MempoolId_t pool =
                pool_opt.has_value() ? pool_opt.value() : c10::MempoolId_t{0, 0};

            ::foundry::GraphLoadResult load_result;
            {
              py::gil_scoped_release release;
              load_result = ::foundry::CUDAGraph::load(json_path, pool);
            }

            py::object output_tensors = py::none();
            if (load_result.output_type == ::foundry::OutputTensorType::Single) {
              if (std::holds_alternative<at::Tensor>(load_result.outputs)) {
                output_tensors = py::cast(std::get<at::Tensor>(load_result.outputs));
              }
            } else if (load_result.output_type == ::foundry::OutputTensorType::List) {
              if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                py::list tensor_list;
                for (const auto& t : std::get<std::vector<at::Tensor>>(load_result.outputs)) {
                  tensor_list.append(py::cast(t));
                }
                output_tensors = tensor_list;
              }
            } else if (load_result.output_type == ::foundry::OutputTensorType::Tuple) {
              if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                const auto& tensors = std::get<std::vector<at::Tensor>>(load_result.outputs);
                py::tuple tensor_tuple(tensors.size());
                for (size_t i = 0; i < tensors.size(); ++i) {
                  tensor_tuple[i] = py::cast(tensors[i]);
                }
                output_tensors = tensor_tuple;
              }
            }

            return py::make_tuple(load_result.graph, output_tensors);
          },
          py::arg("json_path"), py::arg("pool") = py::none())
      .def_static(
          "start_exec_pool_prewarm",
          [](const std::vector<std::string>& json_paths) {
            py::gil_scoped_release release;
            ::foundry::CUDAGraph::start_exec_pool_prewarm(json_paths);
          },
          py::arg("json_paths"))
      .def_static(
          "start_graph_builds",
          [](const std::vector<std::string>& json_paths, std::optional<c10::MempoolId_t> pool_opt,
             int num_threads) {
            c10::MempoolId_t pool =
                pool_opt.has_value() ? pool_opt.value() : c10::MempoolId_t{0, 0};

            std::shared_ptr<::foundry::PendingGraphLoads> pending;
            {
              py::gil_scoped_release release;
              pending = ::foundry::CUDAGraph::start_graph_builds(json_paths, pool, num_threads);
            }
            return pending;
          },
          py::arg("json_paths"), py::arg("pool") = py::none(), py::arg("num_threads") = 4)
      .def_static(
          "finish_graph_loads",
          [](std::shared_ptr<::foundry::PendingGraphLoads> pending) {
            std::vector<::foundry::GraphLoadResult> load_results;
            {
              py::gil_scoped_release release;
              load_results = ::foundry::CUDAGraph::finish_graph_loads(std::move(pending));
            }

            py::list result_list;
            for (auto& load_result : load_results) {
              py::object output_tensors = py::none();
              if (load_result.output_type == ::foundry::OutputTensorType::Single) {
                if (std::holds_alternative<at::Tensor>(load_result.outputs)) {
                  output_tensors = py::cast(std::get<at::Tensor>(load_result.outputs));
                }
              } else if (load_result.output_type == ::foundry::OutputTensorType::List) {
                if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                  py::list tensor_list;
                  for (const auto& t : std::get<std::vector<at::Tensor>>(load_result.outputs)) {
                    tensor_list.append(py::cast(t));
                  }
                  output_tensors = tensor_list;
                }
              } else if (load_result.output_type == ::foundry::OutputTensorType::Tuple) {
                if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                  const auto& tensors = std::get<std::vector<at::Tensor>>(load_result.outputs);
                  py::tuple tensor_tuple(tensors.size());
                  for (size_t i = 0; i < tensors.size(); ++i) {
                    tensor_tuple[i] = py::cast(tensors[i]);
                  }
                  output_tensors = tensor_tuple;
                }
              }

              result_list.append(py::make_tuple(load_result.graph, output_tensors));
            }

            return result_list;
          },
          py::arg("pending"))
      .def_static(
          "finish_one_graph_load",
          [](std::shared_ptr<::foundry::PendingGraphLoads> pending, size_t index) {
            ::foundry::GraphLoadResult load_result;
            {
              py::gil_scoped_release release;
              load_result = ::foundry::CUDAGraph::finish_one_graph_load(std::move(pending), index);
            }
            py::object output_tensors = py::none();
            if (load_result.output_type == ::foundry::OutputTensorType::Single) {
              if (std::holds_alternative<at::Tensor>(load_result.outputs)) {
                output_tensors = py::cast(std::get<at::Tensor>(load_result.outputs));
              }
            } else if (load_result.output_type == ::foundry::OutputTensorType::List) {
              if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                py::list tensor_list;
                for (const auto& t : std::get<std::vector<at::Tensor>>(load_result.outputs)) {
                  tensor_list.append(py::cast(t));
                }
                output_tensors = tensor_list;
              }
            } else if (load_result.output_type == ::foundry::OutputTensorType::Tuple) {
              if (std::holds_alternative<std::vector<at::Tensor>>(load_result.outputs)) {
                const auto& tensors = std::get<std::vector<at::Tensor>>(load_result.outputs);
                py::tuple tensor_tuple(tensors.size());
                for (size_t i = 0; i < tensors.size(); ++i) {
                  tensor_tuple[i] = py::cast(tensors[i]);
                }
                output_tensors = tensor_tuple;
              }
            }
            return py::make_tuple(load_result.graph, output_tensors);
          },
          py::arg("pending"), py::arg("index"));
}
