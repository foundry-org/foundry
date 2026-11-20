#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"
#include "metadata.h"
#include "hook.h"

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace {

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

// Tensors cross the binding as PyObject* through the stable 2.14 shims
// (torch_tensor_from_pyobject / torch_tensor_to_pyobject); requires_grad has no
// stable getter, so it is read from the Python object here.
::foundry::Tensor tensor_from_py(py::handle obj) {
  AtenTensorHandle h = nullptr;
  if (torch_tensor_from_pyobject(obj.ptr(), &h) != AOTI_TORCH_SUCCESS) {
    const char* what = torch_exception_get_what_without_backtrace();
    throw std::runtime_error(
        std::string("output_tensors must be a Tensor, list[Tensor], or tuple[Tensor]: ") +
        (what ? what : ""));
  }
  ::foundry::Tensor t;
  t.handle = std::make_shared<::foundry::TensorHandle>(h);
  t.requires_grad = obj.attr("requires_grad").cast<bool>();
  return t;
}

py::object tensor_to_py(const ::foundry::Tensor& t) {
  void* raw = nullptr;
  FOUNDRY_SHIM_CHECK(torch_tensor_to_pyobject(t.handle->get(), nullptr, &raw));
  return py::reinterpret_steal<py::object>(static_cast<PyObject*>(raw));
}

py::tuple load_result_to_py(const ::foundry::GraphLoadResult& load_result) {
  py::object output_tensors = py::none();
  if (load_result.output_type == ::foundry::OutputTensorType::Single) {
    if (std::holds_alternative<::foundry::Tensor>(load_result.outputs)) {
      output_tensors = tensor_to_py(std::get<::foundry::Tensor>(load_result.outputs));
    }
  } else if (load_result.output_type == ::foundry::OutputTensorType::List ||
             load_result.output_type == ::foundry::OutputTensorType::Tuple) {
    if (std::holds_alternative<std::vector<::foundry::Tensor>>(load_result.outputs)) {
      const auto& tensors = std::get<std::vector<::foundry::Tensor>>(load_result.outputs);
      py::list tensor_list;
      for (const auto& t : tensors) {
        tensor_list.append(tensor_to_py(t));
      }
      if (load_result.output_type == ::foundry::OutputTensorType::List) {
        output_tensors = tensor_list;
      } else {
        output_tensors = py::tuple(tensor_list);
      }
    }
  }
  return py::make_tuple(load_result.graph, output_tensors);
}
}  // namespace

PYBIND11_MODULE(ops, m) {
  // SAVE: hook record window around torch's own capture (foundry/graph.py)
  m.def("native_record_begin", &::foundry::native_record_begin);
  m.def("native_record_abort", &::foundry::native_record_abort);
  m.def(
      "scan_graph_words",
      [](uintptr_t raw_graph, const std::vector<uint64_t>& words) {
        std::unordered_set<uint64_t> cand(words.begin(), words.end());
        auto found = ::foundry::scan_raw_graph_words(reinterpret_cast<CUgraph>(raw_graph), cand);
        std::vector<uint64_t> out(found.begin(), found.end());
        std::sort(out.begin(), out.end());
        return out;
      },
      py::arg("raw_graph"), py::arg("words"));
  m.def(
      "recorded_allocation_contains",
      [](uint64_t addr, size_t len) { return ::foundry::recorded_allocation_contains(addr, len); },
      py::arg("addr"), py::arg("len"),
      "SAVE/LOAD: [addr, addr+len) lies in a live driver allocation Foundry recorded / replayed");

  m.def(
      "get_graph_ptr",
      [](::foundry::CUDAGraph& graph) { return reinterpret_cast<uintptr_t>(&graph); },
      py::arg("graph"), "Get pointer address of CUDAGraph object for hook integration");

  m.def(
      "set_allocation_region",
      [](uintptr_t base, size_t size) {
        return ::foundry::set_allocation_region(reinterpret_cast<void*>(base), size);
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
      "preallocate_cublas_workspaces",
      []() {
        // Running torch's major.minor ("2.14.1+cu130", "2.16.0.dev20261102", "2.16.0a0+git").
        const auto version = py::module_::import("torch").attr("__version__").cast<std::string>();
        int major = 0, minor = 0;
        FOUNDRY_CHECK(std::sscanf(version.c_str(), "%d.%d", &major, &minor) == 2,
                      "preallocate_cublas_workspaces: cannot parse torch version ", version);
        ::foundry::preallocate_cublas_workspaces(major > 2 || (major == 2 && minor >= 15));
      },
      "Pre-allocate the cuBLAS handle and workspace (= the unified cuBLASLt workspace) for "
      "the current stream");

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
          "adopt_native_capture",
          [](::foundry::CUDAGraph& self, uintptr_t raw_graph, ::foundry::MempoolId_t pool,
             const std::vector<::foundry::NativeGenerator>& generators,
             const std::vector<uint64_t>& rng_candidates) {
            self.adopt_native_capture(raw_graph, pool, generators, rng_candidates);
          },
          py::arg("raw_graph"), py::arg("pool"), py::arg("generators"),
          py::arg("rng_candidates") = std::vector<uint64_t>{},
          py::call_guard<py::gil_scoped_release>())
      .def("replay",
           [](::foundry::CUDAGraph& self) {
             // A loaded graph with RNG takes (seed, offset) per replay from torch's default
             // generator of its device, like torch's replay_prologue: philox_state(inc)
             // outside capture returns the current seed and offset and advances the offset
             // by inc under the generator's lock. Graphs without RNG skip this.
             const size_t n = self.native_rng_count();
             for (size_t i = 0; i < n; ++i) {
               const uint64_t inc = self.native_rng_increment(i);
               if (inc == 0)
                 continue;
               static PyObject* generators =  // torch.cuda.default_generators (a tuple), kept
                   py::object(py::module_::import("torch.cuda").attr("default_generators"))
                       .release()
                       .ptr();
               py::tuple st = py::reinterpret_borrow<py::tuple>(generators)[self.native_rng_device()]
                                  .attr("philox_state")(inc);
               // int64 tensors holding the uint64 seed/offset bits (torch fills its own
               // buffers with static_cast<int64_t> of them too).
               self.set_native_rng_philox(
                   i, static_cast<uint64_t>(st[0].attr("item")().cast<int64_t>()),
                   static_cast<uint64_t>(st[1].attr("item")().cast<int64_t>()));
             }
             py::gil_scoped_release release;
             self.replay();
           })
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
                FOUNDRY_CHECK(py::hasattr(value, "cuda_event"),
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

            if (output_tensors_opt.has_value() && !output_tensors_opt->is_none()) {
              py::object output_tensors = output_tensors_opt.value();
              if (py::isinstance<py::list>(output_tensors) ||
                  py::isinstance<py::tuple>(output_tensors)) {
                std::vector<::foundry::Tensor> tensors;
                for (auto item : output_tensors) {
                  tensors.push_back(tensor_from_py(item));
                }
                outputs = std::move(tensors);
                output_type = py::isinstance<py::list>(output_tensors)
                                  ? ::foundry::OutputTensorType::List
                                  : ::foundry::OutputTensorType::Tuple;
              } else {
                outputs = tensor_from_py(output_tensors);
                output_type = ::foundry::OutputTensorType::Single;
              }
            }

            py::gil_scoped_release release;
            self.save(json_path, outputs, output_type);
          },
          py::arg("json_path"), py::arg("output_tensors") = py::none())
      .def_static(
          "load",
          [](const std::string& json_path, std::optional<::foundry::MempoolId_t> pool_opt) {
            ::foundry::MempoolId_t pool = pool_opt.value_or(::foundry::MempoolId_t{0, 0});
            ::foundry::GraphLoadResult load_result;
            {
              py::gil_scoped_release release;
              load_result = ::foundry::CUDAGraph::load(json_path, pool);
            }
            return load_result_to_py(load_result);
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
          [](const std::vector<std::string>& json_paths,
             std::optional<::foundry::MempoolId_t> pool_opt, int num_threads) {
            ::foundry::MempoolId_t pool = pool_opt.value_or(::foundry::MempoolId_t{0, 0});
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
              result_list.append(load_result_to_py(load_result));
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
            return load_result_to_py(load_result);
          },
          py::arg("pending"), py::arg("index"));
}
