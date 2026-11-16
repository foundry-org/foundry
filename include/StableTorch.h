#pragma once

// Everything foundry.ops needs from torch, through the LibTorch Stable ABI only
// (torch/csrc/stable, torch/headeronly, the aoti C shim; torch >= 2.14), plus
// foundry's own error macros over the CUDA runtime/driver. No ATen/c10 header and
// no at::/c10::/torch:: C++ symbol of libtorch: every libtorch call below is an
// extern "C" shim function (aoti_torch_*/torch_*).

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/c/shim.h>

namespace foundry {

// Plain integer stand-ins for c10::MempoolId_t / c10::CaptureId_t / c10::DeviceIndex.
// Foundry only passes them through (torch.cuda pool handles are (int, int) tuples).
using MempoolId_t = std::pair<unsigned long long, unsigned long long>;
using CaptureId_t = unsigned long long;
using DeviceIndex = int;

namespace detail {

template <typename... Args>
inline std::string str_cat(const Args&... args) {
  std::ostringstream ss;
  (void)(ss << ... << args);
  return ss.str();
}

inline std::string check_msg(const char* cond) {
  return std::string("Expected ") + cond + " to be true, but got false.";
}
template <typename A, typename... Args>
inline std::string check_msg(const char*, const A& a, const Args&... args) {
  return str_cat(a, args...);
}

[[noreturn]] inline void throw_runtime(const std::string& msg) {
  throw std::runtime_error(msg);
}

inline std::string cuda_driver_error(CUresult rc) {
  const char* name = nullptr;
  const char* text = nullptr;
  if (cuGetErrorName(rc, &name) != CUDA_SUCCESS || !name)
    name = "unknown CUresult";
  if (cuGetErrorString(rc, &text) != CUDA_SUCCESS || !text)
    text = "";
  return str_cat("CUDA driver error: ", name, " (", static_cast<int>(rc), "): ", text);
}

inline std::string cuda_runtime_error(cudaError_t err) {
  return str_cat("CUDA error: ", cudaGetErrorString(err), " (", cudaGetErrorName(err), " ",
                 static_cast<int>(err), ")");
}

}  // namespace detail

}  // namespace foundry

// Errors surface as std::runtime_error (RuntimeError in Python).
#define FOUNDRY_CHECK(cond, ...)                                                            \
  do {                                                                                      \
    if (__builtin_expect(!(cond), 0)) {                                                     \
      ::foundry::detail::throw_runtime(::foundry::detail::check_msg(#cond, ##__VA_ARGS__)); \
    }                                                                                       \
  } while (0)

#define FOUNDRY_INTERNAL_ASSERT(cond, ...)                                                        \
  do {                                                                                            \
    if (__builtin_expect(!(cond), 0)) {                                                           \
      ::foundry::detail::throw_runtime(::foundry::detail::str_cat(                                \
          "INTERNAL ASSERT FAILED at ", __FILE__, ":", __LINE__, ": " #cond " ", ##__VA_ARGS__)); \
    }                                                                                             \
  } while (0)

#define FOUNDRY_WARN(...)                                          \
  do {                                                             \
    std::fprintf(stderr, "[foundry] Warning: %s\n",                \
                 ::foundry::detail::str_cat(__VA_ARGS__).c_str()); \
  } while (0)

// CUDA runtime call: throws with the runtime's error string; clears the
// (non-sticky) error like c10's check does.
#define FOUNDRY_CUDA_CHECK(expr)                                                                  \
  do {                                                                                            \
    const cudaError_t foundry_err_ = (expr);                                                      \
    if (__builtin_expect(foundry_err_ != cudaSuccess, 0)) {                                       \
      (void)cudaGetLastError();                                                                   \
      ::foundry::detail::throw_runtime(::foundry::detail::str_cat(                                \
          ::foundry::detail::cuda_runtime_error(foundry_err_), " at ", __FILE__, ":", __LINE__)); \
    }                                                                                             \
  } while (0)

#define FOUNDRY_CUDA_CHECK_WARN(expr)                                    \
  do {                                                                   \
    const cudaError_t foundry_err_ = (expr);                             \
    if (foundry_err_ != cudaSuccess) {                                   \
      (void)cudaGetLastError();                                          \
      FOUNDRY_WARN(::foundry::detail::cuda_runtime_error(foundry_err_)); \
    }                                                                    \
  } while (0)

// CUDA driver call.
#define FOUNDRY_CU_CHECK(expr)                                                                  \
  do {                                                                                          \
    const CUresult foundry_rc_ = (expr);                                                        \
    if (__builtin_expect(foundry_rc_ != CUDA_SUCCESS, 0)) {                                     \
      ::foundry::detail::throw_runtime(::foundry::detail::str_cat(                              \
          ::foundry::detail::cuda_driver_error(foundry_rc_), " at ", __FILE__, ":", __LINE__)); \
    }                                                                                           \
  } while (0)

// Stable shim call (AOTITorchError): throws with torch's message.
#define FOUNDRY_SHIM_CHECK(expr)                                                \
  do {                                                                          \
    if (__builtin_expect((expr) != AOTI_TORCH_SUCCESS, 0)) {                    \
      const char* foundry_what_ = torch_exception_get_what_without_backtrace(); \
      ::foundry::detail::throw_runtime(::foundry::detail::str_cat(              \
          #expr " failed: ", foundry_what_ ? foundry_what_ : "(no message)"));  \
    }                                                                           \
  } while (0)

namespace foundry {

// Owning AtenTensorHandle (the stable ABI's tensor). Move-only.
class TensorHandle {
 public:
  TensorHandle() = default;
  explicit TensorHandle(AtenTensorHandle h) : h_(h) {}
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  TensorHandle(TensorHandle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
  TensorHandle& operator=(TensorHandle&& o) noexcept {
    if (this != &o) {
      reset();
      h_ = std::exchange(o.h_, nullptr);
    }
    return *this;
  }
  ~TensorHandle() { reset(); }
  void reset() {
    if (h_) {
      (void)aoti_torch_delete_tensor_object(h_);
      h_ = nullptr;
    }
  }
  AtenTensorHandle get() const { return h_; }
  AtenTensorHandle release() { return std::exchange(h_, nullptr); }
  explicit operator bool() const { return h_ != nullptr; }
  void* data_ptr() const {
    void* p = nullptr;
    FOUNDRY_SHIM_CHECK(aoti_torch_get_data_ptr(h_, &p));
    return p;
  }

 private:
  AtenTensorHandle h_ = nullptr;
};

// A tensor crossing the binding: the handle plus what the stable ABI cannot read
// back (requires_grad has no getter in 2.14; the binding reads it from Python).
struct Tensor {
  std::shared_ptr<TensorHandle> handle;
  bool requires_grad = false;
};

// Current device (torch's view, = c10::cuda::current_device()).
inline DeviceIndex current_device() {
  int32_t dev = -1;
  FOUNDRY_SHIM_CHECK(aoti_torch_get_current_device_index(&dev));
  return dev;
}

// torch's current CUDA stream of `device` (= at::cuda::getCurrentCUDAStream(device)).
inline cudaStream_t current_stream(DeviceIndex device) {
  void* s = nullptr;
  FOUNDRY_SHIM_CHECK(aoti_torch_get_current_cuda_stream(device, &s));
  return static_cast<cudaStream_t>(s);
}

// Runtime-API device guard (what c10::OptionalDeviceGuard does for CUDA):
// switches only when needed, restores on exit; a negative index is a no-op.
class DeviceGuard {
 public:
  explicit DeviceGuard(DeviceIndex device) {
    if (device < 0)
      return;
    FOUNDRY_CUDA_CHECK(cudaGetDevice(&prev_));
    if (prev_ != device) {
      FOUNDRY_CUDA_CHECK(cudaSetDevice(device));
      switched_ = true;
    }
  }
  ~DeviceGuard() {
    if (switched_)
      (void)cudaSetDevice(prev_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = -1;
  bool switched_ = false;
};

}  // namespace foundry
