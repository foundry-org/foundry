#ifndef HOOK_H
#define HOOK_H

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <boost/json.hpp>
#undef cuGetProcAddress
#undef cuCtxCreate
#undef cuMemAlloc
#undef cuMemAllocPitch
#undef cuMemFree

#ifdef __cplusplus
extern "C" {
#endif

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags);

#ifdef __cplusplus
}

namespace foundry {

void set_allocation_region(void* base, size_t size);
void stop_allocation_region();
void resume_allocation_region();
bool preallocate_region(size_t size);
void free_preallocated_region();
size_t get_current_alloc_offset();
void set_current_alloc_offset(size_t offset);
void set_pack_fatbins_on_exit(bool enabled);
void pack_fatbins_to_folder(const std::string& folder_path);
void set_skip_fatbin_processing(bool enabled);
void set_nvshmem_auto_init(bool enabled);
int init_nvshmem_for_loaded_modules();

void start_hook_record();
void end_hook_record();
void clear_hook_events();
boost::json::object save_hook_events_to_json();
void replay_hook_events_from_json(const boost::json::object& events_obj);

std::variant<CUfunction, CUkernel> query_function_handle(uint64_t binary_hash,
                                                         const std::string& function_name);
uint64_t query_binary_hash(std::variant<CUmodule, CUlibrary> handle);
void mark_binary_used(uint64_t binary_hash);
void load_cuda_modules_and_libraries(const std::string& archive_dir);

}  // namespace foundry
#endif

#endif
