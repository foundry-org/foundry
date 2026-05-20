#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstdlib>
#endif

// Enable debug logging to trace memory allocations (uncomment for debugging)
// #define HOOK_DEBUG

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <atomic>
#include <mutex>
#include <variant>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <string_view>
#include <unistd.h>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/crc.hpp>
#include <boost/format.hpp>
#include <boost/json.hpp>
#include "hook.h"

namespace fs = boost::filesystem;

typedef void* (*fp_dlsym)(void*, const char*);
static fp_dlsym real_dlsym = nullptr;

typedef struct {
    void* fn_ptr;
    const char* name;
} cuda_driver_entry_t;

enum CudaDriverAPIIndex {
    CUDA_ENTRY_cuModuleLoadData = 0,
    CUDA_ENTRY_cuModuleLoadDataEx,
    CUDA_ENTRY_cuModuleLoadFatBinary,
    CUDA_ENTRY_cuLibraryLoadData,
    CUDA_ENTRY_cuModuleLoad,
    CUDA_ENTRY_cuLibraryLoadFromFile,
    CUDA_ENTRY_cuGetProcAddress,
    CUDA_ENTRY_cuGetProcAddress_v2,
    CUDA_ENTRY_cuCtxCreate,
    CUDA_ENTRY_cuCtxCreate_v2,
    CUDA_ENTRY_cuCtxCreate_v3,
    CUDA_ENTRY_cuCtxCreate_v4,
    CUDA_ENTRY_cuMemAlloc_v2,
    CUDA_ENTRY_cuMemAllocPitch_v2,
    CUDA_ENTRY_cuMemFree_v2,
    CUDA_ENTRY_cuMemCreate,
    CUDA_ENTRY_cuMemAddressReserve,
    CUDA_ENTRY_cuMemMap,
    CUDA_ENTRY_cuMemSetAccess,
    CUDA_ENTRY_cuMemUnmap,
    CUDA_ENTRY_cuMemRelease,
    CUDA_ENTRY_cuMemAddressFree,
    CUDA_ENTRY_cuMemGetAllocationGranularity,
    CUDA_ENTRY_cuCtxGetDevice,
    CUDA_ENTRY_cuLibraryGetKernelCount,
    CUDA_ENTRY_cuLibraryEnumerateKernels,
    CUDA_ENTRY_cuKernelGetName,
    CUDA_ENTRY_cuKernelGetLibrary,
    CUDA_ENTRY_cuModuleGetFunctionCount,
    CUDA_ENTRY_cuModuleEnumerateFunctions,
    CUDA_ENTRY_cuFuncGetName,
    CUDA_ENTRY_cuFuncGetModule,
    CUDA_ENTRY_cuMemExportToShareableHandle,
    CUDA_ENTRY_cuMemImportFromShareableHandle,
    CUDA_ENTRY_cuIpcGetMemHandle,
    CUDA_ENTRY_cuIpcOpenMemHandle,
    CUDA_ENTRY_cuIpcCloseMemHandle,
    CUDA_ENTRY_cuLinkCreate,
    CUDA_ENTRY_cuLinkAddData,
    CUDA_ENTRY_cuLinkComplete,
    CUDA_ENTRY_cuLinkDestroy,
    CUDA_ENTRY_cuModuleGetGlobal,
    CUDA_ENTRY_cuLibraryGetGlobal,
    CUDA_ENTRY_END
};

static cuda_driver_entry_t cuda_driver_entry_table[] = {
    {nullptr, "cuModuleLoadData"},
    {nullptr, "cuModuleLoadDataEx"},
    {nullptr, "cuModuleLoadFatBinary"},
    {nullptr, "cuLibraryLoadData"},
    {nullptr, "cuModuleLoad"},
    {nullptr, "cuLibraryLoadFromFile"},
    {nullptr, "cuGetProcAddress"},
    {nullptr, "cuGetProcAddress_v2"},
    {nullptr, "cuCtxCreate"},
    {nullptr, "cuCtxCreate_v2"},
    {nullptr, "cuCtxCreate_v3"},
    {nullptr, "cuCtxCreate_v4"},
    {nullptr, "cuMemAlloc_v2"},
    {nullptr, "cuMemAllocPitch_v2"},
    {nullptr, "cuMemFree_v2"},
    {nullptr, "cuMemCreate"},
    {nullptr, "cuMemAddressReserve"},
    {nullptr, "cuMemMap"},
    {nullptr, "cuMemSetAccess"},
    {nullptr, "cuMemUnmap"},
    {nullptr, "cuMemRelease"},
    {nullptr, "cuMemAddressFree"},
    {nullptr, "cuMemGetAllocationGranularity"},
    {nullptr, "cuCtxGetDevice"},
    {nullptr, "cuLibraryGetKernelCount"},
    {nullptr, "cuLibraryEnumerateKernels"},
    {nullptr, "cuKernelGetName"},
    {nullptr, "cuKernelGetLibrary"},
    {nullptr, "cuModuleGetFunctionCount"},
    {nullptr, "cuModuleEnumerateFunctions"},
    {nullptr, "cuFuncGetName"},
    {nullptr, "cuFuncGetModule"},
    {nullptr, "cuMemExportToShareableHandle"},
    {nullptr, "cuMemImportFromShareableHandle"},
    {nullptr, "cuIpcGetMemHandle"},
    {nullptr, "cuIpcOpenMemHandle"},
    {nullptr, "cuIpcCloseMemHandle"},
    {nullptr, "cuLinkCreate"},
    {nullptr, "cuLinkAddData"},
    {nullptr, "cuLinkComplete"},
    {nullptr, "cuLinkDestroy"},
    {nullptr, "cuModuleGetGlobal_v2"},
    {nullptr, "cuLibraryGetGlobal"},
    {nullptr, nullptr}
};

#define CUDA_DRIVER_CALL(table, idx) (table[idx].fn_ptr)

inline constexpr uint32_t FATBINC_MAGIC = 0x466243B1;
inline constexpr uint16_t FATBINC_VERSION = 1;
inline constexpr uint16_t FATBINC_LINK_VERSION = 2;
inline constexpr uint32_t FATBIN_MAGIC = 0xBA55ED50;
inline constexpr uint32_t ELF_MAGIC = 0x464C457F;

typedef struct {
    int magic;
    int version;
    const unsigned long long* data;
    void *filename_or_fatbins;
} __fatBinC_Wrapper_t;

struct __attribute__((__packed__)) fat_elf_header {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size;
};

enum class BinaryFormat {
    WRAPPER,
    FATBIN,
    CUBIN_ELF,
    PTX,
    UNKNOWN
};

// Bit flags for binary metadata (stored as single integer in metadata file)
enum BinaryFlags : uint32_t {
    BINARY_FLAG_NONE              = 0,
    BINARY_FLAG_NEEDS_DEVICE_LINK = 1 << 0,  // Binary needs device linking at LOAD time (segments stored separately)
    BINARY_FLAG_REQUIRES_NVSHMEM  = 1 << 1,  // Module requires NVSHMEM initialization
};

using ModuleHandles = std::tuple<CUmodule, std::unordered_map<std::string, CUfunction>>;
using LibraryHandles = std::tuple<CUlibrary, std::unordered_map<std::string, CUkernel>>;

struct BinaryMetadata {
    std::vector<uint8_t> binary_data;
    std::string base_func_name;
    std::string filename;
    std::vector<CUjit_option> jit_options;
    std::vector<void*> jit_option_values;
    std::vector<CUlibraryOption> library_options;
    std::vector<void*> library_option_values;
    std::vector<std::string> entrypoint_names;
    bool used = false;
    std::vector<std::vector<uint8_t>> linked_fatbin_segments;
    uint32_t binary_flags = BINARY_FLAG_NONE;  // Bit flags for binary properties
};

static std::atomic<int> dumped_binary_counter{0};
static std::once_flag load_once_flag;
static std::atomic<bool> binary_loaded{false};
static std::atomic<bool> pack_fatbins_on_exit_enabled{false};
static std::atomic<bool> skip_fatbin_processing{false};
static boost::unordered::concurrent_flat_map<uint64_t, std::variant<ModuleHandles, LibraryHandles>> binary_hash_to_handles;
static boost::unordered::concurrent_flat_map<uint64_t, BinaryMetadata> binary_hash_to_metadata;

struct VariantHash {
    std::size_t operator()(const std::variant<CUmodule, CUlibrary>& v) const {
        return std::visit([](auto&& arg) -> std::size_t {
            return std::hash<std::decay_t<decltype(arg)>>{}(arg);
        }, v);
    }
};

static boost::unordered::concurrent_flat_map<std::variant<CUmodule, CUlibrary>, uint64_t, VariantHash> module_or_library_handle_to_hash;

constexpr size_t kAllocAlignment = 2 * 1024 * 1024;
constexpr uintptr_t kAllocDefaultRegionBase = 0x500000000000;
constexpr size_t kAllocDefaultRegionSize = 2ULL << 30;

struct AllocRegion {
    void* base;
    size_t size;
};

struct AllocMetadata {
    CUdeviceptr ptr;
    size_t size;
    CUmemGenericAllocationHandle handle;
    CUdeviceptr region_base;
    bool from_preallocation;  // If true, this allocation is carved from preallocated memory
};

struct ThreadLocalStorage {
    AllocRegion region;
    size_t current_alloc_base_addr;
    size_t current_vmm_reserve_addr;
    bool enabled;
    bool region_initialized;

    // Preallocation state for fast allocation path
    CUmemGenericAllocationHandle preallocated_handle;
    size_t preallocated_start_addr;
    size_t preallocated_end_addr;
    bool has_preallocation;

    // Cached values to avoid repeated driver calls
    CUdevice cached_device;
    size_t cached_granularity;
    bool device_cached;

    ThreadLocalStorage() : region{nullptr, 0}, current_alloc_base_addr(0), current_vmm_reserve_addr(0),
                           enabled(false), region_initialized(false),
                           preallocated_handle(0), preallocated_start_addr(0), preallocated_end_addr(0),
                           has_preallocation(false),
                           cached_device(0), cached_granularity(0), device_cached(false) {}
};

static thread_local ThreadLocalStorage tls_storage;
static std::once_flag default_allocation_region_flag;
static boost::unordered::concurrent_flat_map<CUdeviceptr, AllocMetadata> global_alloc_metadata;
static boost::unordered::concurrent_flat_map<CUdeviceptr, size_t> global_carved_reserve_metadata;

struct HookAllocationEvent {
    enum class Type { Alloc, Free, Reserve };
    Type type;
    size_t size;
    CUdeviceptr ptr;
    size_t alignment;  // Used for Reserve events
};

static std::atomic<bool> hook_recording_enabled{false};
static std::vector<HookAllocationEvent> hook_alloc_events;
static std::mutex hook_events_mutex;
static CUdeviceptr hook_recording_start_base_addr{0};

static inline size_t align_to(size_t addr, size_t alignment) {
    return (addr + alignment - 1) & ~(alignment - 1);
}

// Cache allocation granularity per device to avoid repeated driver calls
static std::unordered_map<CUdevice, size_t> cached_granularity;
static std::mutex granularity_cache_mutex;

static size_t get_allocation_granularity(CUdevice device) {
    // Fast path: check cache without lock first (read is safe for unordered_map if no concurrent writes)
    {
        std::lock_guard<std::mutex> lock(granularity_cache_mutex);
        auto it = cached_granularity.find(device);
        if (it != cached_granularity.end()) {
            return it->second;
        }
    }

    // Slow path: query driver and cache result
    typedef CUresult (*cuMemGetAllocationGranularity_t)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    auto real_func = (cuMemGetAllocationGranularity_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemGetAllocationGranularity);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    size_t granularity = 0;
    CUresult result = real_func(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] FATAL ERROR: cuMemGetAllocationGranularity failed with error %d\n", result);
        abort();
    }

    {
        std::lock_guard<std::mutex> lock(granularity_cache_mutex);
        cached_granularity[device] = granularity;
    }

    return granularity;
}

static uint64_t compute_hash(const std::vector<uint8_t>& data) {
    boost::crc_optimal<64, 0x42F0E1EBA9EA3693ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, true, true> crc;
    crc.process_bytes(data.data(), data.size());
    return crc.checksum();
}

static BinaryFormat detect_binary_format(const void *data_ptr) {
    if (!data_ptr) {
        return BinaryFormat::UNKNOWN;
    }

    const auto* magic_ptr = static_cast<const uint32_t*>(data_ptr);
    const uint32_t magic = *magic_ptr;

    if (magic == FATBINC_MAGIC) {
        return BinaryFormat::WRAPPER;
    }

    if (magic == FATBIN_MAGIC) {
        return BinaryFormat::FATBIN;
    }

    if (magic == ELF_MAGIC) {
        return BinaryFormat::CUBIN_ELF;
    }

    return BinaryFormat::PTX;
}

static size_t compute_fatbin_size(const uint8_t* fatbin_data) {
    size_t size = 0;
    size_t offset = 0;

    while (true) {
        const auto* header = reinterpret_cast<const fat_elf_header*>(fatbin_data + offset);

        if (offset > 0 && header->magic != FATBIN_MAGIC) {
            break;
        }

        const size_t fatbin_size = header->header_size + header->size;
        size = offset + fatbin_size;
        offset += fatbin_size;

        if (header->magic != FATBIN_MAGIC) {
            break;
        }
    }

    return size == 0 ? sizeof(fat_elf_header) : size;
}

static size_t compute_elf_size(const uint8_t* elf_data) {
    struct elf64_hdr {
        uint8_t e_ident[16];
        uint16_t e_type;
        uint16_t e_machine;
        uint32_t e_version;
        uint64_t e_entry;
        uint64_t e_phoff;
        uint64_t e_shoff;
        uint32_t e_flags;
        uint16_t e_ehsize;
        uint16_t e_phentsize;
        uint16_t e_phnum;
        uint16_t e_shentsize;
        uint16_t e_shnum;
        uint16_t e_shstrndx;
    };

    const auto* elf_hdr = reinterpret_cast<const elf64_hdr*>(elf_data);

    const uint64_t section_end = elf_hdr->e_shoff + (elf_hdr->e_shentsize * elf_hdr->e_shnum);
    const uint64_t program_end = elf_hdr->e_phoff + (elf_hdr->e_phentsize * elf_hdr->e_phnum);

    return std::max(section_end, program_end);
}

static std::string_view jit_option_to_string(CUjit_option opt) {
    using namespace std::string_view_literals;
    constexpr std::string_view options[] = {
        "CU_JIT_MAX_REGISTERS"sv,
        "CU_JIT_THREADS_PER_BLOCK"sv,
        "CU_JIT_WALL_TIME"sv,
        "CU_JIT_INFO_LOG_BUFFER"sv,
        "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES"sv,
        "CU_JIT_ERROR_LOG_BUFFER"sv,
        "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES"sv,
        "CU_JIT_OPTIMIZATION_LEVEL"sv,
        "CU_JIT_TARGET_FROM_CUCONTEXT"sv,
        "CU_JIT_TARGET"sv,
        "CU_JIT_FALLBACK_STRATEGY"sv,
        "CU_JIT_GENERATE_DEBUG_INFO"sv,
        "CU_JIT_LOG_VERBOSE"sv,
        "CU_JIT_GENERATE_LINE_INFO"sv,
        "CU_JIT_CACHE_MODE"sv,
        "CU_JIT_NEW_SM3X_OPT"sv,
        "CU_JIT_FAST_COMPILE"sv,
        "CU_JIT_GLOBAL_SYMBOL_NAMES"sv,
        "CU_JIT_GLOBAL_SYMBOL_ADDRESSES"sv,
        "CU_JIT_GLOBAL_SYMBOL_COUNT"sv,
        "CU_JIT_LTO"sv,
        "CU_JIT_FTZ"sv,
        "CU_JIT_PREC_DIV"sv,
        "CU_JIT_PREC_SQRT"sv,
        "CU_JIT_FMA"sv,
        "CU_JIT_REFERENCED_KERNEL_NAMES"sv,
        "CU_JIT_REFERENCED_KERNEL_COUNT"sv,
        "CU_JIT_REFERENCED_VARIABLE_NAMES"sv,
        "CU_JIT_REFERENCED_VARIABLE_COUNT"sv,
        "CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES"sv,
        "CU_JIT_POSITION_INDEPENDENT_CODE"sv,
        "CU_JIT_MIN_CTA_PER_SM"sv,
        "CU_JIT_MAX_THREADS_PER_BLOCK"sv,
        "CU_JIT_OVERRIDE_DIRECTIVE_VALUES"sv,
        #if (defined(CUDA_VERSION) && CUDA_VERSION >= 12900)  
        "CU_JIT_SPLIT_COMPILE"sv
        #endif
    };
    if (opt >= 0 && opt < 35) {
        return options[opt];
    }
    fprintf(stderr, "[HOOK] FATAL ERROR: Unknown JIT option: %d\n", static_cast<int>(opt));
    abort();
}

static std::string_view library_option_to_string(CUlibraryOption opt) {
    using namespace std::string_view_literals;
    switch(opt) {
        case 0: return "CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE"sv;
        case 1: return "CU_LIBRARY_BINARY_IS_PRESERVED"sv;
        default:
            fprintf(stderr, "[HOOK] FATAL ERROR: Unknown library option: %d\n", static_cast<int>(opt));
            abort();
    }
}

static bool is_jit_option_ignored(CUjit_option opt) {
    return opt == CU_JIT_INFO_LOG_BUFFER ||
           opt == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES ||
           opt == CU_JIT_ERROR_LOG_BUFFER ||
           opt == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES ||
           opt == CU_JIT_GLOBAL_SYMBOL_NAMES ||
           opt == CU_JIT_GLOBAL_SYMBOL_ADDRESSES ||
           opt == CU_JIT_GLOBAL_SYMBOL_COUNT ||
           opt == CU_JIT_REFERENCED_KERNEL_NAMES ||
           opt == CU_JIT_REFERENCED_KERNEL_COUNT ||
           opt == CU_JIT_REFERENCED_VARIABLE_NAMES ||
           opt == CU_JIT_REFERENCED_VARIABLE_COUNT;
}

static bool jit_option_needs_value(CUjit_option opt) {
    return opt != CU_JIT_TARGET_FROM_CUCONTEXT &&
           opt != CU_JIT_NEW_SM3X_OPT &&
           opt != CU_JIT_FAST_COMPILE;
}

static bool library_option_ignored(CUlibraryOption opt) {
    return opt == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE;
}

static bool library_option_needs_value(CUlibraryOption opt) {
    return opt != CU_LIBRARY_BINARY_IS_PRESERVED;
}

static std::string jit_option_value_to_string(CUjit_option opt, void* value, bool values_array_is_null) {
    if (is_jit_option_ignored(opt)) {
        return "null";
    }

    if (values_array_is_null) {
        return "null";
    }

    switch (opt) {
        case CU_JIT_MAX_REGISTERS:
        case CU_JIT_THREADS_PER_BLOCK:
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_OPTIMIZATION_LEVEL:
        case CU_JIT_TARGET:
        case CU_JIT_FALLBACK_STRATEGY:
        case CU_JIT_CACHE_MODE:
        case CU_JIT_GLOBAL_SYMBOL_COUNT:
        case CU_JIT_TARGET_FROM_CUCONTEXT:
        case CU_JIT_REFERENCED_KERNEL_COUNT:
        case CU_JIT_REFERENCED_VARIABLE_COUNT:
        case CU_JIT_MIN_CTA_PER_SM:
        #if (defined(CUDA_VERSION) && CUDA_VERSION >= 12900)
        case CU_JIT_SPLIT_COMPILE:
        #endif
        case CU_JIT_NEW_SM3X_OPT:
        case CU_JIT_FAST_COMPILE:
            return std::to_string(reinterpret_cast<uintptr_t>(value));

        case CU_JIT_GENERATE_DEBUG_INFO:
        case CU_JIT_LOG_VERBOSE:
        case CU_JIT_GENERATE_LINE_INFO:
        case CU_JIT_LTO:
        case CU_JIT_FTZ:
        case CU_JIT_PREC_DIV:
        case CU_JIT_PREC_SQRT:
        case CU_JIT_FMA:
        case CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES:
        case CU_JIT_POSITION_INDEPENDENT_CODE:
        case CU_JIT_MAX_THREADS_PER_BLOCK:
        case CU_JIT_OVERRIDE_DIRECTIVE_VALUES: {
            int val = static_cast<int>(reinterpret_cast<intptr_t>(value));
            return std::to_string(val);
        }

        case CU_JIT_WALL_TIME: {
            float val;
            std::memcpy(&val, &value, sizeof(float));
            return std::to_string(val);
        }

        default:
            return "null";
    }
}

static std::string library_option_value_to_string(CUlibraryOption opt, void* value, bool values_array_is_null) {
    if (library_option_ignored(opt)) {
        return "null";
    }

    if (values_array_is_null) {
        return "null";
    }

    switch (opt) {
        case CU_LIBRARY_BINARY_IS_PRESERVED:
            return std::to_string(reinterpret_cast<uintptr_t>(value));

        default:
            fprintf(stderr, "[HOOK] FATAL ERROR: Unhandled library option in library_option_value_to_string: %d\n", static_cast<int>(opt));
            abort();
    }
}

static void dump_fatbin_and_info(const void *data_ptr, std::string_view func_name,
                                 unsigned int numJitOptions, CUjit_option *jitOptions, void **jitOptionsValues,
                                 unsigned int numLibraryOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues,
                                 int idx, uint64_t* out_hash) {
    const BinaryFormat format = detect_binary_format(data_ptr);

    const uint8_t* binary_data = nullptr;
    size_t total_size = 0;

    // For FATBINC_LINK_VERSION, we may need to concatenate multiple fatbins
    std::vector<uint8_t> concatenated_fatbin;
    // Store individual segments for device linking during load
    std::vector<std::vector<uint8_t>> linked_fatbin_segments;

    if (format == BinaryFormat::WRAPPER) {
        const auto* wrapper = static_cast<const __fatBinC_Wrapper_t*>(data_ptr);

        if (wrapper->version != FATBINC_VERSION && wrapper->version != FATBINC_LINK_VERSION) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Invalid fatbin wrapper version %d\n", wrapper->version);
            abort();
        }

        if (wrapper->version == FATBINC_VERSION) {
            binary_data = reinterpret_cast<const uint8_t*>(wrapper->data);
            total_size = compute_fatbin_size(binary_data);
        } else if (wrapper->version == FATBINC_LINK_VERSION) {
            // For linked fatbins, iterate through all entries in the array
            auto** fatbin_array = static_cast<void**>(wrapper->filename_or_fatbins);
            if (!fatbin_array || !fatbin_array[0]) {
                fprintf(stderr, "[HOOK] FATAL ERROR: Null fatbin array in link wrapper\n");
                abort();
            }

            // Count and concatenate all fatbin entries
            size_t num_fatbins = 0;
            for (void** ptr = fatbin_array; *ptr != nullptr; ptr++) {
                num_fatbins++;
            }
#ifdef HOOK_DEBUG
            fprintf(stderr, "[HOOK] DEBUG: FATBINC_LINK_VERSION wrapper has %zu fatbin entries\n", num_fatbins);
#endif
            if (num_fatbins == 1) {
                // Single fatbin, use directly
                binary_data = static_cast<const uint8_t*>(fatbin_array[0]);
                total_size = compute_fatbin_size(binary_data);
            } else {
                // Multiple fatbins - store each separately for device linking during load
                // Also concatenate for hash computation
                for (size_t i = 0; i < num_fatbins; i++) {
                    const uint8_t* fb_data = static_cast<const uint8_t*>(fatbin_array[i]);
                    size_t fb_size = compute_fatbin_size(fb_data);
#ifdef HOOK_DEBUG
                    fprintf(stderr, "[HOOK] DEBUG:   fatbin[%zu] size: %zu bytes\n", i, fb_size);
#endif
                    // Store each segment separately
                    linked_fatbin_segments.emplace_back(fb_data, fb_data + fb_size);
                    // Also concatenate for hash
                    concatenated_fatbin.insert(concatenated_fatbin.end(), fb_data, fb_data + fb_size);
                }
                // Point to the concatenated buffer for hash computation
                binary_data = concatenated_fatbin.data();
                total_size = concatenated_fatbin.size();
#ifdef HOOK_DEBUG
                fprintf(stderr, "[HOOK] DEBUG: Total size for %zu linked fatbins: %zu bytes (will use device linker)\n",
                        num_fatbins, total_size);
#endif
            }
        }

        if (!binary_data) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Null binary data in wrapper\n");
            abort();
        }

    } else if (format == BinaryFormat::FATBIN) {
        binary_data = static_cast<const uint8_t*>(data_ptr);
        total_size = compute_fatbin_size(binary_data);

    } else if (format == BinaryFormat::CUBIN_ELF) {
        binary_data = static_cast<const uint8_t*>(data_ptr);
        total_size = compute_elf_size(binary_data);

    } else if (format == BinaryFormat::PTX) {
        binary_data = static_cast<const uint8_t*>(data_ptr);
        total_size = strlen(reinterpret_cast<const char*>(data_ptr));

    } else {
        fprintf(stderr, "[HOOK] FATAL ERROR: Unknown binary format\n");
        abort();
    }

    if (!binary_data) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Null binary data\n");
        abort();
    }

    if (total_size == 0) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Zero size binary\n");
        abort();
    }

    if (numJitOptions > 0 && !jitOptionsValues) {
        for (unsigned int i = 0; i < numJitOptions; i++) {
            if (jit_option_needs_value(jitOptions[i]) && !is_jit_option_ignored(jitOptions[i])) {
                fprintf(stderr, "[HOOK] FATAL ERROR: JIT option %s requires a value but jitOptionsValues is nullptr\n",
                        std::string(jit_option_to_string(jitOptions[i])).c_str());
                abort();
            }
        }
    }

    if (numLibraryOptions > 0 && !libraryOptionValues) {
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            if (library_option_needs_value(libraryOptions[i]) && !library_option_ignored(libraryOptions[i])) {
                fprintf(stderr, "[HOOK] FATAL ERROR: Library option %s requires a value but libraryOptionValues is nullptr\n",
                        std::string(library_option_to_string(libraryOptions[i])).c_str());
                abort();
            }
        }
    }

    const std::vector<uint8_t> binary_vec(binary_data, binary_data + total_size);
    const uint64_t hash = compute_hash(binary_vec);
    if (out_hash) *out_hash = hash;

    BinaryMetadata metadata;
    metadata.binary_data = binary_vec;
    metadata.base_func_name = std::string(func_name);

    if (numJitOptions > 0 && jitOptions) {
        for (unsigned int i = 0; i < numJitOptions; i++) {
            if (!is_jit_option_ignored(jitOptions[i])) {
                metadata.jit_options.push_back(jitOptions[i]);
                metadata.jit_option_values.push_back(jitOptionsValues ? jitOptionsValues[i] : nullptr);
            }
        }
    }

    if (numLibraryOptions > 0 && libraryOptions) {
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            if (!library_option_ignored(libraryOptions[i])) {
                metadata.library_options.push_back(libraryOptions[i]);
                metadata.library_option_values.push_back(libraryOptionValues ? libraryOptionValues[i] : nullptr);
            }
        }
    }

    // Store linked fatbin segments if this is a device-linked library
    if (!linked_fatbin_segments.empty()) {
        metadata.binary_flags |= BINARY_FLAG_NEEDS_DEVICE_LINK;
        metadata.linked_fatbin_segments = std::move(linked_fatbin_segments);
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] DEBUG: Stored %zu fatbin segments for device linking (hash %016llx)\n",
                metadata.linked_fatbin_segments.size(), (unsigned long long)hash);
#endif
    }

    binary_hash_to_metadata.emplace(hash, std::move(metadata));
}

static void dump_fatbin_from_file_and_info(const char *filename, std::string_view func_name,
                                            unsigned int numJitOptions, CUjit_option *jitOptions, void **jitOptionsValues,
                                            unsigned int numLibraryOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues,
                                            int idx, uint64_t* out_hash) {
    if (!filename) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Null filename\n");
        abort();
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Failed to open file %s\n", filename);
        abort();
    }

    file.seekg(0, std::ios::end);
    const size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == 0) {
        fprintf(stderr, "[HOOK] FATAL ERROR: File %s is empty\n", filename);
        abort();
    }

    std::vector<uint8_t> file_contents(file_size);
    file.read(reinterpret_cast<char*>(file_contents.data()), file_size);
    if (!file) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Failed to read file %s\n", filename);
        abort();
    }

    if (numJitOptions > 0 && !jitOptionsValues) {
        for (unsigned int i = 0; i < numJitOptions; i++) {
            if (jit_option_needs_value(jitOptions[i]) && !is_jit_option_ignored(jitOptions[i])) {
                fprintf(stderr, "[HOOK] FATAL ERROR: JIT option %s requires a value but jitOptionsValues is nullptr\n",
                        std::string(jit_option_to_string(jitOptions[i])).c_str());
                abort();
            }
        }
    }

    if (numLibraryOptions > 0 && !libraryOptionValues) {
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            if (library_option_needs_value(libraryOptions[i]) && !library_option_ignored(libraryOptions[i])) {
                fprintf(stderr, "[HOOK] FATAL ERROR: Library option %s requires a value but libraryOptionValues is nullptr\n",
                        std::string(library_option_to_string(libraryOptions[i])).c_str());
                abort();
            }
        }
    }

    const uint64_t hash = compute_hash(file_contents);
    if (out_hash) *out_hash = hash;

    BinaryMetadata metadata;
    metadata.binary_data = file_contents;
    metadata.base_func_name = std::string(func_name);
    metadata.filename = filename;

    if (numJitOptions > 0 && jitOptions) {
        for (unsigned int i = 0; i < numJitOptions; i++) {
            if (!is_jit_option_ignored(jitOptions[i])) {
                metadata.jit_options.push_back(jitOptions[i]);
                metadata.jit_option_values.push_back(jitOptionsValues ? jitOptionsValues[i] : nullptr);
            }
        }
    }

    if (numLibraryOptions > 0 && libraryOptions) {
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            if (!library_option_ignored(libraryOptions[i])) {
                metadata.library_options.push_back(libraryOptions[i]);
                metadata.library_option_values.push_back(libraryOptionValues ? libraryOptionValues[i] : nullptr);
            }
        }
    }

    binary_hash_to_metadata.emplace(hash, std::move(metadata));
}

static CUjit_option string_to_jit_option(const std::string& str) {
    if (str == "CU_JIT_MAX_REGISTERS") return CU_JIT_MAX_REGISTERS;
    if (str == "CU_JIT_THREADS_PER_BLOCK") return CU_JIT_THREADS_PER_BLOCK;
    if (str == "CU_JIT_WALL_TIME") return CU_JIT_WALL_TIME;
    if (str == "CU_JIT_INFO_LOG_BUFFER") return CU_JIT_INFO_LOG_BUFFER;
    if (str == "CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES") return CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    if (str == "CU_JIT_ERROR_LOG_BUFFER") return CU_JIT_ERROR_LOG_BUFFER;
    if (str == "CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES") return CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    if (str == "CU_JIT_OPTIMIZATION_LEVEL") return CU_JIT_OPTIMIZATION_LEVEL;
    if (str == "CU_JIT_TARGET_FROM_CUCONTEXT") return CU_JIT_TARGET_FROM_CUCONTEXT;
    if (str == "CU_JIT_TARGET") return CU_JIT_TARGET;
    if (str == "CU_JIT_FALLBACK_STRATEGY") return CU_JIT_FALLBACK_STRATEGY;
    if (str == "CU_JIT_GENERATE_DEBUG_INFO") return CU_JIT_GENERATE_DEBUG_INFO;
    if (str == "CU_JIT_LOG_VERBOSE") return CU_JIT_LOG_VERBOSE;
    if (str == "CU_JIT_GENERATE_LINE_INFO") return CU_JIT_GENERATE_LINE_INFO;
    if (str == "CU_JIT_CACHE_MODE") return CU_JIT_CACHE_MODE;
    if (str == "CU_JIT_NEW_SM3X_OPT") return CU_JIT_NEW_SM3X_OPT;
    if (str == "CU_JIT_FAST_COMPILE") return CU_JIT_FAST_COMPILE;
    if (str == "CU_JIT_GLOBAL_SYMBOL_NAMES") return CU_JIT_GLOBAL_SYMBOL_NAMES;
    if (str == "CU_JIT_GLOBAL_SYMBOL_ADDRESSES") return CU_JIT_GLOBAL_SYMBOL_ADDRESSES;
    if (str == "CU_JIT_GLOBAL_SYMBOL_COUNT") return CU_JIT_GLOBAL_SYMBOL_COUNT;
    if (str == "CU_JIT_LTO") return CU_JIT_LTO;
    if (str == "CU_JIT_FTZ") return CU_JIT_FTZ;
    if (str == "CU_JIT_PREC_DIV") return CU_JIT_PREC_DIV;
    if (str == "CU_JIT_PREC_SQRT") return CU_JIT_PREC_SQRT;
    if (str == "CU_JIT_FMA") return CU_JIT_FMA;
    if (str == "CU_JIT_REFERENCED_KERNEL_NAMES") return CU_JIT_REFERENCED_KERNEL_NAMES;
    if (str == "CU_JIT_REFERENCED_KERNEL_COUNT") return CU_JIT_REFERENCED_KERNEL_COUNT;
    if (str == "CU_JIT_REFERENCED_VARIABLE_NAMES") return CU_JIT_REFERENCED_VARIABLE_NAMES;
    if (str == "CU_JIT_REFERENCED_VARIABLE_COUNT") return CU_JIT_REFERENCED_VARIABLE_COUNT;
    if (str == "CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES") return CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES;
    if (str == "CU_JIT_POSITION_INDEPENDENT_CODE") return CU_JIT_POSITION_INDEPENDENT_CODE;
    if (str == "CU_JIT_MIN_CTA_PER_SM") return CU_JIT_MIN_CTA_PER_SM;
    if (str == "CU_JIT_MAX_THREADS_PER_BLOCK") return CU_JIT_MAX_THREADS_PER_BLOCK;
    if (str == "CU_JIT_OVERRIDE_DIRECTIVE_VALUES") return CU_JIT_OVERRIDE_DIRECTIVE_VALUES;
    #if (defined(CUDA_VERSION) && CUDA_VERSION >= 12900)
    if (str == "CU_JIT_SPLIT_COMPILE") return CU_JIT_SPLIT_COMPILE;
    #endif
    fprintf(stderr, "[HOOK] FATAL ERROR: Unknown JIT option string: %s\n", str.c_str());
    abort();
}

static CUlibraryOption string_to_library_option(const std::string& str) {
    if (str == "CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE") return CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE;
    if (str == "CU_LIBRARY_BINARY_IS_PRESERVED") return CU_LIBRARY_BINARY_IS_PRESERVED;
    fprintf(stderr, "[HOOK] FATAL ERROR: Unknown library option string: %s\n", str.c_str());
    abort();
}

static void* string_to_jit_option_value(CUjit_option opt, const std::string& value_str) {
    if (value_str == "null" || is_jit_option_ignored(opt)) {
        return nullptr;
    }

    switch (opt) {
        case CU_JIT_MAX_REGISTERS:
        case CU_JIT_THREADS_PER_BLOCK:
        case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        case CU_JIT_OPTIMIZATION_LEVEL:
        case CU_JIT_TARGET:
        case CU_JIT_FALLBACK_STRATEGY:
        case CU_JIT_CACHE_MODE:
        case CU_JIT_GLOBAL_SYMBOL_COUNT:
        case CU_JIT_REFERENCED_KERNEL_COUNT:
        case CU_JIT_REFERENCED_VARIABLE_COUNT:
        case CU_JIT_MIN_CTA_PER_SM:
        #if (defined(CUDA_VERSION) && CUDA_VERSION >= 12900)
        case CU_JIT_SPLIT_COMPILE: {
            unsigned int val = std::stoul(value_str);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(val));
        }
        #endif

        case CU_JIT_GENERATE_DEBUG_INFO:
        case CU_JIT_LOG_VERBOSE:
        case CU_JIT_GENERATE_LINE_INFO:
        case CU_JIT_LTO:
        case CU_JIT_FTZ:
        case CU_JIT_PREC_DIV:
        case CU_JIT_PREC_SQRT:
        case CU_JIT_FMA:
        case CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES:
        case CU_JIT_POSITION_INDEPENDENT_CODE:
        case CU_JIT_MAX_THREADS_PER_BLOCK:
        case CU_JIT_OVERRIDE_DIRECTIVE_VALUES: {
            int val = std::stoi(value_str);
            return reinterpret_cast<void*>(static_cast<intptr_t>(val));
        }

        case CU_JIT_TARGET_FROM_CUCONTEXT:
        case CU_JIT_NEW_SM3X_OPT:
        case CU_JIT_FAST_COMPILE: {
            unsigned int val = std::stoul(value_str);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(val));
        }

        case CU_JIT_WALL_TIME: {
            float val = std::stof(value_str);
            void* result;
            std::memcpy(&result, &val, sizeof(float));
            return result;
        }

        default:
            fprintf(stderr, "[HOOK] FATAL ERROR: Unhandled JIT option in string_to_jit_option_value: %d\n", static_cast<int>(opt));
            abort();
    }
}

static void* string_to_library_option_value(CUlibraryOption opt, const std::string& value_str) {
    if (value_str == "null" || library_option_ignored(opt)) {
        return nullptr;
    }

    switch (opt) {
        case CU_LIBRARY_BINARY_IS_PRESERVED: {
            unsigned int val = std::stoul(value_str);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(val));
        }

        default:
            fprintf(stderr, "[HOOK] FATAL ERROR: Unhandled library option in string_to_library_option_value: %d\n", static_cast<int>(opt));
            abort();
    }
}

// ============================================================================
// NVSHMEM Detection - Used at SAVE time to determine if modules need NVSHMEM init
// ============================================================================
// Known NVSHMEM device symbols that indicate NVSHMEM usage.
// These are internal symbols from libnvshmem_device.a that get linked into
// any CUDA module/library that uses NVSHMEM device APIs.
static const char* const NVSHMEM_DEVICE_SYMBOLS[] = {
    "nvshmemi_device_state_d",
    "nvshmemi_team_pool_d",
    "nvshmemi_team_same_mype_node_pool_d",
    "nvshmemi_mype_d",
    "nvshmemi_npes_d",
    "nvshmemi_node_mype_d",
    "nvshmemi_node_npes_d",
    "nvshmemi_heap_base_d",
    nullptr  // sentinel
};

// Detect if a CUmodule requires NVSHMEM initialization by probing for NVSHMEM device symbols
static bool detect_nvshmem_requirement_for_module(CUmodule module) {
    typedef CUresult (*cuModuleGetGlobal_t)(CUdeviceptr*, size_t*, CUmodule, const char*);
    auto module_get_global = (cuModuleGetGlobal_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleGetGlobal);

    if (!module_get_global) {
        return false;
    }

    CUdeviceptr ptr;
    size_t size;

    for (const char* const* sym = NVSHMEM_DEVICE_SYMBOLS; *sym != nullptr; ++sym) {
        CUresult res = module_get_global(&ptr, &size, module, *sym);
        if (res == CUDA_SUCCESS) {
#ifdef HOOK_DEBUG
            fprintf(stderr, "[HOOK] DEBUG: Found NVSHMEM symbol '%s' in module (ptr=%p, size=%zu)\n",
                    *sym, (void*)ptr, size);
#endif
            return true;
        }
    }

    return false;
}

// Detect if a CUlibrary requires NVSHMEM initialization by probing for NVSHMEM device symbols
static bool detect_nvshmem_requirement_for_library(CUlibrary library) {
    typedef CUresult (*cuLibraryGetGlobal_t)(CUdeviceptr*, size_t*, CUlibrary, const char*);
    auto library_get_global = (cuLibraryGetGlobal_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryGetGlobal);

    if (!library_get_global) {
        return false;
    }

    CUdeviceptr ptr;
    size_t size;

    for (const char* const* sym = NVSHMEM_DEVICE_SYMBOLS; *sym != nullptr; ++sym) {
        CUresult res = library_get_global(&ptr, &size, library, *sym);
        if (res == CUDA_SUCCESS) {
#ifdef HOOK_DEBUG
            fprintf(stderr, "[HOOK] DEBUG: Found NVSHMEM symbol '%s' in library (ptr=%p, size=%zu)\n",
                    *sym, (void*)ptr, size);
#endif
            return true;
        }
    }

    return false;
}

static void dump_module_or_library_entrypoints(std::variant<CUmodule, CUlibrary> handle, uint64_t hash) {
    std::vector<std::string> entrypoint_names;
    bool requires_nvshmem = false;

    if (std::holds_alternative<CUlibrary>(handle)) {
        CUlibrary library = std::get<CUlibrary>(handle);

        typedef CUresult (*cuLibraryGetKernelCount_t)(unsigned int*, CUlibrary);
        typedef CUresult (*cuLibraryEnumerateKernels_t)(CUkernel*, unsigned int, CUlibrary);
        typedef CUresult (*cuKernelGetName_t)(const char**, CUkernel);

        auto get_count_func = (cuLibraryGetKernelCount_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryGetKernelCount);
        auto enum_func = (cuLibraryEnumerateKernels_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryEnumerateKernels);
        auto get_name_func = (cuKernelGetName_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuKernelGetName);

        unsigned int kernel_count = 0;
        CUresult res = get_count_func(&kernel_count, library);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] FATAL ERROR: cuLibraryGetKernelCount failed with error %d\n", res);
            abort();
        }
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] DEBUG: dump_module_or_library_entrypoints for hash %016llx: kernel_count=%u\n",
                (unsigned long long)hash, kernel_count);
#endif
        if (kernel_count > 0) {
            std::vector<CUkernel> kernels(kernel_count);
            res = enum_func(kernels.data(), kernel_count, library);
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "[HOOK] FATAL ERROR: cuLibraryEnumerateKernels failed with error %d\n", res);
                abort();
            }

            for (const auto& kernel : kernels) {
                const char* name = nullptr;
                res = get_name_func(&name, kernel);
                if (res != CUDA_SUCCESS) {
                    fprintf(stderr, "[HOOK] FATAL ERROR: cuKernelGetName failed with error %d\n", res);
                    abort();
                }
                if (!name) {
                    fprintf(stderr, "[HOOK] FATAL ERROR: cuKernelGetName returned null name\n");
                    abort();
                }
                entrypoint_names.push_back(name);
            }
        }

        // Detect NVSHMEM requirement at SAVE time by probing for NVSHMEM device symbols
        requires_nvshmem = detect_nvshmem_requirement_for_library(library);
#ifdef HOOK_DEBUG
        if (requires_nvshmem) {
            fprintf(stderr, "[HOOK] DEBUG: Library hash %016llx requires NVSHMEM initialization\n",
                    (unsigned long long)hash);
        }
#endif
    } else {
        CUmodule module = std::get<CUmodule>(handle);

        typedef CUresult (*cuModuleGetFunctionCount_t)(unsigned int*, CUmodule);
        typedef CUresult (*cuModuleEnumerateFunctions_t)(CUfunction*, unsigned int, CUmodule);
        typedef CUresult (*cuFuncGetName_t)(const char**, CUfunction);

        auto get_count_func = (cuModuleGetFunctionCount_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleGetFunctionCount);
        auto enum_func = (cuModuleEnumerateFunctions_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleEnumerateFunctions);
        auto get_name_func = (cuFuncGetName_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuFuncGetName);

        unsigned int func_count = 0;
        CUresult res = get_count_func(&func_count, module);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] FATAL ERROR: cuModuleGetFunctionCount failed with error %d\n", res);
            abort();
        }

        if (func_count > 0) {
            std::vector<CUfunction> functions(func_count);
            res = enum_func(functions.data(), func_count, module);
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "[HOOK] FATAL ERROR: cuModuleEnumerateFunctions failed with error %d\n", res);
                abort();
            }

            for (const auto& function : functions) {
                const char* name = nullptr;
                res = get_name_func(&name, function);
                if (res != CUDA_SUCCESS) {
                    fprintf(stderr, "[HOOK] FATAL ERROR: cuFuncGetName failed with error %d\n", res);
                    abort();
                }
                if (!name) {
                    fprintf(stderr, "[HOOK] FATAL ERROR: cuFuncGetName returned null name\n");
                    abort();
                }
                entrypoint_names.push_back(name);
            }
        }

        // Detect NVSHMEM requirement at SAVE time by probing for NVSHMEM device symbols
        requires_nvshmem = detect_nvshmem_requirement_for_module(module);
#ifdef HOOK_DEBUG
        if (requires_nvshmem) {
            fprintf(stderr, "[HOOK] DEBUG: Module hash %016llx requires NVSHMEM initialization\n",
                    (unsigned long long)hash);
        }
#endif
    }

    binary_hash_to_metadata.visit(hash, [&entrypoint_names, requires_nvshmem](auto& pair) {
        pair.second.entrypoint_names = std::move(entrypoint_names);
        if (requires_nvshmem) {
            pair.second.binary_flags |= BINARY_FLAG_REQUIRES_NVSHMEM;
        }
    });
}

// ============================================================================
// NVSHMEM Auto-Initialization for DeepEP Kernels
// ============================================================================
// When loading CUDA modules/libraries that contain DeepEP kernels (which use
// NVSHMEM for inter-GPU communication), we need to call nvshmemx_cumodule_init
// or nvshmemx_culibrary_init to properly link the NVSHMEM device-side state.
// Without this, kernels will fail with IBGDA assertion errors on replay.
// Default to false since most graphs don't use NVSHMEM; enable via set_nvshmem_auto_init(true).

static std::atomic<bool> nvshmem_auto_init_enabled{false};

// Track modules/libraries that need deferred NVSHMEM init.
// This is populated during load_cuda_modules_and_libraries when nvshmem_auto_init is disabled.
// The variant holds either CUmodule or CUlibrary.
using NvshmemPendingHandle = std::variant<CUmodule, CUlibrary>;
struct NvshmemPendingEntry {
    uint64_t hash;
    NvshmemPendingHandle handle;
};
static std::vector<NvshmemPendingEntry> pending_nvshmem_init;
static std::mutex pending_nvshmem_mutex;

// Initialize NVSHMEM for a loaded CUmodule
// The requires_nvshmem flag should have been determined at SAVE time by probing for NVSHMEM symbols
static void maybe_init_nvshmem_for_module(CUmodule module, uint64_t hash, bool requires_nvshmem) {
    if (!requires_nvshmem) {
        return;
    }

    // If nvshmem_auto_init is disabled, add to pending list for deferred init
    if (!nvshmem_auto_init_enabled.load()) {
        std::lock_guard<std::mutex> lock(pending_nvshmem_mutex);
        pending_nvshmem_init.push_back({hash, module});
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] DEBUG: Added module hash %016llx to pending NVSHMEM init list\n",
                (unsigned long long)hash);
#endif
        return;
    }

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] DEBUG: Initializing NVSHMEM for module hash %016llx\n",
            (unsigned long long)hash);
#endif

    using init_fn_t = int (*)(CUmodule);
    static init_fn_t init_fn = reinterpret_cast<init_fn_t>(
        dlsym(RTLD_DEFAULT, "nvshmemx_cumodule_init"));

    if (!init_fn) {
        fprintf(stderr, "[HOOK] WARNING: nvshmemx_cumodule_init not found; "
                "NVSHMEM-using kernels may fail (hash %016llx). "
                "Ensure libnvshmem_host.so is in LD_PRELOAD or LD_LIBRARY_PATH.\n",
                (unsigned long long)hash);
        return;
    }

    int res = init_fn(module);
    if (res != 0) {
        fprintf(stderr, "[HOOK] WARNING: nvshmemx_cumodule_init failed for hash %016llx (res=%d)\n",
                (unsigned long long)hash, res);
    }
}

// Initialize NVSHMEM for a loaded CUlibrary
// The requires_nvshmem flag should have been determined at SAVE time by probing for NVSHMEM symbols
static void maybe_init_nvshmem_for_library(CUlibrary library, uint64_t hash, bool requires_nvshmem) {
    if (!requires_nvshmem) {
        return;
    }

    // If nvshmem_auto_init is disabled, add to pending list for deferred init
    if (!nvshmem_auto_init_enabled.load()) {
        std::lock_guard<std::mutex> lock(pending_nvshmem_mutex);
        pending_nvshmem_init.push_back({hash, library});
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] DEBUG: Added library hash %016llx to pending NVSHMEM init list\n",
                (unsigned long long)hash);
#endif
        return;
    }

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] DEBUG: Initializing NVSHMEM for library hash %016llx\n",
            (unsigned long long)hash);
#endif

    using init_fn_t = int (*)(CUlibrary);
    static init_fn_t init_fn = reinterpret_cast<init_fn_t>(
        dlsym(RTLD_DEFAULT, "nvshmemx_culibrary_init"));

    if (!init_fn) {
        fprintf(stderr, "[HOOK] WARNING: nvshmemx_culibrary_init not found; "
                "NVSHMEM-using kernels may fail (hash %016llx). "
                "Ensure libnvshmem_host.so is in LD_PRELOAD or LD_LIBRARY_PATH.\n",
                (unsigned long long)hash);
        return;
    }

    int res = init_fn(library);
    if (res != 0) {
        fprintf(stderr, "[HOOK] WARNING: nvshmemx_culibrary_init failed for hash %016llx (res=%d)\n",
                (unsigned long long)hash, res);
    }
}

// Initialize NVSHMEM for all already-loaded modules/libraries that require it.
// This should be called AFTER NVSHMEM runtime is initialized (e.g., by DeepEP Buffer creation).
// Processes the pending_nvshmem_init list which is populated during load_cuda_modules_and_libraries
// when nvshmem_auto_init is disabled.
// Returns the number of modules/libraries initialized.
int init_nvshmem_for_loaded_modules() {
    using cumodule_init_fn_t = int (*)(CUmodule);
    using culibrary_init_fn_t = int (*)(CUlibrary);

    static cumodule_init_fn_t cumodule_init_fn = reinterpret_cast<cumodule_init_fn_t>(
        dlsym(RTLD_DEFAULT, "nvshmemx_cumodule_init"));
    static culibrary_init_fn_t culibrary_init_fn = reinterpret_cast<culibrary_init_fn_t>(
        dlsym(RTLD_DEFAULT, "nvshmemx_culibrary_init"));

    if (!cumodule_init_fn && !culibrary_init_fn) {
        fprintf(stderr, "[HOOK] WARNING: Neither nvshmemx_cumodule_init nor nvshmemx_culibrary_init found; "
                "NVSHMEM-using kernels may fail. "
                "Ensure libnvshmem_host.so is in LD_PRELOAD or LD_LIBRARY_PATH.\n");
        return 0;
    }

    // Take ownership of the pending list
    std::vector<NvshmemPendingEntry> pending;
    {
        std::lock_guard<std::mutex> lock(pending_nvshmem_mutex);
        pending = std::move(pending_nvshmem_init);
        pending_nvshmem_init.clear();
    }

    if (pending.empty()) {
        fprintf(stderr, "[HOOK] INFO: No modules pending NVSHMEM initialization\n");
        return 0;
    }

    fprintf(stderr, "[HOOK] INFO: Processing %zu modules pending NVSHMEM initialization\n", pending.size());

    int count = 0;
    for (const auto& entry : pending) {
        if (std::holds_alternative<CUlibrary>(entry.handle)) {
            if (culibrary_init_fn) {
                CUlibrary library = std::get<CUlibrary>(entry.handle);
                int res = culibrary_init_fn(library);
                if (res != 0) {
                    fprintf(stderr, "[HOOK] WARNING: nvshmemx_culibrary_init failed for hash %016llx (res=%d)\n",
                            (unsigned long long)entry.hash, res);
                } else {
                    fprintf(stderr, "[HOOK] INFO: Initialized NVSHMEM for library hash %016llx\n",
                            (unsigned long long)entry.hash);
                    count++;
                }
            }
        } else if (std::holds_alternative<CUmodule>(entry.handle)) {
            if (cumodule_init_fn) {
                CUmodule module = std::get<CUmodule>(entry.handle);
                int res = cumodule_init_fn(module);
                if (res != 0) {
                    fprintf(stderr, "[HOOK] WARNING: nvshmemx_cumodule_init failed for hash %016llx (res=%d)\n",
                            (unsigned long long)entry.hash, res);
                } else {
                    fprintf(stderr, "[HOOK] INFO: Initialized NVSHMEM for module hash %016llx\n",
                            (unsigned long long)entry.hash);
                    count++;
                }
            }
        }
    }

    return count;
}

static void setup_handles_for_module(CUmodule module, uint64_t hash, const std::vector<std::string>& expected_names,
                                     bool requires_nvshmem = false) {
    typedef CUresult (*cuModuleGetFunctionCount_t)(unsigned int*, CUmodule);
    typedef CUresult (*cuModuleEnumerateFunctions_t)(CUfunction*, unsigned int, CUmodule);
    typedef CUresult (*cuFuncGetName_t)(const char**, CUfunction);

    auto get_count_func = (cuModuleGetFunctionCount_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleGetFunctionCount);
    auto enum_func = (cuModuleEnumerateFunctions_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleEnumerateFunctions);
    auto get_name_func = (cuFuncGetName_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuFuncGetName);

    unsigned int func_count = 0;
    if (CUresult res = get_count_func(&func_count, module); res != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuModuleGetFunctionCount failed for hash %016llx\n", (unsigned long long)hash);
        abort();
    }

    if (func_count != expected_names.size()) {
        fprintf(stderr, "[HOOK] ERROR: Function count mismatch for hash %016llx: got %u, expected %zu\n",
                (unsigned long long)hash, func_count, expected_names.size());
        abort();
    }

    std::unordered_map<std::string, CUfunction> func_map;

    if (func_count > 0) {
        std::vector<CUfunction> functions(func_count);
        if (CUresult res = enum_func(functions.data(), func_count, module); res != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] ERROR: cuModuleEnumerateFunctions failed for hash %016llx\n", (unsigned long long)hash);
            abort();
        }

        for (const auto& function : functions) {
            const char* name = nullptr;
            if (CUresult res = get_name_func(&name, function); res != CUDA_SUCCESS || !name) {
                fprintf(stderr, "[HOOK] ERROR: cuFuncGetName failed for hash %016llx\n", (unsigned long long)hash);
                abort();
            }
            func_map[name] = function;
        }
    }

    for (const auto& expected_name : expected_names) {
        if (func_map.find(expected_name) == func_map.end()) {
            fprintf(stderr, "[HOOK] ERROR: Expected function '%s' not found in module for hash %016llx\n",
                    expected_name.c_str(), (unsigned long long)hash);
            abort();
        }
    }

    // Initialize NVSHMEM if required (flag determined at SAVE time via symbol probing)
    maybe_init_nvshmem_for_module(module, hash, requires_nvshmem);

    binary_hash_to_handles.insert_or_assign(hash, std::make_tuple(module, func_map));
}

static void setup_handles_for_library(CUlibrary library, uint64_t hash, const std::vector<std::string>& expected_names,
                                      bool requires_nvshmem = false) {
    typedef CUresult (*cuLibraryGetKernelCount_t)(unsigned int*, CUlibrary);
    typedef CUresult (*cuLibraryEnumerateKernels_t)(CUkernel*, unsigned int, CUlibrary);
    typedef CUresult (*cuKernelGetName_t)(const char**, CUkernel);

    auto get_count_func = (cuLibraryGetKernelCount_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryGetKernelCount);
    auto enum_func = (cuLibraryEnumerateKernels_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryEnumerateKernels);
    auto get_name_func = (cuKernelGetName_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuKernelGetName);

    unsigned int kernel_count = 0;
    if (CUresult res = get_count_func(&kernel_count, library); res != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuLibraryGetKernelCount failed for hash %016llx\n", (unsigned long long)hash);
        abort();
    }

    if (kernel_count != expected_names.size()) {
        fprintf(stderr, "[HOOK] ERROR: Kernel count mismatch for hash %016llx: got %u, expected %zu\n",
                (unsigned long long)hash, kernel_count, expected_names.size());

        // Debug: enumerate and print all kernels that ARE available
        if (kernel_count > 0) {
            std::vector<CUkernel> avail_kernels(kernel_count);
            if (enum_func(avail_kernels.data(), kernel_count, library) == CUDA_SUCCESS) {
                fprintf(stderr, "[HOOK] DEBUG: Available kernels in library:\n");
                std::unordered_set<std::string> avail_names;
                for (const auto& kernel : avail_kernels) {
                    const char* name = nullptr;
                    if (get_name_func(&name, kernel) == CUDA_SUCCESS && name) {
                        avail_names.insert(name);
                        fprintf(stderr, "[HOOK] DEBUG:   - %s\n", name);
                    }
                }

                // Print missing kernels
                fprintf(stderr, "[HOOK] DEBUG: Missing kernels (%zu):\n", expected_names.size() - avail_names.size());
                for (const auto& expected : expected_names) {
                    if (avail_names.find(expected) == avail_names.end()) {
                        fprintf(stderr, "[HOOK] DEBUG:   - MISSING: %s\n", expected.c_str());
                    }
                }
            }
        }

        abort();
    }

    std::unordered_map<std::string, CUkernel> kernel_map;

    if (kernel_count > 0) {
        std::vector<CUkernel> kernels(kernel_count);
        if (CUresult res = enum_func(kernels.data(), kernel_count, library); res == CUDA_SUCCESS) {
            for (const auto& kernel : kernels) {
                const char* name = nullptr;
                if (CUresult res = get_name_func(&name, kernel); res != CUDA_SUCCESS || !name) {
                    fprintf(stderr, "[HOOK] ERROR: cuKernelGetName failed for hash %016llx\n", (unsigned long long)hash);
                    abort();
                }
                kernel_map[name] = kernel;
            }
        }
    }

    for (const auto& expected_name : expected_names) {
        if (kernel_map.find(expected_name) == kernel_map.end()) {
            fprintf(stderr, "[HOOK] ERROR: Expected kernel '%s' not found in library for hash %016llx\n",
                    expected_name.c_str(), (unsigned long long)hash);
            abort();
        }
    }

    // Initialize NVSHMEM if required (flag determined at SAVE time via symbol probing)
    maybe_init_nvshmem_for_library(library, hash, requires_nvshmem);

    binary_hash_to_handles.insert_or_assign(hash, std::make_tuple(library, kernel_map));
}

// Pre-link fatbin segments during SAVE mode to avoid runtime linking during LOAD
static std::vector<uint8_t> prelink_fatbin_segments(const std::vector<std::vector<uint8_t>>& segments,
                                                      const std::vector<CUjit_option>& jit_options,
                                                      const std::vector<void*>& jit_option_values,
                                                      uint64_t hash) {
    typedef CUresult (*cuLinkCreate_t)(unsigned int, CUjit_option*, void**, CUlinkState*);
    typedef CUresult (*cuLinkAddData_t)(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
    typedef CUresult (*cuLinkComplete_t)(CUlinkState, void**, size_t*);
    typedef CUresult (*cuLinkDestroy_t)(CUlinkState);

    auto link_create = (cuLinkCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkCreate);
    auto link_add_data = (cuLinkAddData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkAddData);
    auto link_complete = (cuLinkComplete_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkComplete);
    auto link_destroy = (cuLinkDestroy_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkDestroy);

    if (!link_create || !link_add_data || !link_complete || !link_destroy) {
        fprintf(stderr, "[HOOK] WARNING: CUDA linker API not available, cannot pre-link\n");
        return {};
    }

    unsigned int num_jit = jit_options.size();
    CUjit_option* jit_opts = num_jit > 0 ? const_cast<CUjit_option*>(jit_options.data()) : nullptr;
    void** jit_vals = num_jit > 0 ? const_cast<void**>(jit_option_values.data()) : nullptr;

    CUlinkState link_state;
    CUresult res = link_create(num_jit, jit_opts, jit_vals, &link_state);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] WARNING: cuLinkCreate failed with error %d during pre-link, falling back to segments\n", res);
        return {};
    }

    // Add each segment to the linker
    for (size_t i = 0; i < segments.size(); i++) {
        const auto& segment = segments[i];
        res = link_add_data(link_state, CU_JIT_INPUT_FATBINARY,
                           const_cast<uint8_t*>(segment.data()), segment.size(),
                           nullptr, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] WARNING: cuLinkAddData failed for segment %zu with error %d during pre-link\n", i, res);
            link_destroy(link_state);
            return {};
        }
    }

    // Complete linking
    void* cubin_out = nullptr;
    size_t cubin_size = 0;
    res = link_complete(link_state, &cubin_out, &cubin_size);
    if (res != CUDA_SUCCESS || !cubin_out || cubin_size == 0) {
        fprintf(stderr, "[HOOK] WARNING: cuLinkComplete failed with error %d during pre-link\n", res);
        link_destroy(link_state);
        return {};
    }

    // Copy the linked cubin (the linker state owns the memory, so we must copy before destroying)
    std::vector<uint8_t> linked_cubin(static_cast<uint8_t*>(cubin_out),
                                       static_cast<uint8_t*>(cubin_out) + cubin_size);

    link_destroy(link_state);

    fprintf(stderr, "[HOOK] DEBUG: Pre-linked %zu segments into %zu byte cubin for hash %016llx\n",
            segments.size(), cubin_size, (unsigned long long)hash);

    return linked_cubin;
}

static void pack_fatbins_to_folder_impl(const fs::path& archive_dir) {
    fs::create_directories(archive_dir);

    const fs::path packed_img_path = archive_dir / "fatbin_image_packed.img";
    const fs::path packed_txt_path = archive_dir / "fatbin_entrypoint_packed.txt";

    std::ofstream packed_img(packed_img_path.string(), std::ios::binary);
    std::ofstream packed_txt(packed_txt_path.string());

    if (!packed_img || !packed_txt) {
        return;
    }

    binary_hash_to_metadata.cvisit_all([&](const auto& pair) {
        const uint64_t hash = pair.first;
        const BinaryMetadata& metadata = pair.second;

        if (!metadata.used) {
            return;
        }

        // Write binary data to img file
        packed_img.write(reinterpret_cast<const char*>(&hash), sizeof(uint64_t));

        // Compute final binary_flags for this binary
        uint32_t binary_flags = metadata.binary_flags;
        std::vector<uint8_t> linked_cubin;
        bool has_segments = (binary_flags & BINARY_FLAG_NEEDS_DEVICE_LINK) && !metadata.linked_fatbin_segments.empty();

        if (has_segments) {
            // Try to pre-link the segments to avoid runtime linking during LOAD
            linked_cubin = prelink_fatbin_segments(metadata.linked_fatbin_segments,
                                                   metadata.jit_options,
                                                   metadata.jit_option_values,
                                                   hash);
            if (!linked_cubin.empty()) {
                // Pre-linking succeeded - clear NEEDS_DEVICE_LINK, will be stored as normal binary
                binary_flags &= ~BINARY_FLAG_NEEDS_DEVICE_LINK;
                has_segments = false;
            }
        }

        if (!linked_cubin.empty()) {
            // Write pre-linked cubin (stored same as normal binary)
            const size_t size = linked_cubin.size();
            packed_img.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
            packed_img.write(reinterpret_cast<const char*>(linked_cubin.data()), size);
            fprintf(stderr, "[HOOK] DEBUG: Packed pre-linked cubin (%zu bytes) for hash %016llx\n",
                    size, (unsigned long long)hash);
        } else if (has_segments) {
            // Fallback: Write marker for device-linked binary (size = 0, followed by segment count)
            const size_t marker = 0;
            const size_t num_segments = metadata.linked_fatbin_segments.size();
            packed_img.write(reinterpret_cast<const char*>(&marker), sizeof(size_t));
            packed_img.write(reinterpret_cast<const char*>(&num_segments), sizeof(size_t));

            // Write each segment
            for (const auto& segment : metadata.linked_fatbin_segments) {
                const size_t seg_size = segment.size();
                packed_img.write(reinterpret_cast<const char*>(&seg_size), sizeof(size_t));
                packed_img.write(reinterpret_cast<const char*>(segment.data()), seg_size);
            }
            fprintf(stderr, "[HOOK] DEBUG: Packed %zu linked fatbin segments for hash %016llx (pre-link failed)\n",
                    num_segments, (unsigned long long)hash);
        } else {
            // Regular single binary
            const size_t size = metadata.binary_data.size();
            packed_img.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
            packed_img.write(reinterpret_cast<const char*>(metadata.binary_data.data()), size);
        }

        // Write metadata to txt file
        packed_txt << str(boost::format("%016llx") % hash) << "\n";
        packed_txt << metadata.base_func_name;

        if (!metadata.filename.empty()) {
            packed_txt << ",filename=" << metadata.filename;
        }

        if (!metadata.jit_options.empty()) {
            packed_txt << ",numJitOptions=" << metadata.jit_options.size();
            for (size_t i = 0; i < metadata.jit_options.size(); i++) {
                packed_txt << "," << jit_option_to_string(metadata.jit_options[i]) << "="
                          << jit_option_value_to_string(metadata.jit_options[i], metadata.jit_option_values[i], false);
            }
        }

        if (!metadata.library_options.empty()) {
            packed_txt << ",numLibraryOptions=" << metadata.library_options.size();
            for (size_t i = 0; i < metadata.library_options.size(); i++) {
                packed_txt << "," << library_option_to_string(metadata.library_options[i]) << "="
                          << library_option_value_to_string(metadata.library_options[i], metadata.library_option_values[i], false);
            }
        }

        // Write binary_flags and num_segments if needed
        packed_txt << ",binary_flags=" << binary_flags;
        if (has_segments) {
            packed_txt << ",num_segments=" << metadata.linked_fatbin_segments.size();
        }

        packed_txt << "\n";
        packed_txt << metadata.entrypoint_names.size() << "\n";
        for (const auto& name : metadata.entrypoint_names) {
            packed_txt << name << "\n";
        }
        packed_txt << "---\n";
    });
}

static void pack_fatbins_on_exit() {
    if (!pack_fatbins_on_exit_enabled.load()) {
        return;
    }
    pack_fatbins_to_folder_impl(fs::path("hook_archive"));
}

static void* get_real_dlsym() {
    if (!real_dlsym) {
#if defined(__x86_64__)
        real_dlsym = (fp_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
#elif defined(__i386__)
        real_dlsym = (fp_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");
#elif defined(__aarch64__)
        real_dlsym = (fp_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
#else
        // Fallback: try common versions
        real_dlsym = (fp_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
        if (!real_dlsym) {
            real_dlsym = (fp_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        }
#endif
        if (!real_dlsym) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to get real dlsym\n");
            abort();
        }
    }
    return (void*)real_dlsym;
}

static const char* prior_function(const char* symbol) {
    static char buf[256];

    size_t len = strlen(symbol);
    if (len >= 3 && strcmp(symbol + len - 3, "_v3") == 0) {
        size_t copy_len = len - 3;
        memcpy(buf, symbol, copy_len);
        buf[copy_len] = '\0';
        strcat(buf, "_v2");
        return buf;
    } else if (len >= 3 && strcmp(symbol + len - 3, "_v2") == 0) {
        size_t copy_len = len - 3;
        memcpy(buf, symbol, copy_len);
        buf[copy_len] = '\0';
        return buf;
    }
    return nullptr;
}

static void* try_load_symbol(void* handle, const char* symbol) {
    void* ptr = real_dlsym(handle, symbol);
    if (ptr) return ptr;

    ptr = real_dlsym(RTLD_NEXT, symbol);
    if (ptr) return ptr;

    char tmpfunc[256];
    strncpy(tmpfunc, symbol, sizeof(tmpfunc) - 1);
    tmpfunc[sizeof(tmpfunc) - 1] = '\0';

    const char* prior = prior_function(tmpfunc);
    while (prior) {
        ptr = real_dlsym(RTLD_NEXT, prior);
        if (ptr) return ptr;
        strncpy(tmpfunc, prior, sizeof(tmpfunc) - 1);
        tmpfunc[sizeof(tmpfunc) - 1] = '\0';
        prior = prior_function(tmpfunc);
    }

    return nullptr;
}

static void load_cuda_library() {
    get_real_dlsym();

    void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NODELETE);
    if (!handle) {
        handle = dlopen("libcuda.so", RTLD_NOW | RTLD_NODELETE);
    }

    if (!handle) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Failed to load CUDA library\n");
        abort();
    }

    for (int i = 0; cuda_driver_entry_table[i].name != nullptr; i++) {
        cuda_driver_entry_table[i].fn_ptr = try_load_symbol(handle, cuda_driver_entry_table[i].name);
        if (!cuda_driver_entry_table[i].fn_ptr) {
            // cuLibraryGetGlobal is optional (CUDA 12.0+), don't abort if not found
            if (i == CUDA_ENTRY_cuLibraryGetGlobal) {
#ifdef HOOK_DEBUG
                fprintf(stderr, "[HOOK] DEBUG: Optional symbol %s not found (may require CUDA 12.0+)\n",
                        cuda_driver_entry_table[i].name);
#endif
                continue;
            }
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to load symbol %s\n", cuda_driver_entry_table[i].name);
            abort();
        }
    }

    dlclose(handle);
}

static void __attribute__((constructor)) init_hook() {
    load_cuda_library();

    // Check for CGE_MODE environment variable to enable early skip of fatbin processing
    // This allows skipping fatbin processing before Python code can call set_skip_fatbin_processing()
    const char* cge_mode = std::getenv("CGE_MODE");
    if (cge_mode && std::strcmp(cge_mode, "load") == 0) {
        skip_fatbin_processing.store(true);
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] CGE_MODE=load detected, skipping fatbin processing\n");
#endif
    }
}

static void __attribute__((destructor)) cleanup_hook() {
    pack_fatbins_on_exit();
}

extern "C" {

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuModuleLoadData\n");
#endif

    typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
    auto real_func = (cuModuleLoadData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadData);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(module, image);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_and_info(image, "cuModuleLoadData", 0, nullptr, nullptr, 0, nullptr, nullptr, idx, &hash);

    CUresult res = real_func(module, image);

    if (res == CUDA_SUCCESS) {
        if (!module) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Module pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*module, hash);
        module_or_library_handle_to_hash.insert_or_assign(*module, hash);
    }

    return res;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                             CUjit_option *options, void **optionValues) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuModuleLoadDataEx\n");
#endif

    typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
    auto real_func = (cuModuleLoadDataEx_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadDataEx);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(module, image, numOptions, options, optionValues);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_and_info(image, "cuModuleLoadDataEx", numOptions, options, optionValues, 0, nullptr, nullptr, idx, &hash);

    CUresult res = real_func(module, image, numOptions, options, optionValues);

    if (res == CUDA_SUCCESS) {
        if (!module) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Module pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*module, hash);
        module_or_library_handle_to_hash.insert_or_assign(*module, hash);
    }

    return res;
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuModuleLoadFatBinary\n");
#endif

    typedef CUresult (*cuModuleLoadFatBinary_t)(CUmodule*, const void*);
    auto real_func = (cuModuleLoadFatBinary_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadFatBinary);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(module, fatCubin);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_and_info(fatCubin, "cuModuleLoadFatBinary", 0, nullptr, nullptr, 0, nullptr, nullptr, idx, &hash);

    CUresult res = real_func(module, fatCubin);

    if (res == CUDA_SUCCESS) {
        if (!module) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Module pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*module, hash);
        module_or_library_handle_to_hash.insert_or_assign(*module, hash);
    }

    return res;
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions,
                            void **jitOptionsValues, unsigned int numJitOptions,
                            CUlibraryOption *libraryOptions, void **libraryOptionValues,
                            unsigned int numLibraryOptions) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuLibraryLoadData\n");
#endif

    typedef CUresult (*cuLibraryLoadData_t)(CUlibrary*, const void*, CUjit_option*, void**,
                                             unsigned int, CUlibraryOption*, void**, unsigned int);
    auto real_func = (cuLibraryLoadData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryLoadData);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(library, code, jitOptions, jitOptionsValues, numJitOptions,
                         libraryOptions, libraryOptionValues, numLibraryOptions);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_and_info(code, "cuLibraryLoadData", numJitOptions, jitOptions, jitOptionsValues, numLibraryOptions, libraryOptions, libraryOptionValues, idx, &hash);

    CUresult res = real_func(library, code, jitOptions, jitOptionsValues, numJitOptions,
                     libraryOptions, libraryOptionValues, numLibraryOptions);

    if (res == CUDA_SUCCESS) {
        if (!library) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Library pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*library, hash);
        module_or_library_handle_to_hash.insert_or_assign(*library, hash);
    }

    return res;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuModuleLoad\n");
#endif

    typedef CUresult (*cuModuleLoad_t)(CUmodule*, const char*);
    auto real_func = (cuModuleLoad_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoad);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(module, fname);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_from_file_and_info(fname, "cuModuleLoad", 0, nullptr, nullptr, 0, nullptr, nullptr, idx, &hash);

    CUresult res = real_func(module, fname);

    if (res == CUDA_SUCCESS) {
        if (!module) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Module pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*module, hash);
        module_or_library_handle_to_hash.insert_or_assign(*module, hash);
    }

    return res;
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
                                void **jitOptionsValues, unsigned int numJitOptions,
                                CUlibraryOption *libraryOptions, void **libraryOptionValues,
                                unsigned int numLibraryOptions) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuLibraryLoadFromFile\n");
#endif

    typedef CUresult (*cuLibraryLoadFromFile_t)(CUlibrary*, const char*, CUjit_option*, void**,
                                                 unsigned int, CUlibraryOption*, void**, unsigned int);
    auto real_func = (cuLibraryLoadFromFile_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryLoadFromFile);

    // Skip heavy processing in LOAD mode for faster startup
    if (skip_fatbin_processing.load()) {
        return real_func(library, fileName, jitOptions, jitOptionsValues, numJitOptions,
                         libraryOptions, libraryOptionValues, numLibraryOptions);
    }

    const int idx = dumped_binary_counter.fetch_add(1);
    uint64_t hash = 0;
    dump_fatbin_from_file_and_info(fileName, "cuLibraryLoadFromFile", numJitOptions, jitOptions, jitOptionsValues, numLibraryOptions, libraryOptions, libraryOptionValues, idx, &hash);

    CUresult res = real_func(library, fileName, jitOptions, jitOptionsValues, numJitOptions,
                     libraryOptions, libraryOptionValues, numLibraryOptions);

    if (res == CUDA_SUCCESS) {
        if (!library) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Library pointer is null after successful load\n");
            abort();
        }
        if (hash == 0) {
            fprintf(stderr, "[HOOK] FATAL ERROR: Failed to compute hash for binary\n");
            abort();
        }
        dump_module_or_library_entrypoints(*library, hash);
        module_or_library_handle_to_hash.insert_or_assign(*library, hash);
    }

    return res;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuCtxCreate\n");
#endif

    typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
    auto real_func = (cuCtxCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxCreate);

    CUresult res = real_func(pctx, flags, dev);

    if (res == CUDA_SUCCESS) {
        std::call_once(default_allocation_region_flag, []() {
            foundry::set_allocation_region((void*)kAllocDefaultRegionBase, kAllocDefaultRegionSize);
        });
    }

    return res;
}

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuCtxCreate_v2\n");
#endif

    typedef CUresult (*cuCtxCreate_v2_t)(CUcontext*, unsigned int, CUdevice);
    auto real_func = (cuCtxCreate_v2_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxCreate_v2);

    CUresult res = real_func(pctx, flags, dev);

    if (res == CUDA_SUCCESS) {
        std::call_once(default_allocation_region_flag, []() {
            foundry::set_allocation_region((void*)kAllocDefaultRegionBase, kAllocDefaultRegionSize);
        });
    }

    return res;
}

CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuCtxCreate_v3\n");
#endif

    typedef CUresult (*cuCtxCreate_v3_t)(CUcontext*, CUexecAffinityParam*, int, unsigned int, CUdevice);
    auto real_func = (cuCtxCreate_v3_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxCreate_v3);

    CUresult res = real_func(pctx, paramsArray, numParams, flags, dev);

    if (res == CUDA_SUCCESS) {
        std::call_once(default_allocation_region_flag, []() {
            foundry::set_allocation_region((void*)kAllocDefaultRegionBase, kAllocDefaultRegionSize);
        });
    }

    return res;
}

CUresult cuCtxCreate_v4(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev) {
#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuCtxCreate_v4\n");
#endif

    typedef CUresult (*cuCtxCreate_v4_t)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);
    auto real_func = (cuCtxCreate_v4_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxCreate_v4);

    CUresult res = real_func(pctx, ctxCreateParams, flags, dev);

    if (res == CUDA_SUCCESS) {
        std::call_once(default_allocation_region_flag, []() {
            foundry::set_allocation_region((void*)kAllocDefaultRegionBase, kAllocDefaultRegionSize);
        });
    }

    return res;
}

static void* find_symbol_by_cuda_version(const char* symbol, int cudaVersion) {
    if (strstr(symbol, "_v2") != nullptr || strstr(symbol, "_v3") != nullptr || strstr(symbol, "_v4") != nullptr) {
        fprintf(stderr, "[HOOK] FATAL ERROR: Symbol %s contains version suffix. Version selection is automatic based on CUDA version %d\n",
                symbol, cudaVersion);
        abort();
    }

    if (strcmp(symbol, "cuModuleLoadData") == 0) {
        return (void*)cuModuleLoadData;
    } else if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {
        return (void*)cuModuleLoadDataEx;
    } else if (strcmp(symbol, "cuModuleLoadFatBinary") == 0) {
        return (void*)cuModuleLoadFatBinary;
    } else if (strcmp(symbol, "cuLibraryLoadData") == 0) {
        return (void*)cuLibraryLoadData;
    } else if (strcmp(symbol, "cuModuleLoad") == 0) {
        return (void*)cuModuleLoad;
    } else if (strcmp(symbol, "cuLibraryLoadFromFile") == 0) {
        return (void*)cuLibraryLoadFromFile;
    } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
        if (cudaVersion >= 12000) {
            return (void*)cuGetProcAddress_v2;
        } else if (cudaVersion >= 11030) {
            return (void*)cuGetProcAddress;
        }
        return nullptr;
    } else if (strcmp(symbol, "cuCtxCreate") == 0) {
        if (cudaVersion >= 12050) {
            return (void*)cuCtxCreate_v4;
        } else if (cudaVersion >= 11040) {
            return (void*)cuCtxCreate_v3;
        } else if (cudaVersion >= 3020) {
            return (void*)cuCtxCreate_v2;
        } else if (cudaVersion >= 2000) {
            return (void*)cuCtxCreate;
        }
        return nullptr;
    } else if (strcmp(symbol, "cuMemAlloc") == 0) {
        return (void*)cuMemAlloc_v2;
    } else if (strcmp(symbol, "cuMemAllocPitch") == 0) {
        return (void*)cuMemAllocPitch_v2;
    } else if (strcmp(symbol, "cuMemFree") == 0) {
        return (void*)cuMemFree_v2;
    } else if (strcmp(symbol, "cuMemAddressReserve") == 0) {
        return (void*)cuMemAddressReserve;
    } else if (strcmp(symbol, "cuMemAddressFree") == 0) {
        return (void*)cuMemAddressFree;
    } else if (strcmp(symbol, "cuIpcGetMemHandle") == 0) {
        return (void*)cuIpcGetMemHandle;
    } else if (strcmp(symbol, "cuIpcOpenMemHandle") == 0) {
        return (void*)cuIpcOpenMemHandle;
    } else if (strcmp(symbol, "cuIpcCloseMemHandle") == 0) {
        return (void*)cuIpcCloseMemHandle;
    }

    return nullptr;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
    typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, cuuint64_t);
    auto real_func = (cuGetProcAddress_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuGetProcAddress);

    if (symbol) {
        void* hooked_fn = find_symbol_by_cuda_version(symbol, cudaVersion);
        if (hooked_fn) {
            *pfn = hooked_fn;
            return CUDA_SUCCESS;
        }
    }

    return real_func(symbol, pfn, cudaVersion, flags);
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags,
                              CUdriverProcAddressQueryResult* symbolStatus) {
    typedef CUresult (*cuGetProcAddress_v2_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    auto real_func = (cuGetProcAddress_v2_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuGetProcAddress_v2);

    if (symbol) {
        void* hooked_fn = find_symbol_by_cuda_version(symbol, cudaVersion);
        if (hooked_fn) {
            *pfn = hooked_fn;
            if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
            return CUDA_SUCCESS;
        }
    }

    return real_func(symbol, pfn, cudaVersion, flags, symbolStatus);
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
    auto real_func = (cuMemAlloc_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAlloc_v2);

    if (!tls_storage.enabled || !tls_storage.region_initialized) {
        return real_func(dptr, bytesize);
    }

    // Use cached device and granularity to avoid driver calls on every allocation
    CUdevice device;
    size_t granularity;
    if (tls_storage.device_cached) {
        device = tls_storage.cached_device;
        granularity = tls_storage.cached_granularity;
    } else {
        typedef CUresult (*cuCtxGetDevice_t)(CUdevice*);
        auto get_device_func = (cuCtxGetDevice_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxGetDevice);
        get_device_func(&device);
        granularity = get_allocation_granularity(device);
        tls_storage.cached_device = device;
        tls_storage.cached_granularity = granularity;
        tls_storage.device_cached = true;
    }

    size_t aligned_size = align_to(bytesize, granularity);

    CUdeviceptr target_addr = align_to(tls_storage.current_alloc_base_addr, kAllocAlignment);
    CUdeviceptr region_base_addr = (CUdeviceptr)tls_storage.region.base;
    CUdeviceptr region_end_addr = region_base_addr + tls_storage.region.size;

    // Fast path: check if we can serve from preallocated memory
    if (tls_storage.has_preallocation && (target_addr + aligned_size) <= tls_storage.preallocated_end_addr) {
        *dptr = target_addr;

        AllocMetadata metadata;
        metadata.ptr = *dptr;
        metadata.size = aligned_size;
        metadata.handle = 0;  // No individual handle, part of preallocated chunk
        metadata.region_base = region_base_addr;
        metadata.from_preallocation = true;

        global_alloc_metadata.emplace(*dptr, metadata);
        tls_storage.current_alloc_base_addr = align_to(*dptr + bytesize, kAllocAlignment);

        if (hook_recording_enabled.load()) {
            std::lock_guard<std::mutex> lock(hook_events_mutex);
            HookAllocationEvent event;
            event.type = HookAllocationEvent::Type::Alloc;
            event.size = bytesize;
            event.ptr = *dptr;
            hook_alloc_events.push_back(event);
        }

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemAlloc_v2 (FAST PATH) ptr=%llu size=%zu aligned_size=%zu\n",
                (unsigned long long)*dptr, bytesize, aligned_size);
#endif

        return CUDA_SUCCESS;
    }

    // Slow path: normal VMM allocation

    // Idempotent check: if this address was already allocated (e.g. from replay
    // of comm buffer events), skip physical allocation and reuse existing mapping.
    {
        bool already_allocated = false;
        global_alloc_metadata.visit(target_addr, [&](const auto&) {
            already_allocated = true;
        });
        if (already_allocated) {
            *dptr = target_addr;
            tls_storage.current_alloc_base_addr = align_to(*dptr + bytesize, kAllocAlignment);

            if (hook_recording_enabled.load()) {
                std::lock_guard<std::mutex> lock(hook_events_mutex);
                HookAllocationEvent event;
                event.type = HookAllocationEvent::Type::Alloc;
                event.size = bytesize;
                event.ptr = *dptr;
                hook_alloc_events.push_back(event);
            }

            fprintf(stderr, "[HOOK] cuMemAlloc_v2 (IDEMPOTENT) reusing pre-mapped 0x%llx size=%zu\n",
                    (unsigned long long)target_addr, bytesize);
            return CUDA_SUCCESS;
        }
    }

    typedef CUresult (*cuMemCreate_t)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    auto mem_create_func = (cuMemCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemCreate);

    typedef CUresult (*cuMemMap_t)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    auto mem_map_func = (cuMemMap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemMap);

    typedef CUresult (*cuMemSetAccess_t)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    auto mem_set_access_func = (cuMemSetAccess_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemSetAccess);

    CUmemGenericAllocationHandle allocHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    // Enable IPC via VMM shareable handles (POSIX file descriptor on Linux)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUresult result = mem_create_func(&allocHandle, aligned_size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemCreate failed with error %d for size=%zu\n", result, aligned_size);
        return real_func(dptr, bytesize);
    }

    result = mem_map_func(target_addr, aligned_size, 0, allocHandle, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemMap failed with error %d at cuMemAlloc_v2 "
                "addr=0x%llx size=%zu (requested=%zu) alloc_base=0x%llx has_prealloc=%d "
                "prealloc_start=0x%llx prealloc_end=0x%llx\n",
                result,
                (unsigned long long)target_addr, aligned_size, bytesize,
                (unsigned long long)tls_storage.current_alloc_base_addr,
                (int)tls_storage.has_preallocation,
                (unsigned long long)tls_storage.preallocated_start_addr,
                (unsigned long long)tls_storage.preallocated_end_addr);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_release_func(allocHandle);
        return real_func(dptr, bytesize);
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    result = mem_set_access_func(target_addr, aligned_size, &accessDesc, 1);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemSetAccess failed with error %d\n", result);
        typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
        auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_unmap_func(target_addr, aligned_size);
        mem_release_func(allocHandle);
        return real_func(dptr, bytesize);
    }

    *dptr = target_addr;

    if (*dptr < region_base_addr || (*dptr + bytesize) > region_end_addr) {
        fprintf(stderr, "[HOOK] ERROR: Allocated address [%llu, %llu) is outside allocation region [%llu, %llu)\n",
                (unsigned long long)*dptr, (unsigned long long)(*dptr + bytesize),
                (unsigned long long)region_base_addr, (unsigned long long)region_end_addr);
    }

    if (*dptr != target_addr) {
        fprintf(stderr, "[HOOK] WARNING: Allocated address %llu != requested address %llu\n",
                (unsigned long long)*dptr, (unsigned long long)target_addr);
    }

    AllocMetadata metadata;
    metadata.ptr = *dptr;
    metadata.size = aligned_size;
    metadata.handle = allocHandle;
    metadata.region_base = region_base_addr;
    metadata.from_preallocation = false;

    global_alloc_metadata.emplace(*dptr, metadata);
    tls_storage.current_alloc_base_addr = align_to(*dptr + bytesize, kAllocAlignment);

    if (hook_recording_enabled.load()) {
        std::lock_guard<std::mutex> lock(hook_events_mutex);
        HookAllocationEvent event;
        event.type = HookAllocationEvent::Type::Alloc;
        event.size = bytesize;
        event.ptr = *dptr;
        hook_alloc_events.push_back(event);
    }

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuMemAlloc_v2 (SLOW PATH) ptr=%llu size=%zu aligned_size=%zu\n",
            (unsigned long long)*dptr, bytesize, aligned_size);
#endif

    return CUDA_SUCCESS;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    typedef CUresult (*cuMemAllocPitch_t)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int);
    auto real_func = (cuMemAllocPitch_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAllocPitch_v2);

    if (!tls_storage.enabled || !tls_storage.region_initialized) {
        return real_func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    }

    // First, call real function to get the pitch value, then free immediately
    CUdeviceptr temp_ptr = 0;
    size_t pitch = 0;
    CUresult result = real_func(&temp_ptr, &pitch, WidthInBytes, Height, ElementSizeBytes);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    typedef CUresult (*cuMemFree_t)(CUdeviceptr);
    auto mem_free_func = (cuMemFree_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemFree_v2);
    if (temp_ptr != 0) {
        mem_free_func(temp_ptr);
    }

    size_t total_size = pitch * Height;

    // Use cached device and granularity to avoid driver calls on every allocation
    CUdevice device;
    size_t granularity;
    if (tls_storage.device_cached) {
        device = tls_storage.cached_device;
        granularity = tls_storage.cached_granularity;
    } else {
        typedef CUresult (*cuCtxGetDevice_t)(CUdevice*);
        auto get_device_func = (cuCtxGetDevice_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxGetDevice);
        get_device_func(&device);
        granularity = get_allocation_granularity(device);
        tls_storage.cached_device = device;
        tls_storage.cached_granularity = granularity;
        tls_storage.device_cached = true;
    }

    size_t aligned_size = align_to(total_size, granularity);

    CUdeviceptr target_addr = align_to(tls_storage.current_alloc_base_addr, kAllocAlignment);
    CUdeviceptr region_base_addr = (CUdeviceptr)tls_storage.region.base;
    CUdeviceptr region_end_addr = region_base_addr + tls_storage.region.size;

    // Fast path: check if we can serve from preallocated memory
    if (tls_storage.has_preallocation && (target_addr + aligned_size) <= tls_storage.preallocated_end_addr) {
        *dptr = target_addr;
        *pPitch = pitch;

        AllocMetadata metadata;
        metadata.ptr = *dptr;
        metadata.size = aligned_size;
        metadata.handle = 0;  // No individual handle, part of preallocated chunk
        metadata.region_base = region_base_addr;
        metadata.from_preallocation = true;

        global_alloc_metadata.emplace(*dptr, metadata);
        tls_storage.current_alloc_base_addr = align_to(*dptr + total_size, kAllocAlignment);

        if (hook_recording_enabled.load()) {
            std::lock_guard<std::mutex> lock(hook_events_mutex);
            HookAllocationEvent event;
            event.type = HookAllocationEvent::Type::Alloc;
            event.size = total_size;
            event.ptr = *dptr;
            hook_alloc_events.push_back(event);
        }

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemAllocPitch_v2 (FAST PATH) ptr=%llu pitch=%zu size=%zu aligned_size=%zu\n",
                (unsigned long long)*dptr, pitch, total_size, aligned_size);
#endif

        return CUDA_SUCCESS;
    }

    // Slow path: normal VMM allocation
    typedef CUresult (*cuMemCreate_t)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    auto mem_create_func = (cuMemCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemCreate);

    typedef CUresult (*cuMemMap_t)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    auto mem_map_func = (cuMemMap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemMap);

    typedef CUresult (*cuMemSetAccess_t)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    auto mem_set_access_func = (cuMemSetAccess_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemSetAccess);

    CUmemGenericAllocationHandle allocHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    result = mem_create_func(&allocHandle, aligned_size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemCreate failed with error %d\n", result);
        return real_func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    }

    result = mem_map_func(target_addr, aligned_size, 0, allocHandle, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemMap failed with error %d at cuMemAllocPitch_v2\n", result);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_release_func(allocHandle);
        return real_func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    result = mem_set_access_func(target_addr, aligned_size, &accessDesc, 1);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemSetAccess failed with error %d\n", result);
        typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
        auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_unmap_func(target_addr, aligned_size);
        mem_release_func(allocHandle);
        return real_func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    }

    *dptr = target_addr;
    *pPitch = pitch;

    if (*dptr < region_base_addr || (*dptr + total_size) > region_end_addr) {
        fprintf(stderr, "[HOOK] ERROR: Allocated address [%llu, %llu) is outside allocation region [%llu, %llu)\n",
                (unsigned long long)*dptr, (unsigned long long)(*dptr + total_size),
                (unsigned long long)region_base_addr, (unsigned long long)region_end_addr);
    }

    if (*dptr != target_addr) {
        fprintf(stderr, "[HOOK] WARNING: Allocated address %llu != requested address %llu\n",
                (unsigned long long)*dptr, (unsigned long long)target_addr);
    }

    AllocMetadata metadata;
    metadata.ptr = *dptr;
    metadata.size = aligned_size;
    metadata.handle = allocHandle;
    metadata.region_base = region_base_addr;
    metadata.from_preallocation = false;

    global_alloc_metadata.emplace(*dptr, metadata);
    tls_storage.current_alloc_base_addr = align_to(*dptr + total_size, kAllocAlignment);

    if (hook_recording_enabled.load()) {
        std::lock_guard<std::mutex> lock(hook_events_mutex);
        HookAllocationEvent event;
        event.type = HookAllocationEvent::Type::Alloc;
        event.size = total_size;
        event.ptr = *dptr;
        hook_alloc_events.push_back(event);
    }

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuMemAllocPitch_v2 (SLOW PATH) ptr=%llu pitch=%zu size=%zu aligned_size=%zu\n",
            (unsigned long long)*dptr, pitch, total_size, aligned_size);
#endif

    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    typedef CUresult (*cuMemFree_t)(CUdeviceptr);
    auto real_func = (cuMemFree_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemFree_v2);

    bool found = false;
    AllocMetadata metadata;

    size_t visit_count = global_alloc_metadata.visit(dptr, [&](std::pair<const CUdeviceptr, AllocMetadata>& kv) {
        found = true;
        metadata = kv.second;
    });

    if (found) {
        // For preallocated memory, we just remove tracking but don't actually free
        // The memory will be freed when free_preallocated_region() is called
        if (!metadata.from_preallocation) {
            typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
            auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);

            typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
            auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);

            typedef CUresult (*cuMemAddressFree_t)(CUdeviceptr, size_t);
            auto mem_addr_free_func = (cuMemAddressFree_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAddressFree);

            mem_unmap_func(metadata.ptr, metadata.size);
            mem_release_func(metadata.handle);
            mem_addr_free_func(metadata.ptr, metadata.size);
        }

        global_alloc_metadata.erase_if([dptr](const std::pair<const CUdeviceptr, AllocMetadata>& kv) {
            return kv.first == dptr;
        });

        if (hook_recording_enabled.load()) {
            std::lock_guard<std::mutex> lock(hook_events_mutex);
            HookAllocationEvent event;
            event.type = HookAllocationEvent::Type::Free;
            event.size = 0;
            event.ptr = dptr;
            hook_alloc_events.push_back(event);
        }

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemFree_v2 ptr=%llu size=%zu (%s)\n",
                (unsigned long long)dptr, metadata.size,
                metadata.from_preallocation ? "PREALLOCATED" : "VMM");
#endif

        return CUDA_SUCCESS;
    }

    return real_func(dptr);
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    typedef CUresult (*cuMemAddressReserve_t)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    auto real_func = (cuMemAddressReserve_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAddressReserve);

    if (!tls_storage.enabled || !tls_storage.region_initialized) {
        return real_func(ptr, size, alignment, addr, flags);
    }

    if (addr != 0) {
        CUresult result = real_func(ptr, size, alignment, addr, flags);
#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemAddressReserve ptr=0x%llx size=0x%zx alignment=0x%zx addr=0x%llx (user-specified)\n",
                (unsigned long long)*ptr, size, alignment, (unsigned long long)addr);
#endif
        return result;
    }

    constexpr size_t kSmallAllocThreshold = 2ULL << 30;
    if (size < kSmallAllocThreshold) {
        CUdeviceptr aligned_addr = align_to(tls_storage.current_alloc_base_addr, alignment);

        size_t region_base = (size_t)tls_storage.region.base;
        size_t region_end = region_base + tls_storage.region.size;

        if (aligned_addr + size > region_end) {
            fprintf(stderr, "[HOOK] ERROR: Not enough space in reserved region for allocation (size=0x%zx, available=0x%llx)\n",
                    size, region_end - aligned_addr);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        *ptr = aligned_addr;
        tls_storage.current_alloc_base_addr = align_to(aligned_addr + size, kAllocAlignment);

        global_carved_reserve_metadata.emplace(*ptr, size);

        if (hook_recording_enabled.load()) {
            std::lock_guard<std::mutex> lock(hook_events_mutex);
            HookAllocationEvent event;
            event.type = HookAllocationEvent::Type::Reserve;
            event.size = size;
            event.ptr = *ptr;
            event.alignment = alignment;
            hook_alloc_events.push_back(event);
        }

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemAddressReserve ptr=0x%llx size=0x%zx alignment=0x%zx (carved from reserved region), next_alloc_base=0x%llx\n",
                (unsigned long long)*ptr, size, alignment, (unsigned long long)tls_storage.current_alloc_base_addr);
#endif
        return CUDA_SUCCESS;
    }

    // constexpr size_t kReserveAlignment = 2ULL << 37; // FIXME(liuxs): This value should be equal to the first reservation size 
    CUdeviceptr hint_addr = align_to(tls_storage.current_vmm_reserve_addr, alignment);
    CUresult result = real_func(ptr, size, alignment, hint_addr, flags);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemAddressReserve failed with error 0x%x\n", result);
        return result;
    }

    if (*ptr != hint_addr) {
        fprintf(stderr, "[HOOK] WARNING: cuMemAddressReserve returned address 0x%llx != hint address 0x%llx (alignment=0x%zx, size=0x%zx, current_vmm_reserve=0x%llx)\n",
                (unsigned long long)*ptr, (unsigned long long)hint_addr, alignment, size,
                (unsigned long long)tls_storage.current_vmm_reserve_addr);
    }

    tls_storage.current_vmm_reserve_addr = align_to(*ptr + size, alignment);

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuMemAddressReserve ptr=0x%llx size=0x%zx alignment=0x%zx (large alloc, VMM hint), next_vmm_reserve=0x%llx\n",
            (unsigned long long)*ptr, size, alignment, (unsigned long long)tls_storage.current_vmm_reserve_addr);
#endif

    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    typedef CUresult (*cuMemAddressFree_t)(CUdeviceptr, size_t);
    auto real_func = (cuMemAddressFree_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAddressFree);

    bool is_carved = false;
    global_carved_reserve_metadata.visit(ptr, [&is_carved](const auto&) {
        is_carved = true;
    });

    if (is_carved) {
        global_carved_reserve_metadata.erase_if([ptr](const std::pair<const CUdeviceptr, size_t>& kv) {
            return kv.first == ptr;
        });

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuMemAddressFree ptr=0x%llx size=0x%zx (carved reservation, skipped real free)\n",
                (unsigned long long)ptr, size);
#endif
        return CUDA_SUCCESS;
    }

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] cuMemAddressFree ptr=0x%llx size=0x%zx (real reservation, forwarding to driver)\n",
            (unsigned long long)ptr, size);
#endif

    return real_func(ptr, size);
}

void* dlsym(void* handle, const char* symbol) {
    if (!real_dlsym) {
        get_real_dlsym();
    }
    if (symbol && strncmp(symbol, "cu", 2) == 0) {
        if (strcmp(symbol, "cuModuleLoadData") == 0) {
            return (void*)cuModuleLoadData;
        } else if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {
            return (void*)cuModuleLoadDataEx;
        } else if (strcmp(symbol, "cuModuleLoadFatBinary") == 0) {
            return (void*)cuModuleLoadFatBinary;
        } else if (strcmp(symbol, "cuModuleLoad") == 0) {
            return (void*)cuModuleLoad;
        } else if (strcmp(symbol, "cuLibraryLoadData") == 0) {
            return (void*)cuLibraryLoadData;
        } else if (strcmp(symbol, "cuLibraryLoadFromFile") == 0) {
            return (void*)cuLibraryLoadFromFile;
        } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
            return (void*)cuGetProcAddress;
        } else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
            return (void*)cuGetProcAddress_v2;
        } else if (strcmp(symbol, "cuCtxCreate") == 0) {
            return (void*)cuCtxCreate;
        } else if (strcmp(symbol, "cuCtxCreate_v2") == 0) {
            return (void*)cuCtxCreate_v2;
        } else if (strcmp(symbol, "cuCtxCreate_v3") == 0) {
            return (void*)cuCtxCreate_v3;
        } else if (strcmp(symbol, "cuCtxCreate_v4") == 0) {
            return (void*)cuCtxCreate_v4;
        } else if (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) {
            return (void*)cuMemAlloc_v2;
        } else if (strcmp(symbol, "cuMemAllocPitch") == 0 || strcmp(symbol, "cuMemAllocPitch_v2") == 0) {
            return (void*)cuMemAllocPitch_v2;
        } else if (strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0) {
            return (void*)cuMemFree_v2;
        } else if (strcmp(symbol, "cuMemAddressReserve") == 0) {
            return (void*)cuMemAddressReserve;
        } else if (strcmp(symbol, "cuMemAddressFree") == 0) {
            return (void*)cuMemAddressFree;
        } else if (strcmp(symbol, "cuIpcGetMemHandle") == 0) {
            return (void*)cuIpcGetMemHandle;
        } else if (strcmp(symbol, "cuIpcOpenMemHandle") == 0) {
            return (void*)cuIpcOpenMemHandle;
        } else if (strcmp(symbol, "cuIpcCloseMemHandle") == 0) {
            return (void*)cuIpcCloseMemHandle;
        }
    }

    return real_dlsym(handle, symbol);
}

// =============================================================================
// VMM IPC Support - Hook cuIpcGetMemHandle/cuIpcOpenMemHandle to work with
// VMM allocations by translating to cuMemExportToShareableHandle API
// =============================================================================

// Magic marker to identify VMM IPC handles in CUipcMemHandle
static constexpr uint32_t VMM_IPC_MAGIC = 0x564D4D49; // "VMMI"

// Hook for cuIpcGetMemHandle - intercept Driver API to support VMM allocations
CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
    typedef CUresult (*cuIpcGetMemHandle_t)(CUipcMemHandle*, CUdeviceptr);
    auto real_func = (cuIpcGetMemHandle_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuIpcGetMemHandle);

    // Try to find this pointer in our VMM allocations
    AllocMetadata metadata;
    bool found = false;

    global_alloc_metadata.visit(dptr, [&](const std::pair<const CUdeviceptr, AllocMetadata>& kv) {
        found = true;
        metadata = kv.second;
    });

    if (found && metadata.handle != 0) {
        // This is a VMM allocation - export via shareable handle
        typedef CUresult (*cuMemExportToShareableHandle_t)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
        auto export_func = (cuMemExportToShareableHandle_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemExportToShareableHandle);

        if (export_func == nullptr) {
            fprintf(stderr, "[HOOK] ERROR: cuMemExportToShareableHandle not found\n");
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        int fd = -1;
        CUresult result = export_func(&fd, metadata.handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);

        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] ERROR: cuMemExportToShareableHandle failed with error %d for ptr=0x%llx\n",
                    result, (unsigned long long)dptr);
            return result;
        }

        // Pack our custom data into the handle structure
        // CUipcMemHandle has 64 reserved bytes - we use them to store our info:
        // - Magic marker (4 bytes) to identify VMM handles
        // - File descriptor (4 bytes)
        // - Original pointer address (8 bytes) - critical for CUDA graph replay!
        // - Size (8 bytes)
        memset(pHandle, 0, sizeof(CUipcMemHandle));

        memcpy(pHandle->reserved, &VMM_IPC_MAGIC, sizeof(uint32_t));
        memcpy(pHandle->reserved + 4, &fd, sizeof(int));
        memcpy(pHandle->reserved + 8, &dptr, sizeof(CUdeviceptr));
        memcpy(pHandle->reserved + 16, &metadata.size, sizeof(size_t));

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuIpcGetMemHandle: VMM ptr=0x%llx, fd=%d, size=%zu\n",
                (unsigned long long)dptr, fd, metadata.size);
#endif
        return CUDA_SUCCESS;
    }

    // Not a VMM allocation - fall back to real function
    if (real_func) {
        return real_func(pHandle, dptr);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// Hook for cuIpcOpenMemHandle - import VMM shareable handle and map to original address
CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) {
    typedef CUresult (*cuIpcOpenMemHandle_t)(CUdeviceptr*, CUipcMemHandle, unsigned int);
    auto real_func = (cuIpcOpenMemHandle_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuIpcOpenMemHandle);

    // Check for our VMM magic marker
    uint32_t magic;
    memcpy(&magic, handle.reserved, sizeof(uint32_t));

    if (magic == VMM_IPC_MAGIC) {
        // This is a VMM IPC handle - extract the packed data
        int fd;
        CUdeviceptr original_ptr;
        size_t size;

        memcpy(&fd, handle.reserved + 4, sizeof(int));
        memcpy(&original_ptr, handle.reserved + 8, sizeof(CUdeviceptr));
        memcpy(&size, handle.reserved + 16, sizeof(size_t));

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuIpcOpenMemHandle: VMM fd=%d, original_ptr=0x%llx, size=%zu\n",
                fd, (unsigned long long)original_ptr, size);
#endif

        // Import the allocation handle from file descriptor
        typedef CUresult (*cuMemImportFromShareableHandle_t)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
        auto import_func = (cuMemImportFromShareableHandle_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemImportFromShareableHandle);

        if (import_func == nullptr) {
            fprintf(stderr, "[HOOK] ERROR: cuMemImportFromShareableHandle not found\n");
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        CUmemGenericAllocationHandle imported_handle;
        CUresult result = import_func(&imported_handle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] ERROR: cuMemImportFromShareableHandle failed with error %d\n", result);
            return result;
        }

        // Map to the SAME address as original (critical for CUDA graph replay!)
        typedef CUresult (*cuMemMap_t)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
        auto map_func = (cuMemMap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemMap);

        result = map_func(original_ptr, size, 0, imported_handle, 0);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] ERROR: cuMemMap for IPC failed with error %d at addr=0x%llx size=%zu\n",
                    result, (unsigned long long)original_ptr, size);
            return result;
        }

        // Set access permissions
        CUdevice device;
        typedef CUresult (*cuCtxGetDevice_t)(CUdevice*);
        auto get_device_func = (cuCtxGetDevice_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxGetDevice);
        get_device_func(&device);

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        typedef CUresult (*cuMemSetAccess_t)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
        auto set_access_func = (cuMemSetAccess_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemSetAccess);

        result = set_access_func(original_ptr, size, &accessDesc, 1);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[HOOK] ERROR: cuMemSetAccess for IPC failed with error %d\n", result);
            return result;
        }

        *pdptr = original_ptr;

        // Track this imported allocation
        AllocMetadata alloc_metadata;
        alloc_metadata.ptr = original_ptr;
        alloc_metadata.size = size;
        alloc_metadata.handle = imported_handle;
        alloc_metadata.region_base = 0;  // region_base == 0 indicates IPC-imported
        alloc_metadata.from_preallocation = false;
        global_alloc_metadata.emplace(original_ptr, alloc_metadata);

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuIpcOpenMemHandle: Successfully mapped VMM fd=%d -> ptr=0x%llx, size=%zu\n",
                fd, (unsigned long long)*pdptr, size);
#endif
        return CUDA_SUCCESS;
    }

    // Not a VMM IPC handle - fall back to real function
    if (real_func) {
        return real_func(pdptr, handle, Flags);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// Hook for cuIpcCloseMemHandle - clean up VMM IPC mappings
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    typedef CUresult (*cuIpcCloseMemHandle_t)(CUdeviceptr);
    auto real_func = (cuIpcCloseMemHandle_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuIpcCloseMemHandle);

    AllocMetadata metadata;
    bool found = false;

    global_alloc_metadata.visit(dptr, [&](const std::pair<const CUdeviceptr, AllocMetadata>& kv) {
        found = true;
        metadata = kv.second;
    });

    if (found && metadata.handle != 0 && metadata.region_base == 0) {
        // This is an IPC-imported VMM allocation (region_base == 0 indicates imported)
        typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
        auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);

        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);

        mem_unmap_func(dptr, metadata.size);
        mem_release_func(metadata.handle);

        global_alloc_metadata.erase_if([dptr](const std::pair<const CUdeviceptr, AllocMetadata>& kv) {
            return kv.first == dptr;
        });

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] cuIpcCloseMemHandle: Closed VMM IPC ptr=0x%llx\n", (unsigned long long)dptr);
#endif
        return CUDA_SUCCESS;
    }

    // Not a VMM IPC allocation - fall back to real function
    if (real_func) {
        return real_func(dptr);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

}

namespace foundry {

// Forward declarations
void free_preallocated_region();

void set_allocation_region(void* base, size_t size) {
    size_t aligned_base = align_to((size_t)base, kAllocAlignment);
    if (aligned_base != (size_t)base) {
        fprintf(stderr, "[HOOK] ERROR: Allocation region base %p is not aligned to %zu bytes, adjusted to %p\n",
                base, kAllocAlignment, (void*)aligned_base);
    }

    typedef CUresult (*cuMemAddressReserve_t)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    auto reserve_func = (cuMemAddressReserve_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAddressReserve);

    CUdeviceptr reserved_ptr = 0;
    CUresult result = reserve_func(&reserved_ptr, size, kAllocAlignment, aligned_base, 0);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: cuMemAddressReserve failed with error %d for base=%p size=%zu\n",
                result, (void*)aligned_base, size);
        fprintf(stderr, "[HOOK] ERROR: This may indicate the address range is still in use or not available\n");
        fprintf(stderr, "[HOOK] HINT: Try restarting the system or use a different base_addr\n");
        tls_storage.enabled = false;
        tls_storage.region_initialized = false;
        return;
    }

    if (reserved_ptr != aligned_base) {
        fprintf(stderr, "[HOOK] ERROR: Reserved address %llu != requested base %p, disabling allocation region\n",
                (unsigned long long)reserved_ptr, (void*)aligned_base);

        typedef CUresult (*cuMemAddressFree_t)(CUdeviceptr, size_t);
        auto free_func = (cuMemAddressFree_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemAddressFree);
        free_func(reserved_ptr, size);

        tls_storage.enabled = false;
        tls_storage.region_initialized = false;
        return;
    }

    tls_storage.region.base = (void*)aligned_base;
    tls_storage.region.size = size;
    tls_storage.current_alloc_base_addr = aligned_base;
    tls_storage.current_vmm_reserve_addr = align_to(aligned_base + size, kAllocAlignment);
    tls_storage.enabled = true;
    tls_storage.region_initialized = true;

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] Allocation region set: base=%p size=%zu, vmm_reserve_addr=%p\n",
            (void*)aligned_base, size, (void*)tls_storage.current_vmm_reserve_addr);
#endif
}

void stop_allocation_region() {
    tls_storage.enabled = false;

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] Allocation region stopped\n");
#endif
}

void resume_allocation_region() {
    tls_storage.enabled = true;

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] Allocation region resumed: base=%p size=%zu\n",
            tls_storage.region.base, tls_storage.region.size);
#endif
}

bool preallocate_region(size_t size) {
    if (!tls_storage.region_initialized) {
        fprintf(stderr, "[HOOK] ERROR: Cannot preallocate before allocation region is set\n");
        return false;
    }

    if (tls_storage.has_preallocation) {
        fprintf(stderr, "[HOOK] WARNING: Preallocation already exists, freeing previous allocation\n");
        free_preallocated_region();
    }

    // Get and cache device and granularity for subsequent allocations
    typedef CUresult (*cuCtxGetDevice_t)(CUdevice*);
    auto get_device_func = (cuCtxGetDevice_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuCtxGetDevice);
    CUdevice device = 0;
    get_device_func(&device);

    size_t granularity = get_allocation_granularity(device);

    // Cache for fast path allocations
    tls_storage.cached_device = device;
    tls_storage.cached_granularity = granularity;
    tls_storage.device_cached = true;

    size_t aligned_size = align_to(size, granularity);

    typedef CUresult (*cuMemCreate_t)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    auto mem_create_func = (cuMemCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemCreate);

    typedef CUresult (*cuMemMap_t)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    auto mem_map_func = (cuMemMap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemMap);

    typedef CUresult (*cuMemSetAccess_t)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    auto mem_set_access_func = (cuMemSetAccess_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemSetAccess);

    CUmemGenericAllocationHandle allocHandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    // Enable IPC via VMM shareable handles (POSIX file descriptor on Linux)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUresult result = mem_create_func(&allocHandle, aligned_size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: preallocate_region: cuMemCreate failed with error %d for size=%zu\n",
                result, aligned_size);
        return false;
    }

    CUdeviceptr target_addr = align_to(tls_storage.current_alloc_base_addr, kAllocAlignment);

    // Check if we have enough space in the reserved region
    CUdeviceptr region_end = (CUdeviceptr)tls_storage.region.base + tls_storage.region.size;
    if (target_addr + aligned_size > region_end) {
        fprintf(stderr, "[HOOK] ERROR: preallocate_region: Not enough space in region (need=%zu, available=%llu)\n",
                aligned_size, (unsigned long long)(region_end - target_addr));
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_release_func(allocHandle);
        return false;
    }

    result = mem_map_func(target_addr, aligned_size, 0, allocHandle, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: preallocate_region: cuMemMap failed with error %d at addr=0x%llx size=%zu\n",
                result, (unsigned long long)target_addr, aligned_size);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_release_func(allocHandle);
        return false;
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    result = mem_set_access_func(target_addr, aligned_size, &accessDesc, 1);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[HOOK] ERROR: preallocate_region: cuMemSetAccess failed with error %d\n", result);
        typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
        auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);
        typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
        auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);
        mem_unmap_func(target_addr, aligned_size);
        mem_release_func(allocHandle);
        return false;
    }

    // Store preallocation state
    tls_storage.preallocated_handle = allocHandle;
    tls_storage.preallocated_start_addr = target_addr;
    tls_storage.preallocated_end_addr = target_addr + aligned_size;
    tls_storage.has_preallocation = true;

    // Note: we do NOT advance current_alloc_base_addr here.
    // The alloc calls will advance it as they consume the preallocated memory.
    return true;
}

void free_preallocated_region() {
    if (!tls_storage.has_preallocation) {
        return;
    }

    typedef CUresult (*cuMemUnmap_t)(CUdeviceptr, size_t);
    auto mem_unmap_func = (cuMemUnmap_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemUnmap);

    typedef CUresult (*cuMemRelease_t)(CUmemGenericAllocationHandle);
    auto mem_release_func = (cuMemRelease_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuMemRelease);

    size_t preallocated_size = tls_storage.preallocated_end_addr - tls_storage.preallocated_start_addr;

    mem_unmap_func(tls_storage.preallocated_start_addr, preallocated_size);
    mem_release_func(tls_storage.preallocated_handle);

    tls_storage.preallocated_handle = 0;
    tls_storage.preallocated_start_addr = 0;
    tls_storage.preallocated_end_addr = 0;
    tls_storage.has_preallocation = false;
}

size_t get_current_alloc_offset() {
    if (!tls_storage.region_initialized) {
        return 0;
    }
    return tls_storage.current_alloc_base_addr - (size_t)tls_storage.region.base;
}

void set_current_alloc_offset(size_t offset) {
    if (!tls_storage.region_initialized) {
        fprintf(stderr, "[HOOK] ERROR: Cannot set offset before allocation region is initialized\n");
        return;
    }

    size_t new_alloc_addr = (size_t)tls_storage.region.base + offset;

    // Validate the offset is within region bounds
    size_t region_end = (size_t)tls_storage.region.base + tls_storage.region.size;
    if (new_alloc_addr > region_end) {
        fprintf(stderr, "[HOOK] ERROR: Offset 0x%llx extends beyond region (base=0x%llx, size=%zu)\n",
                (unsigned long long)offset, (unsigned long long)(size_t)tls_storage.region.base,
                tls_storage.region.size);
        return;
    }

    // Ensure the new offset is not less than the current offset
    // (we don't support moving backwards as it could corrupt existing allocations)
    if (new_alloc_addr < tls_storage.current_alloc_base_addr) {
        fprintf(stderr, "[HOOK] WARNING: New offset 0x%llx is less than current offset 0x%llx, skipping\n",
                (unsigned long long)offset,
                (unsigned long long)(tls_storage.current_alloc_base_addr - (size_t)tls_storage.region.base));
        return;
    }

    tls_storage.current_alloc_base_addr = new_alloc_addr;
    // NOTE: Do NOT update current_vmm_reserve_addr here.
    // current_alloc_base_addr tracks the cursor WITHIN the CGE region.
    // current_vmm_reserve_addr tracks where the NEXT cuMemAddressReserve
    // (outside the region) should hint — it was set to base + region_size
    // by set_allocation_region and must stay there.

#ifdef HOOK_DEBUG
    fprintf(stderr, "[HOOK] Set allocation offset: 0x%llx (absolute addr: 0x%llx)\n",
            (unsigned long long)offset, (unsigned long long)new_alloc_addr);
#endif
}

void start_hook_record() {
    hook_recording_start_base_addr = tls_storage.current_alloc_base_addr;
    hook_recording_enabled.store(true);
}

void end_hook_record() {
    hook_recording_enabled.store(false);
}

void clear_hook_events() {
    std::lock_guard<std::mutex> lock(hook_events_mutex);
    hook_alloc_events.clear();
}

boost::json::object save_hook_events_to_json() {
    std::lock_guard<std::mutex> lock(hook_events_mutex);

    namespace json = boost::json;
    json::array events_array;

    for (const auto& event : hook_alloc_events) {
        json::object event_obj;
        if (event.type == HookAllocationEvent::Type::Alloc) {
            event_obj["type"] = "alloc";
            event_obj["size"] = event.size;
            event_obj["ptr"] = static_cast<uint64_t>(event.ptr);
        } else if (event.type == HookAllocationEvent::Type::Free) {
            event_obj["type"] = "free";
            event_obj["ptr"] = static_cast<uint64_t>(event.ptr);
        } else if (event.type == HookAllocationEvent::Type::Reserve) {
            event_obj["type"] = "reserve";
            event_obj["size"] = event.size;
            event_obj["ptr"] = static_cast<uint64_t>(event.ptr);
            event_obj["alignment"] = event.alignment;
        }
        events_array.push_back(event_obj);
    }

    json::object root;
    root["start_base_addr"] = static_cast<uint64_t>(hook_recording_start_base_addr);
    root["events"] = events_array;
    return root;
}

void replay_hook_events_from_json(const boost::json::object& events_obj) {
    namespace json = boost::json;

    if (events_obj.contains("start_base_addr")) {
        uint64_t start_base_addr = events_obj.at("start_base_addr").to_number<uint64_t>();
        if (start_base_addr >= tls_storage.current_alloc_base_addr) {
            tls_storage.current_alloc_base_addr = static_cast<CUdeviceptr>(start_base_addr);
        } else {
            // LOAD mode consumed more memory than SAVE mode before graph loading.
            // This means allocations happened in a different order or additional allocations
            // occurred that weren't present during SAVE.
            fprintf(stderr, "[HOOK] ERROR: Memory offset mismatch during replay\n");
            fprintf(stderr, "[HOOK]   Current offset: 0x%llx (%.2f MB from base)\n",
                (unsigned long long)tls_storage.current_alloc_base_addr,
                (tls_storage.current_alloc_base_addr - (size_t)tls_storage.region.base) / (1024.0 * 1024.0));
            fprintf(stderr, "[HOOK]   Expected start: 0x%llx (%.2f MB from base)\n",
                (unsigned long long)start_base_addr,
                (start_base_addr - (size_t)tls_storage.region.base) / (1024.0 * 1024.0));
            fprintf(stderr, "[HOOK]   Difference: %.2f MB\n",
                (tls_storage.current_alloc_base_addr - start_base_addr) / (1024.0 * 1024.0));
            fprintf(stderr, "[HOOK] HINT: Ensure LOAD mode has identical initialization as SAVE mode\n");
            abort();
        }
    }

    const json::array& events_array = events_obj.at("events").as_array();

    for (const auto& event_val : events_array) {
        const json::object& event_obj = event_val.as_object();
        std::string type = event_obj.at("type").as_string().c_str();

        if (type == "alloc") {
            size_t size = event_obj.at("size").to_number<uint64_t>();
            uint64_t expected_ptr = event_obj.at("ptr").to_number<uint64_t>();

            CUdeviceptr actual_ptr = 0;
            CUresult result = cuMemAlloc_v2(&actual_ptr, size);

            if (result != CUDA_SUCCESS) {
                fprintf(stderr, "[REPLAY] FATAL: cuMemAlloc_v2 failed for size=%zu expected_ptr=0x%llx error=%d\n",
                        size, (unsigned long long)expected_ptr, result);
                fprintf(stderr, "[REPLAY] FATAL: Graph load cannot continue with allocation failure\n");
                abort();
            }

            if (actual_ptr != expected_ptr) {
                fprintf(stderr, "[REPLAY] FATAL: Allocation address mismatch!\n");
                fprintf(stderr, "[REPLAY]   Expected: 0x%llx\n", (unsigned long long)expected_ptr);
                fprintf(stderr, "[REPLAY]   Actual:   0x%llx\n", (unsigned long long)actual_ptr);
                fprintf(stderr, "[REPLAY]   Size:     %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));
                fprintf(stderr, "[REPLAY]   Offset:   0x%llx\n", (unsigned long long)(actual_ptr > expected_ptr ?
                        actual_ptr - expected_ptr : expected_ptr - actual_ptr));
                fprintf(stderr, "[REPLAY] FATAL: Continuing would cause memory corruption and incorrect results\n");
                fprintf(stderr, "[REPLAY] HINT: Try a different base_addr in your TOML config\n");
                abort();
            }
#ifdef HOOK_DEBUG
            fprintf(stderr, "[REPLAY] OK: Allocated %zu bytes at expected address 0x%llx\n",
                    size, (unsigned long long)actual_ptr);
#endif

        } else if (type == "reserve") {
            // Just advance the pointer without physical mapping.
            // NVSHMEM will do its own cuMemCreate+cuMemMap when prepare_comm_buffer runs later.
            size_t size = event_obj.at("size").to_number<uint64_t>();
            size_t alignment = event_obj.at("alignment").to_number<uint64_t>();
            uint64_t expected_ptr = event_obj.at("ptr").to_number<uint64_t>();

            CUdeviceptr aligned_addr = align_to(tls_storage.current_alloc_base_addr, alignment);

            if (aligned_addr != expected_ptr) {
                fprintf(stderr, "[REPLAY] FATAL: Reserve address mismatch!\n");
                fprintf(stderr, "[REPLAY]   Expected: 0x%llx\n", (unsigned long long)expected_ptr);
                fprintf(stderr, "[REPLAY]   Actual:   0x%llx\n", (unsigned long long)aligned_addr);
                abort();
            }

            tls_storage.current_alloc_base_addr = align_to(aligned_addr + size, kAllocAlignment);

#ifdef HOOK_DEBUG
            fprintf(stderr, "[REPLAY] OK: Reserved %zu bytes at 0x%llx (pointer advance only)\n",
                    size, (unsigned long long)aligned_addr);
#endif

        } else if (type == "free") {
            uint64_t ptr = event_obj.at("ptr").to_number<uint64_t>();
            CUresult result = cuMemFree_v2(static_cast<CUdeviceptr>(ptr));
            if (result != CUDA_SUCCESS) {
                fprintf(stderr, "[REPLAY] ERROR: cuMemFree_v2 failed for ptr=0x%llx\n",
                        (unsigned long long)ptr);
            }
        }
    }
}

void set_pack_fatbins_on_exit(bool enabled) {
    pack_fatbins_on_exit_enabled.store(enabled);
}

void set_skip_fatbin_processing(bool enabled) {
    skip_fatbin_processing.store(enabled);
}

void set_nvshmem_auto_init(bool enabled) {
    nvshmem_auto_init_enabled.store(enabled);
}

int init_nvshmem_for_loaded_modules() {
    return ::init_nvshmem_for_loaded_modules();
}

void pack_fatbins_to_folder(const std::string& folder_path) {
    pack_fatbins_to_folder_impl(fs::path(folder_path));
}

std::variant<CUfunction, CUkernel> query_function_handle(uint64_t binary_hash, const std::string& function_name) {
    std::variant<CUfunction, CUkernel> result;
    bool found = false;

    binary_hash_to_handles.cvisit(binary_hash, [&](const auto& pair) {
        found = true;
        const auto& handles_variant = pair.second;

        if (std::holds_alternative<ModuleHandles>(handles_variant)) {
            const auto& [mod, func_map] = std::get<ModuleHandles>(handles_variant);
            auto func_it = func_map.find(function_name);
            if (func_it == func_map.end()) {
                fprintf(stderr, "[HOOK] ERROR: Function %s not found in module for hash %016llx\n",
                        function_name.c_str(), (unsigned long long)binary_hash);
                abort();
            }
            result = func_it->second;
        } else {
            const auto& [lib, kernel_map] = std::get<LibraryHandles>(handles_variant);
            auto kern_it = kernel_map.find(function_name);
            if (kern_it == kernel_map.end()) {
                fprintf(stderr, "[HOOK] ERROR: Kernel %s not found in library for hash %016llx\n",
                        function_name.c_str(), (unsigned long long)binary_hash);
                abort();
            }
            result = kern_it->second;
        }
    });

    if (!found) {
        fprintf(stderr, "[HOOK] ERROR: Binary hash %016llx not found\n", (unsigned long long)binary_hash);
        abort();
    }

    return result;
}

uint64_t query_binary_hash(std::variant<CUmodule, CUlibrary> handle) {
    uint64_t hash = 0;
    bool found = false;

    module_or_library_handle_to_hash.visit(handle, [&](const auto& pair) {
        hash = pair.second;
        found = true;
    });

    if (!found) {
        fprintf(stderr, "[HOOK] ERROR: Module/Library handle not found in hash map\n");
        abort();
    }

    return hash;
}

void mark_binary_used(uint64_t binary_hash) {
    binary_hash_to_metadata.visit(binary_hash, [](auto& pair) {
        pair.second.used = true;
    });
}

struct ParsedOptions {
    std::string base_func_name;
    std::string filename;
    std::vector<CUjit_option> jit_options;
    std::vector<void*> jit_option_values;
    std::vector<CUlibraryOption> library_options;
    std::vector<void*> library_option_values;
    uint32_t binary_flags = BINARY_FLAG_NONE;  // Bit flags (NEEDS_DEVICE_LINK, REQUIRES_NVSHMEM)
    size_t num_segments = 0;
};

static ParsedOptions parse_function_and_options(const std::string& func_line) {
    ParsedOptions result;

    size_t first_comma = func_line.find(',');
    if (first_comma == std::string::npos) {
        result.base_func_name = func_line;
        return result;
    }

    result.base_func_name = func_line.substr(0, first_comma);

    std::string options_part = func_line.substr(first_comma + 1);
    size_t pos = 0;

    std::vector<std::pair<CUjit_option, std::string>> jit_option_pairs;
    std::vector<std::pair<CUlibraryOption, std::string>> library_option_pairs;

    while (pos < options_part.length()) {
        size_t comma_pos = options_part.find(',', pos);
        if (comma_pos == std::string::npos) {
            comma_pos = options_part.length();
        }

        std::string token = options_part.substr(pos, comma_pos - pos);
        size_t eq_pos = token.find('=');

        if (eq_pos != std::string::npos) {
            std::string key = token.substr(0, eq_pos);
            std::string value = token.substr(eq_pos + 1);

            if (key == "numJitOptions" || key == "numLibraryOptions") {
            } else if (key == "filename") {
                result.filename = value;
            } else if (key == "binary_flags") {
                result.binary_flags = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "num_segments") {
                result.num_segments = std::stoul(value);
            } else if (key.find("CU_JIT_") == 0) {
                CUjit_option opt = string_to_jit_option(key);
                if (!is_jit_option_ignored(opt)) {
                    jit_option_pairs.push_back({opt, value});
                }
            } else if (key.find("CU_LIBRARY_") == 0) {
                CUlibraryOption opt = string_to_library_option(key);
                if (!library_option_ignored(opt)) {
                    library_option_pairs.push_back({opt, value});
                }
            }
        }

        pos = comma_pos + 1;
    }

    bool all_jit_values_null = true;
    for (const auto& [opt, value] : jit_option_pairs) {
        if (value != "null") {
            all_jit_values_null = false;
            break;
        }
    }

    for (const auto& [opt, value] : jit_option_pairs) {
        result.jit_options.push_back(opt);
        if (all_jit_values_null) {
        } else {
            if (value == "null") {
                fprintf(stderr, "[HOOK] FATAL ERROR: Inconsistent state - JIT option %s has null value but not all values are null\n",
                        std::string(jit_option_to_string(opt)).c_str());
                abort();
            }
            result.jit_option_values.push_back(string_to_jit_option_value(opt, value));
        }
    }

    bool all_library_values_null = true;
    for (const auto& [opt, value] : library_option_pairs) {
        if (value != "null") {
            all_library_values_null = false;
            break;
        }
    }

    for (const auto& [opt, value] : library_option_pairs) {
        result.library_options.push_back(opt);
        if (all_library_values_null) {
        } else {
            if (value == "null") {
                fprintf(stderr, "[HOOK] FATAL ERROR: Inconsistent state - Library option %s has null value but not all values are null\n",
                        std::string(library_option_to_string(opt)).c_str());
                abort();
            }
            result.library_option_values.push_back(string_to_library_option_value(opt, value));
        }
    }

    return result;
}

void load_cuda_modules_and_libraries(const std::string& archive_dir) {
    if (binary_loaded.load()) {
        fprintf(stderr, "[HOOK] ERROR: Binaries already loaded\n");
        abort();
    }

    // FIXME: this will reset memory region, try to avoid that
    // Ensure CUDA context exists before loading modules.
    //
    // NOTE: the four dlsym lookups below use RTLD_DEFAULT with RTLD_NEXT
    // fallback. Reason: under forkserver, libcudart gets loaded with
    // RTLD_LOCAL scope (ctypes default), which is reachable via
    // RTLD_DEFAULT but not via RTLD_NEXT from this preload. Under spawn,
    // libcudart is loaded via DT_NEEDED with a scope visible to RTLD_NEXT.
    // Querying DEFAULT first + falling back to NEXT makes both cases work.
    // We do NOT change RTLD_NEXT usage elsewhere in this file — those are
    // genuine interposer lookups (hook wants the *real* function, not its
    // own wrapper) and must stay on RTLD_NEXT.
    typedef CUresult (*cuCtxGetCurrent_t)(CUcontext*);
    auto ctx_get_current = (cuCtxGetCurrent_t)real_dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
    if (!ctx_get_current)
        ctx_get_current = (cuCtxGetCurrent_t)real_dlsym(RTLD_NEXT, "cuCtxGetCurrent");
    CUcontext ctx = nullptr;
    if (ctx_get_current) {
        ctx_get_current(&ctx);
    }
    if (!ctx) {
        // Get current device from CUDA runtime (set by PyTorch/framework)
        // cudaError_t is an enum, cudaSuccess = 0
        typedef int (*cudaGetDevice_t)(int*);
        auto get_device = (cudaGetDevice_t)real_dlsym(RTLD_DEFAULT, "cudaGetDevice");
        if (!get_device)
            get_device = (cudaGetDevice_t)real_dlsym(RTLD_NEXT, "cudaGetDevice");
        int device = 0;
        if (get_device) {
            int err = get_device(&device);
            if (err != 0) {  // cudaSuccess = 0
                fprintf(stderr, "[HOOK] WARNING: cudaGetDevice failed with error %d, defaulting to device 0\n", err);
                device = 0;
            }
        }

#ifdef HOOK_DEBUG
        fprintf(stderr, "[HOOK] DEBUG: No CUDA context found, creating context on device %d\n", device);
#endif

        // Initialize via CUDA runtime API (cudaSetDevice + cudaFree) instead of
        // cuCtxCreate_v2. cuCtxCreate_v2 creates a non-primary context that gets
        // destroyed when NCCL calls cuDevicePrimaryCtxReset during DP init.
        // This orphans loaded modules in the dead context, causing cross-context
        // graph execution overhead (~7x slowdown on DP>1).
        // cudaSetDevice + cudaFree(0) creates the runtime's primary context which
        // survives NCCL's context management.
        typedef int (*cudaSetDevice_t)(int);
        typedef int (*cudaFree_t)(void*);
        auto set_dev = (cudaSetDevice_t)real_dlsym(RTLD_DEFAULT, "cudaSetDevice");
        if (!set_dev)
            set_dev = (cudaSetDevice_t)real_dlsym(RTLD_NEXT, "cudaSetDevice");
        auto cuda_free = (cudaFree_t)real_dlsym(RTLD_DEFAULT, "cudaFree");
        if (!cuda_free)
            cuda_free = (cudaFree_t)real_dlsym(RTLD_NEXT, "cudaFree");
        if (set_dev && cuda_free) {
            int err = set_dev(device);
            if (err != 0) {
                fprintf(stderr, "[HOOK] ERROR: cudaSetDevice(%d) failed with error %d\n", device, err);
                abort();
            }
            err = cuda_free(nullptr);
            if (err != 0) {
                fprintf(stderr, "[HOOK] ERROR: cudaFree(0) on device %d failed with error %d\n", device, err);
                abort();
            }
            if (ctx_get_current) ctx_get_current(&ctx);
            fprintf(stderr, "[HOOK] Initialized CUDA runtime primary context on device %d: ctx=%p\n", device, (void*)ctx);
        } else {
            fprintf(stderr, "[HOOK] ERROR: Could not find cudaSetDevice/cudaFree\n");
            abort();
        }
    } else {
#ifdef HOOK_DEBUG
        // Context already exists - log which device it's on
        typedef CUresult (*cuCtxGetDevice_t)(CUdevice*);
        auto ctx_get_device = (cuCtxGetDevice_t)real_dlsym(RTLD_NEXT, "cuCtxGetDevice");
        if (ctx_get_device) {
            CUdevice existing_device = -1;
            CUresult res = ctx_get_device(&existing_device);
            if (res == CUDA_SUCCESS) {
                fprintf(stderr, "[HOOK] DEBUG: Using existing CUDA context on device %d\n", existing_device);
            }
        }
#endif
    }

    std::call_once(load_once_flag, [&archive_dir]() {
        const fs::path packed_img_path = fs::path(archive_dir) / "fatbin_image_packed.img";
        const fs::path packed_txt_path = fs::path(archive_dir) / "fatbin_entrypoint_packed.txt";

        if (!fs::exists(packed_img_path) || !fs::exists(packed_txt_path)) {
            fprintf(stderr, "[HOOK] ERROR: Packed files not found in %s\n", archive_dir.c_str());
            abort();
        }

        std::ifstream img_file(packed_img_path.string(), std::ios::binary);
        std::ifstream txt_file(packed_txt_path.string());

        if (!img_file || !txt_file) {
            fprintf(stderr, "[HOOK] ERROR: Failed to open packed files\n");
            abort();
        }

        struct BinaryEntry {
            uint64_t hash;
            std::vector<uint8_t> data;
            std::string func_name;
            ParsedOptions options;
            std::vector<std::string> entry_names;
            // For device-linked binaries
            std::vector<std::vector<uint8_t>> linked_segments;
        };

        std::vector<BinaryEntry> binaries;

        while (img_file) {
            uint64_t hash = 0;
            size_t size = 0;

            img_file.read(reinterpret_cast<char*>(&hash), sizeof(uint64_t));
            if (!img_file) break;

            img_file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
            if (!img_file) {
                fprintf(stderr, "[HOOK] ERROR: Failed to read size\n");
                abort();
            }

            BinaryEntry entry;
            entry.hash = hash;

            if (size == 0) {
                // Device-linked binary marker - read segment count and segments
                size_t num_segments = 0;
                img_file.read(reinterpret_cast<char*>(&num_segments), sizeof(size_t));
                if (!img_file || num_segments == 0) {
                    fprintf(stderr, "[HOOK] ERROR: Invalid segment count for device-linked binary\n");
                    abort();
                }

#ifdef HOOK_DEBUG
                fprintf(stderr, "[HOOK] DEBUG: Reading %zu linked segments for hash %016llx\n",
                        num_segments, (unsigned long long)hash);
#endif

                for (size_t i = 0; i < num_segments; i++) {
                    size_t seg_size = 0;
                    img_file.read(reinterpret_cast<char*>(&seg_size), sizeof(size_t));
                    if (!img_file || seg_size == 0) {
                        fprintf(stderr, "[HOOK] ERROR: Invalid segment size\n");
                        abort();
                    }

                    std::vector<uint8_t> seg_data(seg_size);
                    img_file.read(reinterpret_cast<char*>(seg_data.data()), seg_size);
                    if (!img_file) {
                        fprintf(stderr, "[HOOK] ERROR: Failed to read segment data\n");
                        abort();
                    }

                    entry.linked_segments.push_back(std::move(seg_data));
#ifdef HOOK_DEBUG
                    fprintf(stderr, "[HOOK] DEBUG:   segment[%zu] size: %zu bytes\n", i, seg_size);
#endif
                }
            } else {
                // Regular single binary
                entry.data.resize(size);
                img_file.read(reinterpret_cast<char*>(entry.data.data()), size);
                if (!img_file) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to read binary data\n");
                    abort();
                }
            }

            binaries.push_back(std::move(entry));
        }

        size_t binary_idx = 0;
        std::string line;
        int state = 0;
        unsigned int expected_count = 0;

        while (std::getline(txt_file, line)) {
            if (line == "---") {
                binary_idx++;
                state = 0;
                expected_count = 0;
                continue;
            }

            if (binary_idx >= binaries.size()) {
                fprintf(stderr, "[HOOK] ERROR: Mismatch between img and txt files\n");
                abort();
            }

            auto& binary = binaries[binary_idx];

            if (state == 0) {
                uint64_t txt_hash = std::strtoull(line.c_str(), nullptr, 16);
                if (txt_hash != binary.hash) {
                    fprintf(stderr, "[HOOK] ERROR: Hash mismatch: img=%016llx, txt=%016llx\n",
                            (unsigned long long)binary.hash, (unsigned long long)txt_hash);
                    abort();
                }
                state = 1;
            } else if (state == 1) {
                binary.func_name = line;
                binary.options = parse_function_and_options(line);
                state = 2;
            } else if (state == 2) {
                expected_count = std::stoul(line);
                state = 3;
            } else if (state == 3) {
                binary.entry_names.push_back(line);
                if (binary.entry_names.size() >= expected_count) {
                    state = 4;
                }
            }
        }

        typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
        typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
        typedef CUresult (*cuModuleLoadFatBinary_t)(CUmodule*, const void*);
        typedef CUresult (*cuLibraryLoadData_t)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int);
        typedef CUresult (*cuModuleLoad_t)(CUmodule*, const char*);
        typedef CUresult (*cuLibraryLoadFromFile_t)(CUlibrary*, const char*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int);
        typedef CUresult (*cuLinkCreate_t)(unsigned int, CUjit_option*, void**, CUlinkState*);
        typedef CUresult (*cuLinkAddData_t)(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
        typedef CUresult (*cuLinkComplete_t)(CUlinkState, void**, size_t*);
        typedef CUresult (*cuLinkDestroy_t)(CUlinkState);

        auto module_load_data = (cuModuleLoadData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadData);
        auto module_load_data_ex = (cuModuleLoadDataEx_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadDataEx);
        auto module_load_fatbinary = (cuModuleLoadFatBinary_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoadFatBinary);
        auto library_load_data = (cuLibraryLoadData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryLoadData);
        auto module_load = (cuModuleLoad_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuModuleLoad);
        auto library_load_from_file = (cuLibraryLoadFromFile_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLibraryLoadFromFile);
        auto link_create = (cuLinkCreate_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkCreate);
        auto link_add_data = (cuLinkAddData_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkAddData);
        auto link_complete = (cuLinkComplete_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkComplete);
        auto link_destroy = (cuLinkDestroy_t)CUDA_DRIVER_CALL(cuda_driver_entry_table, CUDA_ENTRY_cuLinkDestroy);

        for (auto& binary : binaries) {
            if (binary.options.base_func_name == "cuLibraryLoadData") {
                CUlibrary library = nullptr;
                CUjit_option* jit_opts = binary.options.jit_options.empty() ? nullptr : binary.options.jit_options.data();
                void** jit_vals = binary.options.jit_option_values.empty() ? nullptr : binary.options.jit_option_values.data();
                unsigned int num_jit = binary.options.jit_options.size();
                CUlibraryOption* lib_opts = binary.options.library_options.empty() ? nullptr : binary.options.library_options.data();
                void** lib_vals = binary.options.library_option_values.empty() ? nullptr : binary.options.library_option_values.data();
                unsigned int num_lib = binary.options.library_options.size();

                CUresult res;

                if (!binary.linked_segments.empty()) {
                    // Device-linked binary - use CUDA linker to combine segments (fallback path)
#ifdef HOOK_DEBUG
                    fprintf(stderr, "[HOOK] DEBUG: Using device linker for %zu segments (hash %016llx)\n",
                            binary.linked_segments.size(), (unsigned long long)binary.hash);
#endif

                    CUlinkState link_state;
                    res = link_create(num_jit, jit_opts, jit_vals, &link_state);
                    if (res != CUDA_SUCCESS) {
                        fprintf(stderr, "[HOOK] ERROR: cuLinkCreate failed with error %d\n", res);
                        abort();
                    }

                    // Add each segment to the linker
                    for (size_t i = 0; i < binary.linked_segments.size(); i++) {
                        auto& segment = binary.linked_segments[i];
                        res = link_add_data(link_state, CU_JIT_INPUT_FATBINARY,
                                           const_cast<uint8_t*>(segment.data()), segment.size(),
                                           nullptr, 0, nullptr, nullptr);
                        if (res != CUDA_SUCCESS) {
                            fprintf(stderr, "[HOOK] ERROR: cuLinkAddData failed for segment %zu with error %d\n", i, res);
                            link_destroy(link_state);
                            abort();
                        }
#ifdef HOOK_DEBUG
                        fprintf(stderr, "[HOOK] DEBUG:   Added segment %zu (%zu bytes) to linker\n", i, segment.size());
#endif
                    }

                    // Complete linking
                    void* cubin_out = nullptr;
                    size_t cubin_size = 0;
                    res = link_complete(link_state, &cubin_out, &cubin_size);
                    if (res != CUDA_SUCCESS || !cubin_out) {
                        fprintf(stderr, "[HOOK] ERROR: cuLinkComplete failed with error %d\n", res);
                        link_destroy(link_state);
                        abort();
                    }
#ifdef HOOK_DEBUG
                    fprintf(stderr, "[HOOK] DEBUG: Device linking complete, output size: %zu bytes\n", cubin_size);
#endif

                    // Load the linked cubin as a library
                    res = library_load_data(&library, cubin_out, jit_opts, jit_vals, num_jit, lib_opts, lib_vals, num_lib);
                    link_destroy(link_state);

                    if (res != CUDA_SUCCESS || !library) {
                        fprintf(stderr, "[HOOK] ERROR: Failed to load device-linked library for hash %016llx, error=%d\n",
                                (unsigned long long)binary.hash, res);
                        abort();
                    }
                } else {
                    // Regular single binary (or pre-linked cubin from SAVE mode)
#ifdef HOOK_DEBUG
                    fprintf(stderr, "[HOOK] DEBUG: Loading library hash %016llx, size: %zu bytes, %zu kernels\n",
                            (unsigned long long)binary.hash, binary.data.size(), binary.entry_names.size());
#endif

                    res = library_load_data(&library, binary.data.data(), jit_opts, jit_vals, num_jit, lib_opts, lib_vals, num_lib);
                    if (res != CUDA_SUCCESS || !library) {
                        fprintf(stderr, "[HOOK] ERROR: Failed to load library for hash %016llx, error=%d\n",
                                (unsigned long long)binary.hash, res);
                        abort();
                    }
                }
                setup_handles_for_library(library, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else if (binary.options.base_func_name == "cuModuleLoadDataEx") {
                CUmodule module = nullptr;
                CUjit_option* jit_opts = binary.options.jit_options.empty() ? nullptr : binary.options.jit_options.data();
                void** jit_vals = binary.options.jit_option_values.empty() ? nullptr : binary.options.jit_option_values.data();
                unsigned int num_jit = binary.options.jit_options.size();

                CUresult res = module_load_data_ex(&module, binary.data.data(), num_jit, jit_opts, jit_vals);
                if (res != CUDA_SUCCESS || !module) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to load module (Ex) for hash %016llx, error=%d\n",
                            (unsigned long long)binary.hash, res);
                    abort();
                }
                setup_handles_for_module(module, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else if (binary.options.base_func_name == "cuModuleLoadData") {
                CUmodule module = nullptr;
                CUresult res = module_load_data(&module, binary.data.data());
                if (res != CUDA_SUCCESS || !module) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to load module for hash %016llx, error=%d\n",
                            (unsigned long long)binary.hash, res);
                    abort();
                }
                setup_handles_for_module(module, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else if (binary.options.base_func_name == "cuModuleLoadFatBinary") {
                CUmodule module = nullptr;
                CUresult res = module_load_fatbinary(&module, binary.data.data());
                if (res != CUDA_SUCCESS || !module) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to load fatbinary for hash %016llx, error=%d\n",
                            (unsigned long long)binary.hash, res);
                    abort();
                }
                setup_handles_for_module(module, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else if (binary.options.base_func_name == "cuModuleLoad") {
                if (binary.options.filename.empty()) {
                    fprintf(stderr, "[HOOK] ERROR: No filename found for cuModuleLoad\n");
                    abort();
                }

                const fs::path temp_file_path = fs::path(archive_dir) / str(boost::format("temp_module_%016llx.bin") % binary.hash);
                std::ofstream temp_file(temp_file_path.string(), std::ios::binary);
                if (!temp_file) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to create temporary file for module loading\n");
                    abort();
                }
                temp_file.write(reinterpret_cast<const char*>(binary.data.data()), binary.data.size());
                temp_file.close();

                CUmodule module = nullptr;
                CUresult res = module_load(&module, temp_file_path.string().c_str());

                fs::remove(temp_file_path);

                if (res != CUDA_SUCCESS || !module) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to load module from file for hash %016llx, error=%d\n",
                            (unsigned long long)binary.hash, res);
                    abort();
                }
                setup_handles_for_module(module, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else if (binary.options.base_func_name == "cuLibraryLoadFromFile") {
                if (binary.options.filename.empty()) {
                    fprintf(stderr, "[HOOK] ERROR: No filename found for cuLibraryLoadFromFile\n");
                    abort();
                }

                const fs::path temp_file_path = fs::path(archive_dir) / str(boost::format("temp_library_%016llx.bin") % binary.hash);
                std::ofstream temp_file(temp_file_path.string(), std::ios::binary);
                if (!temp_file) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to create temporary file for library loading\n");
                    abort();
                }
                temp_file.write(reinterpret_cast<const char*>(binary.data.data()), binary.data.size());
                temp_file.close();

                CUlibrary library = nullptr;
                CUjit_option* jit_opts = binary.options.jit_options.empty() ? nullptr : binary.options.jit_options.data();
                void** jit_vals = binary.options.jit_option_values.empty() ? nullptr : binary.options.jit_option_values.data();
                unsigned int num_jit = binary.options.jit_options.size();
                CUlibraryOption* lib_opts = binary.options.library_options.empty() ? nullptr : binary.options.library_options.data();
                void** lib_vals = binary.options.library_option_values.empty() ? nullptr : binary.options.library_option_values.data();
                unsigned int num_lib = binary.options.library_options.size();

                CUresult res = library_load_from_file(&library, temp_file_path.string().c_str(), jit_opts, jit_vals, num_jit, lib_opts, lib_vals, num_lib);

                fs::remove(temp_file_path);

                if (res != CUDA_SUCCESS || !library) {
                    fprintf(stderr, "[HOOK] ERROR: Failed to load library from file for hash %016llx, error=%d\n",
                            (unsigned long long)binary.hash, res);
                    abort();
                }
                setup_handles_for_library(library, binary.hash, binary.entry_names, binary.options.binary_flags & BINARY_FLAG_REQUIRES_NVSHMEM);
            } else {
                fprintf(stderr, "[HOOK] ERROR: Unknown function name: %s\n", binary.options.base_func_name.c_str());
                abort();
            }
        }

        bool expected = false;
        if (!binary_loaded.compare_exchange_strong(expected, true)) {
            fprintf(stderr, "[HOOK] ERROR: Binary already loaded by another thread\n");
            abort();
        }
    });

    if (!binary_loaded.load()) {
        fprintf(stderr, "[HOOK] ERROR: Binary loading failed\n");
        abort();
    }
}

} // namespace foundry
