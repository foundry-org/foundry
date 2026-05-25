#pragma once

#include <cstdint>
#include <cstring>
#include <string>

// Forward declare to avoid pulling in boost/json.hpp in the header
namespace boost {
namespace json {
class value;
}
}  // namespace boost

// Binary CUDA Graph format (.cugraph) — Version 1
//
// Layout:
//   [FileHeader]           64 bytes
//   [SectionDirectory]     num_sections * 24 bytes
//   [Section data...]      variable, referenced by directory
//
// Sections:
//   STRING_TABLE       — deduplicated null-terminated strings
//   NODE_TABLE         — fixed-size node descriptors (160 bytes each)
//   DEPENDENCY_TABLE   — (from_id, to_id) pairs
//   KERNEL_PARAM_INDEX — per-kernel param descriptors
//   KERNEL_PARAM_DATA  — raw binary param bytes
//   ARG_BUFFER_DATA    — raw binary extra arg buffers
//   COMMON_KERNEL_ATTRS — single struct for shared attrs
//   GENERATORS_TABLE   — generator state entries
//   ALLOCATOR_EVENTS   — JSON text (complex, small)
//   OUTPUT_TENSORS     — JSON text (complex, small)
//   TOPOLOGY_KEY       — raw string

namespace foundry {
namespace binary_format {

static constexpr uint8_t MAGIC[8] = {'C', 'U', 'G', 'R', 'A', 'P', 'H', '\0'};
static constexpr uint32_t FORMAT_VERSION = 1;

enum SectionType : uint32_t {
  SECTION_STRING_TABLE = 0,
  SECTION_NODE_TABLE = 1,
  SECTION_DEPENDENCY_TABLE = 2,
  SECTION_KERNEL_PARAM_INDEX = 3,
  SECTION_KERNEL_PARAM_DATA = 4,
  SECTION_ARG_BUFFER_DATA = 5,
  SECTION_COMMON_KERNEL_ATTRS = 6,
  SECTION_GENERATORS_TABLE = 7,
  SECTION_ALLOCATOR_EVENTS = 8,
  SECTION_OUTPUT_TENSORS = 9,
  SECTION_TOPOLOGY_KEY = 10,
  SECTION_COUNT = 11,
};

enum HeaderFlags : uint32_t {
  FLAG_HAS_COMMON_KERNEL_ATTRS = 1 << 0,
  FLAG_HAS_DEPENDENCIES = 1 << 1,
  FLAG_HAS_ALLOCATOR_EVENTS = 1 << 2,
  FLAG_HAS_OUTPUT_TENSORS = 1 << 3,
  FLAG_HAS_GENERATORS = 1 << 4,
};

#pragma pack(push, 1)

struct FileHeader {
  uint8_t magic[8];  // "CUGRAPH\0"
  uint32_t version;  // FORMAT_VERSION
  uint32_t flags;    // HeaderFlags bitmask
  uint32_t num_nodes;
  uint32_t num_dependencies;
  uint32_t num_generators;
  uint32_t num_sections;
  uint32_t num_strings;
  uint8_t reserved[28];  // pad to 64 bytes
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

struct SectionEntry {
  uint32_t section_type;
  uint32_t _pad;
  uint64_t offset;  // from file start
  uint64_t size;    // in bytes
};
static_assert(sizeof(SectionEntry) == 24, "SectionEntry must be 24 bytes");

// ---- Node types ----

enum BinNodeType : uint8_t {
  NODE_KERNEL = 0,
  NODE_MEMSET = 1,
  NODE_MEMCPY = 2,
  NODE_EVENT_RECORD = 3,
  NODE_EVENT_WAIT = 4,
  NODE_EMPTY = 5,
};

// Bitmask for which kernel_node_attrs are set (per-node overrides)
enum KernelNodeAttrFlags : uint32_t {
  KNA_CLUSTER_DIM = 1 << 0,
  KNA_PREFERRED_CLUSTER_DIM = 1 << 1,
  KNA_CLUSTER_SCHEDULING = 1 << 2,
  KNA_COOPERATIVE = 1 << 3,
  KNA_PRIORITY = 1 << 4,
  KNA_MEM_SYNC_DOMAIN = 1 << 5,
  KNA_MEM_SYNC_DOMAIN_MAP = 1 << 6,
  KNA_SHARED_MEM_CARVEOUT = 1 << 7,
  KNA_ATTR_QUERY_AVAILABLE = 1 << 8,
};

struct BinKernelNode {
  uint32_t blockDimX, blockDimY, blockDimZ;
  uint32_t gridDimX, gridDimY, gridDimZ;
  uint32_t sharedMemBytes;
  uint64_t binary_hash;
  uint32_t function_name_offset;  // into StringTable
  uint32_t function_name_length;
  // func_attrs
  int32_t max_dynamic_shared_size_bytes;
  int32_t preferred_shared_memory_carveout;
  int32_t cluster_scheduling_policy_preference;
  int32_t required_cluster_width;
  int32_t required_cluster_height;
  int32_t required_cluster_depth;
  // kernel_node_attrs
  uint32_t kna_flags;  // KernelNodeAttrFlags bitmask
  uint32_t kna_clusterDimX, kna_clusterDimY, kna_clusterDimZ;
  uint32_t kna_preferredClusterDimX, kna_preferredClusterDimY, kna_preferredClusterDimZ;
  int32_t kna_clusterSchedulingPolicy;
  int32_t kna_cooperative;
  int32_t kna_priority;
  int32_t kna_memSyncDomain;
  uint8_t kna_memSyncDomainMapDefault;
  uint8_t kna_memSyncDomainMapRemote;
  uint8_t kna_attrQueryAvailable;
  uint8_t _kna_pad;
  uint32_t kna_preferredSharedMemCarveout;
  // Param references
  uint32_t param_index_offset;  // byte offset into KERNEL_PARAM_INDEX section
  uint32_t num_params;
  uint32_t arg_buffer_offset;  // byte offset into ARG_BUFFER_DATA (0xFFFFFFFF = none)
  uint32_t arg_buffer_size;
};

struct BinMemsetNode {
  uint64_t dst;
  uint32_t elementSize;
  uint32_t _pad;
  uint64_t height;
  uint64_t pitch;
  uint32_t value;
  uint32_t _pad2;
  uint64_t width;
};

struct BinMemcpyNode {
  uint64_t Depth, Height, WidthInBytes;
  uint64_t dstDevice;
  uint64_t dstHeight, dstLOD, dstPitch, dstXInBytes, dstY, dstZ;
  int32_t dstMemoryType;
  int32_t _pad1;
  uint64_t srcDevice;
  uint64_t srcHeight, srcLOD, srcPitch, srcXInBytes, srcY, srcZ;
  int32_t srcMemoryType;
  int32_t _pad2;
};

struct BinEventNode {
  int32_t event_id;
};

struct BinNodeEntry {
  uint32_t node_id;
  BinNodeType type;
  uint8_t _pad[3];
  union {
    BinKernelNode kernel;  // 136 bytes
    BinMemsetNode memset;  // 48 bytes
    BinMemcpyNode memcpy;  // 152 bytes (largest)
    BinEventNode event;    // 4 bytes
                           // EmptyNode has no data
  };
  // Pad to fixed 168 bytes (8 header + 152 memcpy + 8 tail)
  uint8_t _tail_pad[168 - 8 - sizeof(BinMemcpyNode)];
};
static_assert(sizeof(BinNodeEntry) == 168, "BinNodeEntry must be 168 bytes");

// ---- Param index ----

struct BinParamEntry {
  uint32_t data_offset;  // byte offset into KERNEL_PARAM_DATA section
  uint32_t size;         // param size in bytes
};
static_assert(sizeof(BinParamEntry) == 8, "BinParamEntry must be 8 bytes");

// ---- Common kernel attrs ----

struct BinCommonKernelAttrs {
  uint32_t flags;  // KernelNodeAttrFlags bitmask (which are common)
  uint32_t clusterDimX, clusterDimY, clusterDimZ;
  uint32_t preferredClusterDimX, preferredClusterDimY, preferredClusterDimZ;
  int32_t clusterSchedulingPolicy;
  int32_t cooperative;
  int32_t priority;
  int32_t memSyncDomain;
  uint8_t memSyncDomainMapDefault;
  uint8_t memSyncDomainMapRemote;
  uint8_t attrQueryAvailable;
  uint8_t _pad;
  uint32_t preferredSharedMemCarveout;
};

// ---- Generators ----

struct BinGenerator {
  uint64_t id;
  uint64_t seed;
  uint64_t wholegraph_increment;
};
static_assert(sizeof(BinGenerator) == 24, "BinGenerator must be 24 bytes");

// ---- Dependency ----

struct BinDependency {
  uint32_t from_id;
  uint32_t to_id;
};
static_assert(sizeof(BinDependency) == 8, "BinDependency must be 8 bytes");

#pragma pack(pop)

// ---- Helpers ----

inline bool validate_header(const FileHeader& h) {
  return memcmp(h.magic, MAGIC, 8) == 0 && h.version == FORMAT_VERSION;
}

}  // namespace binary_format

// Binary graph I/O functions
// Returns empty json::value{} if file doesn't exist or is invalid.
boost::json::value read_and_parse_binary_graph(const std::string& bin_path);

// Lightweight binary file handle for direct-access reading.
// Holds the file data and section pointers. No JSON construction.
struct BinaryGraphFile {
  std::vector<uint8_t> data;
  binary_format::FileHeader header;
  const binary_format::SectionEntry* sections = nullptr;

  // Section accessors
  template <typename T = uint8_t>
  const T* section_ptr(binary_format::SectionType type) const {
    return reinterpret_cast<const T*>(data.data() + sections[type].offset);
  }
  size_t section_size(binary_format::SectionType type) const { return sections[type].size; }
  const binary_format::BinNodeEntry* node_table() const {
    return section_ptr<binary_format::BinNodeEntry>(binary_format::SECTION_NODE_TABLE);
  }
  const binary_format::BinParamEntry* param_index() const {
    return section_ptr<binary_format::BinParamEntry>(binary_format::SECTION_KERNEL_PARAM_INDEX);
  }
  const uint8_t* param_data() const {
    return section_ptr<uint8_t>(binary_format::SECTION_KERNEL_PARAM_DATA);
  }
  const uint8_t* arg_buffer_data() const {
    return section_ptr<uint8_t>(binary_format::SECTION_ARG_BUFFER_DATA);
  }
  const char* string_table() const {
    return section_ptr<char>(binary_format::SECTION_STRING_TABLE);
  }
  std::string get_string(uint32_t offset, uint32_t length) const {
    return std::string(string_table() + offset, length);
  }
  std::string topology_key() const {
    auto [ptr, sz] = std::make_pair(section_ptr<char>(binary_format::SECTION_TOPOLOGY_KEY),
                                    section_size(binary_format::SECTION_TOPOLOGY_KEY));
    return std::string(ptr, sz);
  }
  bool valid() const { return !data.empty() && sections != nullptr; }
};

// Read a .cugraph file into a BinaryGraphFile (lightweight, no JSON construction).
// Returns an invalid BinaryGraphFile if the file doesn't exist or has bad format.
BinaryGraphFile read_binary_graph_file(const std::string& bin_path);

}  // namespace foundry
