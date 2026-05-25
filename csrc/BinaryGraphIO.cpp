// Binary CUDA Graph I/O — write and read .cugraph format
// See include/BinaryGraphFormat.h for format specification.

#include "CUDAGraph.h"
#include "BinaryGraphFormat.h"
#include <boost/json.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <cstring>

namespace foundry {

void CUDAGraph::save_binary(const std::string& bin_path, const boost::json::object& root) {
  namespace json = boost::json;
  namespace bf = binary_format;

  const json::array& nodes_array = root.at("nodes").as_array();
  const json::array& deps_array =
      root.contains("dependencies") ? root.at("dependencies").as_array() : json::array{};
  const json::array& generators_array = root.at("generators").as_array();

  // ---- Build string table (deduplicated function names) ----
  std::vector<char> string_table;
  std::unordered_map<std::string, uint32_t> string_offsets;
  auto intern_string = [&](const std::string& s) -> uint32_t {
    auto it = string_offsets.find(s);
    if (it != string_offsets.end())
      return it->second;
    uint32_t offset = static_cast<uint32_t>(string_table.size());
    string_table.insert(string_table.end(), s.begin(), s.end());
    string_table.push_back('\0');
    string_offsets[s] = offset;
    return offset;
  };

  std::string topology_key =
      root.contains("topology_key") ? std::string(root.at("topology_key").as_string()) : "";

  // ---- Build node entries and param data ----
  std::vector<bf::BinNodeEntry> node_entries;
  std::vector<bf::BinParamEntry> param_index;
  std::vector<uint8_t> param_data;
  std::vector<uint8_t> arg_buffer_data;
  node_entries.reserve(nodes_array.size());

  // Helper to decode hex string to binary
  auto hex_decode = [](const std::string& hex, std::vector<uint8_t>& out) {
    for (size_t j = 0; j < hex.size(); j += 2) {
      uint8_t byte = static_cast<uint8_t>(std::stoul(hex.substr(j, 2), nullptr, 16));
      out.push_back(byte);
    }
  };

  // Helper to parse kernel_node_attrs flags from JSON
  auto parse_kna_flags = [](const json::object& kna, bf::BinKernelNode& k) {
    k.kna_flags = 0;
    if (kna.contains("attrQueryAvailable")) {
      k.kna_flags |= bf::KNA_ATTR_QUERY_AVAILABLE;
      k.kna_attrQueryAvailable = kna.at("attrQueryAvailable").as_bool() ? 1 : 0;
    }
    if (kna.contains("clusterDimX")) {
      k.kna_flags |= bf::KNA_CLUSTER_DIM;
      k.kna_clusterDimX = kna.at("clusterDimX").to_number<uint32_t>();
      k.kna_clusterDimY = kna.at("clusterDimY").to_number<uint32_t>();
      k.kna_clusterDimZ = kna.at("clusterDimZ").to_number<uint32_t>();
    }
    if (kna.contains("preferredClusterDimX")) {
      k.kna_flags |= bf::KNA_PREFERRED_CLUSTER_DIM;
      k.kna_preferredClusterDimX = kna.at("preferredClusterDimX").to_number<uint32_t>();
      k.kna_preferredClusterDimY = kna.at("preferredClusterDimY").to_number<uint32_t>();
      k.kna_preferredClusterDimZ = kna.at("preferredClusterDimZ").to_number<uint32_t>();
    }
    if (kna.contains("clusterSchedulingPolicyPreference")) {
      k.kna_flags |= bf::KNA_CLUSTER_SCHEDULING;
      k.kna_clusterSchedulingPolicy =
          kna.at("clusterSchedulingPolicyPreference").to_number<int32_t>();
    }
    if (kna.contains("cooperative")) {
      k.kna_flags |= bf::KNA_COOPERATIVE;
      k.kna_cooperative = kna.at("cooperative").to_number<int32_t>();
    }
    if (kna.contains("priority")) {
      k.kna_flags |= bf::KNA_PRIORITY;
      k.kna_priority = kna.at("priority").to_number<int32_t>();
    }
    if (kna.contains("memSyncDomain")) {
      k.kna_flags |= bf::KNA_MEM_SYNC_DOMAIN;
      k.kna_memSyncDomain = kna.at("memSyncDomain").to_number<int32_t>();
    }
    if (kna.contains("memSyncDomainMapDefault")) {
      k.kna_flags |= bf::KNA_MEM_SYNC_DOMAIN_MAP;
      k.kna_memSyncDomainMapDefault =
          static_cast<uint8_t>(kna.at("memSyncDomainMapDefault").to_number<int>());
      k.kna_memSyncDomainMapRemote =
          static_cast<uint8_t>(kna.at("memSyncDomainMapRemote").to_number<int>());
    }
    if (kna.contains("preferredSharedMemCarveout")) {
      k.kna_flags |= bf::KNA_SHARED_MEM_CARVEOUT;
      k.kna_preferredSharedMemCarveout = kna.at("preferredSharedMemCarveout").to_number<uint32_t>();
    }
  };

  // Common kernel attrs
  bool has_common_attrs = root.contains("common_kernel_node_attrs");
  bf::BinCommonKernelAttrs common_attrs = {};
  if (has_common_attrs) {
    const json::object& ca = root.at("common_kernel_node_attrs").as_object();
    // Reuse the kna parser with a temporary BinKernelNode to extract flags
    bf::BinKernelNode tmp = {};
    parse_kna_flags(ca, tmp);
    common_attrs.flags = tmp.kna_flags;
    common_attrs.clusterDimX = tmp.kna_clusterDimX;
    common_attrs.clusterDimY = tmp.kna_clusterDimY;
    common_attrs.clusterDimZ = tmp.kna_clusterDimZ;
    common_attrs.preferredClusterDimX = tmp.kna_preferredClusterDimX;
    common_attrs.preferredClusterDimY = tmp.kna_preferredClusterDimY;
    common_attrs.preferredClusterDimZ = tmp.kna_preferredClusterDimZ;
    common_attrs.clusterSchedulingPolicy = tmp.kna_clusterSchedulingPolicy;
    common_attrs.cooperative = tmp.kna_cooperative;
    common_attrs.priority = tmp.kna_priority;
    common_attrs.memSyncDomain = tmp.kna_memSyncDomain;
    common_attrs.memSyncDomainMapDefault = tmp.kna_memSyncDomainMapDefault;
    common_attrs.memSyncDomainMapRemote = tmp.kna_memSyncDomainMapRemote;
    common_attrs.attrQueryAvailable = tmp.kna_attrQueryAvailable;
    common_attrs.preferredSharedMemCarveout = tmp.kna_preferredSharedMemCarveout;
  }

  // ---- Build node entries ----
  for (const auto& node_val : nodes_array) {
    const json::object& no = node_val.as_object();
    bf::BinNodeEntry entry = {};
    entry.node_id = no.at("id").to_number<uint32_t>();

    std::string type_str(no.at("type").as_string());

    if (type_str == "KernelNode") {
      entry.type = bf::NODE_KERNEL;
      const json::object& p = no.at("params").as_object();
      auto& k = entry.kernel;

      k.blockDimX = p.at("blockDimX").to_number<uint32_t>();
      k.blockDimY = p.at("blockDimY").to_number<uint32_t>();
      k.blockDimZ = p.at("blockDimZ").to_number<uint32_t>();
      k.gridDimX = p.at("gridDimX").to_number<uint32_t>();
      k.gridDimY = p.at("gridDimY").to_number<uint32_t>();
      k.gridDimZ = p.at("gridDimZ").to_number<uint32_t>();
      k.sharedMemBytes = p.at("sharedMemBytes").to_number<uint32_t>();

      std::string fn(p.at("function_name").as_string());
      k.function_name_offset = intern_string(fn);
      k.function_name_length = static_cast<uint32_t>(fn.size());
      k.binary_hash = p.at("kernel_source_binary_hash").to_number<uint64_t>();

      // func_attrs
      if (p.contains("func_attrs")) {
        const json::object& fa = p.at("func_attrs").as_object();
        k.max_dynamic_shared_size_bytes =
            fa.at("max_dynamic_shared_size_bytes").to_number<int32_t>();
        k.preferred_shared_memory_carveout =
            fa.at("preferred_shared_memory_carveout").to_number<int32_t>();
        k.cluster_scheduling_policy_preference =
            fa.contains("cluster_scheduling_policy_preference")
                ? fa.at("cluster_scheduling_policy_preference").to_number<int32_t>()
                : 0;
        k.required_cluster_width = fa.contains("required_cluster_width")
                                       ? fa.at("required_cluster_width").to_number<int32_t>()
                                       : 0;
        k.required_cluster_height = fa.contains("required_cluster_height")
                                        ? fa.at("required_cluster_height").to_number<int32_t>()
                                        : 0;
        k.required_cluster_depth = fa.contains("required_cluster_depth")
                                       ? fa.at("required_cluster_depth").to_number<int32_t>()
                                       : 0;
      }

      // kernel_node_attrs (per-node overrides)
      if (p.contains("kernel_node_attrs")) {
        parse_kna_flags(p.at("kernel_node_attrs").as_object(), k);
      }

      // Kernel params
      k.param_index_offset = static_cast<uint32_t>(param_index.size() * sizeof(bf::BinParamEntry));
      const json::array& kp_array = p.at("kernelParams").as_array();
      k.num_params = static_cast<uint32_t>(kp_array.size());

      for (const auto& kp_val : kp_array) {
        const json::object& kp = kp_val.as_object();
        bf::BinParamEntry pe = {};

        if (kp.contains("value_hex")) {
          const std::string hex_str(kp.at("value_hex").as_string());
          pe.data_offset = static_cast<uint32_t>(param_data.size());
          pe.size = static_cast<uint32_t>(hex_str.size() / 2);
          hex_decode(hex_str, param_data);
        } else {
          // Metadata-only (extra-style) — offset/size descriptors
          pe.data_offset = kp.at("offset").to_number<uint32_t>();
          pe.size = kp.at("size").to_number<uint32_t>();
        }
        param_index.push_back(pe);
      }

      // Extra arg buffer
      const std::string extra_hex(p.contains("extra_argBuffer_hex")
                                      ? std::string(p.at("extra_argBuffer_hex").as_string())
                                      : "");
      if (!extra_hex.empty()) {
        k.arg_buffer_offset = static_cast<uint32_t>(arg_buffer_data.size());
        k.arg_buffer_size = static_cast<uint32_t>(extra_hex.size() / 2);
        hex_decode(extra_hex, arg_buffer_data);
      } else {
        k.arg_buffer_offset = 0xFFFFFFFF;
        k.arg_buffer_size = 0;
      }

    } else if (type_str == "MemcpyNode") {
      entry.type = bf::NODE_MEMCPY;
      const json::object& p = no.at("params").as_object();
      auto& m = entry.memcpy;
      m.Depth = p.at("Depth").to_number<uint64_t>();
      m.Height = p.at("Height").to_number<uint64_t>();
      m.WidthInBytes = p.at("WidthInBytes").to_number<uint64_t>();
      m.dstDevice = p.at("dstDevice").to_number<uint64_t>();
      m.dstHeight = p.at("dstHeight").to_number<uint64_t>();
      m.dstLOD = p.at("dstLOD").to_number<uint64_t>();
      m.dstMemoryType = p.at("dstMemoryType").to_number<int32_t>();
      m.dstPitch = p.at("dstPitch").to_number<uint64_t>();
      m.dstXInBytes = p.at("dstXInBytes").to_number<uint64_t>();
      m.dstY = p.at("dstY").to_number<uint64_t>();
      m.dstZ = p.at("dstZ").to_number<uint64_t>();
      m.srcDevice = p.at("srcDevice").to_number<uint64_t>();
      m.srcHeight = p.at("srcHeight").to_number<uint64_t>();
      m.srcLOD = p.at("srcLOD").to_number<uint64_t>();
      m.srcMemoryType = p.at("srcMemoryType").to_number<int32_t>();
      m.srcPitch = p.at("srcPitch").to_number<uint64_t>();
      m.srcXInBytes = p.at("srcXInBytes").to_number<uint64_t>();
      m.srcY = p.at("srcY").to_number<uint64_t>();
      m.srcZ = p.at("srcZ").to_number<uint64_t>();

    } else if (type_str == "MemsetNode") {
      entry.type = bf::NODE_MEMSET;
      const json::object& p = no.at("params").as_object();
      auto& m = entry.memset;
      m.dst = p.at("dst").to_number<uint64_t>();
      m.elementSize = p.at("elementSize").to_number<uint32_t>();
      m.height = p.at("height").to_number<uint64_t>();
      m.pitch = p.at("pitch").to_number<uint64_t>();
      m.value = p.at("value").to_number<uint32_t>();
      m.width = p.at("width").to_number<uint64_t>();

    } else if (type_str == "EventRecordNode") {
      entry.type = bf::NODE_EVENT_RECORD;
      entry.event.event_id = no.at("params").as_object().at("event_id").to_number<int32_t>();

    } else if (type_str == "EventWaitNode") {
      entry.type = bf::NODE_EVENT_WAIT;
      entry.event.event_id = no.at("params").as_object().at("event_id").to_number<int32_t>();

    } else if (type_str == "EmptyNode") {
      entry.type = bf::NODE_EMPTY;
    }

    node_entries.push_back(entry);
  }

  // ---- Dependencies ----
  std::vector<bf::BinDependency> deps;
  deps.reserve(deps_array.size());
  for (const auto& dep_val : deps_array) {
    const json::object& d = dep_val.as_object();
    deps.push_back({d.at("from").to_number<uint32_t>(), d.at("to").to_number<uint32_t>()});
  }

  // ---- Generators ----
  std::vector<bf::BinGenerator> gens;
  gens.reserve(generators_array.size());
  for (const auto& gen_val : generators_array) {
    const json::object& g = gen_val.as_object();
    gens.push_back({g.at("id").to_number<uint64_t>(), g.at("seed").to_number<uint64_t>(),
                    g.at("wholegraph_increment").to_number<uint64_t>()});
  }

  // ---- JSON sections (allocator_events, output_tensors) ----
  std::string alloc_events_json =
      root.contains("allocator_events") ? json::serialize(root.at("allocator_events")) : "";
  std::string output_tensors_json =
      root.contains("output_tensors") ? json::serialize(root.at("output_tensors")) : "";

  // ---- Compute layout ----
  uint32_t num_sections = bf::SECTION_COUNT;
  uint64_t offset = sizeof(bf::FileHeader) + num_sections * sizeof(bf::SectionEntry);

  bf::SectionEntry sections[bf::SECTION_COUNT] = {};
  auto place_section = [&](bf::SectionType type, uint64_t size) {
    sections[type].section_type = type;
    sections[type].offset = offset;
    sections[type].size = size;
    offset += size;
    offset = (offset + 7) & ~7ULL;  // align to 8 bytes
  };

  place_section(bf::SECTION_STRING_TABLE, string_table.size());
  place_section(bf::SECTION_NODE_TABLE, node_entries.size() * sizeof(bf::BinNodeEntry));
  place_section(bf::SECTION_DEPENDENCY_TABLE, deps.size() * sizeof(bf::BinDependency));
  place_section(bf::SECTION_KERNEL_PARAM_INDEX, param_index.size() * sizeof(bf::BinParamEntry));
  place_section(bf::SECTION_KERNEL_PARAM_DATA, param_data.size());
  place_section(bf::SECTION_ARG_BUFFER_DATA, arg_buffer_data.size());
  place_section(bf::SECTION_COMMON_KERNEL_ATTRS,
                has_common_attrs ? sizeof(bf::BinCommonKernelAttrs) : 0);
  place_section(bf::SECTION_GENERATORS_TABLE, gens.size() * sizeof(bf::BinGenerator));
  place_section(bf::SECTION_ALLOCATOR_EVENTS, alloc_events_json.size());
  place_section(bf::SECTION_OUTPUT_TENSORS, output_tensors_json.size());
  place_section(bf::SECTION_TOPOLOGY_KEY, topology_key.size());

  // ---- Write ----
  std::ofstream out(bin_path, std::ios::binary);
  if (!out.is_open()) {
    fprintf(stderr, "[foundry] WARNING: Failed to write binary graph: %s\n", bin_path.c_str());
    return;
  }

  bf::FileHeader header = {};
  memcpy(header.magic, bf::MAGIC, 8);
  header.version = bf::FORMAT_VERSION;
  header.flags = 0;
  if (has_common_attrs)
    header.flags |= bf::FLAG_HAS_COMMON_KERNEL_ATTRS;
  if (!deps.empty())
    header.flags |= bf::FLAG_HAS_DEPENDENCIES;
  if (!alloc_events_json.empty())
    header.flags |= bf::FLAG_HAS_ALLOCATOR_EVENTS;
  if (!output_tensors_json.empty())
    header.flags |= bf::FLAG_HAS_OUTPUT_TENSORS;
  if (!gens.empty())
    header.flags |= bf::FLAG_HAS_GENERATORS;
  header.num_nodes = static_cast<uint32_t>(node_entries.size());
  header.num_dependencies = static_cast<uint32_t>(deps.size());
  header.num_generators = static_cast<uint32_t>(gens.size());
  header.num_sections = num_sections;
  header.num_strings = static_cast<uint32_t>(string_offsets.size());

  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  out.write(reinterpret_cast<const char*>(sections), sizeof(sections));

  auto write_padded = [&](const void* data, size_t size) {
    if (size > 0)
      out.write(static_cast<const char*>(data), size);
    size_t pad = ((size + 7) & ~7ULL) - size;
    if (pad > 0) {
      static const char zeros[8] = {};
      out.write(zeros, pad);
    }
  };

  write_padded(string_table.data(), string_table.size());
  write_padded(node_entries.data(), node_entries.size() * sizeof(bf::BinNodeEntry));
  write_padded(deps.data(), deps.size() * sizeof(bf::BinDependency));
  write_padded(param_index.data(), param_index.size() * sizeof(bf::BinParamEntry));
  write_padded(param_data.data(), param_data.size());
  write_padded(arg_buffer_data.data(), arg_buffer_data.size());
  if (has_common_attrs)
    write_padded(&common_attrs, sizeof(common_attrs));
  write_padded(gens.data(), gens.size() * sizeof(bf::BinGenerator));
  write_padded(alloc_events_json.data(), alloc_events_json.size());
  write_padded(output_tensors_json.data(), output_tensors_json.size());
  write_padded(topology_key.data(), topology_key.size());

  out.close();
}

// ============================================================================
// Binary reader: .cugraph -> boost::json::value
// Reconstructs the same JSON structure that read_and_parse_graph_json produces,
// but from binary data (no text tokenization, no hex decoding).
// ============================================================================

boost::json::value read_and_parse_binary_graph(const std::string& bin_path) {
  namespace json = boost::json;
  namespace bf = binary_format;

  // Read entire file
  std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
  if (!in.is_open())
    return json::value{};

  size_t file_size = in.tellg();
  in.seekg(0);
  std::vector<uint8_t> buf(file_size);
  in.read(reinterpret_cast<char*>(buf.data()), file_size);
  in.close();

  const uint8_t* data = buf.data();

  // Validate header
  if (file_size < sizeof(bf::FileHeader))
    return json::value{};
  const bf::FileHeader& header = *reinterpret_cast<const bf::FileHeader*>(data);
  if (!bf::validate_header(header))
    return json::value{};

  // Read section directory
  const bf::SectionEntry* sections =
      reinterpret_cast<const bf::SectionEntry*>(data + sizeof(bf::FileHeader));

  auto section = [&](bf::SectionType type) -> std::pair<const uint8_t*, size_t> {
    const auto& s = sections[type];
    return {data + s.offset, s.size};
  };

  // String table
  auto [str_data, str_size] = section(bf::SECTION_STRING_TABLE);

  // Helper to get string from table
  auto get_string = [&](uint32_t offset, uint32_t length) -> std::string {
    return std::string(reinterpret_cast<const char*>(str_data + offset), length);
  };

  // Helper to hex-encode binary data (for JSON compatibility with downstream)
  auto hex_encode = [](const uint8_t* bytes, size_t len) -> std::string {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; i++) {
      result.push_back(hex_chars[bytes[i] >> 4]);
      result.push_back(hex_chars[bytes[i] & 0x0f]);
    }
    return result;
  };

  // Node table
  auto [node_data, node_size] = section(bf::SECTION_NODE_TABLE);
  const bf::BinNodeEntry* node_entries = reinterpret_cast<const bf::BinNodeEntry*>(node_data);

  // Param index and data
  auto [pi_data, pi_size] = section(bf::SECTION_KERNEL_PARAM_INDEX);
  auto [pd_data, pd_size] = section(bf::SECTION_KERNEL_PARAM_DATA);
  auto [ab_data, ab_size] = section(bf::SECTION_ARG_BUFFER_DATA);

  // Common kernel attrs
  bf::BinCommonKernelAttrs common_attrs = {};
  bool has_common = header.flags & bf::FLAG_HAS_COMMON_KERNEL_ATTRS;
  if (has_common) {
    auto [ca_data, ca_size] = section(bf::SECTION_COMMON_KERNEL_ATTRS);
    if (ca_size >= sizeof(bf::BinCommonKernelAttrs)) {
      common_attrs = *reinterpret_cast<const bf::BinCommonKernelAttrs*>(ca_data);
    }
  }

  // Helper to build kernel_node_attrs JSON from binary flags
  auto build_kna_json = [](const bf::BinKernelNode& k) -> json::object {
    json::object kna;
    if (k.kna_flags & bf::KNA_ATTR_QUERY_AVAILABLE)
      kna["attrQueryAvailable"] = (bool)k.kna_attrQueryAvailable;
    if (k.kna_flags & bf::KNA_CLUSTER_DIM) {
      kna["clusterDimX"] = k.kna_clusterDimX;
      kna["clusterDimY"] = k.kna_clusterDimY;
      kna["clusterDimZ"] = k.kna_clusterDimZ;
    }
    if (k.kna_flags & bf::KNA_PREFERRED_CLUSTER_DIM) {
      kna["preferredClusterDimX"] = k.kna_preferredClusterDimX;
      kna["preferredClusterDimY"] = k.kna_preferredClusterDimY;
      kna["preferredClusterDimZ"] = k.kna_preferredClusterDimZ;
    }
    if (k.kna_flags & bf::KNA_CLUSTER_SCHEDULING)
      kna["clusterSchedulingPolicyPreference"] = k.kna_clusterSchedulingPolicy;
    if (k.kna_flags & bf::KNA_COOPERATIVE)
      kna["cooperative"] = k.kna_cooperative;
    if (k.kna_flags & bf::KNA_PRIORITY)
      kna["priority"] = k.kna_priority;
    if (k.kna_flags & bf::KNA_MEM_SYNC_DOMAIN)
      kna["memSyncDomain"] = k.kna_memSyncDomain;
    if (k.kna_flags & bf::KNA_MEM_SYNC_DOMAIN_MAP) {
      kna["memSyncDomainMapDefault"] = (int)k.kna_memSyncDomainMapDefault;
      kna["memSyncDomainMapRemote"] = (int)k.kna_memSyncDomainMapRemote;
    }
    if (k.kna_flags & bf::KNA_SHARED_MEM_CARVEOUT)
      kna["preferredSharedMemCarveout"] = k.kna_preferredSharedMemCarveout;
    return kna;
  };

  // ---- Build JSON nodes array ----
  json::array nodes_array;
  nodes_array.reserve(header.num_nodes);

  for (uint32_t i = 0; i < header.num_nodes; i++) {
    const bf::BinNodeEntry& entry = node_entries[i];
    json::object node_obj;
    node_obj["id"] = entry.node_id;

    switch (entry.type) {
      case bf::NODE_KERNEL: {
        node_obj["type"] = "KernelNode";
        const auto& k = entry.kernel;
        json::object params;

        params["blockDimX"] = k.blockDimX;
        params["blockDimY"] = k.blockDimY;
        params["blockDimZ"] = k.blockDimZ;
        params["gridDimX"] = k.gridDimX;
        params["gridDimY"] = k.gridDimY;
        params["gridDimZ"] = k.gridDimZ;
        params["sharedMemBytes"] = k.sharedMemBytes;

        params["function_name"] = get_string(k.function_name_offset, k.function_name_length);
        params["kernel_source_binary_hash"] = k.binary_hash;

        // func_attrs
        json::object func_attrs;
        func_attrs["max_dynamic_shared_size_bytes"] = k.max_dynamic_shared_size_bytes;
        func_attrs["preferred_shared_memory_carveout"] = k.preferred_shared_memory_carveout;
        func_attrs["cluster_scheduling_policy_preference"] = k.cluster_scheduling_policy_preference;
        func_attrs["required_cluster_width"] = k.required_cluster_width;
        func_attrs["required_cluster_height"] = k.required_cluster_height;
        func_attrs["required_cluster_depth"] = k.required_cluster_depth;
        params["func_attrs"] = func_attrs;

        // kernel_node_attrs
        params["kernel_node_attrs"] = build_kna_json(k);

        // Kernel params
        json::array kp_array;
        const bf::BinParamEntry* param_entries =
            reinterpret_cast<const bf::BinParamEntry*>(pi_data + k.param_index_offset);

        if (k.arg_buffer_offset != 0xFFFFFFFF) {
          // Extra-style params (metadata only, data in arg buffer)
          for (uint32_t j = 0; j < k.num_params; j++) {
            json::object kp;
            kp["index"] = (int)j;
            kp["offset"] = param_entries[j].data_offset;
            kp["size"] = param_entries[j].size;
            kp_array.push_back(kp);
          }

          // Reconstruct extra_argBuffer_hex from binary
          params["extra_argBuffer_hex"] =
              hex_encode(ab_data + k.arg_buffer_offset, k.arg_buffer_size);

          json::array extra;
          extra.push_back("CU_LAUNCH_PARAM_BUFFER_POINTER");
          extra.push_back("null");
          extra.push_back("CU_LAUNCH_PARAM_BUFFER_SIZE");
          extra.push_back((uint64_t)k.arg_buffer_size);
          extra.push_back("CU_LAUNCH_PARAM_END");
          params["extra"] = extra;
        } else {
          // Per-param hex data style
          for (uint32_t j = 0; j < k.num_params; j++) {
            json::object kp;
            kp["index"] = (int)j;
            kp["offset"] = param_entries[j].data_offset;
            kp["size"] = param_entries[j].size;
            kp["value_hex"] =
                hex_encode(pd_data + param_entries[j].data_offset, param_entries[j].size);
            kp_array.push_back(kp);
          }
          params["extra"] = json::array{};
          params["extra_argBuffer_hex"] = "";
        }
        params["kernelParams"] = kp_array;

        node_obj["params"] = params;
        break;
      }
      case bf::NODE_MEMCPY: {
        node_obj["type"] = "MemcpyNode";
        const auto& m = entry.memcpy;
        json::object params;
        params["Depth"] = m.Depth;
        params["Height"] = m.Height;
        params["WidthInBytes"] = m.WidthInBytes;
        params["dstDevice"] = m.dstDevice;
        params["dstHeight"] = m.dstHeight;
        params["dstLOD"] = m.dstLOD;
        params["dstMemoryType"] = m.dstMemoryType;
        params["dstPitch"] = m.dstPitch;
        params["dstXInBytes"] = m.dstXInBytes;
        params["dstY"] = m.dstY;
        params["dstZ"] = m.dstZ;
        params["srcDevice"] = m.srcDevice;
        params["srcHeight"] = m.srcHeight;
        params["srcLOD"] = m.srcLOD;
        params["srcMemoryType"] = m.srcMemoryType;
        params["srcPitch"] = m.srcPitch;
        params["srcXInBytes"] = m.srcXInBytes;
        params["srcY"] = m.srcY;
        params["srcZ"] = m.srcZ;
        node_obj["params"] = params;
        break;
      }
      case bf::NODE_MEMSET: {
        node_obj["type"] = "MemsetNode";
        const auto& m = entry.memset;
        json::object params;
        params["dst"] = m.dst;
        params["elementSize"] = m.elementSize;
        params["height"] = m.height;
        params["pitch"] = m.pitch;
        params["value"] = m.value;
        params["width"] = m.width;
        node_obj["params"] = params;
        break;
      }
      case bf::NODE_EVENT_RECORD:
        node_obj["type"] = "EventRecordNode";
        node_obj["params"] = json::object{{"event_id", entry.event.event_id}};
        break;
      case bf::NODE_EVENT_WAIT:
        node_obj["type"] = "EventWaitNode";
        node_obj["params"] = json::object{{"event_id", entry.event.event_id}};
        break;
      case bf::NODE_EMPTY:
        node_obj["type"] = "EmptyNode";
        node_obj["params"] = json::object{};
        break;
    }

    nodes_array.push_back(std::move(node_obj));
  }

  // ---- Build root JSON object ----
  json::object root;
  root["nodes"] = std::move(nodes_array);

  // Topology key
  auto [tk_data, tk_size] = section(bf::SECTION_TOPOLOGY_KEY);
  root["topology_key"] = std::string(reinterpret_cast<const char*>(tk_data), tk_size);

  // Common kernel attrs
  if (has_common) {
    // Build from the BinCommonKernelAttrs using a fake BinKernelNode
    bf::BinKernelNode fake = {};
    fake.kna_flags = common_attrs.flags;
    fake.kna_clusterDimX = common_attrs.clusterDimX;
    fake.kna_clusterDimY = common_attrs.clusterDimY;
    fake.kna_clusterDimZ = common_attrs.clusterDimZ;
    fake.kna_preferredClusterDimX = common_attrs.preferredClusterDimX;
    fake.kna_preferredClusterDimY = common_attrs.preferredClusterDimY;
    fake.kna_preferredClusterDimZ = common_attrs.preferredClusterDimZ;
    fake.kna_clusterSchedulingPolicy = common_attrs.clusterSchedulingPolicy;
    fake.kna_cooperative = common_attrs.cooperative;
    fake.kna_priority = common_attrs.priority;
    fake.kna_memSyncDomain = common_attrs.memSyncDomain;
    fake.kna_memSyncDomainMapDefault = common_attrs.memSyncDomainMapDefault;
    fake.kna_memSyncDomainMapRemote = common_attrs.memSyncDomainMapRemote;
    fake.kna_attrQueryAvailable = common_attrs.attrQueryAvailable;
    fake.kna_preferredSharedMemCarveout = common_attrs.preferredSharedMemCarveout;
    root["common_kernel_node_attrs"] = build_kna_json(fake);
  }

  // Dependencies
  if (header.flags & bf::FLAG_HAS_DEPENDENCIES) {
    auto [dep_data, dep_size] = section(bf::SECTION_DEPENDENCY_TABLE);
    const bf::BinDependency* deps = reinterpret_cast<const bf::BinDependency*>(dep_data);
    json::array deps_array;
    deps_array.reserve(header.num_dependencies);
    for (uint32_t i = 0; i < header.num_dependencies; i++) {
      json::object d;
      d["from"] = deps[i].from_id;
      d["to"] = deps[i].to_id;
      deps_array.push_back(d);
    }
    root["dependencies"] = std::move(deps_array);
  }

  // Generators
  if (header.flags & bf::FLAG_HAS_GENERATORS) {
    auto [gen_data, gen_size] = section(bf::SECTION_GENERATORS_TABLE);
    const bf::BinGenerator* gens = reinterpret_cast<const bf::BinGenerator*>(gen_data);
    json::array gens_array;
    gens_array.reserve(header.num_generators);
    for (uint32_t i = 0; i < header.num_generators; i++) {
      json::object g;
      g["id"] = gens[i].id;
      g["seed"] = gens[i].seed;
      g["wholegraph_increment"] = gens[i].wholegraph_increment;
      gens_array.push_back(g);
    }
    root["generators"] = std::move(gens_array);
  } else {
    root["generators"] = json::array{};
  }

  // Allocator events (embedded JSON text)
  if (header.flags & bf::FLAG_HAS_ALLOCATOR_EVENTS) {
    auto [ae_data, ae_size] = section(bf::SECTION_ALLOCATOR_EVENTS);
    std::string ae_str(reinterpret_cast<const char*>(ae_data), ae_size);
    root["allocator_events"] = json::parse(ae_str);
  } else {
    root["allocator_events"] = json::object{};
  }

  // Output tensors (embedded JSON text)
  if (header.flags & bf::FLAG_HAS_OUTPUT_TENSORS) {
    auto [ot_data, ot_size] = section(bf::SECTION_OUTPUT_TENSORS);
    std::string ot_str(reinterpret_cast<const char*>(ot_data), ot_size);
    root["output_tensors"] = json::parse(ot_str);
  }

  return json::value(std::move(root));
}

// ============================================================================
// Lightweight binary reader: just reads file + validates header.
// No JSON construction. Used for direct-access in prepare_on_demand_graph.
// ============================================================================

BinaryGraphFile read_binary_graph_file(const std::string& bin_path) {
  namespace bf = binary_format;
  BinaryGraphFile result;

  std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
  if (!in.is_open())
    return result;

  size_t file_size = in.tellg();
  if (file_size < sizeof(bf::FileHeader))
    return result;

  in.seekg(0);
  result.data.resize(file_size);
  in.read(reinterpret_cast<char*>(result.data.data()), file_size);
  in.close();

  memcpy(&result.header, result.data.data(), sizeof(bf::FileHeader));
  if (!bf::validate_header(result.header)) {
    result.data.clear();
    return result;
  }

  result.sections =
      reinterpret_cast<const bf::SectionEntry*>(result.data.data() + sizeof(bf::FileHeader));

  return result;
}

}  // namespace foundry
