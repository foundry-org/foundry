#pragma once

// Pure CPU policy for the optional archive LOAD experiment. Event identities
// may be normalized only when both handles were privately created by LOAD and
// every record/wait use has exactly the same node position and kind.
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace foundry {
using ResearchEventUsage = std::map<uintptr_t, std::vector<std::pair<size_t, int>>>;

inline std::map<uintptr_t, uintptr_t> research_private_event_aliases(
    const ResearchEventUsage& source, const ResearchEventUsage& target,
    const std::set<uintptr_t>& source_owned, const std::set<uintptr_t>& target_owned) {
  if (source.size() != target.size())
    throw std::runtime_error("research private event group counts differ");
  for (const auto& [event, usage] : source)
    if (!event || !source_owned.count(event) || usage.empty())
      throw std::runtime_error("research source event lacks private LOAD ownership");
  std::map<uintptr_t, uintptr_t> aliases;
  std::set<uintptr_t> used;
  for (const auto& [event, usage] : target) {
    if (!event || !target_owned.count(event) || usage.empty())
      throw std::runtime_error("research target event lacks private LOAD ownership");
    uintptr_t match = 0;
    for (const auto& [candidate, candidate_usage] : source) {
      if (usage == candidate_usage) {
        if (match)
          throw std::runtime_error("research private event usage mapping is ambiguous");
        match = candidate;
      }
    }
    if (!match || !used.insert(match).second)
      throw std::runtime_error("research private event usage lacks a unique bijection");
    aliases[event] = match;
  }
  return aliases;
}
}  // namespace foundry
