#pragma once

// LOAD exec mode, read from FOUNDRY_ENABLE_EXEC_SHARE.
//   unset / 0 (eager, default): every graph gets its own exec in Phase 2, plus the
//     exec-pool prewarm.
//   1 (share): only template execs are instantiated (no prewarm, no member exec); a
//     member replay switches its template's exec to the member with cuGraphExecUpdate.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "CgpWeak.h"

namespace foundry::exec_share {

inline bool enabled() {
  const char* env = std::getenv("FOUNDRY_ENABLE_EXEC_SHARE");
  return env && std::strcmp(env, "1") == 0;
}

// The member-switch route, from the weak cgp_* pointers (CgpWeak.h).
inline bool patched_route() {
  return cgp_exec_update != nullptr;
}

// Once per process, at the first share-mode Phase 2.
inline void log_mode_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    fprintf(stderr, patched_route() ? "[foundry] exec share: patched route (cgp)\n"
                                    : "[foundry] exec share: real cuGraphExecUpdate\n");
  });
}

}  // namespace foundry::exec_share
