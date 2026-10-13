#pragma once

// cudagraph-patch entry points used by exec share mode, declared weak so that
// foundry.ops links and runs without the library. They resolve only when
// libcudagraph_patch.so is in the global scope before foundry.ops loads
// (LD_PRELOAD); otherwise they are null and share mode uses the real
// cuGraphExecUpdate_v2. The pointers are the whole detection: no probe, no env.
//
// Prototypes and cgp_census_t copied from cudagraph-patch
// include/cudagraph_patch/cgp.h, CGP_ABI_VERSION 1 (commit f0351bd). Return 0
// on success, a negative cgp_error_t, or (cgp_exec_update) the positive
// CUresult of the real update.

extern "C" {
typedef struct cgp_census {
  unsigned struct_size;   /* sizeof(cgp_census_t), for forward compatibility */
  void* const* functions; /* CUfunction per public kernel node, any order; may repeat */
  unsigned function_count;
  unsigned pdl_edges;            /* programmatic dependency edges in the graph */
  unsigned memcpy_nodes;         /* public memcpy nodes (0 is a valid, explicit value) */
  const unsigned* helper_counts; /* optional allowlist of acceptable internal helper counts */
  unsigned helper_count_n;       /* 0: use CUDAGRAPH_PATCH_HELPERS or the default rule */
  unsigned flags;                /* cgp_census_flags */
} cgp_census_t;

// Around the pristine instantiate of the exec, on one thread:
// begin, expect(&census), instantiate, end(exec, public_kernel_count).
int cgp_exec_register_begin(void) __attribute__((weak));
int cgp_exec_register_expect(const cgp_census_t* census) __attribute__((weak));
int cgp_exec_register_end(void* exec, unsigned public_kernel_count) __attribute__((weak));
// prepare + cuGraphExecUpdate_v2 + finish; update_info is a CUgraphExecUpdateResultInfo*.
int cgp_exec_update(void* exec, void* graph, void* update_info) __attribute__((weak));
// Before the exec is destroyed.
int cgp_exec_forget(void* exec) __attribute__((weak));
}
