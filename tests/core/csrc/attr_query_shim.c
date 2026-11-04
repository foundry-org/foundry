// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foundry project
//
// LD_PRELOAD shim for tests/core/test_attr_query.py: cuGraphKernelNodeGetAttribute
// returns CUDA_ERROR_INVALID_VALUE for the attribute ids listed in FOUNDRY_TEST_FAIL_ATTRS
// (comma-separated ints) and forwards every other query to the driver. At exit it prints
// how many queries it failed.
//
// The driver function is looked up with dlsym(RTLD_NEXT) on purpose: the harness
// preloads libcuda_hook.so ahead of this library and the hook interposes dlsym, so
// this doubles as the regression test for the hook's RTLD_NEXT handling (it must
// resolve relative to this shim, not to the hook; the self-check below aborts
// otherwise).

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef void* CUgraphNode;
typedef int CUkernelNodeAttrID;
typedef union CUkernelNodeAttrValue CUkernelNodeAttrValue;
typedef CUresult (*get_attr_fn)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue*);

#define CUDA_ERROR_INVALID_VALUE 1
#define MAX_FAIL 32

static get_attr_fn real_get_attr;
static int fail_ids[MAX_FAIL];
static int num_fail;
static unsigned long failed_queries;
static pthread_once_t once = PTHREAD_ONCE_INIT;

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode node, CUkernelNodeAttrID id,
                                       CUkernelNodeAttrValue* value);

static void init(void) {
  const char* env = getenv("FOUNDRY_TEST_FAIL_ATTRS");
  if (env) {
    char buf[256];
    strncpy(buf, env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char* tok = strtok(buf, ","); tok && num_fail < MAX_FAIL; tok = strtok(NULL, ","))
      fail_ids[num_fail++] = atoi(tok);
  }
  real_get_attr = (get_attr_fn)dlsym(RTLD_NEXT, "cuGraphKernelNodeGetAttribute");
  if (!real_get_attr || real_get_attr == cuGraphKernelNodeGetAttribute) {
    fprintf(stderr,
            "[attr_query_shim] cannot resolve the driver's cuGraphKernelNodeGetAttribute\n");
    abort();
  }
  fprintf(stderr, "[attr_query_shim] active, failing attribute ids: %s\n", env ? env : "(none)");
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode node, CUkernelNodeAttrID id,
                                       CUkernelNodeAttrValue* value) {
  pthread_once(&once, init);
  for (int i = 0; i < num_fail; i++) {
    if (fail_ids[i] == id) {
      __atomic_add_fetch(&failed_queries, 1, __ATOMIC_RELAXED);
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  return real_get_attr(node, id, value);
}

__attribute__((destructor)) static void report(void) {
  if (num_fail)
    fprintf(stderr, "[attr_query_shim] failed %lu queries\n", failed_queries);
}
