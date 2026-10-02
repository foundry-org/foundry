# What a CUDA graph needs to be rebuilt in another process: a driver-API checklist against Foundry

Branch `sglang-registry` at `6d5bec5` (includes the cluster opt-in fix `8298912`). Local reading only: nothing
here was compiled or run. File:line refers to `foundry/` in the workspace. Claims about driver behaviour come from
the CUDA driver API documentation as I know it. They are marked **[doc]** where the code does not show them, and
**[unverified]** where I could not confirm them from either source.

Legend for each row:

- **Rec**: SAVE writes it to the archive.
- **App**: LOAD applies it before or while it adds the node.
- **Det**: a missing or mismatched item fails loudly (at SAVE or at LOAD) rather than silently.
- **Test**: a test in `foundry/tests` covers it.

The load paths are named explicitly, because a fact that holds on one path is often false on another:

- **SER**: `CUDAGraph::load`, the serial JSON path. Exposed as `foundry.graph.CUDAGraph.load`
  (python/foundry/graph.py:71).
- **JT**: `build_graph_from_parsed`, the JSON template builder (CUDAGraphParallel.cpp:266). Used when a graph has
  no `.cugraph` file or no `FLAG_COMPLETE_KERNEL_ATTRS`.
- **BT**: `build_template_graph_binary`, the binary template builder (CUDAGraphParallel.cpp:1707).
- **MEM**: the member rewrite. `prepare_on_demand_graph[_binary]` (CUDAGraphParallel.cpp:998 / :1321) prepares it,
  and `apply_on_demand_updates` (CUDAGraph.cpp:521) applies it before each member's own `cudaGraphInstantiate`
  (CUDAGraph.cpp:722).
- **PW**: `build_prewarm_graph` (CUDAGraphParallel.cpp:1936). It builds a kernel-only pool-warming graph and sets
  no node attributes on purpose.

## 0. The root cause of the whole class

On LOAD, the hook passes every module and library load straight through (`skip_fatbin_processing`, set from
`FOUNDRY_MODE=load` in `init_hook`, hook.cpp:1995, and checked in each load hook, e.g. `cuLibraryLoadData`
hook.cpp:2142). `load_cuda_modules_and_libraries` (hook.cpp:4544) then loads a second, separate copy of every
archived binary. Graph nodes resolve to that copy (`query_function_handle`, hook.cpp:4371).

As a result, all per-module and per-function state lives on the copy the process's own eager code uses (cudart's
registration, or the library's own `cuModuleLoadData`). That state never reaches the functions and module globals the
rebuilt graph launches:

- `cudaFuncSetAttribute`;
- `cudaMemcpyToSymbol` / `__device__` and `__constant__` initialisation;
- `cuFuncSetCacheConfig`;
- NVSHMEM's device state.

This is true even when the LOAD process runs exactly the same init code as SAVE. So every such item has to be
recorded and re-applied (or derived) by Foundry itself. Today Foundry does that for four items:

- dynamic smem cap;
- carveout;
- the non-portable cluster opt-in, which is derived rather than recorded;
- NVSHMEM device state (`init_nvshmem_for_loaded_modules`, hook.cpp:1483).

## 1. Function-level state (CUfunction / CUkernel)

| Item | Rec | App | Det | Test | Notes / code |
|---|---|---|---|---|---|
| `MAX_DYNAMIC_SHARED_SIZE_BYTES` (opt-in above 48 KB) | yes: `func_attrs.max_dynamic_shared_size_bytes`, floored at the node's `sharedMemBytes` (CUDAGraph.cpp:1541-1570). BinKernelNode `max_dynamic_shared_size_bytes` | yes, on all paths: `apply_saved_function_attributes` (CUDAGraph.cpp:2516; SER :2060, JT CUDAGraphParallel.cpp:519, member JSON :1120, binary :1446) plus the per-node `ensure_dynamic_smem_optin` in `add_restored_kernel_node` (CUDAGraph.cpp:2546) and MEM (:545). High-water mark per handle, and on the CUkernel **and** its per-context CUfunction (:2473) | partly: a failed set is only logged (:2463); `cuGraphAddKernelNode` then fails loudly | no unit test; model runs only | Correct design: recording the node's own smem as the cap was an earlier bug (comment :1532) |
| `PREFERRED_SHARED_MEMORY_CARVEOUT` (function attribute) | yes (:1545/:1559) | yes (:2526/:2533), `C10_CUDA_DRIVER_CHECK` so fatal on error | yes (fatal) | no | Set on the CUkernel only, not on `cuKernelGetFunction`'s per-context function, unlike smem and the cluster opt-in. ClusterOptIn.h says a per-context function does not inherit the CUkernel's value. Performance hint only |
| `CLUSTER_SCHEDULING_POLICY_PREFERENCE` (function) | yes (:1548/:1562) | **no**, on purpose (comment :2518: fails with invalid-cluster-size) | n/a | no | Hint only. The node-level attribute of the same name is applied (section 2) |
| `REQUIRED_CLUSTER_WIDTH/HEIGHT/DEPTH` | yes (:1550-1565), binary `required_cluster_*` | not set on the function. Merged into the node's `CLUSTER_DIMENSION` on every path (SER :2045-2058, JT :500-515, BT `apply_binary_kernel_attrs` :1618-1625, MEM prep :1321+166) | yes: `add_kernel_node_cluster_optin` retries from the driver's compiled dims on 912 | `test_cluster_optin.py` cases 2-3 | **[doc]** these are settable at runtime when not fixed at compile time (NOT_PERMITTED otherwise). A runtime-set value is recorded and replayed as a node cluster dim, which gives the same launch shape |
| `NON_PORTABLE_CLUSTER_SIZE_ALLOWED` | no (derived) | yes, all LOAD paths plus MEM, via ClusterOptIn.h `ensure_cluster_size_optin` when the cluster has more than 8 blocks (CUkernel + per-context function + CUfunction) | yes (912 retry and log) | `test_cluster_optin.py` 1-5 | **Gap:** non-portable *common* (graph-wide) cluster dims are not opted in (dsv4_cluster_fix.md). They would fail at the batch `cuGraphKernelNodeSetAttribute` (CUDAGraphParallel.cpp:815) |
| `cuFuncSetCacheConfig` / `cuCtxSetCacheConfig` (L1/shared preference) | **no** | no | no | no | Performance only. Superseded by the carveout on sm_70+ **[doc]** |
| `cuFuncSetSharedMemConfig` (bank size) | no | no | no | no | Deprecated, no effect on current GPUs **[doc]** |
| Read-only attributes (`MAX_THREADS_PER_BLOCK`, `NUM_REGS`, `LOCAL_SIZE_BYTES`, `PTX/BINARY_VERSION`, `CACHE_MODE_CA`, `CLUSTER_SIZE_MUST_BE_SET`) | no | n/a | not compared | no | They follow from the binary, which is identical by hash. They would only matter if LOAD picked different SASS (other GPU or driver; section 5) |
| CUkernel vs CUfunction semantics (`cuKernelSetAttribute(dev)` vs the per-context function) | n/a | handled for smem and the cluster opt-in (both handles). Carveout sets the CUkernel only | n/a | cluster test only | MEM retries `cuGraphKernelNodeSetParams` with the per-context function when the kern-based call is refused (CUDAGraph.cpp:556-572) |
| Lazy loading (`CUDA_MODULE_LOADING=LAZY`), `cuKernelGetFunction` | not recorded | implicit: the per-context function is created at `cuKernelGetFunction` / node add | failures logged (:2490) | no | Correctness-neutral as long as attributes are applied to both handles (smem and the opt-in are; the carveout is not) |
| **Any other `cudaFuncSetAttribute` the capturing process made** (the general class) | **no** | **no** | **no** until the driver refuses the node (912 was this) | no | Foundry derives three attributes instead of snapshotting what was set. See the gap list, item 1 |

## 2. Launch attributes on kernel nodes

The SAVE query is in `analyze_captured_graph` (CUDAGraph.cpp:960-1063). If the first query
(`CLUSTER_DIMENSION`, :963) fails, `attr_query_available=false` and **every** other attribute is skipped silently.

| Attribute | Rec (JSON) | Binary node table | Applied: SER / JT / BT / MEM | Notes |
|---|---|---|---|---|
| `CLUSTER_DIMENSION` | yes when >1 (:1344) | `KNA_CLUSTER_DIM` | yes / yes / yes / yes (MEM first resets to 1x1x1, :531-543) | In the topology key (:1664) unless `FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0` |
| `PREFERRED_CLUSTER_DIMENSION` | yes (:1351) | `KNA_PREFERRED_CLUSTER_DIM` | yes / yes / yes / yes | |
| `CLUSTER_SCHEDULING_POLICY_PREFERENCE` | yes (:1360) | yes | yes / yes / yes / yes | |
| `COOPERATIVE` | yes (:1365) | yes | yes / yes / yes / yes (only when the member has a value; sentinel -1 keeps the template's) | **SER ignores `common_kernel_node_attrs`**. A graph where every kernel is cooperative would silently lose it on SER |
| `PRIORITY` | yes (:1368) | yes | yes / yes / yes / yes (same sentinel) | Only effective with `INSTANTIATE_FLAG_USE_NODE_PRIORITY`, which no path uses (section 4) **[doc]** |
| `MEM_SYNC_DOMAIN`, `MEM_SYNC_DOMAIN_MAP` | yes (:1371/:1374) | yes | yes / yes / yes / yes | |
| `PREFERRED_SHARED_MEMORY_CARVEOUT` (node) | yes (:1381) | yes | yes / yes / yes / yes | |
| `ACCESS_POLICY_WINDOW` | yes, per node, with the raw `base_ptr` (:1387) | **no** (clears `FLAG_COMPLETE_KERNEL_ATTRS`, BinaryGraphIO.cpp:110) | yes (:2229) / per node yes (:688), **common no** / n/a (falls back to JT) / **no** | No effect without an L2 persisting set-aside, which is not recorded (section 5). The pointer is in the VMM region if the window was |
| `DEVICE_UPDATABLE_KERNEL_NODE` | flag yes. `devNode` handle recorded in memory only, not written | `KNX_DEVICE_UPDATABLE` (ext) | yes / per node yes, **common no** / yes / **no** | The `CUgraphDeviceNode` handle the app keeps for device-side updates changes at LOAD. Not handled. Not detected |
| `PROGRAMMATIC_STREAM_SERIALIZATION` (PDL) | yes (:1052-1062, :1405) | `KNX_PROGRAMMATIC_STREAM_SERIALIZATION` (ext) | **SER: no** (never read) / yes, per node and common (:891) / yes, per node / **no** (member inherits the template's) | The topology key does not contain it |
| `PROGRAMMATIC_EVENT`, `LAUNCH_COMPLETION_EVENT` | captured as edge data (programmatic / launch-order ports) **[doc]** | `BinDependency` v2 ports and type | yes on SER, JT and BT (`add_graph_dependencies`). MEM: deps are **stripped** from member files (graph.py:243-255), so the template's edges are used | `test_graph_dependencies.py` |
| `IGNORE`-style / newer attributes (`NVLINK_UTIL_CENTRIC_SCHEDULING`, portable-cluster-size mode) **[unverified names/versions]** | no | no | no | Not queried at SAVE: a node carrying one would be rebuilt without it, silently |
| Node params: grid, block, `sharedMemBytes`, `kernelParams` bytes, `extra` arg buffer | yes (:1331-1490) | yes | yes on all paths | `kernelParams` sizes come from `cuFuncGetParamInfo(params.func, …)` (:1068). **[unverified]** If a node reports only `kern` (func NULL), this returns no params and the node is saved with none. `cuKernelGetParamInfo` would be the kern-side call |
| `CUDA_KERNEL_NODE_PARAMS.ctx` | read (:955), not written | no | LOAD uses the current context | See green contexts (section 5) |

## 3. Node types

| Node type | Handling | How an unsupported one fails |
|---|---|---|
| Kernel | supported (all of the above) | |
| Memcpy (`CUDA_MEMCPY3D`, 1D/2D/3D, pitch, offsets) | supported, device to device. The context comes from the current one at LOAD | Arrays, host pointers and `reserved*` are rejected by `TORCH_CHECK` in `save()` (CUDAGraph.cpp:1591-1596), i.e. at SAVE, after capture. **Not detected:** a copy recorded with `CU_MEMORYTYPE_UNIFIED`/DEVICE whose address is pinned host memory under UVA **[unverified whether capture reports srcHost or a UVA device pointer]**. LOAD would copy from whatever lives at that address |
| Memset | supported | |
| Event record / wait | supported *structurally*. SER, JT and BT create a new `CU_EVENT_DEFAULT` event per `event_id` per graph (CUDAGraph.cpp:2317-2340, CUDAGraphParallel.cpp:772-790, :1321+238) | **Silent:** these nodes exist only for external events (`cudaEventRecordWithFlags(External)` / `cudaStreamWaitEvent(External)`) **[doc]**. Their purpose is a link to something outside the graph (host code, another stream or another graph), and LOAD's private event breaks that link. A wait node on a never-recorded event completes at once, so the result is a race, not an error. Event flags (DISABLE_TIMING, INTERPROCESS) are lost |
| Empty | supported | |
| Host (`cudaLaunchHostFunc` in capture) | rejected | SAVE: `TORCH_CHECK(false, "Graph contains unsupported node type!")` (CUDAGraph.cpp:1205), raised in `analyze_captured_graph`. The message gives no type number and no neighbouring kernel |
| Child graph | rejected | same (SAVE) |
| Mem alloc / free (graph-owned, `cudaMallocAsync` in capture) | rejected | same (SAVE). Instantiate flags diverge anyway: the template uses `AutoFreeOnLaunch` (CUDAGraph.cpp:504), members use 0 (:727) |
| Conditional (if/while/switch, 12.3+) | rejected | same (SAVE) |
| External semaphore signal / wait | rejected | same (SAVE) |
| Batch mem-op (`cuStreamWaitValue`/`WriteValue` in capture) | rejected | same (SAVE) |
| Unknown future types | rejected | same (SAVE) |

## 4. Graph level

| Item | Rec | App | Det | Notes |
|---|---|---|---|---|
| Instantiate flags | no | template exec: `AutoFreeOnLaunch` when the driver is >= 11.4 (CUDAGraph.cpp:479-513). Member exec: 0 (:727) | n/a | Matches torch's capture-time flags for the template. `USE_NODE_PRIORITY`, `DEVICE_LAUNCH` and `UPLOAD` are never used. A graph meant for device launch, or whose node priorities matter, would behave differently **[doc]**. No upload: the first launch pays it |
| Common kernel attributes ("applied last") | yes: extracted only when **all** kernel nodes have identical attributes (CUDAGraph.cpp:1726-1777) | JT (:811-903) and BT (:1806-1814) apply them. **SER does not read them at all**. JT's common block skips `accessPolicyWindow` and `deviceUpdatable` | no | The binary writer sends common PSS / deviceUpdatable / APW graphs to JT (BinaryGraphIO.cpp:110-122). JT applies common PSS but drops common APW and deviceUpdatable silently |
| Edges (ports, type; `cuGraphAddDependencies_v2`) | yes (CUDAGraph.cpp:1211-1245, :1782-1798), binary v2 | SER, JT, BT (`add_graph_dependencies`), PW (:2002-2015) | `add_graph_dependencies` fails loudly | `test_graph_dependencies.py` |
| Topology key (member grouping) | node type sequence + cluster dims only (CUDAGraph.cpp:1664-1723, LOAD fallback CUDAGraphParallel.cpp:2404) | n/a | **no** | **It excludes edges and ports, function identity, PSS, APW, deviceUpdatable, and whether priority / cooperative / carveout are set.** `save_graph_manifest` strips member deps without comparing them (graph.py:243-255). MEM leaves template values for anything the member does not set (sentinel -1) |
| Node order / ids | yes | yes (`ordered_nodes`) | `updates.size()==num_nodes` check (:1719) | |
| RNG generator state (Philox seed and offset per graph) | yes (`generators`, :1800-1815) | yes (`register_generator_state`, under `SuspendAllocationRegion`, :1881-1898) | registry lookup | Torch-level, not driver, but also per-process implicit state |
| Output tensors | yes (metadata plus raw `data_ptr`) | `from_blob` at the recorded address | through the VMM address checks | |

## 5. Module / binary and per-process state

| Item | Rec | App | Det | Notes |
|---|---|---|---|---|
| Code image the node needs (fatbin / cubin / PTX, by CRC64 hash + mangled name) | yes: hooks on `cuModuleLoadData(Ex)`, `cuModuleLoadFatBinary`, `cuModuleLoad`, `cuLibraryLoadData`, `cuLibraryLoadFromFile` (hook.cpp:2015-2250). JIT options and library options are recorded (hook.cpp:580-740; log-buffer options ignored :640) | loaded by `load_cuda_modules_and_libraries` (hook.cpp:4544) with the same options | SAVE: a node whose module came through an unhooked entry point aborts in `query_binary_hash` ("handle not found", hook.cpp:4410). LOAD: missing hash or name aborts (:4371-4405). Function-count mismatch aborts (:1570) | JIT-produced cubins (Triton, DeepGEMM, nvrtc, flashinfer, inductor) are covered because they arrive through `cuModuleLoadData`/`cuLibraryLoad*` |
| Multi-segment device-linked fatbins (`FATBINC_LINK_VERSION`, rdc, NVSHMEM) | segments recorded | `cuLink*` at pack time (`prelink_fatbin_segments`, hook.cpp:1700) or at LOAD (:4877) | a link failure is a warning with fallback (:1731-1757) | The link result depends on the driver's JIT linker, so a compat-driver change can change it |
| PTX inputs | stored as PTX | re-JITted at LOAD | no | Different driver means possibly different SASS, and a JIT cost at LOAD |
| GPU arch / SM count / driver version / compat driver / `CUDA_MODULE_LOADING` | integration `warmup_state.json` stores `gpu_name` and `cuda_version` (sglang runtime.py:94-106, vllm runtime.py:74-80) | **never compared at LOAD** | only indirectly (no SASS: module load fails; cluster occupancy differs: 912, etc.) | A different driver can pick different SASS or PTX-JIT output. A different SM count changes occupancy-derived grids that are baked into params |
| `__device__` / `__constant__` globals written from the host before capture (`cudaMemcpyToSymbol`, `cuModuleGetGlobal` + copy) | **no** | **no** (LOAD's archive copy of the module starts from the ELF initialisers, see section 0) | **no** | Exception: NVSHMEM's `nvshmemi_device_state_d`, detected at SAVE by symbol probe (hook.cpp:1191-1240) and initialised at LOAD (hook.cpp:1393-1483, called from the integration after `prepare_communication_buffer_for_model`) |
| Addresses of module globals passed as kernel params (`cudaGetSymbolAddress`) | raw bytes only | replayed raw | **no** | Module globals are allocated by the driver at module load, not through the hooked `cuMemAlloc_v2`, so the LOAD copy's addresses differ **[unverified that no model hits this]** |
| Texture / surface objects (64-bit handles in params) | raw bytes | replayed raw | **no** | Handles are per-process. Not hooked |
| `cuFuncSetAttribute` calls (history) | no | no | no | See section 1 |
| Device limits: `cudaLimitMallocHeapSize` (device `malloc`), `StackSize`, `PrintfFifoSize`, `DevRuntimePendingLaunchCount`, `PersistingL2CacheSize` | no | no | no | Only the heap size can break correctness (device malloc returns NULL). The L2 set-aside makes recorded access-policy windows no-ops |
| Contexts: green contexts / exec affinity (`cuCtxCreate_v3/v4` are hooked for context bookkeeping only, hook.cpp:2291-2333), MPS | node `ctx` not written | LOAD adds nodes in the current primary context | no | SM partitioning of the capturing context is lost |
| Stream priorities | n/a | graphs launch on the framework's current stream | n/a | Same framework code in SAVE and LOAD |
| Peer access (legacy `cudaDeviceEnablePeerAccess`) | no | region memory is granted to the local device only (hook.cpp:2567-2572) | no | Multi-GPU pointers come through the VMM IPC path instead: `cuIpcOpenMemHandle` maps a peer's chunk at the exporter's original VA (hook.cpp:3501-3700) |
| NVSHMEM / DeepEP buffers | reserve events (`type: reserve`) replayed as a pointer advance (hook.cpp:4316-4336) | NVSHMEM maps its own buffers later; module init as above | reserve-address mismatch aborts | `test_deepep_fabric.py` |
| torch symm-mem / multicast (NVLS) | integration | integration re-rendezvous | integration validates the rendezvous before replay | `test_symm_mem.py` (two_shot / multimem / one_shot) |
| NCCL communicators / NCCL-registered buffers | outside Foundry (the integration re-inits the communicators) | — | — | Graph kernels embed communicator device pointers. They are correct only if NCCL's allocations replay into the region at the same addresses |

## 6. Memory

| Item | Rec | App | Det | Notes |
|---|---|---|---|---|
| `cuMemAlloc_v2` / `cuMemAllocPitch_v2` inside the VMM region (torch caching-allocator segments and graph pool included) | alloc/free event log per graph (`allocator_events`), plus `start_base_addr` | replayed in order (hook.cpp:4251-4345) | LOAD: `Memory offset mismatch` abort (:4260) and `Allocation address mismatch` abort (:4296) | `test_vmm_alloc.py`, `test_preallocation.py`, `test_load_graph.py` |
| `cuMemAddressReserve` (expandable segments, symm-mem, NVSHMEM) | reserve events | pointer advance | mismatch abort | |
| Allocation that lands outside the region | — | — | `[HOOK] ERROR: Allocated address … is outside allocation region` (hook.cpp:2589 / :2769). **Printed only, not fatal** | |
| Pointers in kernel params / memcpy / memset that were **never** in the region: memory from before `set_allocation_region`, `cuMemAllocAsync` / `cudaMallocAsync` pools, `cuMemAllocManaged`, `cuMemHostAlloc` / pinned or zero-copy host memory, driver-internal allocations | raw bytes | replayed raw | **no** | None of these entry points is hooked. There is no SAVE-time scan of param words. LOAD works only if the same object happens to sit at the same address |
| Device memory **contents** the graph reads but does not produce (weights, warm-up-initialised workspaces, lookup tables, split-k semaphores, TMA descriptors stored in memory) | no | no | no | Correct only if 2nd SAVE and LOAD run identical init (CLAUDE.md rule). Nothing checks it |
| Host-pinned buffers used by memcpy nodes | rejected when reported as host (section 3) | — | partly | |
| Unified memory | no | no | no | |

## Ranked gaps (likelihood x blast radius for sglang/vLLM serving)

1. **Function attributes in general.** Foundry derives smem, the carveout and the non-portable opt-in, but it
   neither records nor diffs what the capturing process set. The next `cudaFuncSetAttribute`-style bug fails the same
   way 912 did, or silently. How to detect at SAVE: for every function a node uses, query all settable attributes
   (`cuKernelGetAttribute` for the CUkernel **and** for the per-context function) and store them in `func_attrs`. At
   LOAD, after adding the node, re-query the resolved handle and log a one-line diff for any attribute that differs.
   Cheap and generic. Also set the carveout on the per-context function too.
2. **Module globals / `__constant__` written by the host before capture.** LOAD's archive module copy never sees
   those writes (section 0). Only NVSHMEM's state is handled. Result: silent wrong results. How to detect at SAVE:
   list the `STT_OBJECT` symbols of each used binary from the ELF symbol table, read them at capture end with
   `cuModuleGetGlobal`/`cuLibraryGetGlobal` and D2H, and store any that differ from the image's initialiser. At
   LOAD, write them into the archive copy. Alternatively, hook `cuModuleGetGlobal` + `cuMemcpyHtoD*` to flag writes.
3. **The topology key ignores edges, ports and the attributes MEM does not rewrite.** Those attributes are PSS,
   APW, deviceUpdatable, and whether priority / cooperative / carveout are set. Member deps are stripped unchecked
   (graph.py:243-255), so a member can run with the template's ordering, silently. How to detect at SAVE: add a hash
   of (deps + ports + types) and of those attributes to `topology_key`, or compare member deps to the template's in
   `save_graph_manifest` before stripping.
4. **Pointers outside the VMM region.** The sources are async pools, managed memory, pinned or zero-copy host
   memory, pre-region allocations and symbol addresses. The out-of-region message is non-fatal, and nothing scans
   params. How to detect at SAVE: scan 8-byte-aligned words of each node's params, memcpy and memset addresses. For
   each word that `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` accepts, check that it is inside the
   region (or a known IPC/NVSHMEM range), and warn with the kernel name and param index.
5. **Device memory contents not produced by the graph.** Warm-up-initialised workspaces and tables rely on SAVE2 and
   LOAD taking the same init path. Nothing verifies it. How to detect: a debug mode that checksums the live region
   ranges (`get_live_region_ranges`, hook.cpp:4139) right before the first replay, in SAVE and in LOAD, and diffs
   them.
6. **External event record / wait nodes.** Each is rebuilt with a private new event, so the external link is lost
   and a wait becomes a no-op race. Event flags are lost too. How to detect at SAVE: fail, or at least warn, on any
   event node whose partner is not in the same graph. Better, record the event's creator or site.
7. **Path divergence.**
   - SER (`CUDAGraph.load`, graph.py:71) ignores `common_kernel_node_attrs` and PSS completely. A graph whose
     kernels are all cooperative or all PDL loses those attributes.
   - JT drops *common* APW and deviceUpdatable.
   - MEM never applies PSS, APW or deviceUpdatable.
   - Non-portable common cluster dims get no opt-in.

   Fix: route all paths through one attribute-apply helper, and give it one unit test that runs the JSON builder
   and the binary builder on the same archive.
8. **Environment identity is never checked at LOAD.** That covers GPU CC and SM count, driver and compat version,
   and PTX-JIT or cuLink output. `warmup_state.json` already stores `gpu_name` and `cuda_version` but does not
   compare them. How to detect: record `cuDriverGetVersion`, CC, name, multiprocessor count and
   `CUDA_MODULE_LOADING` in the archive, and refuse with a clear message on a mismatch.
9. **SAVE-time diagnostics for unsupported nodes.** The error is a generic "unsupported node type!" with no type
   and no neighbouring kernel. The host-memcpy rejection only happens in `save()`. A UNIFIED/UVA memcpy to pinned
   host memory is not detected. How to improve: print the `CUgraphNodeType` and the preceding kernel's name, and
   check `cuPointerGetAttribute(MEMORY_TYPE)` on memcpy endpoints.
10. **Rare per-process context state.**
    - green contexts / exec affinity (node `ctx` is dropped);
    - device limits (malloc heap, stack, L2 set-aside);
    - texture and surface objects;
    - `CUgraphDeviceNode` handles;
    - device-launch or node-priority instantiate flags;
    - **[unverified]** kern-only nodes losing their params through `cuFuncGetParamInfo(func=NULL)`.

    How to detect at SAVE: hook `cuTexObjectCreate`/`cuSurfObjectCreate`/`cuCtxSetLimit`/green-context creation and
    warn if any was called before or during capture. Assert `params.func != nullptr || num_params > 0` for every
    kernel node that has `kernelParams`.
