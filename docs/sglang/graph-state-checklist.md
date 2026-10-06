# What Foundry must take care of to restore an SGLang CUDA graph: merged checklist

Merged from two independent reviews (2026-09-30, code read at sglang upstream `fa090f7755` and foundry
`sglang-registry` 6d5bec5; nothing run): [graph-state-checklist-driver.md](graph-state-checklist-driver.md) (from the CUDA driver API) and
[graph-state-checklist-sglang.md](graph-state-checklist-sglang.md) (from SGLang's flags, backends and kernels). Both files keep the file:line
evidence; this document is the reviewed union, deduplicated and ranked. Trigger: the DeepSeek-V4 LOAD failure
(`cuGraphAddKernelNode` 912) fixed ad hoc in 8298912; the question is the whole class it belongs to.

## 1. Why a restored graph can differ from the captured one: four mechanisms

Every item below is one of these.

- **M1, separate module copies.** LOAD loads its own copy of every archived binary and the graph nodes resolve
  to that copy (hook.cpp:4544, :4371). Everything the process or a library sets on its *own* handles never
  reaches the kernels the restored graph launches, even when LOAD runs exactly the same init code:
  `cudaFuncSetAttribute`, `cudaMemcpyToSymbol` / `__device__`/`__constant__` initialisers, cache config, library
  device state. Foundry re-applies exactly four things: max dynamic smem (recorded), carveout (recorded, CUkernel
  only), the non-portable cluster opt-in (derived, 8298912) and NVSHMEM device state (init_nvshmem_for_loaded_modules).
- **M2, SAVE warm-ups run in a private pool.** Since 2026-10-06 the two upstream warm-up forwards per shape run on
  SAVE in a preparation pass, inside a private `torch.cuda.MemPool` with recording stopped, released before the
  capture pass (`warmup_pool.py`); LOAD runs no forward. First calls of kernels and lazy initialisers therefore
  happen before the capture again, but a persistent resource they create lands in the private pool and fails SAVE
  unless the persistent bootstrap creates it first. Before that change the first call of every kernel and every
  lazy initialiser happened inside the first stream capture. Lazy work not
  in the pre-capture bootstrap list ends up (a) failing the capture loudly, (b) allocated from graph #1's pool with
  a memset/arange node only graph #1 contains, or (c) silently on a different code path than native (probes that
  answer "not ready" while capturing).
- **M3, LOAD substitutes only the capture.** Since c5e43c4 both runners re-run SGLang's own capture loop on
  LOAD and replace only `capture_one` with the archived graph, so every per-shape host step (wrapper creation,
  attention metadata, chunked-prefix buffers, TBO children, LoRA prepare, the DeepEP adapter mode) runs in SAVE's
  order and allocates at the recorded addresses. What is still not run on LOAD: `forward_fn` (the two warm-up
  forwards and the captured forward itself) and anything done on `graph_capture()` *exit* (custom all-reduce IPC
  registration, pinned off by the plugin).
- **M4, host-side latches set by capture, read by replay.** Handled: `prefill_graph_has_dp_gather`, the shared-read
  event fence, the DeepEP adapter mode. Anything else of this kind is a candidate gap.

## 2. Checklist

Rec = recorded at SAVE, App = applied at LOAD, Det = detected when missing, Test = exercised by our rows.
"—" = not applicable. Code references are in the two source documents.

### 2.1 Function-level state (M1)

| item | Rec | App | Det | Test | status |
|---|---|---|---|---|---|
| `MAX_DYNAMIC_SHARED_SIZE_BYTES` | yes | yes, all paths | fatal on error | all rows | handled |
| `PREFERRED_SHARED_MEMORY_CARVEOUT` (function) | yes | CUkernel only, not the per-context function | fatal | no | partial |
| `NON_PORTABLE_CLUSTER_SIZE_ALLOWED` | no (derived from recorded cluster dims > 8) | yes, all paths, both handles | 912 retry | unit test only | handled since 8298912; graphs whose every kernel shares a >8 cluster still uncovered |
| `CLUSTER_SCHEDULING_POLICY_PREFERENCE` (function) | yes | no, on purpose | — | no | hint only; node-level attribute applied |
| `REQUIRED_CLUSTER_*` | yes | merged into the node's cluster dims | — | DSV4 row | handled |
| cache config, shared-mem bank config | no | no | no | no | performance only on sm70+ |
| **any other attribute the process set** (trtllm-gen, FA4 CuTe, DeepGEMM, cutlass set attributes on their own handles) | **no** | **no** | only when the driver refuses the node | no | **gap G1** |
| SAVE-end snapshot of all function attributes, LOAD diff | no | — | no | no | **gap G1** |

### 2.2 Launch attributes, edges, graph-level

| item | Rec | App (serial / JSON template / binary template / member) | status |
|---|---|---|---|
| cluster dims, preferred cluster dims, scheduling policy, cooperative, priority, mem-sync domain and map, carveout | yes (JSON + binary) | yes on all four | handled; `PRIORITY` only effective with an instantiate flag no path uses |
| `PROGRAMMATIC_STREAM_SERIALIZATION` (PDL; on for sm>=90 in ~200 kernel files, DeepGEMM, trtllm) | yes | serial **no** / yes / yes / member **no** (inherits template) | **gap G3** |
| programmatic / launch-completion event edges (ports) | yes, edge data v2 | yes on serial/JSON/binary; members: deps stripped, inherit template | **gap G3** |
| `ACCESS_POLICY_WINDOW`, `DEVICE_UPDATABLE_KERNEL_NODE` | JSON yes; binary no (clears the complete-attrs flag) | per node yes; graph-wide **no**; member **no** | **gap G3** |
| graph-wide "common" attributes | yes (only when all nodes agree) | JSON + binary yes; **serial path ignores them** | **gap G3** |
| newer attributes (e.g. NVLink-utilisation scheduling) | not queried | rebuilt without them, silently | **gap G1** |
| instantiate flags | no | template: auto-free-on-launch; member: 0 | matches torch; device-launch / node-priority unsupported |
| topology key = node types + cluster dims only | — | — | **gap G2**: excludes edges, ports, function identity, PSS, APW, device-updatable; member dependencies are deleted without comparison |
| RNG generator state, output tensors, node order | yes | yes | handled |

### 2.3 Node types

| type | status |
|---|---|
| kernel, memcpy (device to device), memset, empty | supported |
| event record / wait | rebuilt with a **private event per graph**: an external event's link to host code is lost, a wait becomes a silent race (only the shared-read fence is handled as an M4 latch) — **gap G6** |
| host function, child graph, mem alloc/free, conditional, external semaphore, batch mem-op | rejected at SAVE with a generic "unsupported node type!" that names neither the type nor a neighbour — **gap G8** |
| memcpy from pinned host memory (`--cpu-offload-gb`, possibly DSV4 c_plan H2D) | rejected when reported as host pointer; a UVA-reported host copy passes silently — **gap G5** |

### 2.4 Modules, binaries, GPU and driver identity

| item | status |
|---|---|
| code images (fatbin/cubin/PTX by CRC64 + mangled name), device-linked multi-segment fatbins, cuLink at pack time | handled |
| PTX inputs | re-JITted at LOAD by the possibly different driver; not checked |
| `__device__`/`__constant__` globals written from the host before capture (`cudaMemcpyToSymbol`) | **not recorded, not applied**: LOAD's copy starts from the ELF initialisers; only NVSHMEM's state is handled — **gap G2b** |
| addresses of module globals passed as kernel params; texture/surface handles | raw bytes replayed; per-process, not detected — **gap G5** |
| GPU architecture, SM count, driver + compat version, `CUDA_MODULE_LOADING`, kernel-selecting `SGLANG_*` envs | `gpu_name`/`cuda_version` stored in warmup_state.json, **never compared** — **gap G7**; sglang bakes the SM count into grids (EP MoE, marlin, tiny_gemm, mega_moe) and topk_v2 bakes occupancy macros into its build |
| device limits (malloc heap, stack, L2 set-aside), green contexts / exec affinity, MPS | not handled; rare — gap G10 |

### 2.5 Memory and pointers

| item | status |
|---|---|
| `cuMemAlloc_v2` / pitch inside the VMM region (caching allocator, graph pool), `cuMemAddressReserve` (NVSHMEM, symm-mem) | handled: event log replayed in order, reserve = pointer advance, mismatch aborts |
| allocation landing outside the region | `[HOOK] ERROR ... outside allocation region` **printed, not fatal** — gap G5 |
| pointers never in the region (pre-region memory, `cudaMallocAsync` pools, managed, pinned/zero-copy host, fabric memory of the flashinfer MNNVL dispatcher) | **no scan of kernel params** — **gap G5** |
| device memory *contents* the graph reads but does not write (warm-up-initialised workspaces, tables, split-k semaphores, TMA descriptors in memory, trtllm multi-CTA counters) | correct only if SAVE and LOAD take identical init paths; **nothing checks it** — **gap G4** |
| unified memory | not handled |

### 2.6 SGLang runtime state Foundry bootstraps or must bootstrap (M2/M3)

| feature | one-time work native does in warm-up / capture | Foundry |
|---|---|---|
| NCCL communicators | first collective connects and allocates | handled: `bootstrap_collective_connections`; `NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0` |
| torch symm-mem all-reduce, logits `MultimemAllGatherer` | rendezvous / buffer on first eager forward | handled (bootstrap + re-rendezvous) |
| DeepEP / DeepEP v2 / Mooncake buffer | created on first dispatch | handled (`bootstrap_deepep_buffer` and v2 variant) |
| inductor lazy init, DeepGEMM runtime init, Triton autotune | first call | handled: per-key lazy_init warm-up, autotune guard, **caches must be warm before SAVE** |
| FlashInfer per-bs wrappers and workspaces | created by the capture loop between shapes | handled since c5e43c4: the loop runs on LOAD too, no pre-pass |
| **backend detection by attribute** (`hasattr(attn_backend, "indices_updater_decode")`): wrapper backends such as `HybridLinearAttnBackend` hide their FlashInfer child, so hybrid + flashinfer took the non-FlashInfer branch and LOAD re-created the per-bs `_int_workspace_buffer`s after load_all_graphs at new addresses; graphs read plans never updated (Qwen3.5-2B: same wrong tokens every run) | per-bs wrapper creation between captures | **closed 2026-09-30 by c5e43c4: decode LOAD re-runs SGLang's capture loop and substitutes only capture_one (as prefill did), so every per-shape host object is re-created by SGLang in SAVE's order; pre-pass, reuse shim, backend detection and post-load metadata init deleted; archives need `capture_loop_version: 2` (re-SAVE)** |
| **custom all-reduce v1/v2 (v2 default ON in sglang)** | IPC handle exchange + device table fill on `graph_capture()` **exit** | **not handled, not rejected** (our recipes disable it) — **gap G0** |
| `--enable-symm-mem` (NCCL windows), `--enable-nccl-nvls`, `--enable-mscclpp` | `ncclCommWindowRegister` on context exit inside the forward; first-use registration | not handled, not rejected — gap G0b |
| `flashinfer_megamoe` layers, DeepGEMM `megamoe` symmetric buffer, K3 AR-fusion named buffers, `flashinfer_cutedsl` MoE wrapper, nixl buffer | built on the layer's first forward with a collective | **not handled** — gap G0c |
| cutedsl GDN `_cu_seqlens_cache` + state-pool dummy tensors, qsa 128 MB workspace | allocated inside the first capture; later graphs only read them | replay order dependence; not checked — gap G4 |
| `--dsv4-attn-backend trtllm` semaphore getter | asserts not-capturing on first call | SAVE fails loudly; needs a bootstrap — gap G0d |
| two-batch overlap children metadata | per-shape `capture_one_batch_size` | built on LOAD since c5e43c4 (the loop runs); still pinned off by the plugin until a TBO row is validated |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` (default on: first-call sweep with a sync), `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0` | — | pinned by the plugin on branch sglang-registry-pins (2440370, pending host validation); was recipe-only — G9 |
| `--enable-memory-saver` graph context, `SGLANG_ENABLE_METADATA_GLUE_GRAPH`, `SGLANG_ENABLE_GRAPH_POOL_PRECARVE` | tms-owned graph memory / a second sglang-owned graph / pool layout | pinned off by the plugin on sglang-registry-pins (pending host validation) — G9 |

### 2.7 SGLang flags the plugin already pins or rejects

Decode backend full; prefill full or disabled; no breakable / tc_piecewise; no `--cuda-graph-config` contradiction;
no `--disable-cuda-graph`; profiling and autotune off; speculative decoding, LoRA, pdmux, elastic-EP recapture and
attention graph variants rejected; missing bs at LOAD is an error; graph-pool borrow forced off.

## 3. Ranked gaps (deduplicated)

| # | gap | mechanism | blast radius | evidence |
|---|---|---|---|---|
| G0 | custom all-reduce (sglang default) registration on `graph_capture()` exit never replayed; symm-mem NCCL windows, NVLS, mscclpp, TBO, dsv4-trtllm: pinned off on sglang-registry-pins (pending validation); megamoe/cutedsl/K3/nixl lazy collective buffers still not bootstrapped (G0c open) | M2/M3 | any user who does not pass our recipe flags gets a silent wrong or hanging LOAD | sglang B1, B2, B5, B6, B7 |
| G1 | function attributes: three re-applied, none recorded or compared; the next attribute kind fails like 912 or silently | M1 | every kernel library that sets attributes on its own handles | driver A1, sglang B4 |
| G2 | member grouping ignores edges/ports/attributes; member dependencies deleted without comparison; `__device__`/`__constant__` globals written before capture are lost | M1 | silent wrong results | driver A2, A3 |
| G3 | the three load paths disagree on graph-wide attributes, PDL and access-policy window; members never get PSS/APW/device-updatable | — | silent perf or correctness loss on PDL-heavy graphs | driver A7 |
| G4 | read-only memory contents and in-capture lazy caches rely on identical init paths; nothing checks | M2 | silent wrong results | driver A5, sglang B3 |
| G5 | pointers outside the region never scanned; "outside region" is not fatal; UVA host copies undetected | — | crash or silent garbage | driver A4, A9, sglang B9 |
| G6 | external event nodes rebuilt as private events | — | silent races | driver A6 |
| G7 | GPU/driver identity (arch, SM count, compat driver, kernel-selecting envs, baked occupancy macros) never checked at LOAD | — | wrong grids on a different GPU | driver A8, sglang B8 |
| G8 | unsupported node types rejected with an uninformative message | — | debuggability | driver A9 |
| G9 | (pinned on sglang-registry-pins, pending validation) env pins and modes formerly set only by the recipe | — | users of the plugin without the recipe | sglang B10 |
| G10 | rare per-process state: green contexts, device limits, texture objects, PTX re-JIT | — | rare | driver A10 |
| G11 | (closed) backend dispatch by `hasattr` on the top-level attention backend missed composite backends; per-bs resources created between captures moved on LOAD | M3, removed by c5e43c4 (decode LOAD runs the upstream loop) | was: silent wrong tokens on Qwen3.5 + flashinfer | qwen35_flashinfer_load_divergence.md, decode_load_loop.md |

## 4. Generic mechanisms instead of per-kernel fixes

1. **Record-and-replay function-attribute writes** (closes G1, hardens 2.1): interpose `cuFuncSetAttribute`,
   `cuKernelSetAttribute` and the cache-config setters in the hook's `cuGetProcAddress` table; SAVE stores
   (binary hash, kernel name, attribute, value); LOAD applies them to its own copies right after module load, on
   both the CUkernel and the per-context function; at SAVE end snapshot every settable attribute of every archived
   kernel, at LOAD re-query and print a diff.
2. **Fail or warn on in-capture persistent state** (G0c, G4): at SAVE, report allocations made inside one graph's
   capture that stay live and are read by other graphs, with the allocation stack; make symmetric-memory, multicast
   or NVSHMEM allocations inside capture a hard error; add a debug checksum of live region ranges before the first
   replay, compared between SAVE and LOAD.
3. **Replay `graph_capture()`-exit work or refuse it** (G0): until custom all-reduce registration can be replayed
   (enter the same capture contexts on LOAD and restore the recorded pointer/size list), the plugin must reject
   custom all-reduce unless `--disable-custom-all-reduce` is set, and reject `--enable-symm-mem`,
   `--enable-nccl-nvls`, `--enable-mscclpp`, two-batch overlap, `--dsv4-attn-backend trtllm`, memory saver and
   the metadata glue graph, with one-line reasons.
4. **Environment identity in the archive** (G7, G9): store compute capability, SM count, driver and compat version,
   `CUDA_MODULE_LOADING` and the kernel-selecting `SGLANG_*` envs at SAVE; compare at LOAD; move the recipe-only env
   pins into the plugin.
5. **A registry of pre-capture bootstraps** run at the same sequence point on SAVE and LOAD (G0c, G0d), data-driven
   instead of the current hand-written list: add megamoe prebuild, dsv4-trtllm semaphore, nixl buffer, K3 buffers,
   the qsa workspace.
6. **Member grouping by full topology** (G2, G3): include edges/ports and the attributes the member rewrite does not
   reapply in the topology key, or make the rewrite reapply them; compare member dependencies with the template's
   before stripping; make the serial path honour graph-wide attributes.
7. **Pointer and node hygiene** (G5, G6, G8): scan 8-byte kernel-param words with `cuPointerGetAttribute` at SAVE and
   require them inside the region; make "outside allocation region" fatal in SAVE; name the node type and a
   neighbouring kernel in the unsupported-node error; record external events as such and refuse them unless the
   integration owns the fence.

8. **Re-run SGLang's capture loop on LOAD, substitute only the capture** (G11, done in c5e43c4 for decode; prefill already did): no emulation of per-shape host work, hence no backend-specific dispatch; SAVE and LOAD take the same allocation path by construction, checked by `final_alloc_offset == after_load_all_graphs`.

Recommended order: 3 (a plugin change, no C++), 8, 4, 1, 2, 5, 6, 7.

## 5. What the plugin should pin before the PR ships (decided 2026-09-30)

Foundry's rule: state that cannot be made static across processes is not supported; use the cuMem-backed
alternatives instead. Everything on that list that a flag or environment variable controls is a CONFIG the plugin
PINS (resolution hook, same as the graph backends today), not a rejection:

| pinned by the plugin | value | why |
|---|---|---|
| custom all-reduce (sglang default on) | `--disable-custom-all-reduce`; torch symm-mem all-reduce on | IPC registration on `graph_capture()` exit is not replayable |
| NCCL buffer registration | `NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0` (already), `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0` | registrations live outside the recorded region |
| `--enable-symm-mem` (NCCL windows), `--enable-nccl-nvls`, `--enable-mscclpp` | off | first-use registration inside the forward |
| DeepGEMM first-call precompile sweep | `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` | a sync inside capture |
| two-batch overlap | off | per-shape children metadata built only by the capture loop (moot once decode LOAD re-runs the loop) |
| memory saver graph context, metadata glue graph, graph-pool precarve, graph-pool borrow | off | graph memory owned outside Foundry's region |
| `--dsv4-attn-backend` | `auto`/`flashmla` | the trtllm variant asserts at first call during capture; needs a bootstrap first |

Rejected with an error (no Foundry logic for them): speculative decoding, LoRA, pdmux, elastic-EP recapture,
attention graph variants, non-full graph backends, `--disable-cuda-graph`, profiling/autotune on.

Documented requirements stay: dense no-padding capture with an explicit bs list, DeepEP low-latency on EP rows,
JIT caches warm before SAVE, pinned hybrid state pools, identical flags on SAVE and LOAD.

## 6. Coverage

Every row we ran was on H200 (sm90) with custom all-reduce disabled, torch symm-mem on, fa3/FlashInfer/triton
attention, DeepEP low-latency or no a2a, deep_gemm or triton MoE runners. No sm100/sm120 path (fa4, trtllm-gen,
cutedsl, qsa, K3 fusion, flashinfer megamoe) has been exercised; the function-attribute class (G1) is therefore
only known to be safe for the kernels those rows launched.
