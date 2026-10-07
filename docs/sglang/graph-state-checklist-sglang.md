# SGLang graph-state checklist: which flags change a captured graph, and what one-time state LOAD misses

sglang upstream main `fa090f7755` (read from `git archive origin/main`), foundry `sglang-registry` `6d5bec5`
(includes the cluster opt-in fix `8298912`). Local reading only: nothing was compiled or run.

- sglang paths are relative to `python/sglang/`. Foundry paths are relative to `foundry/`.
- `kernels/aot/csrc` is the vendored sgl-kernel. There is no separate `sgl-kernel/` tree at this commit.
- **[spec]** marks a claim about code that is not in either tree (flashinfer, FA3, FlashMLA, DeepGEMM, NCCL,
  torch, cutlass-dsl, mooncake), or anything I did not confirm by reading.
- Companion document: [graph-state-checklist-driver.md](graph-state-checklist-driver.md) covers the same problem at driver level (function
  attributes, node types, launch attributes, module globals, memory). This file is the sglang side: which server
  flag, env var, backend, model or GPU puts that state into a graph.

## 0. Why LOAD misses things: four mechanisms

Every row below is one of these four mechanisms.

- **M1. Separate module copies.** LOAD loads its own copy of every archived binary
  (`load_cuda_modules_and_libraries`, csrc/hook.cpp:4544). Graph nodes resolve to that copy
  (`query_function_handle`, hook.cpp:4371). So anything sglang or a library sets on its *own* handle never
  reaches the handle a restored graph launches, even when LOAD runs the same init code:
    - `cudaFuncSetAttribute`
    - module globals
    - cache config

  Foundry re-applies exactly three function attributes and nothing else:
    - `MAX_DYNAMIC_SHARED_SIZE_BYTES` (recorded);
    - `PREFERRED_SHARED_MEMORY_CARVEOUT` (recorded);
    - `NON_PORTABLE_CLUSTER_SIZE_ALLOWED` (derived from cluster > 8, ClusterOptIn.h).

  Foundry records `CLUSTER_SCHEDULING_POLICY_PREFERENCE` but does not apply it (CUDAGraph.cpp:2516-2540).
  The dsv4 912 bug was this mechanism.
- **M2. SAVE runs no eager warmup.** `patched_capture_one` (integration/sglang/hooks.py:470-517) skips upstream's
  two warmup forwards (`full_cuda_graph_backend.py:137-151`) and their `post_warmup_hook`. So on SAVE, the
  **first call of every kernel and every lazy initialiser happens inside the first stream capture.** Native sglang
  does all of that eagerly.

  Anything lazy that is not in Foundry's pre-capture bootstrap list (hooks.py:405-446) then ends up in one of
  three states:
    - **(a)** it fails the capture loudly: a sync, a collective, or an assert;
    - **(b)** it allocates from graph #1's pool, sometimes with a memset or arange node that only graph #1
    contains;
    - **(c)** it silently captures a different code path from native sglang, because many probes return
    "not ready" while capturing.
- **M3. LOAD skips the whole capture loop,** decode only. LOAD's decode path (hooks.py:568-670) never enters:
    - `graph_capture()` (`srt/distributed/parallel_state.py:2289-2313`);
    - `backend.capture_session`;
    - `capture_one_shape`, so the per-shape pre-capture steps do not run: `tbo_plugin.capture_one_batch_size`,
    LoRA prepare, `deepep_adapter.capture`, and the capture-time `init_forward_metadata_out_graph`
    (`decode_cuda_graph_runner.py:1160-1262`).

  LOAD re-creates only these pieces:
    - `warmup()`;
    - the graph pool plus `set_graph_pool_id`;
    - `out_graph(in_capture=True)` per bs (graph_ops.py:880-926);
    - `deepep_adapter.capture(is_extend_in_batch=False)`.

  Anything done on `graph_capture()` **exit** is missing, which is the custom all-reduce registration. Prefill
  LOAD does run `orig_prefill_capture`, so it enters `graph_capture()`, but with nothing captured.
- **M4. Host-side latches that capture sets and replay reads.** Handled today:
    - `prefill_graph_has_dp_gather` (graph_ops.py:245, hooks.py:537);
    - the in-graph shared-read event (hooks.py:746-758);
    - the DeepEP adapter mode.

  Anything else of this kind is a candidate gap.

## 1. Matrix

Columns:

- **Graph content**: kernels, node types and launch attributes the feature puts in the captured graph.
- **One-time side effect LOAD misses**: yes or no, plus what.
- **Foundry status**: handled (where), not handled, or rejected. Rejected means the plugin refuses the
  configuration (plugin.py:130-254, hooks.py:867-900).
- **Detect at SAVE**: how Foundry could see the problem at SAVE time.
- **Coverage**: which of our rows exercised it.

Row keys:

| key | model / config |
|---|---|
| Q1 | Qwen3-1.7B single / dp2 / dp4 |
| Q1t | Qwen3-1.7B tp2, `--enable-torch-symm-mem --disable-custom-all-reduce` |
| Q30 | Qwen3-30B-A3B ep2/ep4/tpep2/tpep4/tp2ep4: fa3 + DP attention + DeepEP LL + deep_gemm + torch symm-mem, custom AR off |
| Q235 | qwen3-235b-a22b-fp8 tpep4 |
| B1/B5 | qwen3.5-122b-a10b-fp8 ep4 / tp2ep4 (GDN hybrid, default linear-attn backends) |
| B2 | DeepSeek-V4-Flash ep4 (dsv4 flashmla backend, topk_v2 cluster-16) |
| B3 | GLM-5.3-Flash ep4 (DSA) |
| B4 | gpt-oss-120b ep4 (mxfp4, `--moe-a2a-backend none`, FULL prefill graphs) |

Every row ran on H200 (sm90). **Nothing ran on sm100/sm120**, and every multi-GPU row passed
`--disable-custom-all-reduce` (recipe/sglang/serve_common.sh:143-158).

### 1.1 Capture machinery and graph flags

| flag / feature | graph content | one-time side effect LOAD misses | Foundry status | detect at SAVE | coverage |
|---|---|---|---|---|---|
| two warmup forwards + `post_warmup_hook` (`full_cuda_graph_backend.py:137-151`) | none (eager) | Native sglang uses them for every lazy init (M2) | intentionally skipped on SAVE. Replaced by the bootstraps: NCCL connect, logits gatherer, DeepEP/v2/mooncake buffer, inductor lazy_init, DeepGEMM runtime (hooks.py:405-446) | alloc events that happen inside the capture of graph i and stay live after the loop (section 3.2) | all rows |
| `--cuda-graph-backend-decode/prefill`, `--cuda-graph-config`, disaggregation roles | full vs breakable vs tc_piecewise (dedup mixin: breakable only, `cuda_graph_dedup_mixin.py:303`) | n/a | pinned: decode `full`, prefill `full`/`disabled` (plugin.py:130-205, hooks.py:838-865) | already enforced | all; prefill full: B4 |
| `--cuda-graph-bs-*`, `--cuda-graph-max-bs-*`, `--disable-cuda-graph-padding`, `--cuda-graph-max-seq-len-prefill` | set of shapes | no; LOAD errors on a missing bs (hooks.py:648-653) | handled | enforced at LOAD | all |
| attention graph variants (`ShapeKey.attention_variant`: DSV4.1 candidate indexer on SM100, HIP DSA dual) | several graphs per bs | n/a | rejected at runtime (hooks.py:885-890) | enforced | none (sm90 rows do not create them) |
| `--enable-profile-cuda-graph`, `SGLANG_GRAPH_BATCH_CAPTURE` | profiler steps | n/a | rejected (plugin.py:163) | enforced | - |
| `SGLANG_ENABLE_GRAPH_POOL_BORROW` | pool extents | n/a | forced off (hooks.py:383) | - | - |
| `SGLANG_ENABLE_GRAPH_POOL_PRECARVE` (`GraphPoolPrecarve`, `full_cuda_graph_backend.py:143,180`) | changes pool layout | no (skipped on both SAVE and LOAD, because Foundry's capture_one bypasses it) | implicitly off. Not stated anywhere | log if the env is set | - |
| `--enable-memory-saver` + `SGLANG_MEMORY_SAVER_CUDA_GRAPH` (`full_cuda_graph_backend.py:99-102,163-174`) | tms-owned graph memory | SAVE bypasses the tms graph context. A later pause/resume would not back up restored graph memory | **not rejected**, untested | reject in `reject_unsupported_features` | - |
| `--enable-cudagraph-gc` / `freeze_gc` | none | none (host GC) | n/a | - | all (known harmless `freeze_gc` traceback) |
| in-graph shared-read external event (`decode_cuda_graph_runner.py:500-515`) | EVENT_RECORD node with an external event | LOAD rebuilds it with a private event (CUDAGraphParallel.cpp:771-793). The host-side event stays None | handled: POST_REPLAY fence (hooks.py:746-758) | the only `make_external_event` site in srt | all |
| `model_runner.capture_tail_hooks` (`decode_cuda_graph_runner.py:1224`) | extra kernels (dflash/dspark) | only via speculative | speculative rejected | - | - |
| `tbo_plugin.capture_one_batch_size` (`--enable-two-batch-overlap`; `two_batch_overlap.py:342-371`, `tbo_backend.py:56-81`) | children micro-batch metadata, DeepEP `Buffer.capture()` events | **yes**: builds `forward_batch.tbo_children`, which the children's `out_graph` needs. LOAD's `initialize_attention_metadata_for_bs` never calls it (graph_ops.py:880-913) | **not handled, not rejected** | reject TBO, or call the plugin step in the LOAD pre-pass | - |
| LoRA dual graph, spec draft/verify, pdmux, elastic-EP recapture, `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` | variant graphs, multi-stream | various | rejected (plugin.py:215-254, hooks.py:891-899) | enforced | - |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH` (`metadata_glue_graph.py`) | a second, sglang-owned graph of the metadata prep | eager warmup + capture at replay time. Not persisted | not rejected; the plugin does not touch it | - | - |
| PP (`--pp-size`) | PPProxyTensors outputs | none known | allowed, persisted, never run on GPU (plugin.py:217-219) | - | - |
| `--enable-torch-compile` (`compilation/torch_compile_decoration.py:71-83`: `coordinate_descent_tuning=True`) | inductor Triton kernels | first-call compile + autotune inside capture (M2) | lazy_init bootstrap + autotune guard (graph_ops.py:605-704); requires a warm cache | guard only wraps `benchmark_all_configs`. **[spec]** coordinate-descent tuning benchmarks through another entry point and may bypass the guard | GLM (DSA indexer compile), gpt-oss |
| `compile_in_capture_mode` (DSV4 helpers, `runner_utils/capture_mode.py:48-57`) | torch.compile'd helpers only under capture | same as above | same | same | B2 |

### 1.2 Attention backends (`--attention-backend`, `--decode-/--prefill-attention-backend`; choices at `srt/arg_groups/choices.py:68-97`)

Rule for all backends: the backend's metadata buffers come from `init_cuda_graph_state` and
`out_graph(in_capture=True)`. LOAD reruns them in SAVE's order, so the graph's baked addresses match
(graph_ops.py:880-926). That is what makes fa3 and flashinfer work.

| backend / flag | graph content | one-time side effect LOAD misses | Foundry status | detect at SAVE | coverage |
|---|---|---|---|---|---|
| fa3 (sm90 default for MHA models; sgl_kernel `flash_attn`) | FA3 fwd + combine. `scheduler_metadata` is computed out-of-graph into `_sched_meta_buf` (`flashattention_backend.py:629-639, 2305-2313`) | no. Metadata is rerun by the LOAD pre-pass. Without sched meta (DP attention, CP), FA3 allocates semaphore/LSE buffers inside capture **[spec]** | handled | - | Q30, Q235 |
| flashinfer / flashinfer_mla | `run()` + merge. `fast_decode_plan` swap only in capture (`flashinfer_backend.py:810-853`). Plan H2D from pinned memory stays out-of-graph | per-bs wrappers + `_int_workspace_buffer` (G1) | handled: pre-pass + reuse shim on SAVE (hooks.py:672-725) | - | Q1 **[spec: the default backend of the Q1 rows is not recorded in the notes]** |
| triton | decode kernels with `launch_pdl` (sm>=9, `triton_backend.py:328-331`). Lean vs standard is chosen by `get_is_capture_mode()` (`:2270-2293`) | no | handled by the generic metadata rerun | - | none |
| fa4 (CuTe DSL; sm100 cluster incl. 2-CTA, CLC scheduler, PDL with bias: `flash_fwd_sm100.py:1625-1638`; sm120 variant) | cluster launch + dyn smem on the CuTe CUfunction | M1: cluster <= 8 (portable) and smem are covered. Any other attribute the DSL sets is not **[spec]** | partly (three attributes) | generic attribute record (section 3.1) | none (no sm100) |
| trtllm_mha (XQA/fmha_v2 on sm90/120, trtllm-gen on sm100) | cubin kernels. sglang owns a persistent zeroed multi-CTA counter (`trtllm_mha_backend.py:375-385`) | M1 attributes on flashinfer's cubin handles **[spec]** | partly | section 3.1 | none |
| trtllm_mla / cutedsl_mla / tokenspeed_mla | trtllm-gen / cute-dsl MLA with PDL (`trtllm_mla_backend.py:176,1084`). Module-global cute workspace (`:155-161`). tokenspeed workspace grows lazily (`tokenspeed_mla_backend.py:81-104`) | tokenspeed / cute-dsl lazy growth inside the first capture (M2b) | not handled | section 3.2 | none |
| flashmla (MLA) | decode + combine. `get_mla_metadata` runs out-of-graph (`flashmla_backend.py:258-289, 369-401`) | no | handled | - | none |
| dsv4, `--dsv4-attn-backend auto/flashmla` | `in_graph` raw→full metadata upgrade inside the graph (`deepseek_v4_backend.py:2022-2060`). `FlashMLASchedMeta` is lazy behind `have_initialized`, so SAVE records the scheduler kernel in the graph (`kernels/aot/python/sgl_kernel/flash_mla.py:232-305`). topk_v2 cluster-16 kernel | M1: `NonPortableClusterSizeAllowed` (`kernels/jit/csrc/deepseek_v4/topk_v2.cuh:763-775`) | **fixed** in 8298912 (derived opt-in); LOAD not yet re-validated on GPU | cluster > 8 now derived | B2 (the failure) |
| dsv4, `--dsv4-attn-backend trtllm` | `trtllm_batch_decode_sparse_mla_dsv4` | **yes, and loud on SAVE**: the patched flashinfer counter getter asserts `not is_current_stream_capturing()` ("first call is expected during eager warmup", `deepseek_v4_trtllm_backend.py:97-104`) | **not handled** | the assert fires at SAVE. Fix: call the patched getter in a pre-capture bootstrap | none |
| dsa / nsa (+ `--dsa-*-backend`, `SGLANG_DSA_TOPK_BROADCAST`, `SGLANG_OPT_USE_TOPK_V2`) | DeepGEMM paged-MQA logits (JIT). Topk broadcast via pynccl **inside** capture (`dsa/dsa_indexer.py:170-199`). Dual-stream indexer under capture mode (`:1645-1650`) | topk_v2 occupancy-probe macros are device-dependent (section 1.5). The counter buffer grows in `init_cuda_graph_state`, with an assert if it grows during capture (`dsa_backend.py:1365-1395`) | handled for the tested path | - | B3 |
| qsa (Qwen sparse; sm100/120 trtllm sparse decode) | trtllm decode | **yes**: 128 MB `torch.zeros` workspace on the **first decode call** (`qwen_sparse_attn_backend.py:1484-1487`), so graph-pool memory plus a memset in graph #1 only. No counter is passed, so flashinfer allocates and zeroes one per layer inside the graph **[spec]** | not handled | section 3.2 | none |
| hybrid linear: GDN/KDA/Mamba2 (`--linear-attn-*`, `--mamba-backend`) | FLA Triton (sm90 default), CuTe-DSL GDN/KDA, flashinfer GDN (sm>=10 state pool) | **cutedsl GDN**: the first-call compile allocates dummy zero tensors sized to the whole state pool (`kernels/ops/attention/cutedsl_gdn.py:1280-1307`). `_cu_seqlens_cache` is filled by a `torch.arange` in graph #1 only and read by later graphs with the same N (`:1428-1437`; `cutedsl_kda.py:1484`). That is a replay-order dependency on SAVE too | Triton path handled. cutedsl path not handled | section 3.2 | B1, B5 (Triton path) |
| minimax sparse | Triton TMA; `triton.set_allocator(robust_allocator)` on every call, scratch kept in a deque (`minimax_sparse/common/utils.py:87-94`) | allocations inside capture (recorded) | probably fine **[spec]** | - | none |
| tbo backend | children's kernels | see TBO row in 1.1 | not handled | - | - |

### 1.3 MoE runner and A2A

Choices: `--moe-runner-backend` (`srt/arg_groups/choices.py:133-161`) and `--moe-a2a-backend`
(`srt/layers/moe/utils.py:35-48`).

| flag / feature | graph content | one-time side effect LOAD misses | Foundry status | detect at SAVE | coverage |
|---|---|---|---|---|---|
| deepep (+ `--deepep-mode`) | DeepEP LL/normal kernels, NVSHMEM | buffer created lazily on the first dispatch. The adapter mode is set in `capture()` and asserted in `replay()` (`runner_utils/deepep_adapter.py:30-42`) | handled: `bootstrap_deepep_buffer` (graph_ops.py:707), NVSHMEM module init (graph_ops.py:447,953), `deepep_adapter.capture(False)` (hooks.py:669) | - | Q30, Q235 |
| deepep_v2 / mooncake | ElasticBuffer (`deepep_v2.py:148-200`) / mooncake Buffer (`mooncake.py:86-121`). Mooncake passes timeout=-1 to the **first** captured dispatch only (`:164,241,279-284`), a per-graph scalar | lazy buffer | handled: `_bootstrap_deepep_v2_buffer`, `_bootstrap_mooncake_buffer` (graph_ops.py:786-877) | - | none (v2 not runnable on the host, pre_pr_tests A2) |
| nixl | lazy `get_nixl_buffer` (`nixl.py:115-180`) | **yes**, same class as DeepEP | not handled | extend `bootstrap_deepep_buffer` | none |
| flashinfer (MNNVL all-to-all) | fabric-mapped workspace from dispatcher init (`token_dispatcher/flashinfer.py:228-260`) | not lazy, but cuMem fabric memory the hook may not replay at the same VA **[spec]** | unknown | archive_check pointer scan | none |
| `megamoe` (DeepGEMM mega MoE, `SGLANG_OPT_DEEPGEMM_MEGA_MOE_*`) | mega kernel on an alt stream (`layers/moe/mega_moe.py:219-231`) | **yes**: `deep_gemm.get_symm_buffer_for_mega_moe` is a collective symm-mem allocation, lazily in the forward (`mega_moe.py:113-153,336`; also `models/kimi_k3.py:795`) | not handled | section 3.2, plus a symm-mem allocation inside capture | none |
| `flashinfer_megamoe` (runner or a2a) | `MoEEpMegaLayer` per layer | **yes**: built on the layer's first forward with a bootstrap collective (`layers/moe/flashinfer_megamoe.py:254-310`). The eager prebuild is **commented out** upstream (`model_executor/model_runner.py:1079-1091`). NVFP4 raises inside capture (`flashinfer_megamoe.py:341-345`) | not handled (loud on SAVE for NVFP4) | loud | none |
| `flashinfer_cutedsl` MoE | `CuteDslMoEWrapper` (w4a4/w4a16) | **yes**: the wrapper and its CUDA-graph buffers are built on the first forward (`moe_runner/flashinfer_cutedsl.py:302-360`, called at `quantization/modelopt_quant.py:3199`), inside capture on SAVE | not handled | section 3.2 | none |
| `deep_gemm` (+ `SGLANG_ENABLE_JIT_DEEPGEMM`, `SGLANG_DEEPGEMM_PDL`) | DeepGEMM JIT kernels with cluster (<= 2 **[spec]**), TMA descriptors in params, PDL via `deep_gemm.set_pdl` (`deep_gemm_wrapper/entrypoint.py:292-296`). Masked vs compact layout from a free-memory budget measured before capture (`moe_runner/deep_gemm.py:220-249`) | the runtime init is handled. **`SGLANG_JIT_DEEPGEMM_PRECOMPILE` (default True, `environ.py:1193`)**: the first call sweeps every M, then `synchronize()` + `empty_cache()` (`deep_gemm_wrapper/compile_utils.py:144-186, 262-274, 445-451`), rank-0-of-node only | runtime init: `bootstrap_lazy_runtimes`. Precompile: pinned to 0 **only by the recipe** (serve_common.sh), not by the plugin | loud (sync in capture), or an asymmetric rank-0 layout | Q30, Q235, B3 |
| `flashinfer_trtllm` / `_routed` (+ `SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS`) | trtllm-gen cubins. PDL on only when `num_tokens <= 8192` (per-bs graph attribute, `moe_runner/flashinfer_trtllm.py:50-56`) | M1 attributes on flashinfer's cubin handles **[spec]**. The tactic stays default while autotune is off | partly. Autotune pinned off (plugin.py:175) | section 3.1 | none |
| `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE` (default on) | JIT `moe_finalize_fuse_shared` with PDL (`kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu:372-374`) | no | handled (PDL edges are recorded) | - | trtllm path only: none |
| mxfp4 fused finalize + TP all-reduce (`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`) | the probe returns None while capturing, so SAVE captures the **unfused** variant (`quantization/mxfp4_flashinfer_trtllm_moe.py:528-536`) | no for LOAD (self-consistent). It differs from native sglang, which also creates a second push-only CustomAllReduceV2 (`:545-555`) | M2c divergence (perf only) | compare against native kernel lists | B4 **[spec: which runner B4 used]** |
| triton MoE (TMA path) | `triton.set_allocator` once (`kernels/ops/moe/fused_moe_triton_kernels.py:724-739`). B-weight TensorDescriptor cache keyed by data_ptr (`:744-786`) | no (scratch allocated inside capture is recorded) | handled | - | B4? |
| marlin / awq / gptq | per-call `torch.zeros` workspace becomes a memset node + pool alloc (`moe_runner/marlin.py:152-154`). JIT marlin sets MaxDynamicSmem | no | handled (smem recorded) | - | none |
| EPLB (`--enable-eplb`), expert distribution recorder | in-place metadata/weight updates (`eplb/expert_location.py:312-340`). "dynamic" dispatch puts Philox `randint` in the graph (`expert_location_dispatch.py:143-146`) | no, if LOAD rebuilds the tensors at the same addresses. RNG is registered per graph by Foundry | probably fine, untested | - | none |
| elastic EP (`--elastic-ep-backend`, `--max-ep-size`) | recapture after a scale event | n/a | rejected (plugin.py:236-246, hooks.py:891-899) | enforced | - |
| waterfill (`layers/moe/waterfill.py:93-100`) | `_counts_buf` lazily allocated in the first capture, zeroed every call | no content dependency; address from graph #1's pool | fine | - | - |

### 1.4 Communication

| flag / feature | graph content | one-time side effect LOAD misses | Foundry status | detect at SAVE | coverage |
|---|---|---|---|---|---|
| **custom all-reduce v2 (default ON: `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=True`, `environ.py:1348`; sizes `SGLANG_CUSTOM_ALL_REDUCE_V2_*`)** | JIT 1shot/2shot pull `LS_GRAPH` variants read `graph_params[row]` (`kernels/jit/csrc/distributed/custom_all_reduce.cuh:392,412`), plus push/multicast and a PDL memcpy. Workspace is torch symm-mem + multicast | **yes**: on `ca_comm.capture()` exit, i.e. `graph_capture()` exit after the whole capture loop, `_register_graph_inputs` exchanges IPC/VMM handles of every captured input and H2D-writes peer pointers into `graph_params`, then advances `_graph_counter` (`device_communicators/custom_all_reduce_v2.py:396-457`). LOAD's decode path never enters `graph_capture()` (M3) | **not handled and not rejected**. Only the recipes pass `--disable-custom-all-reduce` | reject unless `disable_custom_all_reduce`, or record `(ptr,nbytes)` order at SAVE and re-run the registration at LOAD | none (all rows disable it). pre_pr_tests A2 ran default AR natively only |
| custom all-reduce v1 (v2 env=0 or ineligible; HIP) | `cross_device_reduce_{1,2}stage(RankData*, sg_, self_sg_)` | **yes**: `register_graph_buffers` (IPC open + `rank_data` H2D + `d_rank_data_base_ += n`, `kernels/aot/csrc/allreduce/custom_all_reduce.cuh:648-668`; `custom_all_reduce.py:181-258`) | not handled, not rejected | same | none |
| `--enable-torch-symm-mem` | copy-in, `multimem_all_reduce_` / `two_shot_all_reduce_`, copy-out (`torch_symm_mem.py:170-184`) | buffer + rendezvous at init | handled (symm-mem re-rendezvous; `tests/test_symm_mem.py`) | - | Q1t, Q30, Q235, B* |
| logits `MultimemAllGatherer` (`triton_symm_mem_ag.py`) | multimem gather, or the NCCL fallback while capturing | lazy on the first eager forward | handled: `bootstrap_logits_gatherer` (graph_ops.py:537) | archive_check | Q30 (fixed bug) |
| NCCL (pynccl) collectives | NCCL kernels whose args point to comm device state | first collective connects and allocates | handled: `bootstrap_collective_connections` (graph_ops.py:466-535); `NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0` defaulted (runtime.py:411-412); the recipe sets `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0` | - | all multi-GPU |
| `--enable-symm-mem` (NCCL windows, `pynccl_allocator.py`) | NCCL symmetric kernels **[spec]** | **yes**: `use_symmetric_memory()` at about 30 forward sites calls `ncclCommWindowRegister` on `__exit__` for new segments (`pynccl_allocator.py:95-145,294-299`), so inside capture on SAVE. `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` runs only **after** capture (`cuda_graph_setup.py:431-446`) | not handled, not rejected | reject, or hook `ncclCommWindowRegister` | none |
| `--enable-nccl-nvls`, `--enable-mscclpp` | NVLS multicast objects / mscclpp execute with first-use registration **[spec]** | lazy setup at first use **[spec]** | not handled, not rejected. The recipe's NVLS=0 conflicts with `--enable-nccl-nvls` | reject | none |
| K3 AR fusion (`SGLANG_K3_AR_FUSION`, auto on sm100/103 with v2 multicast) | push-res / push-norm, NVLS pull on the multicast address. Cluster kernel (`kimi_k3/comm/ar_fusion.cuh:245`), PDL | **yes**: named symm buffers `attn_o_proj`, `moe_latent_shared` created lazily inside the forward (`layers/communication/k3_ar_fusion.py:140-201`) | not handled | symm-mem allocation inside capture | none |
| moe_finalize all_reduce_fusion (`distributed/all_reduce_fusion.cuh`, portable cluster <= 8 by static_assert :193) | cluster kernel + epoch counters flipped per launch | no, provided every rank starts from the same fresh-init counters (zeros are valid in either parity) | fine | - | none |
| Inkling custom AR + fused variants (`SGLANG_OPT_USE_INKLING_*`) | multimem / one-shot kernels with flag/state pointers as args | resources built at model init (`models/inkling.py:618-620`). Rotation indices flip per call during capture, so the baked addresses depend on capture order (identical on SAVE and LOAD) | probably fine | - | none (Inkling SAVE blocked, known-issues.md) |
| FlashInfer AR+RMSNorm fusion (`flashinfer_allreduce_fusion_backend`), PCIe IPC AR (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`), flashinfer a2a workspace | lamport / IPC kernels | workspaces pre-built in `BaseRunner.warmup()` (`base_runner.py:254-256, 286-352`), which LOAD also calls (hooks.py:598). Fusion can re-create lazily inside the forward when the size is too small (`flashinfer_comm_fusion.py:449-468`). cuMem FABRIC memory **[spec]** | partly (warmup runs on both modes) | archive_check pointer scan | none |
| `--enable-dp-attention` | gather/scatter via all-reduce or all_gatherv (`layers/dp_attention.py:557-612`) | `prefill_graph_has_dp_gather` latch | handled (graph_ops.py:245-257, hooks.py:537-541) | - | Q30 |
| `--enable-two-batch-overlap`, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | fork/join edges; DeepEP capture events | TBO: see 1.1 | multi-stream edges handled | - | none |

### 1.5 GEMM / quantization / JIT / GPU generation

| flag / feature | graph content | one-time side effect LOAD misses | Foundry status | detect at SAVE | coverage |
|---|---|---|---|---|---|
| sgl JIT kernels (`kernels/jit`, tvm-ffi dlopen, `cache_once` per op: `utils/common.py:27-41`) | compiled on first call, i.e. inside capture on SAVE. Modules come through cudart registration + lazy load | 46 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` sites, 2 carveout sites (`include/sgl_kernel/utils.cuh:379` occupancy-driven, `deepseek_v4/wo_a_fused.cuh:784`), 4 non-portable cluster sites (topk_v2, deep_select v3_cluster + kerutils host.h:246-249, cluster_probe) | smem/carveout recorded. Non-portable derived (8298912) | grep list above; generic record (3.1) | B2, B3 |
| topk_v2 occupancy probe (`kernels/ops/attention/dsv4/topk.py:31-62`, `kernels/jit/utils/occupancy.py:21-53`) | `-DSGL_TOPK_V2_MAX_C8_OCC2/C16_OCC1=N` are baked. They decide whether the cluster-16 kernel exists and size the persistent grid (`topk_v2.cuh:420-433`) | device-dependent kernel **variant**. A LOAD on a different SKU/GPC config replays a grid sized for SAVE's GPU (hang risk **[spec]**) | not checked (warmup_state stores `gpu_name` but LOAD never compares it) | store and compare CC, SM count, driver | B2, B3 |
| deep_select (DeepSelect topk, cluster 8 on sm90, 16 on sm100/103: `kernels/jit/csrc/deep_select/entry.cuh:261-265`) | cluster-16 non-portable + TMA descriptors | same as topk_v2 | covered by 8298912 (cluster > 8) | - | none. Registered op, no srt caller at this commit |
| SM count baked into grids (`ep_moe_kernels.py:30-58`, marlin, `tiny_gemm.py:63`, `hc_mix.py:193`, mega_moe reserved SMs) | grid dims | device-dependent | not checked | same as above | all |
| fp8/fp4/bf16 GEMM runner selectors (`--fp8-gemm-runner-backend`, `--fp4-gemm-runner-backend`, `--bf16-gemm-backend`), `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE` (static JSON) | kernel choice only, lru_cache of capability | no | fine if SAVE and LOAD use the same flags | - | Q235, B* |
| cuBLAS/cuBLASLt (bf16 dense) | Lt kernels + workspace | workspace allocated on the first matmul per handle/stream, inside capture on SAVE (recorded) | handled (dense rows pass) | - | all |
| `--cpu-offload-gb` (OffloaderV1, `utils/offloader.py:137-200`) | **H2D memcpy nodes from pinned host memory** (`.to(device, non_blocking=True)` inside the forward) | pinned host VA is not replayed | SAVE rejects `srcHost != nullptr` (CUDAGraph.cpp:1591-1594). **[spec]** A UVA-reported copy would pass silently | check `cuPointerGetAttribute(MEMORY_TYPE)` on memcpy endpoints | none |
| DSV4 `c_plan` / `c128_online` H2D from pinned plan buffers (`kernels/jit/csrc/deepseek_v4/c128_online_v2.cuh:866-900`; `c_plan.cuh:604-606,783`) | possibly a captured H2D memcpy **[spec: whether it runs inside the prefill graph]** | same as above | same | same | B2 (decode only) |
| PDL (no server flag: on for sm>=90 via `is_arch_support_pdl`, `kernels/jit/utils/arch.py:178-181`; `SGLANG_DEEPGEMM_PDL`; trtllm `enable_pdl`; Triton `launch_pdl` / `ENABLE_PDL` constexpr in ~200 files) | programmatic edges + `PROGRAMMATIC_STREAM_SERIALIZATION` node attribute | none | handled on JT/BT (`add_graph_dependencies`). SER and MEM gaps are listed in the driver checklist, section 2 | driver checklist | all sm90 rows |
| cooperative launch (kerutils `cudaLaunchCooperativeKernel`, host.h:231-237) | COOPERATIVE attribute | none | recorded and applied (not on SER common attributes) | - | none |

## 2. Ranked riskiest unhandled items

Rank = likelihood a real deployment hits it × how silent the failure is.

1. **Custom all-reduce graph-input registration** (v2 is default ON, v1 as fallback). It runs on
   `graph_capture()` exit, and LOAD's decode path skips that. So the replayed pull kernels read an empty
   `graph_params` / `rank_data` table. It is not rejected: only our recipes pass `--disable-custom-all-reduce`.
   Any TP>1 user who follows sglang defaults hits it (custom_all_reduce_v2.py:396-457). The cheapest fix is to
   reject it in `reject_unsupported_features`.
2. **Lazy per-layer collective / symm-mem resources that native sglang builds in its warmup forwards.** They are
   not in Foundry's bootstrap list:
   - `flashinfer_megamoe` (the upstream prebuild is commented out);
   - DeepGEMM `megamoe` symm buffer;
   - K3 AR fusion named buffers;
   - `flashinfer_cutedsl` MoE wrapper;
   - nixl buffer.

   On SAVE they are created inside capture (loud, or graph-pool memory). LOAD never creates them, but the graph
   holds their addresses.
3. **In-capture lazy device caches (M2b).** Examples:
   - cutedsl GDN/KDA `_cu_seqlens_cache` and the state-pool dummies;
   - qsa's 128 MB trtllm workspace;
   - tokenspeed and cute-dsl workspace growth.

   The first graph allocates and writes them, and later graphs only read them. That is a replay-order
   dependency already on SAVE, and a silent wrong-data risk.
4. **Function attributes beyond the three Foundry handles (M1).** trtllm-gen cubins (MHA/MLA/MoE), FA4 CuTe,
   DeepGEMM and cutlass set attributes on their own handles inside flashinfer or DeepGEMM **[spec]**. Nothing
   records what was actually set, so the next attribute kind fails the way 912 did. None of these paths ran on
   sm100.
5. **dsv4 `--dsv4-attn-backend trtllm`:** the persistent semaphore getter asserts outside capture
   (deepseek_v4_trtllm_backend.py:97-104). SAVE fails loudly. It needs a pre-capture bootstrap. This is also the
   natural sm100 backend for DSV4.
6. **`--enable-symm-mem` NCCL window registration** inside capture (pynccl_allocator.py:95-145). LOAD misses the
   registration. Neither rejected nor tested. Likewise `--enable-nccl-nvls` and `--enable-mscclpp` **[spec]**.
7. **Two-batch overlap:** LOAD's metadata pre-pass never runs `tbo_plugin.capture_one_batch_size`, so the
   children's metadata is never built. Not rejected.
8. **Device / SKU-dependent build and sizing:**
   - topk_v2 occupancy macros;
   - SM-count grids;
   - the DeepGEMM masked-layout budget, which comes from free memory at SAVE.

   LOAD never compares GPU identity: `warmup_state.json` stores `gpu_name` but never checks it.
9. **Pinned-host H2D memcpy nodes** (`--cpu-offload-gb`, possibly the DSV4 c_plan prefill path). A UVA-reported
   endpoint would escape the `srcHost` check **[spec]**.
10. **Env pins that live only in the recipe, not in the plugin:**
    - `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` (default True: a first-call sweep with a sync inside capture on rank 0);
    - `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`.

    Also unverified: whether inductor's `coordinate_descent_tuning` (on with `--enable-torch-compile`) gets past
    the autotune guard.

## 3. Proposal: generic mechanisms instead of per-kernel fixes

### 3.1 Record every function-attribute write at SAVE and replay it at LOAD

- The hook already rewrites driver entry points returned by `cuGetProcAddress` (hook.cpp:2340-2432). Add these to
  that table and to the `extern "C"` symbols:
    - `cuFuncSetAttribute`
    - `cuKernelSetAttribute`
    - `cuFuncSetCacheConfig`
    - `cuKernelSetCacheConfig`
    - `cuFuncSetSharedMemConfig`

  cudart's `cudaFuncSetAttribute` reaches the driver through that table, and libraries that call the driver
  directly (flashinfer cubins, DeepGEMM, cutlass-dsl via cuda-python) are covered by the symbol interpose.
- SAVE:
    - map each handle to (binary hash, mangled name) with `cuFuncGetModule` / `cuKernelGetLibrary` and the
    existing `module_or_library_handle_to_hash` + entrypoint tables;
    - append `(hash, name, attr, value)`, last write wins, to `func_attr_writes.json` in the archive;
    - keep only functions that some graph node uses (`mark_binary_used`).
- LOAD: right after `load_cuda_modules_and_libraries`, before any template build or prewarm, apply every entry to
  the archive copy, on the CUkernel **and** its per-context function (the same double-set ClusterOptIn.h does).
  This replaces the three special cases with one path that covers any attribute a library adds later. The
  derived non-portable opt-in stays as a fallback for archives saved before the change.
- Cross-check: at SAVE end, snapshot all settable attributes of every used function (`cuKernelGetAttribute`).
  At LOAD, after applying, re-query and print a one-line diff per mismatch. That catches writes that bypass the
  hook.

### 3.2 Flag in-capture persistent allocations at SAVE (the M2 class)

- The hook already logs allocation and free events per graph. At the end of SAVE, list every allocation made
  **inside the capture window of graph i** that is still live after the capture loop. Resolve which graphs
  reference its range, using the existing param scan in `archive_check.py`.
- Any such range referenced by a graph j ≠ i, or read by eager code, is exactly a lazy cache or workspace that
  native sglang would have created in its warmup (M2b). Report it with the Python stack captured at allocation
  time (the hook can store a short `traceback.extract_stack` through a Python callback only in this debug mode).
- That turns items 2 and 3 of the ranked list from silent into a named SAVE warning, independent of backend.
- Add one more check: an allocation made during capture that goes through torch symm-mem, cuMem multicast or
  NVSHMEM. The hook sees `cuMemCreate` / `cuMulticastCreate` / `cuMemAddressReserve`. Such an allocation is a
  collective resource that LOAD cannot recreate, so make it a hard error with the stack.

### 3.3 Replay `graph_capture()`-exit work, or refuse it

- Generic rule: LOAD's decode path should enter the same `graph_capture()` and `capture_session` contexts that
  SAVE entered, with an empty body, so that context-exit hooks run on both sides. It should also compare what
  they registered.
- For custom AR that is not enough, because LOAD has no captured inputs. Record the ordered
  `(data_ptr, nbytes)` list from `CustomAllReduceV2._graph_inputs` / v1 `graph_unreg_buffers_` at SAVE, feed it
  back before `_register_graph_inputs()` at LOAD, and check `_graph_counter` equality. Until that exists, reject
  custom AR when the resolved `disable_custom_all_reduce` is False.

### 3.4 Put environment identity and env pins in the archive

- Record these, and refuse or warn at LOAD on any difference:
    - CC, SM count, GPU name, driver / compat version, `CUDA_MODULE_LOADING`;
    - the resolved values of the kernel-selecting envs from the sglang `envs` registry, filtered by prefix
    `SGLANG_OPT_`, `SGLANG_*_BACKEND`, `SGLANG_ENABLE_*`, `SGLANG_DEEPGEMM_*`, `SGLANG_TRTLLM_*`,
    `SGLANG_FLASHINFER_*`, `SGLANG_DSV4_*`, `SGLANG_DSA_*` (about 200 of the 654 in `srt/environ.py`).
- Move the recipe-only pins into `pin_graph_fields` / `setup_ld_preload_env`: `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`,
  `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`.

### 3.5 A pre-capture bootstrap registry instead of ad-hoc bootstraps

- Today's bootstraps are hard-coded in `begin_graph_layout` (hooks.py:405-446). Make the list data-driven:
  (predicate on resolved args/model, callable), run at the same sequence point on SAVE and LOAD.
- New entries from this review:
    - `warmup_all_flashinfer_megamoe_layers` (it exists upstream, only commented out);
    - DeepGEMM `get_symm_buffer_for_mega_moe` with the forward's key;
    - the dsv4-trtllm semaphore getter;
    - the nixl buffer;
    - the K3 named symm buffers;
    - the qsa workspace.
- Where an upstream hook exists (`prepare_before_cuda_graph_capture`, `BaseRunner.warmup`), prefer asking
  upstream to move the lazy build there. It then runs on both modes with no Foundry code.

## 4. Test coverage summary

- **Exercised:** fa3, flashinfer (Q1, unconfirmed), dsv4-flashmla (B2), dsa (B3), GDN Triton (B1/B5), DeepEP LL +
  deep_gemm, torch symm-mem AR, DP attention, FULL prefill graphs (B4), inductor compile inside capture (GLM,
  gpt-oss).
- **Never exercised:**
    - anything on sm100/sm120 (fa4, trtllm-gen MHA/MLA/MoE, cutedsl, tokenspeed, qsa, K3 fusion, megamoe, DSV4.1
    variants);
    - custom all-reduce under SAVE/LOAD;
    - `--enable-symm-mem`, NVLS, mscclpp;
    - TBO;
    - nixl, flashinfer a2a, DeepEP v2 (host could not run it);
    - PP;
    - `--cpu-offload-gb`;
    - LoRA and speculative (rejected).
- **The B2 fix (8298912) still needs its GPU validation**, step list in dsv4_cluster_fix.md.
