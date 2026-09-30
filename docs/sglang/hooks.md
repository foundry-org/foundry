# SGLang Hook Surface

Every runtime patch `foundry.integration.sglang.hooks.install(cfg_path)` installs. It runs in each process that loads
the plugin (the launcher and every scheduler; `foundry.integration.sglang.plugin.activate`, reached from the entry
point `foundry_sglang_plugin:load` when `FOUNDRY_GRAPH_EXTENSION_CONFIG` is set). The superseded in-tree route calls
it through `install_hooks(server_args)`. The server-args pins (decode `full`, prefill `full` / `disabled`, profiling
and FlashInfer autotune off) are not patches: the plugin registers them as resolution hooks on three whitelisted
steps (`apply_inkling_prefill_cuda_graph_default`, `handle_cuda_graph_config`, `handle_other_validations`); see
`plugin.py`.

## Patches, in install order

| # | patched symbol | SAVE | LOAD |
|---|---|---|---|
| 1 | `sglang.srt.distributed.bootstrap.init_parallel_runtime` (main; called from `Scheduler.__init__` before any `ModelRunner`) or `ModelRunner.init_torch_distributed` (fork base) | `set_device(gpu_id)`, then `setup_graph_extension` reserves the rank's VMM region before any process group exists | same, plus the binary restore (`load_cuda_modules_and_libraries`) and the exec-memory prewarm thread |
| 1b | `ModelRunner.init_torch_distributed` (end; target runner only, main layout) | `skip_to_scratch_boundary()`: weights start at `scratch_space_size` | same |
| 2 | `KVCacheConfigurator._resolve_memory_pool_config` | run upstream, record the context overrides it issued (e.g. the Mamba/GDN state-pool size) | skip profiling: `gc.collect()` + `empty_cache()` (mirrors SAVE's side effects), return the saved `MemoryPoolConfig`, replay the overrides |
| 2b | `ModelRunner.alloc_memory_pool` | write `warmup_state.json` (pool config + overrides) | pass-through with offset logging |
| 3 | `FullCudaGraphBackend.capture_one` | capture into a `foundry.CUDAGraph` (no warmup forwards), `save_graph` | prefill runner only: hand back the next archived graph for this shape |
| 3b | `DecodeCudaGraphRunner.capture` | pre-capture bootstraps and layout start, FlashInfer metadata pre-pass + reuse shim, upstream capture loop, manifest, `pack_fatbins`, `record_region_layout` | bootstraps, `preallocate_for_load_mode`, pre-pass, `load_all_graphs` (one `start_graph_builds` + `finish_graph_loads`), graphs installed under the runner's `ShapeKey`s, `deepep_adapter.capture()`; upstream capture is not called |
| 3c | `PrefillCudaGraphRunner.capture` | full backend only; bootstraps if it runs first | upstream loop with 3 swapping each capture for the archived graph (order and shape checked) |
| 3d | `DecodeCudaGraphRunner._resolve_shared_read_ends` | - | `POST_REPLAY` fence for restored graphs (they carry no in-graph shared-read marker) |
| 4 | `Engine._launch_scheduler_processes` (kept a classmethod) | check the resolved graph config, `setup_ld_preload_env()` (hook lib, NVSHMEM host lib, optional udev shim; `NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0`) before spawning | same |
| 4b | `DataParallelController.launch_tensor_parallel_group` | `setup_ld_preload_env()` (fork-base layout) | same |

Every patch is wrap-and-call and returns to upstream when the mode is `NONE`; the exception is the decode `capture()` on
LOAD, which replaces the upstream loop. Kernel warmup needs no patch: `BaseRunner.warmup()` runs no model forward and
executes at the same sequence point on SAVE and LOAD.

**Device binding (DP/TP/EP).** `set_allocation_region` binds the region to the device current at call time. The
bring-up sets the device itself, but patch 1 runs before it, so rank > 0 would reserve on `cuda:0` and fault later
with an async illegal memory access. Patch 1 therefore calls `set_device(gpu_id)` first.

**LOAD loads every graph in one call.** `load_all_graphs` calls `start_graph_builds(all_paths)` and
`finish_graph_loads(pending)` once each; the manifest's template / on-demand linking needs all graphs in one build.
Per-graph builds leave on-demand graphs without a `shared_exec` and replay aborts with `Called CUDAGraph::replay
without a preceding successful capture or load`.

The alloc-offset log lines (`[Foundry] SGLang alloc_offset[<point>]=...`) mark each step: `after_setup_graph_ext`,
`after_init_parallel_runtime`, `after_init_torch_dist`, `after_scratch_skip`, `before/after_init_memory_pool`, the
bootstrap points below, `after_preallocate`, `after_load_all_graphs`.

## Pre-capture bootstraps and EP additions

Pre-capture bootstraps run at the first runner capture on SAVE and LOAD alike (same sequence point, no model
forward). The DeepEP buffer step is active for the DeepEP-family backends (DeepEP, DeepEP v2, Mooncake). Both
attention layouts work with EP: DP attention, and TP attention (or attention-TP inside DP groups) with its
all-reduce through torch symmetric memory (validated attn-TP + EP rows up to attn-TP4 + EP8 in
`recipe/sglang/README.md`).

- **NCCL communicator bootstrap** (`bootstrap_collective_connections`). NCCL connects a communicator on its first
  collective, which a capturing stream rejects (seen on the attention-TP sub-group's first reduce-scatter of
  Qwen3.5-35B-A3B attention-TP2 + EP4). Foundry issues small and large all-reduce / all-gather / reduce-scatter on
  every initialized group through sglang's coordinators before capture.
- **DeepEP buffer pre-capture bootstrap** (`bootstrap_deepep_buffer`, graph_ops). sglang
  creates the singleton NVSHMEM `Buffer` lazily on the first MoE dispatch — normally
  during the warmup forwards foundry suppresses, which would push creation *inside* the
  captured stream (`deep_ep_cpp.Buffer(...)` → "operation not permitted when stream is
  capturing"). The hook forces it before the capture loop, unwrapping the
  `MaybeTboDeepEPDispatcher._inners` to reach a `DeepEPDispatcher`. Runs on SAVE and LOAD.
- **Logits all-gather bootstrap** (`bootstrap_logits_gatherer`, `capture()` patch, SAVE and LOAD).
  sglang's `LogitsProcessor` gathers TP-sharded logits through `MultimemAllGatherer`, whose torch
  symmetric-memory buffer, multicast mapping and signal pads are built lazily on the first *eager*
  call and skipped under capture (NCCL fallback). Multicast availability is a host property, so on
  a host with multicast any eager forward before capture activates the gatherer and every graph
  bakes in pointers to memory the hook never saw (torch symmetric memory is not `cudaMalloc`);
  LOAD then faults at the first decode, while the same SAVE passes on a host without multicast.
  The hook builds the state (`_build` with a probe of the per-rank logits shard width) before
  capture at the same sequence point in both modes, so it lands at the same addresses and the
  graphs capture the multicast gather like native sglang.
- **SAVE-side runtime-init bootstrap** (`bootstrap_lazy_runtimes`, `capture()` patch). Two
  one-time initializations are illegal while a stream is capturing and would otherwise fire
  inside the first captured forward: inductor's pattern-matcher lazy init (an unpinned
  host-to-device copy) and DeepGEMM's runtime init (`cudaFree(nullptr)`). SAVE runs them
  directly, with the allocation region suspended so their transient tensors never move the
  deterministic cursor. Everything else the model does lazily (torch.compile of its
  functions, DeepGEMM per-shape JIT compile and `cuLibraryLoadFromFile`) happens inside the
  captured forward and is recorded like any other capture-time event. No eager forward runs
  on either mode, and SAVE and LOAD reach the layout start at the same cursor.
- **`deepep_adapter` mode on LOAD.** LOAD replaces the capture loop, so the adapter's
  `_captured_deepep_mode` is never set; replay asserts on it. The hook calls
  `deepep_adapter.capture(is_extend_in_batch=False)` after load.
- **FlashInfer pre-pass gated to FlashInfer.** The decode `capture()` pre-pass
  (`initialize_all_attention_metadata`) + the `_prepare_cuda_graph_metadata` reuse shim handle FlashInfer's per-bs
  wrappers. fa3 (`FlashAttentionBackend`) uses a single
  fixed `init_cuda_graph_state` workspace, so the shim is skipped (detected via absence of
  `indices_updater_decode`); but fa3's per-bs `decode_cuda_graph_metadata[bs]` is still
  populated post-load for the replay lookup.
- **C++: bind context on graph-build pool workers** (`CUDAGraphParallel.cpp`). EP graphs
  carry `NODE_EVENT_RECORD/WAIT` nodes → `cuEventCreate` during on-demand prep, which runs
  on `SimpleThreadPool` workers that never `cuCtxSetCurrent(main_ctx)`. Added that call
  (mirrors the bg thread). Dense graphs never hit it (no event nodes), which is why it
  surfaced only on sglang EP.

See [`../../recipe/sglang/README.md`](../../recipe/sglang/README.md) for the EP serve
config and kernel-stack notes (all wheel-provided by the sglang install).
