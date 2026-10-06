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
| 3 | `FullCudaGraphBackend.capture_one` | upstream's two warm-up forwards for the shape in the private pool, then capture into a `foundry.CUDAGraph`, `save_graph` | both runners: hand back the next archived graph for this shape (`restore_next_decode_graph` / `restore_next_prefill_graph`, order checked against the archive); its allocator events replay here (`finish_one_graph_load`); `forward_fn` is not run. `hooks.capture_one` takes the shape, `forward_fn`, the pool, the stream, the prefill request-slot count, the post-warm-up hook and the TP group and returns `(graph, output)`; the caller stores it (the plugin's wrapper here, SGLang's backend on the dependency route) |
| 3a | `DecodeCudaGraphRunner._capture_one_stream`, `PrefillCudaGraphRunner._capture_one_stream` (the per-shape loop, inside the capture session) | `warmup_pool.run_capture_loop`: persistent bootstrap, the loop (3 warming and capturing), pool release (no live block, no segment left) | persistent bootstrap, the loop (3 restoring) |
| 3b | `DecodeCudaGraphRunner.capture` | pre-capture bootstraps and layout start, upstream capture loop, manifest, `pack_fatbins`, `record_region_layout` (with `capture_loop_version`) | bootstraps, `preallocate_for_load_mode`, `start_decode_graph_restore` (one `start_graph_builds`), upstream capture loop with 3 swapping each capture for the archived graph, `finish_decode_graph_restore` (count check), `empty_cache()` |
| 3c | `PrefillCudaGraphRunner.capture` | full backend only; bootstraps if it runs first | same as 3b for the prefill graphs: `start_prefill_graph_restore`, upstream loop with 3 swapping each capture (order and shape checked), `finish_prefill_graph_restore` |
| 3d | `DecodeCudaGraphRunner._resolve_shared_read_ends` | - | `POST_REPLAY` fence for restored graphs (they carry no in-graph shared-read marker; see [`known-issues.md`](known-issues.md#event-record-nodes-todo-2026-10-02)) |
| 4 | `Engine._launch_scheduler_processes` (kept a classmethod) | check the resolved graph config, `setup_ld_preload_env(server_args)` (hook lib, NVSHMEM host lib, optional udev shim; re-asserts the plugin's environment pins, see [`overview.md`](overview.md#what-the-plugin-pins-and-why)) before spawning | same |
| 4b | `DataParallelController.launch_tensor_parallel_group` | `setup_ld_preload_env()` (fork-base layout) | same |

Every patch is wrap-and-call and returns to upstream when the mode is `NONE`; no patch replaces an upstream loop on
either mode. Kernel warmup needs no patch: `BaseRunner.warmup()` runs no model forward and executes at the same
sequence point on SAVE and LOAD (inside the upstream `capture()`).

**Single substitution point.** SAVE and LOAD both run sglang's own `capture()` loop unchanged: warmup, `seq_lens`
fill, `reset_index_buffers`, the `graph_capture` stream, `backend.capture_session` (pool + `set_graph_pool_id`), and
per shape `capture_prepare`, TBO / LoRA prep, `attn_backend.init_forward_metadata_out_graph(in_capture=True)`
(FlashInfer's per-bs wrappers and workspaces), `deepep_adapter.capture()`. Only `FullCudaGraphBackend.capture_one`
(patch 3) differs: SAVE captures and saves, LOAD restores. Each shape's eager allocations therefore happen at the same
point and in the same order in both modes, and every graph's allocator events replay right where SAVE captured it.

**Device binding (DP/TP/EP).** `set_allocation_region` binds the region to the device current at call time. The
bring-up sets the device itself, but patch 1 runs before it, so rank > 0 would reserve on `cuda:0` and fault later
with an async illegal memory access. Patch 1 therefore calls `set_device(gpu_id)` first.

**LOAD starts every decode graph build in one call.** `start_decode_graph_restore` calls
`start_graph_builds(all_paths)` once before the loop (or takes over the builds started at setup); the manifest's
template / on-demand linking needs all graphs in one build. Per-graph builds leave on-demand graphs without a
`shared_exec` and replay aborts with `Called CUDAGraph::replay without a preceding successful capture or load`. The
builds are finished one per shape inside the loop (`finish_one_graph_load`, from patch 3). `finish_decode_graph_restore`
checks the count and logs `[Foundry] Loaded N SGLang graphs in X s (builds B s, handover H s)`, only Foundry's restore
work, and `[Foundry] SGLang decode capture loop on LOAD: Y s (N shapes, restore X s, per-shape prep Z s)` for the loop
wall (the prefill runner logs the same pair; see [`save-load-workflow.md`](save-load-workflow.md)).

**Archive compatibility.** SAVE writes `capture_loop_version = 2` (`runtime.CAPTURE_LOOP_VERSION`) into each rank's
`region_layout.json`. LOAD checks it at setup (`check_capture_loop_version`, right after `setup_graph_extension`,
before the weight load) and refuses an archive without it or with another value; archives saved by the pre-pass code
must be re-saved.

The alloc-offset log lines (`[Foundry] SGLang alloc_offset[<point>]=...`) mark each step: `after_setup_graph_ext`,
`after_init_parallel_runtime`, `after_init_torch_dist`, `after_scratch_skip`, `before/after_init_memory_pool`, the
bootstrap points below, `before/after_preallocate`, `before/after_prefill_restore` (prefill graphs on),
`before_decode_restore`, `after_load_all_graphs` (must equal SAVE's `final_alloc_offset`),
`after_post_load_empty_cache`.

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
- **No backend-specific code.** LOAD runs the upstream loop, so sglang itself re-creates all per-shape host state
  (FlashInfer / fa3 / hybrid decode metadata, the `deepep_adapter` captured mode, TBO / LoRA buffers) exactly as on
  SAVE. The earlier FlashInfer pre-pass, reuse shim, post-load fa3 metadata pass and explicit
  `deepep_adapter.capture()` are gone. What remains is the `_resolve_shared_read_ends` `POST_REPLAY` patch (3d):
  restored graphs never run `run_once`, so they carry no in-graph shared-read marker. It costs the scheduler's
  write/forward overlap on LOAD; see [`known-issues.md`](known-issues.md#event-record-nodes-todo-2026-10-02) for why
  the marker is lost and the plan to restore it.
- **C++: bind context on graph-build pool workers** (`CUDAGraphParallel.cpp`). EP graphs
  carry `NODE_EVENT_RECORD/WAIT` nodes → `cuEventCreate` during on-demand prep, which runs
  on `SimpleThreadPool` workers that never `cuCtxSetCurrent(main_ctx)`. Added that call
  (mirrors the bg thread). Dense graphs never hit it (no event nodes), which is why it
  surfaced only on sglang EP.

See [`../../recipe/sglang/README.md`](../../recipe/sglang/README.md) for the EP serve
config and kernel-stack notes (all wheel-provided by the sglang install).
