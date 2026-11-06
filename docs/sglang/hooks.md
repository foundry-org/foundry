# SGLang Call Sites

What Foundry does at each of SGLang's call sites (`sglang.srt.utils.foundry_adapter` ->
`foundry.integration.sglang.api` -> `hooks.py`; the list is in `api.py`). The server-args pins (decode `full`,
prefill `full` / `disabled`, profiling and FlashInfer autotune off) run in SGLang's resolution steps
`handle_cuda_graph_persistence` and `validate_cuda_graph_persistence`; see `pins.py`.

## Call sites, in process order

| # | SGLang call site | SAVE | LOAD |
|---|---|---|---|
| 1 | `sglang.srt.distributed.bootstrap.init_parallel_runtime` (head; called from `Scheduler.__init__` before any `ModelRunner`) | `set_device(gpu_id)`, then `setup_graph_extension` reserves the rank's VMM region before any process group exists | same, plus the binary restore (`load_cuda_modules_and_libraries`) and the exec-memory prewarm thread |
| 2 | `KVCacheConfigurator._resolve_memory_pool_config` | run upstream, record the context overrides it issued (e.g. the Mamba/GDN state-pool size) and write `warmup_state.json` (pool config + overrides) | skip profiling: `gc.collect()` + `empty_cache()` (mirrors SAVE's side effects), return the saved `MemoryPoolConfig`, replay the overrides |
| 3 | `FullCudaGraphBackend.capture_one` | capture into a `foundry.CUDAGraph` (no warmup forwards), `save_graph` | both runners: hand back the next archived graph for this shape (`restore_next_decode_graph` / `restore_next_prefill_graph`, order checked against the archive); its allocator events replay here (`finish_one_graph_load`); `forward_fn` is not run. `hooks.capture_one` takes the shape, `forward_fn`, the pool and the stream (the prefill request-slot count comes from the enclosing `capture_scope("prefill", ...)`) and returns `(graph, output)`; SGLang's backend stores them |
| 3b | `DecodeCudaGraphRunner.capture` | pre-capture bootstraps and layout start, upstream capture loop, manifest, `pack_fatbins`, `record_region_layout` (with `capture_loop_version`) | bootstraps, `preallocate_for_load_mode`, `start_decode_graph_restore` (one `start_graph_builds`), upstream capture loop with 3 swapping each capture for the archived graph, `finish_decode_graph_restore` (count check), `empty_cache()` |
| 3c | `PrefillCudaGraphRunner.capture` | full backend only; bootstraps if it runs first | same as 3b for the prefill graphs: `start_prefill_graph_restore`, upstream loop with 3 swapping each capture (order and shape checked), `finish_prefill_graph_restore` |
| 4 | `Engine._launch_scheduler_processes`, around each `proc.start()` | `configure_subprocess(server_args)`: check the resolved graph config, re-assert the environment pins (see [`overview.md`](overview.md#what-foundry-pins-and-why)), set the children's preload environment (hook lib, NVSHMEM host lib, optional udev shim) for the spawn only | same |

No call site replaces an upstream loop on
either mode. Kernel warmup needs no call: `BaseRunner.warmup()` runs no model forward and executes at the same
sequence point on SAVE and LOAD (inside the upstream `capture()`).

**Single substitution point.** SAVE and LOAD both run sglang's own `capture()` loop unchanged: warmup, `seq_lens`
fill, `reset_index_buffers`, the `graph_capture` stream, `backend.capture_session` (pool + `set_graph_pool_id`), and
per shape `capture_prepare`, TBO / LoRA prep, `attn_backend.init_forward_metadata_out_graph(in_capture=True)`
(FlashInfer's per-bs wrappers and workspaces), `deepep_adapter.capture()`. Only `FullCudaGraphBackend.capture_one`
(call site 3) differs: SAVE captures and saves, LOAD restores. Each shape's eager allocations therefore happen at the same
point and in the same order in both modes, and every graph's allocator events replay right where SAVE captured it.

**Device binding (DP/TP/EP).** `set_allocation_region` binds the region to the device current at call time. The
bring-up sets the device itself, but call site 1 runs before it, so rank > 0 would reserve on `cuda:0` and fault later
with an async illegal memory access. Call site 1 therefore calls `set_device(gpu_id)` first.

**LOAD starts every decode graph build in one call.** `start_decode_graph_restore` calls
`start_graph_builds(all_paths)` once before the loop; the manifest's
template / on-demand linking needs all graphs in one build. Per-graph builds leave on-demand graphs without a
`shared_exec` and replay aborts with `Called CUDAGraph::replay without a preceding successful capture or load`. The
builds are finished one per shape inside the loop (`finish_one_graph_load`, from call site 3). `finish_decode_graph_restore`
checks the count and logs `[Foundry] Loaded N SGLang graphs in X s (builds B s, handover H s)`, only Foundry's restore
work, and `[Foundry] SGLang decode capture loop on LOAD: Y s (N shapes, restore X s, per-shape prep Z s)` for the loop
wall (the prefill runner logs the same pair; see [`save-load-workflow.md`](save-load-workflow.md)).

**Archive compatibility.** SAVE writes `capture_loop_version = 2` (`runtime.CAPTURE_LOOP_VERSION`) into each rank's
`region_layout.json`. LOAD checks it at setup (`check_capture_loop_version`, right after `setup_graph_extension`,
before the weight load) and refuses an archive without it or with another value; archives saved by the pre-pass code
must be re-saved.

The alloc-offset log lines (`[Foundry] SGLang alloc_offset[<point>]=...`) mark each step: `after_setup_graph_ext`,
`before_memory_pool_config`, `before/after_collective_bootstrap`,
`after_lazy_runtimes` (SAVE), `before/after_preallocate`,
`before/after_prefill_restore` (prefill graphs on),
`before_decode_restore`, `after_load_all_graphs` (must equal SAVE's `final_alloc_offset`).

## Pre-capture bootstraps and EP additions

Pre-capture bootstraps run at the first runner capture on SAVE and LOAD alike (same sequence point, no model
forward). Both attention layouts work with EP: DP attention, and TP attention (or attention-TP inside DP groups)
with its all-reduce through torch symmetric memory (validated attn-TP + EP rows up to attn-TP4 + EP8 in
`recipe/sglang/README.md`).

- **sglang's own pre-capture state** (`check_capture_state_prebuilt`, graph_ops). sglang builds, in its
  EagerRunner warmup (`BaseRunner._prepare_capture_state`, before the first capture scope, both modes), the
  state its first forward would otherwise create lazily: every MoE dispatcher's buffer (DeepEP, DeepEP v2,
  Mooncake, NIXL; `BaseDispatcher.prebuild`), the standard dispatcher's expert mapping (EP > 1,
  `--moe-a2a-backend none`) and the logits all-gather's symmetric-memory state (`MultimemAllGatherer`).
  Foundry only checks, at the first scope, that none is still pending (it raises naming the object), so all
  of it exists before the first capture.
- **NCCL communicator bootstrap** (`bootstrap_collective_connections`). NCCL connects a communicator on its first
  collective, which a capturing stream rejects (seen on the attention-TP sub-group's first reduce-scatter of
  Qwen3.5-35B-A3B attention-TP2 + EP4). Foundry issues small and large all-reduce / all-gather / reduce-scatter on
  every initialized group through sglang's coordinators before capture.
- **SAVE-side runtime-init bootstrap** (`bootstrap_lazy_runtimes`, first capture scope). DeepGEMM's
  one-time runtime init (`cudaFree(nullptr)`) is illegal while a stream is capturing and would
  otherwise fire inside the first captured forward, so SAVE runs it directly (allocation-free).
  `torch.compile` is pinned off (`TORCHDYNAMO_DISABLE=1`). Everything else the model does lazily
  (DeepGEMM per-shape JIT compile and `cuLibraryLoadFromFile`) happens inside the captured forward
  and is recorded like any other capture-time event. No eager forward runs on either mode, and
  SAVE and LOAD reach the layout start at the same cursor.
- **No backend-specific code.** LOAD runs the upstream loop, so sglang itself re-creates all per-shape host state
  (FlashInfer / fa3 / hybrid decode metadata, the `deepep_adapter` captured mode, TBO / LoRA buffers) exactly as on
  SAVE. Restored graphs never run `run_once`, so they carry no in-graph shared-read marker; SGLang's
  `_resolve_shared_read_ends` then falls back to `POST_REPLAY`. It costs the scheduler's write/forward overlap on
  LOAD; see [`known-issues.md`](known-issues.md#event-record-nodes-todo-2026-10-02).
- **C++: bind context on graph-build pool workers** (`CUDAGraphParallel.cpp`). EP graphs
  carry `NODE_EVENT_RECORD/WAIT` nodes → `cuEventCreate` during on-demand prep, which runs
  on `SimpleThreadPool` workers that never `cuCtxSetCurrent(main_ctx)`. Added that call
  (mirrors the bg thread). Dense graphs never hit it (no event nodes), which is why it
  surfaced only on sglang EP.

See [`../../recipe/sglang/README.md`](../../recipe/sglang/README.md) for the EP serve
config and kernel-stack notes (all wheel-provided by the sglang install).
