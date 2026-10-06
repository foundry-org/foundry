# Known issues — SGLang integration

## Archives from before capture_loop_version 2 (RULE, 2026-09-30)

SAVE and LOAD now both run sglang's own decode capture loop and substitute only `FullCudaGraphBackend.capture_one`;
the old code ran a FlashInfer metadata pre-pass before the loop, which allocates the per-bs wrappers in another
order, and LOAD cannot replay that layout. `region_layout.json` (per rank) records `capture_loop_version = 2`
(`runtime.CAPTURE_LOOP_VERSION`); LOAD checks it at setup, before the weights load, and refuses an archive without it
or with another value: `Foundry archive ... has capture_loop_version=None, this code needs 2: re-SAVE with the
current code`. Rule: re-SAVE every archive saved by the old code.

## torch 2.14: dynamo refuses to compile inside the capture window (FIXED: SAVE warm-ups in a private MemPool, 2026-10-06)

**Symptom.** With torch 2.14 (SGLang's pin from upstream f385390be5), SAVE failed at the first captured forward
that reached a `@torch.compile`d helper (e.g. SGLang's MoE top-k post-processing): torch >= 2.14's dynamo refuses
to trace while a stream is capturing.

**Cause.** SAVE ran no warm-up forwards, so the first call of every compiled helper, every kernel load and every
one-time initialization happened inside the capture window. The forwards were skipped because their activations,
freed into the default caching-allocator pool, change which later allocations hit the cache, which LOAD (no
forward) cannot reproduce.

**Fix** (`warmup_pool.py`, both routes). Each runner's capture loop (`_capture_one_stream`) runs twice:

1. *Persistent bootstrap*, both modes, in the normal recorded domain: the capture stream's cuBLAS handle,
   FlashInfer's global ALiBi slopes buffer, the MoE router's int32 placeholder, and with EP > 1 every
   `StandardDispatcher.prepare_local_expert_mapping()`. These are real resources a first forward would create;
   left to the warm-ups they would end up in the private pool and be retained (see overview.md).
2. *Preparation pass*: SGLang's loop prepares every shape. On SAVE `capture_one` runs upstream's two warm-up
   forwards (synchronize, TP barrier, forward, `post_warmup_hook`) inside one `torch.cuda.MemPool` per runner,
   with Foundry's allocation region stopped; the cursor must not move. LOAD runs the same preparation and no
   forward. The pool must then hold no live block; it is released by its destructor (no `empty_cache`), and no
   allocator segment may still carry its id. A retained block fails SAVE, naming the block and its owner.
3. *Capture pass*: the loop runs again; SAVE captures (no warm-ups), LOAD restores, as before. SAVE then scans the
   new graph JSONs for kernel arguments into the pool's former segments and fails on any hit.

Why the layout survives: the hook records every allocation in the region from `mark_layout_start`, and LOAD only
requires its cursor to be at or below SAVE's at each graph's `start_base_addr` (replay fast-forwards to it; it aborts
if LOAD consumed more). The warm-ups allocate outside the region and leave the default pool's cache state as it
was, so the preparation pass allocates identically on both modes. Per-shape preparation must therefore be
idempotent across the two passes: FlashInfer's decode wrappers are reused (sglang-side change), since recreated
wrappers sit in a reference cycle and are freed at a GC-dependent time.

Experiment knob `FOUNDRY_SGLANG_WARM_POLICY` (default `two_pass`, the scheme above): `per_shape` runs the loop once,
and SAVE warms each shape in the private pool right before capturing it (the prototype's `per_bs` policy); the
pool lives across shapes and is released after the loop with the same checks, then the pointer audit runs. LOAD
restores in a single pass. It exists to compare the two placements by experiment; `two_pass` stays the default.

Validated (prototype, `experimental/torch214_capture`): Qwen3-1.7B and Qwen3-30B-A3B-FP8 with FlashInfer, TP1 and
attention-DP2/EP2 with DeepEP v1 low-latency and v2 direct, Dynamo enabled; native/SAVE/LOAD greedy outputs and
logprobs equal in the controlled runs (one intermittent LOAD logprob outlier in DeepEP auto mode, not reproduced
on repeat; native shows the same class of variation). Not yet: FA3 / triton / hybrid-linear / DSV4 backends, prefill graphs, full-model `--enable-torch-compile`.

## DeepEP v2 (NCCL windows + GIN) not validated on the plugin route (TODO, 2026-09-30)

`--moe-a2a-backend deepep_v2` was validated on the earlier fork route only (H100 EP2 / EP4, all 256 decode graphs,
foundry e24186b). On the plugin route (upstream main) it has not run: the plugin leaves `NCCL_CUMEM_ENABLE` to
sglang for v2 (its NCCL windows need cuMem), and that exception is the one pin without a measurement behind it.

Two attempts on the bare 4xH200 radix hosts failed before any Foundry code ran:

- `sgl-deep-ep` 0.1.2 is built against NCCL 2.30.7 while torch pins 2.29.7 (`NCCL library version is too old`);
  fixed for the run with `uv pip install --no-deps nvidia-nccl-cu13==2.30.7` plus the CUDA 13.3 compat libcuda.
- With NCCL 2.30.7, plain sglang asserts at the first MoE dispatch: `NCCL GIN is unavailable` (`ElasticBuffer.__init__`,
  `csrc/kernels/backend/nccl.cu:87`). v2 requires NCCL GIN even for a single-node EP4, i.e. an RDMA NIC visible to
  the process. These hosts are 4-GPU slices of 8xH200 nodes with IB disabled, so v2 cannot start there at all.

What to check once an RDMA-capable host is available (Qwen3-30B-A3B-FP8 EP4, decode graphs; sglang forces prefill
graphs off for v2): the `[Foundry] pin:` line saying `NCCL_CUMEM_ENABLE` was left to sglang; the ElasticBuffer
bootstrap time on SAVE and LOAD; `Loaded 128 SGLang graphs in`; SAVE `final_alloc_offset` == LOAD
`after_load_all_graphs`; no `[HOOK] WARNING: cuMemAddressReserve returned address ... != hint` (NCCL cuMem
allocations made outside the hook's thread-local region would show up there); greedy parity and TPOT vs native.
Why it is expected to work: v2's windows are library-owned physical memory at a hook-carved VA, registered at the
pre-capture bootstrap on both SAVE and LOAD, the same model as DeepEP v1's NVSHMEM heap; the
`NCCL_GRAPH_REGISTER=0` / `NCCL_LOCAL_REGISTER=0` / `NCCL_NVLS_ENABLE=0` pins stay in force.

## Event-record nodes (TODO, 2026-10-02)

**Why SGLang has an in-graph event.** The overlap scheduler writes batch N+1's shared buffers (`req_to_token`,
`seq_lens`, `positions`, ...) on `schedule_stream` while batch N's forward still reads them on `forward_stream`.
`Scheduler._apply_war_barrier` makes `schedule_stream` wait on an event the runner publishes
(`runner.shared_read_done_event`); with no event it falls back to a coarse `wait_stream(forward_stream)`. Each
attention backend declares where its reads of the shared data end (`SharedReadEnds` in `base_attn_backend.py`):

| value | the reads end ... | event published |
|---|---|---|
| `PRE_REPLAY` | before replay: graph-external metadata prep (`init_forward_metadata_out_graph`) copied everything | recorded before `replay` |
| `IN_REPLAY` | inside the graph, up to a marker | the in-graph marker event |
| `POST_REPLAY` | after the whole replay | recorded after `replay` |
| `UNKNOWN` | not audited | none (coarse `wait_stream`) |

The base backend declares `IN_REPLAY` for decode and target-verify; DeepSeek-V4 inherits it for decode and declares it
for DSPARK verify. The `IN_REPLAY` marker is an event-record node inside the captured graph:
`DecodeCudaGraphRunner._record_in_graph_metadata_prep_done` records an external event (`make_external_event`, i.e.
`torch.cuda.Event(external=True)`) right after `attn_backend.init_forward_metadata_in_graph` in `run_once`, near the
start of the graph. The runner keeps it in `self.in_graph_metadata_prep_done` (one event per runner, created at the
first capture) and `_publish_read_done(in_graph=True)` hands it to the scheduler, which can then overwrite the shared
buffers while the rest of the forward still runs.

**What LOAD loses.** LOAD never runs `run_once` (patch 3 in [`hooks.md`](hooks.md) swaps the capture for the
archived graph), so `runner.in_graph_metadata_prep_done` stays `None`. The archive does not help either. SAVE
serializes event-record / event-wait nodes as `EventRecordNode` / `EventWaitNode` with an `event_id` numbered per
graph in first-seen order (`event_to_id` in `CUDAGraph.cpp`); LOAD creates a fresh private `CUevent` per id
(`cuEventCreate`, kept in `loaded_graph_resources_->created_events`) and adds the node with it. A restored graph
therefore records into an event that only Foundry holds: the link from the node to the host object
(`in_graph_metadata_prep_done`) is not stored, and the id is local to one graph, while SGLang shares one event across
all of a runner's graphs. (Not checked against a SGLang archive for this particular node; the statement is
from the SAVE / LOAD code, which handles every event node this way.)

Upstream's fallback for "`IN_REPLAY` declared, no marker" is `PRE_REPLAY`, which fences too early: the scheduler may
overwrite buffers the graph is still about to read (upstream's own TODO in `_resolve_shared_read_ends` says
`POST_REPLAY` is the sound one). Foundry's `shared_read_ends_override` (`integration/sglang/hooks.py`, called from
`_resolve_shared_read_ends`; patch 3d in [`hooks.md`](hooks.md)) therefore returns `POST_REPLAY` on LOAD in that case.
That is correct, but the scheduler's writes for batch N+1 now wait for the whole forward of batch N, so the overlap the
marker buys is lost. The loss was not measurable in our bs 1-128 TPOT runs; it may matter for small-batch, CPU-bound
serving, where the scheduler's next-batch preparation is a large share of the step (not measured).

**TODO (next phase).** Give events a stable identity across the archive instead of a per-graph index: SAVE records,
per event node, an id for the host event (shared by every graph that records it) and, where the integration registers
one, which host object it is. LOAD creates one event per id, adds the nodes with it, and hands the event back to the
integration (SGLang: set `runner.in_graph_metadata_prep_done`). Then `shared_read_ends_override` and its SGLang call
site in `_resolve_shared_read_ends` can go, and upstream's own `IN_REPLAY` path runs unchanged. The generic mechanism
is "graph nodes that reference host objects": events now, conditional handles and child-graph handles later.

## SAVE spends ~30 s in `Init torch distributed` (and ~20 s more elsewhere) copying over-read fatbins (FIXED, 2026-09-25)

SAVE only (LOAD skips fatbin processing). The hook sized each registered fatbin by walking every following
container of the library's `.nv_fatbin` section (`compute_fatbin_size`), so each image was read to the end of the
section: NCCL's 127 MB linked library as 5.95 GB, a 6200-byte cuBLASLt image as 140 MB. SAVE copied and CRC-64'd
7.8 GB per rank during NCCL init (30 s of a 31.5 s dist init) and 14-15 GB / 51-56 s per rank over the run, and
the pack wrote the tails into `fatbin_image_packed.img`. Fixed by sizing a fatbin by its own header
(`header_size + size`). Qwen3-30B-A3B ep4, MAXBS=16, 4xH200: dist init 32 -> 1.6-2.7 s, weight load 4.1 -> 0.26 s,
SAVE health 106 -> 45 s (= native graph), archive 16.7 GB -> 329 MB; parity and every kernel lookup unchanged.
Archives saved before the fix still load (LOAD does not recompute hashes). Details:
`claude-doc/report_coldstart/h200_8gpu_venv/fatbin_overread_fix.md`.

## Bare-host DeepEP engines start ~40 s late: rdma-core's udev wait on blocked verbs devices (FIXED with a shim, 2026-09-25)

Not a Foundry bug, but it hits SAVE and LOAD as hard as the native engines: on a host where
`/dev/infiniband/uverbs*` exist but cannot be opened, DeepEP low-latency's forced IBGDA transport makes
rdma-core wait up to 5 s per HCA for udev in the first DeepEP buffer creation (the first captured shape
natively, the pre-capture bootstrap on SAVE/LOAD). Restore time is unaffected. Cause, evidence and the
`LD_PRELOAD` shim that removes it: [../bare-host-verbs-udev-wait.md](../bare-host-verbs-udev-wait.md)
(`tools/host/no_cdev_wait.c`).

## SAVE fails cold: inductor autotunes a fresh Triton kernel inside the capture window (RESOLVED by the SAVE warm-up pool, 2026-10-06)

Resolved: the warm-ups now run before the capture (entry "torch 2.14: dynamo refuses to compile inside the capture
window"), so autotuning happens outside the capture window; a warm cache is no longer required. Original record:

**Symptom.** GLM-5.3-Flash EP8 (dummy weights) as the FIRST engine ever run for that model in a fresh container:
SAVE's first captured forward fails with `torch.AcceleratorError: CUDA error: operation not permitted when
stream is capturing`, raised from `torch/cuda/__init__.py synchronize` under
`triton_heuristics.autotune_to_one_config -> benchmark_all_configs -> benchmarking.benchmark_gpu`, reached
from the DSA indexer's `@torch.compile` function. The same row passes when plain sglang (or any engine that
ran the model) preceded it in the same container.

**Cause.** SAVE compiles inside capture by design (no eager forward). Codegen, kernel loads and DeepGEMM JIT
are capture-safe; inductor's Triton autotuning is not: a freshly compiled kernel with several candidate
configs is benchmarked on its first run and the benchmark synchronizes the device. With a warm inductor cache
(`autotune_local_cache`, under sglang's `SGLANG_CACHE_DIR/inductor`) the best config is loaded and nothing is
benchmarked. A second, now-fixed cause hid behind the same row: `bootstrap_lazy_runtimes` warmed inductor's
device-keyed pattern initializers in one loop that aborted on `post_grad.lazy_init`'s "Duplicate pattern"
(its patterns are global, the trivial compile had registered them), so the remaining device keys were never
warmed and `_sfdp_init` ran inside capture ("Cannot copy between CPU and CUDA tensors during CUDA graph
capture"). Each key is now warmed on its own, joint-graph initializers only.

**Status.** Requirement, not a bug we can remove: on a machine (cache dir) that has never run the model, run
plain sglang WITH the same decode-graph set once before SAVE (`serve_*.sh <cfg>` without a mode, or
`--warm`), or ship the inductor/Triton cache directory with the archive. SAVE now installs a guard that
replaces the cryptic capture error with a message naming this remedy. An eager run (`--disable-cuda-graph`)
is not a substitute, verified on GLM-5.3-Flash: the compiled function then sees plain tensors instead of the
views of the static batch buffers it sees under capture, dynamo's guards differ
(`L['x']._base.size()[0] == L['x'].size()[0]`, duck-sized `q_scale`/`x` equality), and the inductor artifacts
differ (24 new artifacts from the eager sweep, 64 more from SAVE's own compiles, still autotuning). Plain
sglang's pre-capture warmup forwards use the capture-mode inputs, so their artifacts are the ones SAVE loads.
Disabling pointwise autotuning (`torch._inductor.config.triton.autotune_pointwise=False`) would also avoid it
but changes the kernels SAVE captures relative to native sglang, so it is not the default.

## SAVE capture fails when a first torch.compile inside the capture window copies a CPU constant to the device (RESOLVED by the SAVE warm-up pool, 2026-10-06)

Resolved the same way: the first compile now happens in a warm-up forward, outside the capture. Original record:

**Symptom.** Inkling-Small (bf16, EP8, dummy weights) on 8xH200: the native sglang graph engine captures all
decode graphs, Foundry SAVE fails in the first captured forward:
`hooks.py patched_capture_one -> graph_ops.capture_graph -> run_once -> inkling.py:571 -> dense_mlp.py:81
swiglu_contiguous -> torch._inductor compile_fx` with
`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`.

**Cause.** SAVE runs no eager forward (upstream's pre-capture warmup forwards are suppressed so their
allocations do not enter the layout), so the first call of every `torch.compile`d function happens inside the
capture window by design. Inductor's lazy runtime initialization is done beforehand (`bootstrap_lazy_runtimes`)
and compiles that only generate and load kernels are capture-safe (Qwen3-30B-A3B-FP8 DeepGEMM JIT, gpt-oss's
`swiglu_gpt_oss_sigmoid_alpha` compile). This compile additionally materializes a CPU constant on the device
(a non-pinned host-to-device copy), which the capturing stream rejects. sglang's native path never sees it
because its warmup forwards compile before capture.

**Status.** Not fixed; the model is not servable on this sglang tree anyway (idle DP-attention rank fails in
`inkling.py:940` on a zero-token batch). Candidates, none tested: pin the constant in sglang's
`swiglu_contiguous` (or pass it as a Python scalar); make inductor's constant placement capture-safe upstream;
as a last resort a SAVE-only compile trigger for the affected function under `allocation_region_suspended()`,
which would have to be symmetric with LOAD (LOAD also compiles on first use) and is therefore not preferred.

## Restored EP graphs fault at the first decode on multicast-capable hosts (FIXED, 2026-09-23)

**Symptom.** Qwen3-30B-A3B EP2 (DP attention + DeepEP low-latency + DeepGEMM) on an 8xH200 host:
SAVE completes, LOAD restores all 256 graphs with every offset equal to SAVE's, then the first
decode replay dies with `CUDA error: an illegal memory access`. The same code, and even the exact
stack that had passed this row on maui on 2026-09-02, fails identically on this host; the row
passes on the H100 radix hosts.

**Cause.** sglang's `LogitsProcessor` gathers the TP-sharded logits through
`MultimemAllGatherer` (`triton_symm_mem_ag.py`), whose torch symmetric-memory buffer, multicast
mapping and signal pads are built lazily on the first *eager* forward and skipped under capture
(NCCL all-gather fallback). Whether multicast is granted is a host property (IMEX / fabric):
maui and the H100 hosts log `multimem all-gather disabled (no multicast for world_size=2)`, this
H200 grants it. Foundry's SAVE ran an eager warmup pass before capture, so on this host the
gatherer was built during the pass and every graph baked in `_all_gather_kernel_inner(multicast
pointer, signal pads)` and a `memcpy_triton_kernel` reading the symmetric buffer. Torch symmetric
memory bypasses `cudaMalloc`, so the hook never recorded those allocations, they were absent from
the live ranges, and LOAD never recreated them. Verified causally: the same SAVE with
`TORCH_SYMM_MEM_DISABLE_MULTICAST=1` restores and serves correctly (archive then holds the NCCL
all-gather instead).

**Fix.** Two changes, both following the rule that state sglang builds lazily on the first
forward must be created at the same sequence point on SAVE and LOAD: (1) SAVE no longer runs an
eager warmup pass; the only pre-capture work is the two one-time initializations that stream
capture rejects (inductor's lazy init, DeepGEMM's runtime init), run with the allocation region
suspended, so the model's compiles and JIT loads happen inside the captured forward and are
recorded; (2) `bootstrap_logits_gatherer` builds the gatherer's state before capture on both
modes, so the graphs capture the multicast gather exactly as native sglang does and it lands at
the same addresses on LOAD. Validated: identical output, TPOT within 1.3% of native capture.
`FOUNDRY_SGLANG_CHECK_ARCHIVE=1` (or `python -m foundry.integration.sglang.archive_check`)
reports any graph pointer into the region that no recorded range covers, which names the next
such buffer immediately.

## LOAD-mode first-token corruption on the EP / dp-attention path (FIXED)

**Symptom.** On a foundry-LOADed EP server (Qwen3-30B-A3B, EP=2,
fa3 + dp-attention + DeepEP low-latency), some prompts deterministically get
a wrong *first* generated token at temperature 0 — e.g. `" zeroes"` before
`<think>` — after which generation continues coherently and correctly.
Different prompts get different (but per-prompt deterministic) stray tokens;
many prompts are unaffected. The very first request served by a fresh LOAD
process is clean; corruption appears on later requests as a function of the
per-rank request history. Logprobs show this is not a near-tie flip: the
stray token has logprob ≈ −0.0001 with the correct token ~9 nats down and a
garbage tail — the last-position prefill logits are confidently *wrong*, not
perturbed.

**Ruled out (all verified experimentally, 2026-09-01, H200 ×2, GPUs 0/2):**

- *Two-pass vs single-pass save*: byte-identical archives and byte-identical
  (mis)behavior. Single-pass save is sufficient for sglang — the SAVE path
  has no pass-conditional logic (warmup_state.json stores only the memory
  pool config + final alloc offset, both consumed by LOAD only).
- *Foundry-mode environment* (NCCL_CUMEM/NVLS pins, kernel_warmup
  suppression, capture machinery): a foundry SAVE-mode server answers the
  probe prompts byte-identically to a no-foundry baseline.
- *VMM layout divergence*: per-rank labeled alloc offsets and
  final_alloc_offset match SAVE exactly on LOAD (DP0=137308930048,
  DP1=123247525888 on both sides).
- *The restored decode graphs themselves*: decode continues coherently and
  correctly after the corrupted first token; a 6-request run of one prompt
  replays consistently.
- *The server startup warmup request*: with `--skip-server-warmup` the first
  request is clean, but the corruption then appears from the second request
  on — the warmup was merely the "prior request" in default runs.

**Established mechanism (partially pinned).** Some early forward on the
LOAD path (dp-attention idle/companion batches are the prime suspect)
consumes `attn_backend.forward_metadata` *without initializing it first*:
setting it to `None` after the load-time fa3 metadata pre-pass turns the
corruption into a CUDA illegal-memory-access on the first decode batch
(async, surfaces at `copy_done.synchronize()`), proving the leftover is
consumed. On the normal path the leftover is the pre-pass's last-bs decode
metadata, whose contents (built from initial buffer values, no warmup
forward) differ from what upstream capture would have left — wrong
attention state → confidently wrong last-position logits, entangled with
the other rank via the dp-attention gather.

*Update (capture_loop_version 2).* The leftover described above belongs to the
old LOAD path, which ran a post-load fa3 metadata pre-pass after
`load_all_graphs`. LOAD now runs sglang's own decode capture loop (only
`capture_one` is substituted), so `attn_backend.forward_metadata` after LOAD
is whatever that loop leaves, the same as on SAVE and native sglang; the
"pre-pass's last-bs decode metadata" leftover no longer exists on LOAD. This
dp-attention symptom has not been re-tested on the new path (the DeepEP
buffer fix below resolved it on the old one).

The plain-DP (no dp-attention) flashinfer path shows a much milder analog
(token-trajectory divergence from baseline on 1/4 probe prompts, no stray
tokens) — not yet characterized under the same first-vs-later-request lens.

**Note.** This is not new to the sglang-0.5.12 hook port: the previous
integration also populated fa3 metadata post-load without the capture-time
warmup forwards. Earlier validations measured throughput and coherence,
not token-level parity, so this went unnoticed.

**Deep-dive round 2 (2026-09-01, instrumented; all sglang instrumentation
reverted afterwards).** The earlier metadata theory is REFUTED; the full
localization chain, each step measured:

- The serving rank's prefill runs with CORRECT attention metadata
  (max_seq_len_k == prompt length) and per-layer last-token hidden norms are
  IDENTICAL between a clean and a dirty request through all 48 layers.
- The dp-attention hidden gather is correct: right variant (SUM_LEN /
  all-reduce with pre-zeroed buffer), correct offsets (get_dp_local_info),
  and the all-reduce output norm is identical on both ranks and equal to the
  clean value. Logits-shard assembly offsets are correct too.
- The corruption materializes between the LM-head matmul and the sampled
  token: the logits processor already emits argmax=" zeroes" on dirty
  requests. Serving-rank asymmetry: dp_rank 0 requests are always clean;
  dp_rank 1 requests are corrupted (prompt-dependent visibility).
- NOT an execution-order race: persists under CUDA_LAUNCH_BLOCKING=1.
- NOT torch-cache aliasing alone: `torch.cuda.empty_cache()` after graph
  load does not fix it (kept in hooks as hygiene regardless).
- ADDRESS-DISPLACEMENT SENSITIVE (smoking gun): planting a 64 MB tensor at
  the exact region boundary (final_alloc_offset, 0x601cb2200000 on the dirty
  rank) makes ALL requests clean — and the canary itself records ZERO
  corrupted bytes. So nothing writes into that band; displacing runtime
  eager allocations off their default addresses removes the corruption.
  Conclusion: some component holds a STALE INIT-TIME POINTER to an address
  in the early eager zone; when the runtime allocator reuses that address
  for a live tensor (logits-path tensors by default), the stale reference
  corrupts it. One-sided (NVSHMEM/DeepEP) access explains the
  launch-blocking immunity and the sporadic illegal accesses.

**Prime suspect / exact-divergence hypothesis:** the DeepEP buffer address
differs between SAVE and LOAD. On SAVE it is created mid-warmup-pass
(interleaved with activation allocations); on LOAD, `bootstrap_deepep_buffer`
creates it in isolation → different VMM address for the same object, while
`preallocate_for_load_mode` jumps the cursor to final_alloc_offset and MASKS
the order divergence. Restored graphs and the peer rank's one-sided ops
reference the SAVE address; the runtime eager path uses the LOAD address.

**RESOLUTION (confirmed 2026-09-01).** The exact divergence: the DeepEP
buffer was created at different VMM addresses on SAVE (lazily, mid-warmup
pass, after rank-asymmetric JIT/activation allocations) vs LOAD
(bootstrap_deepep_buffer, in isolation). Cursor logs prove it: with the fix,
both modes create the buffer at 120468799488 -> 122398179328 on both ranks;
before the fix, LOAD's bootstrap landed at the 122398179328 point while
SAVE's warmup-lazy creation landed elsewhere (rank-dependent). One-sided
NVSHMEM traffic and graph-referenced addresses used the SAVE-time address
while the runtime used the LOAD-time one; whichever eager tensor later
reused the stale VA got corrupted (logits-path tensors by default).

**Fix** (hooks.py, no dummy forwards): run `bootstrap_deepep_buffer` BEFORE
the SAVE-side warmup pass, so the buffer is created at the same
allocation-sequence point in both modes and its address matches by
construction. Archives must be RE-SAVED (baked addresses change).
`torch.cuda.empty_cache()` after graph load was also added as allocator
hygiene. Validated: the repro sequence is fully clean, first token
`<think>` at logprob ~0.0, DP2 regression clean.

**Follow-up.** Add a token-parity check (fixed prompt set, temp 0,
logprobs) to the validation recipe so LOAD-vs-baseline divergence is caught
routinely.

Repro: `experimental/expert-parallel/serve_sglang_qwen_30b_a3b.sh 2 --load`,
then two chat requests, temp 0: `"What is 15 - 6? Answer briefly."` — the
second (or the first, if the server warmup ran) starts with `" zeroes"`.
