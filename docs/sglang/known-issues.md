# Known issues — SGLang integration

## Archives from before capture_loop_version 2 (RULE, 2026-09-30)

SAVE and LOAD now both run sglang's own decode capture loop and substitute only `FullCudaGraphBackend.capture_one`;
the old code ran a FlashInfer metadata pre-pass before the loop, which allocates the per-bs wrappers in another
order, and LOAD cannot replay that layout. `region_layout.json` (per rank) records `capture_loop_version = 2`
(`runtime.CAPTURE_LOOP_VERSION`); LOAD checks it at setup, before the weights load, and refuses an archive without it
or with another value: `Foundry archive ... has capture_loop_version=None, this code needs 2: re-SAVE with the
current code`. Rule: re-SAVE every archive saved by the old code.

## DeepEP v2: NCCL's window descriptor is placed by the driver (NOTE, 2026-11-06)

DeepEP v2 kernels take NCCL's window by address, and NCCL allocates the window descriptor from a CUDA memory pool
(`ncclCommWindowRegister`), which the driver places itself: the pool API takes no address, so this is the one device
address in the graphs that the region does not control. The driver's choice is deterministic: SAVE and LOAD get the
same address when they run with the same environment and the same steps before the first capture scope (measured on
drivers 580 and 595, in either position of LOAD's binary restore and exec prewarm). Foundry relies on that and does
not interpose the pool API. Consequence: SAVE and LOAD must use identical environment variables (for example
`CUDA_MODULE_LOADING`); a difference can move the pool by 512 MiB, and the restored dispatch then fails at the first
replay with an illegal memory access while the region offsets still match.

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

**What LOAD loses.** LOAD never runs `run_once` (`hooks.capture_one` swaps the capture for the
archived graph), so `runner.in_graph_metadata_prep_done` stays `None`. The archive does not help either. SAVE
serializes event-record / event-wait nodes as `EventRecordNode` / `EventWaitNode` with an `event_id` numbered per
graph in first-seen order (`event_to_id` in `CUDAGraph.cpp`); LOAD creates a fresh private `CUevent` per id
(`cuEventCreate`, kept in `loaded_graph_resources_->created_events`) and adds the node with it. A restored graph
therefore records into an event that only Foundry holds: the link from the node to the host object
(`in_graph_metadata_prep_done`) is not stored, and the id is local to one graph, while SGLang shares one event across
all of a runner's graphs. (Not checked against a SGLang archive for this particular node; the statement is
from the SAVE / LOAD code, which handles every event node this way.)

SGLang's fallback for "`IN_REPLAY` declared, no marker" is `POST_REPLAY` (`_resolve_shared_read_ends`).
That is correct, but the scheduler's writes for batch N+1 now wait for the whole forward of batch N, so the overlap the
marker buys is lost. The loss was not measurable in our bs 1-128 TPOT runs; it may matter for small-batch, CPU-bound
serving, where the scheduler's next-batch preparation is a large share of the step (not measured).

**TODO (next phase).** Give events a stable identity across the archive instead of a per-graph index: SAVE records,
per event node, an id for the host event (shared by every graph that records it) and, where the integration registers
one, which host object it is. LOAD creates one event per id, adds the nodes with it, and hands the event back to the
integration (SGLang: set `runner.in_graph_metadata_prep_done`). Then upstream's own `IN_REPLAY` path runs on LOAD
too. The generic mechanism
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
capture rejects (DeepGEMM's runtime init), run with the allocation region
suspended, so the model's compiles and JIT loads happen inside the captured forward and are
recorded; (2) the gatherer's state is built before capture on both modes (now by sglang's
warmup, `LogitsProcessor.prebuild_logits_gatherer`), so the graphs capture the multicast gather exactly as native sglang does and it lands at
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
  load does not fix it (the call was kept as hygiene at the time; it no longer exists).
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

**Fix** (no dummy forwards): create the buffer (now sglang's warmup
prebuild, `DeepEPDispatcher.prebuild`) BEFORE the SAVE-side warmup pass, so the buffer is created at the same
allocation-sequence point in both modes and its address matches by
construction. Archives must be RE-SAVED (baked addresses change).
`torch.cuda.empty_cache()` after graph load was also added as allocator
hygiene (since removed). Validated: the repro sequence is fully clean, first token
`<think>` at logprob ~0.0, DP2 regression clean.

**Follow-up.** Add a token-parity check (fixed prompt set, temp 0,
logprobs) to the validation recipe so LOAD-vs-baseline divergence is caught
routinely.

Repro: `experimental/expert-parallel/serve_sglang_qwen_30b_a3b.sh 2 --load`,
then two chat requests, temp 0: `"What is 15 - 6? Answer briefly."` — the
second (or the first, if the server warmup ran) starts with `" zeroes"`.

## Garbage output on both SAVE and LOAD with `--moe-a2a-backend none` at EP > 1 (FIXED, 2026-10-19)

**Symptom.** Qwen3-30B-A3B-FP8 EP2 (attention DP or TP) with the plain EP all-to-all (`--moe-a2a-backend none`,
sglang's standard token dispatcher): the SAVE engine's own output is garbage from the first token of every request,
LOAD's as well (decode-only: first token right, then wrong); native capture is correct; offsets match, no hook
errors. DeepEP v1 / v2 rows are unaffected.

**Cause.** `StandardDispatcher` built `local_expert_mapping` (global expert id -> local slot, or -1) lazily on its
first dispatch. SAVE captures the first shape without warm-up forwards, so that first dispatch ran under stream
capture: the tensor was allocated in the graph pool and its `full`/`arange` fill was only recorded, never
executed. Every later eager forward (and every graph) routed through the uninitialized mapping, and LOAD's graphs
referenced its capture-time address.

**Fix.** sglang's `StandardDispatcher.prebuild()` builds the mapping with the same builder as its lazy path, in
the EagerRunner warmup (`BaseRunner._prepare_capture_state`) before any capture, at the same sequence point on SAVE
and LOAD; Foundry's first capture scope checks it exists (`check_capture_state_prebuilt`).
The rule stands: state sglang creates lazily on the first forward must exist before the first capture.
