# Experimental prefetch repair on real archive LOAD graphs

This branch adds an opt-in, deliberately synchronized experiment. It validates
whether the existing QMD repair works after Foundry SAVE/LOAD, rather than merely
applying the SAVE grouping algorithm to native SGLang captures. It is not yet an
optimized serving or scheduler-overlap implementation.

The upstream documentation describes `FOUNDRY_QMD_REPAIR`, but the checked-out
`sglang-dep` revision `6a67e3247954525cdaec9c663c867da9ab28f500` did not implement
that mode: its C++ replay only materialized dedicated member executables. The
bridge in this branch supplies an explicit experimental implementation.

## What executes

1. SAVE uses the original Foundry archive writer and actual template manifest.
2. LOAD uses Foundry's original binary/JSON builder and allocator restoration.
3. `FOUNDRY_QMD_REPAIR=1` skips eager member instantiation. Before any member
   rewrite, the bridge clones the **LOADed template builder graph** and
   instantiates a separate candidate while the repair ABI inventories pristine
   descriptors. The original LOAD template exec stays untouched.
4. A member switch invokes Foundry's existing C++
   `rewrite_shared_graph_for_member()` / `apply_on_demand_updates()` to apply the
   archive's decoded parameters, functions, attributes, memory and event nodes.
   The bridge retains a clone of that target and compares it to its preceding
   source using full typed-DAG and eight-byte edge checks, the existing strict
   nonkernel guards, and equal launch attributes.
5. Only a successful public guard permits `prepare -> execUpdate -> finish ->
   upload -> synchronize`. The ordinary `graph.replay()` binding then launches
   that persistent candidate through the original RNG prologue/stream logic.
   A same-member replay skips all update work.
6. The benchmark independently instantiates a fresh exec from the same current
   LOAD target clone and compares numerical outputs and steady GPU event time.
   SAVE capture measurements and generated-token comparison are separate
   cross-process controls.

The candidate is **not the original LOAD template exec**. It is an additional,
pristine, registered exec of the actual LOADed graph. This distinction is recorded
in every registration receipt. It avoids falsely treating an already instantiated
exec with no saved inventory as pristine registration.

## Invocation and artifacts

The process must already have loaded the exact tested helper-roles-r2 repair SO
before worker threads start, with `QMDREPAIR_ENABLE=1`,
`QMDREPAIR_HELPER_ROLES=1`, and `QMDREPAIR_V3_WRITE=1`. Nothing replaces the host
CUDA driver. The tested 595.71.05 SO has SHA-256
`a518d9d1e3292691b606d913d6d7ce280ca8e799c02d48862840a5d296fd1acc`.

Set `FOUNDRY_QMD_REPAIR=1` for the experimental binding. Optional
`FOUNDRY_QMD_RESEARCH_MODE=unpatched` uses the same shared LOAD path without repair;
it is an explicit comparison mode, not silent fallback.
`FOUNDRY_QMD_RECEIPT_DIR=/private/experiment/path` preserves one JSON receipt per
registration and actual switch. Receipts include the source/exec provenance,
full public difference, repair counters and upload completion. Failed guards
write a receipt and exit the worker with status 86, without further CUDA cleanup.

The Python API in `foundry.research_qmd` provides:

- `retain_owners(engine, runner, graphs)` to retain all owners for a bounded test;
- `prepare(graph)` to register/update without launch;
- `replay(graph)` used automatically by the C++ replay binding;
- `state(graph)` to obtain serializable `candidate_exec`, `source_graph`,
  `initial_source_graph`, `template_exec`, member IDs and `last_update`.

The `_research_info`, `_research_rewrite`, and `_research_replay_exec` C++ methods
are diagnostic escape hatches; arbitrary handles are not validated by the
launcher. The test controller must use its owned valid candidate/fresh handles.
The pinned standalone guard copies live in `python/foundry/_qmd_research` with
source and vendored hashes. The original memcpy, topology, edge and event predicates remain unchanged.
The first actual LOAD attempt exposed a successful NULL allocation context for
a Foundry VMM memset destination; this branch adds a narrow fixed-parameter 1D
memset path with complete mapped bounds, VMM allocation handle/properties, device
READWRITE access, buffer/block identity, and exact before/after metadata proof.
A NULL context without that proof still rejects.

CPU failure-order checks:

```sh
python3 -m unittest discover -s tests/research_qmd -v
```

## Current boundaries

The bridge synchronizes the device before every mutation and after uploads.
It therefore does not establish update/launch overlap safety or end-to-end TPOT
improvement. Python guards, graph clones and full receipts are research overhead.
All created clones/candidates and their owners stay alive for the bounded worker
lifetime; this is intentionally unsuitable for an unbounded production service.

The bridge refuses graphs that consume RNG offsets, changed launch attributes,
ambiguous topology, unsupported operations, unproven memory ownership or changed
event handles. SAVE grouping alone cannot override these guards.

The second actual LOAD attempt reached a legitimate ownership boundary: each
member creates private archive events, while the strict update guard requires
identical event handles. In the opt-in shared mode, C++ now canonicalizes these
private member events to the template's private events at link time. It requires
both event identities to belong to the respective `LoadedGraphResources` creation
lists and a unique bijection of **every record/wait node position and kind**.
An unknown owner, changed usage, changed kind or ambiguous mapping rejects.
The template is normalized too (the JSON builder and decoded template params
may have separate privately created event handles). All original resources stay
owned and alive. The Python guard still requires identical handles; it has not
been relaxed to accept arbitrary renamed external events.

The pure CPU event policy has 13 positive/negative cases:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/research_qmd/test_event_aliases.cpp -o /tmp/foundry_event_aliases_cpu
/tmp/foundry_event_aliases_cpu
```

The normal dedicated-exec mode remains the default. GPU results from the bounded
integration experiment, including failed attempts, belong in its report; passing
the CPU tests does not establish GPU correctness or replay parity.
