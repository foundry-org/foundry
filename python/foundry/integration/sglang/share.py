# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""One archive for several ranks of one world size (archive share).

SAVE writes every rank's archive as usual plus ``rank_<r>/share_config.json``
(the server arguments that shape the graphs and the rank's device index). The
post-SAVE tool groups ranks whose archives are identical up to kernel
parameters equal to the rank id, writes ``share_manifest.json`` at the
workspace root and can prune the non-canonical rank directories::

    python -m foundry.integration.sglang.share make <workspace> [--control <ws2>]... [--prune]

Only declared library arguments that carry the rank id are patched
(:data:`DECLARED_RANK_ARGS`: DeepEP low-latency dispatch / combine ``rank``); the
tool verifies that each holds the rank on every rank and refuses otherwise. All
other differing parameter words are reused from the canonical rank as captured
and reported by kernel family; ``--control`` (other SAVEs of the same
configuration) only reports how many of them are same-rank noise.

LOAD with a manifest: before anything is set up, this rank's configuration is
checked against the manifest (:func:`check_load`); a mismatch raises
``Unsupported("archive share", ...)`` and SGLang starts natively. Otherwise the
rank restores its canonical rank's directory and every "= rank" site is
written into the parameter bytes in memory as each ``.cugraph`` is read
(``ops.set_param_patches``), before any node is built.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import sys
from typing import Any

MANIFEST_FILE = "share_manifest.json"
CONFIG_FILE = "share_config.json"
FORMAT_VERSION = 1

# Server arguments that shape the captured graphs and the region layout. Launch
# placement (nnodes, node_rank, base_gpu_id, gpu_id_step, ports) is not among
# them: the device index is checked separately.
CONFIG_FIELDS = (
    "model_path",
    "load_format",
    "dtype",
    "kv_cache_dtype",
    "quantization",
    "tp_size",
    "pp_size",
    "ep_size",
    "attn_dp_size",
    "dp_size",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "moe_a2a_backend",
    "deepep_mode",
    "moe_runner_backend",
    "ep_num_redundant_experts",
    "ep_dispatch_algorithm",
    "init_expert_location",
    "enable_eplb",
    "max_running_requests",
    "max_total_tokens",
    "chunked_prefill_size",
    "mem_fraction_static",
    "page_size",
    "cuda_graph_bs_decode",
    "cuda_graph_backend_prefill",
    "cuda_graph_bs_prefill",
    "disable_cuda_graph_padding",
    "attention_backend",
    "disable_radix_cache",
    "enable_torch_symm_mem",
    "disable_custom_all_reduce",
    "speculative_algorithm",
    "context_length",
)


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return repr(v)


def config_fields(server_args) -> dict[str, Any]:
    return {f: _jsonable(getattr(server_args, f, None)) for f in CONFIG_FIELDS}


def expert_slots(fields: dict[str, Any], num_experts: int | None) -> int | None:
    ep = fields.get("ep_size") or 1
    if num_experts is None:
        return None
    return (num_experts + int(fields.get("ep_num_redundant_experts") or 0)) // int(ep)


# ---------------------------------------------------------------------------
# SAVE side
# ---------------------------------------------------------------------------


def _server_args(fallback):
    """The resolved ServerArgs of this scheduler (the call site passes the
    parallel context, which lacks most fields)."""
    try:
        from sglang.srt.runtime_context import get_server_args

        return get_server_args()
    except Exception:
        return fallback


def write_rank_config(workspace_dir: str, rank: int, server_args, gpu_id: int) -> None:
    """SAVE: what the manifest tool needs to know about this rank."""
    server_args = _server_args(server_args)
    num_experts = None
    try:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(server_args.model_path, trust_remote_code=False)
        num_experts = getattr(hf, "num_experts", None) or getattr(hf, "n_routed_experts", None)
    except Exception:
        pass
    doc = {
        "rank": rank,
        "device_index": gpu_id,
        "fields": config_fields(server_args),
        "num_experts": num_experts,
    }
    with open(os.path.join(workspace_dir, CONFIG_FILE), "w") as f:
        json.dump(doc, f, indent=1)


# ---------------------------------------------------------------------------
# LOAD side
# ---------------------------------------------------------------------------


def read_manifest(workspace_root: str) -> dict | None:
    path = os.path.join(workspace_root, MANIFEST_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _differs(field: str, archive, this) -> str:
    return f"{field} differs (archive {archive!r}, this rank {this!r})"


def check_load(manifest: dict, rank: int, server_args, gpu_ids: dict[int, int]) -> tuple[int, list]:
    """LOAD: (canonical rank, patches with this rank's values) or raise
    ``Unsupported``. ``gpu_ids``: the device index of every rank of the world as
    the launcher places them (computed from the launch, so every rank takes the
    same decision); this rank's own entry is its actual device."""
    from foundry.integration.sglang.api import Unsupported

    def refuse(reason: str):
        raise Unsupported("archive share", reason)

    if manifest.get("format_version") != FORMAT_VERSION:
        refuse(_differs("format_version", manifest.get("format_version"), FORMAT_VERSION))
    fields = config_fields(server_args)
    for k, v in manifest["fields"].items():
        if fields.get(k) != v:
            refuse(_differs(k, v, fields.get(k)))
    world = manifest["world_size"]
    if len(gpu_ids) != world:
        refuse(_differs("world_size", world, len(gpu_ids)))
    rank_map = {int(r): int(c) for r, c in manifest["rank_map"].items()}
    if rank not in rank_map:
        refuse(f"rank {rank} is not in the manifest's rank map")
    devices = {int(c): int(d) for c, d in manifest["device_index"].items()}
    for r in sorted(gpu_ids):
        c = rank_map.get(r)
        if c is None or gpu_ids[r] != devices[c]:
            refuse(
                "device index differs (archive "
                f"{devices.get(c)!r} for rank_{c}, rank {r} runs on cuda:{gpu_ids[r]})"
            )
    canonical = rank_map[rank]
    patches = []
    for s in manifest["patch_sites"]:
        if s["value"] != "rank":
            refuse(f"patch site value {s['value']!r} is not supported")
        value = rank + int(s.get("add", 0))
        if value < 0 or value >= 1 << (8 * int(s["width"])):
            refuse(f"patch value {value} does not fit {s['width']} byte(s) at {s['kernel'][:60]}")
        patches.append(
            (s.get("graph", ""), s["kernel"], s["param"], s["offset"], s["width"], value)
        )
    return canonical, patches


# ---------------------------------------------------------------------------
# Archive reading (include/BinaryGraphFormat.h, format v2)
# ---------------------------------------------------------------------------

_NODE_SZ = 168
_KN_FMT = "<7I Q 2I 6i I 6I 4i 4B 5I 4B"
_KN_FIELDS = [
    "bx",
    "by",
    "bz",
    "gx",
    "gy",
    "gz",
    "smem",
    "binary_hash",
    "fn_off",
    "fn_len",
    "max_dyn_smem",
    "carveout",
    "cl_sched",
    "req_cw",
    "req_ch",
    "req_cd",
    "kna_flags",
    "cdx",
    "cdy",
    "cdz",
    "pcdx",
    "pcdy",
    "pcdz",
    "kna_sched",
    "coop",
    "prio",
    "msd",
    "msdm_def",
    "msdm_rem",
    "attrq",
    "_p",
    "kna_carve",
    "pio",
    "npar",
    "abo",
    "abs",
    "ext_flags",
    "pss",
    "devupd",
    "_p2",
]
_IDENTITY_SKIP = ("fn_off", "fn_len", "pio", "abo", "_p", "_p2")


def load_cugraph(path: str) -> dict:
    """Nodes (type, kernel name, launch identity, params) and the non-param sections."""
    with open(path, "rb") as f:
        d = f.read()
    magic, ver, _flags, nn, _nd, _ng, ns, _nstr = struct.unpack_from("<8sIIIIIII", d, 0)
    if magic != b"CUGRAPH\0":
        raise ValueError(f"{path}: not a .cugraph file")
    sec = {}
    for i in range(ns):
        t, _p, off, sz = struct.unpack_from("<IIQQ", d, 64 + 24 * i)
        sec[t] = d[off : off + sz]
    strings = sec.get(0, b"")
    nodes = []
    for i in range(nn):
        e = sec[1][_NODE_SZ * i : _NODE_SZ * (i + 1)]
        typ = e[4]
        body = e[8:]
        if typ == 0:
            k = dict(zip(_KN_FIELDS, struct.unpack_from(_KN_FMT, body, 0)))
            name = strings[k["fn_off"] : k["fn_off"] + k["fn_len"]].decode(errors="replace")
            params = []
            for j in range(k["npar"]):
                doff, size = struct.unpack_from("<II", sec.get(3, b""), k["pio"] + 8 * j)
                params.append(sec[4][doff : doff + size])
            ident = tuple(v for f, v in k.items() if f not in _IDENTITY_SKIP)
            argbuf = (
                sec.get(5, b"")[k["abo"] : k["abo"] + k["abs"]] if k["abo"] != 0xFFFFFFFF else b""
            )
            nodes.append(("kernel", name, ident, params, argbuf))
        else:
            nodes.append((typ, "", body[:152], [], b""))
    return {
        "nodes": nodes,
        "deps": sec.get(2, b""),
        "other": {t: sec.get(t, b"") for t in (6, 7, 8, 9, 10)},
    }


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_without(path: str, drop: tuple[str, ...]) -> Any:
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, dict):
        for k in drop:
            doc.pop(k, None)
    return doc


def rank_signature(rank_dir: str) -> tuple[str, dict]:
    """Hash of everything that must be equal across ranks sharing one archive:
    files, layout, topology, kernel identity / launch dims, allocator events,
    outputs, kernel images and the share config (minus rank / device)."""
    h = hashlib.sha256()
    graphs = {}
    files = sorted(
        os.path.relpath(p, rank_dir)
        for p in glob.glob(os.path.join(rank_dir, "**"), recursive=True)
        if os.path.isfile(p)
    )
    h.update(json.dumps([f for f in files if f not in ("memhist.pickle",)]).encode())
    for f in files:
        p = os.path.join(rank_dir, f)
        if f.endswith(".cugraph"):
            g = load_cugraph(p)
            graphs[f] = g
            h.update(f.encode())
            h.update(g["deps"])
            for t, b in sorted(g["other"].items()):
                h.update(bytes([t]) + b)
            for typ, name, ident, params, _ab in g["nodes"]:
                h.update(repr((typ, name, ident, len(params), [len(x) for x in params])).encode())
        elif f == CONFIG_FILE:
            doc = _json_without(p, ("rank", "device_index"))
            h.update(json.dumps(doc, sort_keys=True).encode())
        elif f == "warmup_state.json":
            h.update(json.dumps(_json_without(p, ("timestamp",)), sort_keys=True).encode())
        elif f.startswith("graph_") and f.endswith(".json"):
            continue  # JSON twin of the .cugraph (params included); the binary is compared
        elif f in ("memhist.pickle", "nixl_diag.json", "nixl_reloc.json"):
            continue  # diagnostic files (FOUNDRY_DIAG_MEMHIST / FOUNDRY_DIAG_NIXL_OUTSIDE_REGION)
        else:
            h.update(f.encode() + _sha(p).encode())
    return h.hexdigest(), graphs


def _words(pa: bytes, pb: bytes):
    """(byte offset, width, value a, value b) for each differing 4-byte word (1-byte tail)."""
    n = len(pa)
    o = 0
    while o < n:
        w = 4 if o + 4 <= n else 1
        if pa[o : o + w] != pb[o : o + w]:
            fmt = "<I" if w == 4 else "<B"
            yield o, w, struct.unpack(fmt, pa[o : o + w])[0], struct.unpack(fmt, pb[o : o + w])[0]
        o += w


def param_diffs(ga: dict, gb: dict) -> dict:
    """{(node index, param index, byte offset, width): (value a, value b)} of two
    graphs with equal topology (argument buffers compared as param -1)."""
    out = {}
    for i, (na, nb) in enumerate(zip(ga["nodes"], gb["nodes"])):
        if na[0] != "kernel":
            if na[2] != nb[2]:
                for o, w, a, b in _words(na[2], nb[2]):
                    out[(i, -2, o, w)] = (a, b)
            continue
        for j, (pa, pb) in enumerate(zip(na[3], nb[3])):
            if pa != pb:
                for o, w, a, b in _words(pa, pb):
                    out[(i, j, o, w)] = (a, b)
        if na[4] != nb[4]:
            for o, w, a, b in _words(na[4], nb[4]):
                out[(i, -1, o, w)] = (a, b)
    return out


# Declared rank arguments: (kernel-name substrings that must all occur, param index, byte
# offset, width). Only these words are patched on LOAD; every other differing word is
# reused from the canonical rank as captured.
DECLARED_RANK_ARGS = (
    (("deep_ep", "internode_ll", "8dispatch"), 22, 0, 4),  # DeepEP LL dispatch(..., int rank, ...)
    (("deep_ep", "internode_ll", "7combine"), 19, 0, 4),  # DeepEP LL combine(..., int rank, ...)
    (("nixl_ep", "ep_kernels", "8dispatch"), 22, 0, 4),  # NIXL EP LL dispatch(..., int rank, ...)
    (("nixl_ep", "ep_kernels", "7combine"), 20, 0, 4),  # NIXL EP LL combine(..., int rank, ...)
)

_FAMILIES = (
    ("deep_ep", "DeepEP"),
    ("nixl_ep", "NIXL EP"),
    ("flashinfer", "FlashInfer"),
    ("flash", "FlashAttention"),
    ("nvjet", "cuBLAS nvjet"),
    ("cublas", "cuBLAS"),
    ("deep_gemm", "DeepGEMM"),
    ("nccl", "NCCL"),
    ("at6native", "ATen"),
    ("sglang", "sglang kernels"),
    ("triton", "Triton"),
)


def kernel_family(name: str) -> str:
    low = name.lower()
    for key, fam in _FAMILIES:
        if key in low:
            return fam
    return "other"


def _declared(name: str):
    for subs, param, off, width in DECLARED_RANK_ARGS:
        if all(s in name for s in subs):
            return param, off, width
    return None


def classify(
    canonical: int,
    members: dict[int, dict],
    controls: dict[int, Any] | None,
) -> dict:
    """Parameter differences of the ranks of one class.

    ``members``: rank -> {graph file: loaded graph} (canonical included); ``controls``:
    rank -> graphs (or a list of them) of other SAVEs of that rank, optional, only used
    to report how many reused words are same-rank noise. Returns
    {"rank_sites": [(graph, kernel, param, offset, width)], "reused": {family: words},
    "control_noise": n, "unexplained": [...]} where "unexplained" lists declared rank
    arguments whose captured value is not the rank id (the manifest is refused)."""
    base = members[canonical]
    rank_sites: set = set()
    unexplained = []
    for gf, g in base.items():
        for i, node in enumerate(g["nodes"]):
            if node[0] != "kernel":
                continue
            decl = _declared(node[1])
            if decl is None:
                continue
            param, off, width = decl
            for r in members:
                p = members[r][gf]["nodes"][i][3]
                ok = (
                    param < len(p)
                    and off + width <= len(p[param])
                    and int.from_bytes(p[param][off : off + width], "little") == r
                )
                if not ok:
                    unexplained.append(
                        (gf, i, node[1][:100], param, off, width, f"rank {r}: not the rank id")
                    )
            rank_sites.add(("", node[1], param, off, width))
    declared_keys = {(k, p, o) for _g, k, p, o, _w in rank_sites}
    noise_keys = set()
    if controls:
        for r, saves in controls.items():
            if r not in members:
                continue
            saves = [saves] if isinstance(saves, dict) else list(saves)
            runs = [members[r]] + saves
            for x in range(len(runs)):
                for y in range(x + 1, len(runs)):
                    for gf, g in runs[x].items():
                        for key in param_diffs(g, runs[y][gf]):
                            noise_keys.add((gf,) + key)
    reused: dict[str, int] = {}
    seen = set()
    control_noise = 0
    for r, graphs in members.items():
        if r == canonical:
            continue
        for gf, g in graphs.items():
            for key in param_diffs(base[gf], g):
                node, param, off, _w = key
                name = base[gf]["nodes"][node][1]
                if (name, param, off) in declared_keys or (gf,) + key in seen:
                    continue
                seen.add((gf,) + key)
                fam = kernel_family(name) if param != -2 else "memcpy/memset"
                reused[fam] = reused.get(fam, 0) + 1
                control_noise += (gf,) + key in noise_keys
    return {
        "rank_sites": sorted(rank_sites),
        "reused": reused,
        "control_noise": control_noise,
        "unexplained": unexplained,
    }


def make_manifest(
    workspace: str,
    control: str | list[str] | None = None,
    prune: bool = False,
    out=print,
) -> int:
    rank_dirs = sorted(
        glob.glob(os.path.join(workspace, "rank_*")), key=lambda p: int(p.rsplit("_", 1)[1])
    )
    if not rank_dirs:
        out(f"[Foundry share] no rank_* dirs in {workspace}")
        return 2
    ranks = {int(p.rsplit("_", 1)[1]): p for p in rank_dirs}
    configs = {}
    for r, p in ranks.items():
        cp = os.path.join(p, CONFIG_FILE)
        if not os.path.exists(cp):
            out(f"[Foundry share] {cp} missing: SAVE with this Foundry version")
            return 2
        with open(cp) as f:
            configs[r] = json.load(f)
    fields = configs[min(ranks)]["fields"]
    for r, c in configs.items():
        if c["fields"] != fields:
            out(f"[Foundry share] rank {r}: share config differs from rank {min(ranks)}")
            return 2
    sigs, graphs = {}, {}
    for r, p in ranks.items():
        sigs[r], graphs[r] = rank_signature(p)
    classes: dict[str, list[int]] = {}
    for r in sorted(ranks):
        classes.setdefault(sigs[r], []).append(r)
    controls = None
    control_list = [control] if isinstance(control, str) else list(control or [])
    if control_list:
        controls = {r: [] for r in ranks}
        for ctl in control_list:
            for r in ranks:
                cdir = os.path.join(ctl, f"rank_{r}")
                csig, cg = rank_signature(cdir)
                if csig != sigs[r]:
                    out(f"[Foundry share] {cdir} differs from rank_{r} structurally: not a control")
                    return 2
                controls[r].append(cg)
    rank_map, device_index, sites, report = {}, {}, set(), {}
    unexplained_all = []
    reused_all: dict[str, int] = {}
    noise_total = 0
    for members in classes.values():
        c = members[0]
        res = classify(c, {r: graphs[r] for r in members}, controls)
        for r in members:
            rank_map[str(r)] = c
            if configs[r]["device_index"] != configs[c]["device_index"]:
                res["unexplained"].append(
                    ("-", -1, "device index", -1, 0, 0, {r: configs[r]["device_index"]})
                )
        device_index[str(c)] = configs[c]["device_index"]
        sites.update(res["rank_sites"])
        unexplained_all += res["unexplained"]
        for fam, n in res["reused"].items():
            reused_all[fam] = reused_all.get(fam, 0) + n
        noise_total += res["control_noise"]
        report[str(c)] = {
            "members": members,
            "rank_sites": len(res["rank_sites"]),
            "reused_words": sum(res["reused"].values()),
        }
    out(
        f"[Foundry share] {len(ranks)} ranks, {len(classes)} class(es): "
        + "; ".join(f"rank_{c} <- {v['members']}" for c, v in report.items())
    )
    out(f"[Foundry share] {len(sites)} declared rank-argument site(s):")
    for _g, k, p, o, w in sorted(sites):
        out(f"    {k[:90]} param {p} +{o} ({w} B) = rank")
    out(
        f"[Foundry share] {sum(reused_all.values())} other differing param words reused as "
        f"captured: {dict(sorted(reused_all.items(), key=lambda kv: -kv[1]))}"
        + (f"; {noise_total} of them differ between SAVEs of one rank" if controls else "")
    )
    if unexplained_all:
        out(
            f"[Foundry share] {len(unexplained_all)} declared rank argument(s) "
            "do not hold the rank:"
        )
        for u in unexplained_all[:20]:
            out(f"    {u}")
        out("[Foundry share] refused: no manifest written")
        return 1
    manifest = {
        "format_version": FORMAT_VERSION,
        "world_size": len(ranks),
        "fields": fields,
        "tp_size": fields.get("tp_size"),
        "ep_size": fields.get("ep_size"),
        "attn_dp_size": fields.get("attn_dp_size"),
        "expert_slots_per_rank": expert_slots(fields, configs[min(ranks)].get("num_experts")),
        "rank_map": rank_map,
        "device_index": device_index,
        "signature": {c: sigs[int(c)] for c in device_index},
        "patch_sites": [
            {"graph": g, "kernel": k, "param": p, "offset": o, "width": w, "value": "rank"}
            for g, k, p, o, w in sorted(sites)
        ],
        "verification": {
            "declared_rank_args": [list(d[0]) + [d[1]] for d in DECLARED_RANK_ARGS],
            "reused_words_by_family": reused_all,
            "controls": [os.path.abspath(c) for c in control_list],
            "reused_words_same_rank_noise": noise_total if controls else None,
            "classes": report,
        },
    }
    with open(os.path.join(workspace, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=1)
    out(f"[Foundry share] wrote {os.path.join(workspace, MANIFEST_FILE)}")
    if prune:
        keep = {f"rank_{c}" for c in map(int, device_index)}
        for r, p in ranks.items():
            if os.path.basename(p) not in keep:
                shutil.rmtree(p)
                out(f"[Foundry share] pruned {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m foundry.integration.sglang.share")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write share_manifest.json for a SAVE workspace")
    mk.add_argument("workspace")
    mk.add_argument(
        "--control",
        action="append",
        help="another SAVE workspace of the same configuration (repeatable; only reports how "
        "many reused words are same-rank noise)",
    )
    mk.add_argument("--prune", action="store_true", help="delete the non-canonical rank dirs")
    a = ap.parse_args(argv)
    return make_manifest(a.workspace, a.control, a.prune)


if __name__ == "__main__":
    sys.exit(main())
