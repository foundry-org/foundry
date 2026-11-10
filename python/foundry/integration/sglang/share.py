# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""One archive for several ranks of one world size (archive share).

SAVE writes every rank's archive as usual plus ``rank_<r>/share_config.json``
(the server arguments that shape the graphs and the rank's device index). The
post-SAVE tool groups ranks whose archives are identical up to kernel
parameters equal to the rank id, writes ``share_manifest.json`` at the
workspace root and can prune the non-canonical rank directories::

    python -m foundry.integration.sglang.share make <workspace> [--control <workspace2>]
        [--assume-noise] [--prune]

``--control`` is a second SAVE of the same configuration: parameter words that
differ between two SAVEs of the same rank are uninitialised host bytes and are
ignored. Without a control, differing words that are not the rank id refuse the
manifest unless ``--assume-noise`` is given (their count is then recorded).

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


def write_rank_config(workspace_dir: str, rank: int, server_args, gpu_id: int) -> None:
    """SAVE: what the manifest tool needs to know about this rank."""
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
        elif f == "memhist.pickle":
            continue
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


def classify(
    canonical: int,
    members: dict[int, dict],
    controls: dict[int, dict] | None,
) -> dict:
    """Classify the parameter differences of the ranks of one class.

    ``members``: rank -> {graph file: loaded graph} (the canonical rank included);
    ``controls``: rank -> graphs of a second SAVE of the same rank, or None.
    Returns {"rank_sites": [(graph, kernel, param, offset, width, add)], "noise": n,
    "unexplained": [(graph, node, kernel, param, offset, width, values)]}."""
    base = members[canonical]
    noise_keys = set()
    if controls:
        for r, graphs in controls.items():
            if r not in members:
                continue
            for gf, g in graphs.items():
                for key in param_diffs(members[r][gf], g):
                    noise_keys.add((gf,) + key)
    # Per (graph, site): the value on every rank.
    values: dict[tuple, dict[int, int]] = {}
    for r, graphs in members.items():
        if r == canonical:
            continue
        for gf, g in graphs.items():
            for key, (a, b) in param_diffs(base[gf], g).items():
                values.setdefault((gf,) + key, {canonical: a})[r] = b
    rank_sites: set = set()
    unexplained = []
    noise = 0
    for site, vals in values.items():
        gf, node, param, off, width = site
        adds = {v - r for r, v in vals.items()}
        if param >= 0 and len(vals) == len(members) and len(adds) == 1:
            # The rank id, or the rank id plus a constant (a per-rank counter).
            name = base[gf]["nodes"][node][1]
            rank_sites.add((gf, name, param, off, width, adds.pop()))
            continue
        if site in noise_keys:
            noise += 1
            continue
        unexplained.append((gf, node, base[gf]["nodes"][node][1][:100], param, off, width, vals))
    # A symbolic site is patched on EVERY node whose kernel name contains it:
    # each such node must hold the rank there on every rank of the class.
    checked = []
    for gfile, name, param, off, width, add in sorted(rank_sites):
        for gf, g in base.items():
            if gf != gfile:
                continue
            for i, node in enumerate(g["nodes"]):
                if node[0] != "kernel" or name not in node[1]:
                    continue
                for r in members:
                    p = members[r][gf]["nodes"][i][3]
                    ok = (
                        param < len(p)
                        and off + width <= len(p[param])
                        and int.from_bytes(p[param][off : off + width], "little") == r + add
                    )
                    if not ok:
                        unexplained.append(
                            (gf, i, node[1][:100], param, off, width, f"rank {r}: not rank + {add}")
                        )
        checked.append((gfile, name, param, off, width, add))
    return {"rank_sites": checked, "noise": noise, "unexplained": unexplained}


def make_manifest(
    workspace: str,
    control: str | None = None,
    assume_noise: bool = False,
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
    if control:
        controls = {}
        for r in ranks:
            cdir = os.path.join(control, f"rank_{r}")
            csig, cg = rank_signature(cdir)
            if csig != sigs[r]:
                out(
                    f"[Foundry share] control rank_{r} differs from rank_{r} beyond params: "
                    "not a control"
                )
                return 2
            controls[r] = cg
    rank_map, device_index, sites, report = {}, {}, set(), {}
    unexplained_all = []
    noise_total = 0
    for members in classes.values():
        c = members[0]
        res = classify(c, {r: graphs[r] for r in members}, controls)
        for r in members:
            rank_map[str(r)] = c
        device_index[str(c)] = configs[c]["device_index"]
        for r in members:
            if configs[r]["device_index"] != configs[c]["device_index"]:
                res["unexplained"].append(
                    ("-", -1, "device index", -1, 0, 0, {r: configs[r]["device_index"]})
                )
        sites.update(res["rank_sites"])
        unexplained_all += res["unexplained"]
        noise_total += res["noise"]
        report[str(c)] = {
            "members": members,
            "rank_sites": len(res["rank_sites"]),
            "noise_words": res["noise"],
            "unexplained_words": len(res["unexplained"]),
        }
    out(
        f"[Foundry share] {len(ranks)} ranks, {len(classes)} class(es): "
        + "; ".join(f"rank_{c} <- {v['members']}" for c, v in report.items())
    )
    out(f"[Foundry share] {len(sites)} '= rank (+ constant)' site(s):")
    for g, k, p, o, w, a in sorted(sites):
        out(f"    {g} {k[:70]} param {p} +{o} ({w} B) = rank + {a}")
    if unexplained_all:
        out(
            f"[Foundry share] {len(unexplained_all)} differing words are neither the rank id nor "
            f"control noise{'' if controls else ' (no control given)'}:"
        )
        for u in unexplained_all[:20]:
            out(f"    {u}")
        if controls or not assume_noise:
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
            {
                "graph": g,
                "kernel": k,
                "param": p,
                "offset": o,
                "width": w,
                "value": "rank",
                "add": a,
            }
            for g, k, p, o, w, a in sorted(sites)
        ],
        "verification": {
            "control": os.path.abspath(control) if control else None,
            "noise_words_ignored": noise_total,
            "assumed_noise_words": len(unexplained_all)
            if (unexplained_all and assume_noise)
            else 0,
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
    mk.add_argument("--control", help="a second SAVE workspace of the same configuration")
    mk.add_argument(
        "--assume-noise",
        action="store_true",
        help="without --control: accept differing non-rank words as noise (recorded)",
    )
    mk.add_argument("--prune", action="store_true", help="delete the non-canonical rank dirs")
    a = ap.parse_args(argv)
    return make_manifest(a.workspace, a.control, a.assume_noise, a.prune)


if __name__ == "__main__":
    sys.exit(main())
