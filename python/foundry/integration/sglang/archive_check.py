# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Post-SAVE consistency check of a rank's graph archive.

Every device pointer a restored graph dereferences must be memory LOAD will
have: an allocation replayed from the graph's own allocator events, or a range
that was still live at the end of SAVE and is therefore backed by LOAD's
preallocation. A pointer into the region that lies in neither is state the
model built outside the recorded layout (typically communication buffers
created lazily on the first eager forward, e.g. torch symmetric memory), and
LOAD will fault on it. This module scans the kernel parameters of every graph
in ``rank_<N>/`` for such pointers and reports them per kernel name.

The scan reads every ``.cugraph`` file, so it is opt-in:
``FOUNDRY_SGLANG_CHECK_ARCHIVE=1`` runs it at the end of SAVE, or run
``python -m foundry.integration.sglang.archive_check <rank_dir> [region_base]``.
Kernel parameters are scanned as 8-byte words; a word is treated as a pointer
when it falls inside the allocation region, which keeps false positives rare
(integers never look like region addresses).
"""

from __future__ import annotations

import bisect
import collections
import glob
import json
import logging
import os
import struct
import sys

logger = logging.getLogger(__name__)

# BinaryGraphFormat.h
_SECTION_STRING_TABLE = 0
_SECTION_NODE_TABLE = 1
_SECTION_KERNEL_PARAM_INDEX = 3
_SECTION_KERNEL_PARAM_DATA = 4
_SECTION_ALLOCATOR_EVENTS = 8
_NODE_ENTRY_SIZE = 168
_NODE_KERNEL = 0
_NODE_MEMSET = 1
_NODE_MEMCPY = 2


def _load(path: str):
    with open(path, "rb") as f:
        data = f.read()
    magic, _version, _flags, n_nodes, _n_deps, _n_gen, n_sections, _n_strings = struct.unpack_from(
        "<8sIIIIIII", data, 0
    )
    if magic != b"CUGRAPH\0":
        raise ValueError(f"{path}: not a .cugraph file")
    sections = {}
    for i in range(n_sections):
        stype, _pad, off, size = struct.unpack_from("<IIQQ", data, 64 + 24 * i)
        sections[stype] = data[off : off + size]
    return n_nodes, sections


def _iter_nodes(n_nodes: int, sections: dict):
    """Yield (kind, name, [param bytes...]) for kernel nodes and (kind, "", [dst, src]) for
    memset/memcpy nodes."""
    nodes = sections.get(_SECTION_NODE_TABLE, b"")
    strings = sections.get(_SECTION_STRING_TABLE, b"")
    pidx = sections.get(_SECTION_KERNEL_PARAM_INDEX, b"")
    pdata = sections.get(_SECTION_KERNEL_PARAM_DATA, b"")
    for i in range(n_nodes):
        entry = nodes[_NODE_ENTRY_SIZE * i : _NODE_ENTRY_SIZE * (i + 1)]
        _node_id, ntype = struct.unpack_from("<IB", entry, 0)
        body = entry[8:]
        if ntype == _NODE_KERNEL:
            fn_off, fn_len = struct.unpack_from("<II", body, 36)
            pio, npar = struct.unpack_from("<II", body, 120)
            name = strings[fn_off : fn_off + fn_len].decode(errors="replace")
            params = []
            for j in range(npar):
                doff, size = struct.unpack_from("<II", pidx, pio + 8 * j)
                params.append(pdata[doff : doff + size])
            yield "kernel", name, params
        elif ntype == _NODE_MEMSET:
            (dst,) = struct.unpack_from("<Q", body, 0)
            yield "memset", "<memset>", [struct.pack("<Q", dst)]
        elif ntype == _NODE_MEMCPY:
            dst = struct.unpack_from("<Q", body, 24)[0]
            src = struct.unpack_from("<Q", body, 88)[0]
            yield "memcpy", "<memcpy>", [struct.pack("<QQ", dst, src)]


def _allocator_ranges(sections: dict) -> list[tuple[int, int]]:
    raw = sections.get(_SECTION_ALLOCATOR_EVENTS, b"")
    if not raw:
        return []
    try:
        doc = json.loads(raw.decode())
    except Exception:
        return []
    out = []

    def walk(x):
        if isinstance(x, dict):
            if "ptr" in x and "size" in x:
                out.append((int(x["ptr"]), int(x["size"])))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(doc)
    return out


class _Ranges:
    def __init__(self, ranges):
        rs = sorted((int(a), int(a) + int(n)) for a, n in ranges if n > 0)
        merged = []
        for a, b in rs:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.starts = [a for a, _ in merged]
        self.ends = [b for _, b in merged]

    def contains(self, addr: int) -> bool:
        i = bisect.bisect_right(self.starts, addr) - 1
        return i >= 0 and addr < self.ends[i]


def check_rank_archive(rank_dir: str, region_base: int, region_size: int) -> dict:
    """Return {kernel name: {"nodes": n, "targets": sorted offsets}} for pointers into the
    region that neither the graph's allocator events nor the recorded live ranges cover."""
    layout_path = os.path.join(rank_dir, "region_layout.json")
    live = []
    if os.path.exists(layout_path):
        with open(layout_path) as f:
            live = [(region_base + int(o), int(n)) for o, n in json.load(f).get("live_ranges", [])]
    live_ranges = _Ranges(live)
    region_end = region_base + region_size
    findings: dict = collections.defaultdict(lambda: {"nodes": 0, "targets": set()})
    for path in sorted(glob.glob(os.path.join(rank_dir, "graph_*.cugraph"))):
        n_nodes, sections = _load(path)
        own = _Ranges(_allocator_ranges(sections))
        for _kind, name, params in _iter_nodes(n_nodes, sections):
            hit = False
            for p in params:
                for (w,) in struct.iter_unpack("<Q", p[: len(p) // 8 * 8]):
                    if (
                        region_base <= w < region_end
                        and not own.contains(w)
                        and not live_ranges.contains(w)
                    ):
                        findings[name]["targets"].add(w - region_base)
                        hit = True
            if hit:
                findings[name]["nodes"] += 1
    return {k: {"nodes": v["nodes"], "targets": sorted(v["targets"])} for k, v in findings.items()}


def report(findings: dict, log=logger) -> None:
    if not findings:
        log.info("[Foundry] archive check: every region pointer is inside a recorded range")
        return
    for name, info in sorted(findings.items(), key=lambda kv: -kv[1]["nodes"]):
        targets = ", ".join(f"0x{t:x}" for t in info["targets"][:4])
        log.warning(
            "[Foundry] archive check: %d node(s) of %s point into the region outside every "
            "recorded range (offsets %s%s): state built outside the recorded layout, LOAD will "
            "not have it",
            info["nodes"],
            name[:80],
            targets,
            ", ..." if len(info["targets"]) > 4 else "",
        )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    rank_dir = argv[0]
    base = int(argv[1], 0) if len(argv) > 1 else 0x600000000000
    size = int(argv[2], 0) if len(argv) > 2 else 256 << 30
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    findings = check_rank_archive(rank_dir, base, size)
    report(findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
