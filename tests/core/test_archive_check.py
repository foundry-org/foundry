# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""archive_check on a real SAVE (1 GPU): a graph reading a tensor allocated
before the capture is clean while the live ranges cover that tensor, and is
reported (kernel name, offset) once they do not.

    pytest tests/core/test_archive_check.py -q -s
"""

import json
import os
import sys

import torch
from _helpers import prefix_region, run_phase
from foundry.integration.sglang import archive_check

BASE_ADDR = 0x5A0000000000
REGION_SIZE_STR = "1GB"
REGION_SIZE = 1 << 30


def _run_save(archive):
    """A rank archive as the sglang SAVE leaves it: graph_*.cugraph plus
    region_layout.json with the live ranges at the end of SAVE."""
    import foundry as fdry

    g, _ = prefix_region(BASE_ADDR, REGION_SIZE_STR)
    x = torch.randn(256, 256, generator=g).cuda()  # outside the graph's own allocations
    graph = fdry.CUDAGraph()
    with fdry.graph(graph):
        y = x * 2.0 + 1.0
    graph.replay()
    torch.cuda.synchronize()
    graph.save(os.path.join(archive, "graph_0.json"), output_tensors=y)
    layout = {"live_ranges": [list(r) for r in fdry.ops.get_live_region_ranges()]}
    with open(os.path.join(archive, "region_layout.json"), "w") as f:
        json.dump(layout, f)
    with open(os.path.join(archive, "input.json"), "w") as f:
        json.dump({"offset": x.data_ptr() - BASE_ADDR}, f)
    fdry.stop_allocation_region()


def test_main_without_arguments_prints_usage():
    assert archive_check.main([]) == 2


def test_uncovered_input_is_reported(workdir):
    run_phase(__file__, "save", workdir)
    archive = workdir / "archive"
    assert list(archive.glob("graph_*.cugraph")), "SAVE wrote no .cugraph"
    assert archive_check.check_rank_archive(str(archive), BASE_ADDR, REGION_SIZE) == {}

    x_off = json.loads((archive / "input.json").read_text())["offset"]
    layout_path = archive / "region_layout.json"
    live = json.loads(layout_path.read_text())["live_ranges"]
    dropped = [r for r in live if r[0] <= x_off < r[0] + r[1]]
    assert dropped, f"input at offset {x_off:#x} not in the live ranges {live}"
    layout_path.write_text(json.dumps({"live_ranges": [r for r in live if r not in dropped]}))

    findings = archive_check.check_rank_archive(str(archive), BASE_ADDR, REGION_SIZE)
    assert findings, "a graph pointer outside every recorded range was not reported"
    targets = {t for f in findings.values() for t in f["targets"]}
    assert x_off in targets, (hex(x_off), findings)
    assert all(any(o <= t < o + n for o, n in dropped) for t in targets), findings
    assert all(f["nodes"] >= 1 for f in findings.values())
    assert archive_check.main([str(archive), hex(BASE_ADDR), hex(REGION_SIZE)]) == 1


if __name__ == "__main__":
    {"--save": _run_save}[sys.argv[1]](sys.argv[2])
