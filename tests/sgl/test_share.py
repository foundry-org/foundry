# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Archive share (one archive for several ranks of one world size): the LOAD
manifest check and the post-SAVE classification of parameter differences, on
synthetic inputs (no GPU, no archive).

    pytest tests/sgl/test_share.py -v
"""

from types import SimpleNamespace

import pytest
from foundry.integration.sglang import share
from foundry.integration.sglang.api import Unsupported


def _args(**over):
    fields = {f: None for f in share.CONFIG_FIELDS}
    fields.update(tp_size=2, ep_size=2, attn_dp_size=2, enable_dp_lm_head=True)
    fields.update(over)
    return SimpleNamespace(**fields)


def _manifest(**over):
    m = {
        "format_version": share.FORMAT_VERSION,
        "world_size": 2,
        "fields": share.config_fields(_args()),
        "rank_map": {"0": 0, "1": 0},
        "device_index": {"0": 0},
        "patch_sites": [
            {
                "kernel": "internode_ll8dispatch",
                "param": 22,
                "offset": 0,
                "width": 4,
                "value": "rank",
            }
        ],
    }
    m.update(over)
    return m


def test_check_load_returns_canonical_and_rank_patches():
    canonical, patches = share.check_load(_manifest(), 1, _args(), {0: 0, 1: 0})
    assert canonical == 0
    assert patches == [("", "internode_ll8dispatch", 22, 0, 4, 1)]


def test_config_mismatch_is_unsupported_with_field_and_values():
    with pytest.raises(Unsupported) as e:
        share.check_load(_manifest(), 1, _args(enable_dp_lm_head=False), {0: 0, 1: 0})
    msg = str(e.value)
    assert msg.startswith("unsupported at archive share: enable_dp_lm_head differs")
    assert "(archive True, this rank False)" in msg
    assert e.value.step == "archive share"


def test_device_index_mismatch_is_unsupported_on_every_rank():
    # rank 1 on cuda:1 while the canonical archive was captured on cuda:0: rank 0
    # takes the same decision because it checks every rank's device.
    for rank in (0, 1):
        with pytest.raises(Unsupported) as e:
            share.check_load(_manifest(), rank, _args(), {0: 0, 1: 1})
        assert "unsupported at archive share: device index differs" in str(e.value)


def test_world_size_and_unknown_rank():
    with pytest.raises(Unsupported, match="world_size differs"):
        share.check_load(_manifest(), 0, _args(), {0: 0, 1: 0, 2: 0})
    with pytest.raises(Unsupported, match="rank 5 is not in the manifest"):
        share.check_load(_manifest(world_size=1, rank_map={"0": 0}), 5, _args(), {0: 0})


def _graph(nodes):
    """nodes: list of (name, [param bytes])."""
    return {
        "nodes": [("kernel", n, (), list(ps), b"") for n, ps in nodes],
        "deps": b"",
        "other": {},
    }


def _i32(v):
    return int(v).to_bytes(4, "little")


DISPATCH = "_ZN7deep_ep6legacy12internode_ll8dispatchILb1E"
COMBINE = "_ZN7deep_ep6legacy12internode_ll7combineILb0E"


def _dispatch_params(r):
    return [b"\0" * 8] * 22 + [_i32(r), _i32(2)]


def _rank_graph(r, junk=b"\0\0\0\0"):
    combine = [b"\0" * 8] * 19 + [_i32(r)]
    return {
        "g.cugraph": _graph(
            [
                (DISPATCH, _dispatch_params(r)),
                ("nvjet_sm90_tst", [b"\x01\x00\x00\x00" + junk]),
                (COMBINE, combine),
            ]
        )
    }


def test_classify_patches_only_declared_rank_args():
    members = {0: _rank_graph(0, junk=_i32(872)), 1: _rank_graph(1, junk=_i32(873))}
    res = share.classify(0, members, None)
    assert res["rank_sites"] == [("", COMBINE, 19, 0, 4), ("", DISPATCH, 22, 0, 4)]
    # The nvjet word (872 + rank) is reused as captured, not patched.
    assert res["reused"] == {"cuBLAS nvjet": 1}
    assert res["unexplained"] == []


def test_classify_control_only_reports_noise():
    members = {0: _rank_graph(0, junk=b"AAAA"), 1: _rank_graph(1, junk=b"BBBB")}
    controls = {0: _rank_graph(0, junk=b"CCCC"), 1: _rank_graph(1, junk=b"DDDD")}
    res = share.classify(0, members, controls)
    assert res["reused"] == {"cuBLAS nvjet": 1} and res["control_noise"] == 1
    assert res["unexplained"] == []


def test_classify_refuses_declared_arg_without_rank():
    def graph(r):
        return {"g.cugraph": _graph([(DISPATCH, _dispatch_params(0))])}

    res = share.classify(0, {0: graph(0), 1: graph(1)}, None)
    assert any("not the rank id" in str(u[-1]) for u in res["unexplained"])
