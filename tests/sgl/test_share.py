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


def _rank_graph(r, junk=b"\0\0\0\0", ptr=b"\x00\x10\x00\x00"):
    return {
        "g.cugraph": _graph(
            [
                ("deep_ep::dispatch", [ptr, _i32(7), _i32(r)]),
                ("fa3", [b"\x01\x00\x00\x00" + junk]),
                ("deep_ep::dispatch", [ptr, _i32(7), _i32(r)]),
            ]
        )
    }


def test_classify_rank_sites_and_control_noise():
    members = {0: _rank_graph(0, junk=b"AAAA"), 1: _rank_graph(1, junk=b"BBBB")}
    controls = {0: _rank_graph(0, junk=b"CCCC"), 1: _rank_graph(1, junk=b"DDDD")}
    res = share.classify(0, members, controls)
    assert res["rank_sites"] == [("g.cugraph", "deep_ep::dispatch", 2, 0, 4, 0)]
    assert res["noise"] == 1
    assert res["unexplained"] == []


def test_classify_refuses_unexplained_words():
    # A pointer that differs between ranks but not between two SAVEs of a rank.
    members = {0: _rank_graph(0), 1: _rank_graph(1, ptr=b"\x00\x12\x00\x00")}
    controls = {0: _rank_graph(0), 1: _rank_graph(1, ptr=b"\x00\x12\x00\x00")}
    res = share.classify(0, members, controls)
    assert res["unexplained"] and res["unexplained"][0][3] == 0


def test_classify_symbolic_site_must_hold_on_every_matching_node():
    # The second dispatch node holds a constant where the first holds the rank:
    # patching by kernel name would corrupt it, so it is reported.
    def graph(r):
        return {
            "g.cugraph": _graph(
                [("deep_ep::dispatch", [_i32(r)]), ("deep_ep::dispatch", [_i32(0)])]
            )
        }

    res = share.classify(0, {0: graph(0), 1: graph(1)}, None)
    assert res["rank_sites"] == [("g.cugraph", "deep_ep::dispatch", 0, 0, 4, 0)]
    assert any("not rank + 0" in str(u[-1]) for u in res["unexplained"])


def test_classify_rank_plus_constant_and_load_value():
    def graph(r):
        return {"g.cugraph": _graph([("nvjet", [_i32(872 + r)]), ("nvjet", [_i32(872 + r)])])}

    res = share.classify(0, {0: graph(0), 1: graph(1)}, None)
    assert res["rank_sites"] == [("g.cugraph", "nvjet", 0, 0, 4, 872)] and res["unexplained"] == []
    m = _manifest(
        patch_sites=[
            {"kernel": "nvjet", "param": 0, "offset": 0, "width": 4, "value": "rank", "add": 872}
        ]
    )
    assert share.check_load(m, 1, _args(), {0: 0, 1: 0})[1] == [("", "nvjet", 0, 0, 4, 873)]
