# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Workspace rank (archive directory rank_<N>) on both sglang DP layouts.

The DP controller passes ``dp_rank`` = replica without attention DP, and
``dp_rank`` = attention-DP group (inside one TP world) with it, before and
after sglang #41818 (``--attn-dp-size``). The archive numbering must not
depend on the layout. Plain namespaces, no sglang or GPU needed.

    pytest tests/integration/sglang/test_sglang_workspace_rank.py -v
"""

from types import SimpleNamespace

from foundry.integration.sglang.config import compute_workspace_rank, dp_replica


def _ranks(parallel, spawns):
    return [compute_workspace_rank(parallel, tp, pp, dp) for tp, pp, dp in spawns]


def test_tp_only():
    p = SimpleNamespace(tp_size=4, pp_size=1, dp_size=1, attn_dp_size=1, attn_dp_enabled=False)
    assert _ranks(p, [(t, 0, None) for t in range(4)]) == [0, 1, 2, 3]


def test_dp_replicas_each_own_tp_world():
    p = SimpleNamespace(tp_size=2, pp_size=1, dp_size=2, attn_dp_size=1, attn_dp_enabled=False)
    spawns = [(t, 0, d) for d in range(2) for t in range(2)]
    assert _ranks(p, spawns) == [0, 1, 2, 3]


def test_pp_and_tp():
    p = SimpleNamespace(tp_size=2, pp_size=2, dp_size=1, attn_dp_size=1, attn_dp_enabled=False)
    spawns = [(t, pp, None) for pp in range(2) for t in range(2)]
    assert _ranks(p, spawns) == [0, 1, 2, 3]


def test_ep4_dp_attention_after_41818():
    """--attn-dp-size 4 (or the deprecated --dp-size 4 --enable-dp-attention):
    dp_size 1, one TP world, dp_rank = attention-DP group = tp_rank."""
    p = SimpleNamespace(
        tp_size=4,
        pp_size=1,
        dp_size=1,
        attn_dp_size=4,
        attn_dp_enabled=True,
        enable_dp_attention=False,
    )
    spawns = [(t, 0, t) for t in range(4)]
    assert _ranks(p, spawns) == [0, 1, 2, 3]
    assert {dp_replica(p, t) for t in range(4)} == {0}


def test_ep4_dp_attention_before_41818():
    """get_parallel() before #41818: enable_dp_attention, dp_size counts the
    groups, attn_dp_size is derived; no attn_dp_enabled."""
    p = SimpleNamespace(tp_size=4, pp_size=1, dp_size=4, attn_dp_size=4, enable_dp_attention=True)
    assert _ranks(p, [(t, 0, t) for t in range(4)]) == [0, 1, 2, 3]


def test_ep4_dp_attention_fork_server_args():
    """Fork base: the record is a ServerArgs without attn_dp_size."""
    p = SimpleNamespace(tp_size=4, pp_size=1, dp_size=4, enable_dp_attention=True)
    assert _ranks(p, [(t, 0, t) for t in range(4)]) == [0, 1, 2, 3]


def test_replica_major_attention_dp_ranks_stay_unique():
    """Should replicas ever combine with attention DP (sglang rejects it
    today), replica-major dp_rank still yields distinct ranks."""
    p = SimpleNamespace(tp_size=2, pp_size=1, dp_size=2, attn_dp_size=2, attn_dp_enabled=True)
    spawns = [(t, 0, r * 2 + t) for r in range(2) for t in range(2)]
    assert _ranks(p, spawns) == [0, 1, 2, 3]
