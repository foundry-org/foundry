# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Breakable (BCG) prefill graphs: the break encoding (breakable_ops) and its
contract with SGLang's eager_on_graph.

CPU only. GPU parity is covered by the e2e case.

    pytest tests/sgl/test_breakable_ops.py -v
"""

import enum
import functools

import pytest

pytest.importorskip("sglang")
import torch  # noqa: E402
from foundry.integration.sglang import breakable_ops  # noqa: E402


class _Color(enum.Enum):
    RED = 1


class _Layer(torch.nn.Module):
    pass


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


def _break_body(layer, x, out, *, flag=True, mode=_Color.RED):
    return None


def _as_eager_on_graph_builds_it(inner, *args, **kwargs):
    """The closure eager_on_graph records without output_factory."""

    def run(*a, **k) -> None:
        out = inner(*a, **k)
        assert out is None

    return functools.partial(run, *args, **kwargs)


def test_break_parts_reads_the_eager_on_graph_partial():
    fn = _as_eager_on_graph_builds_it(_break_body, 1, 2, flag=False)
    inner, args, kwargs = breakable_ops._break_parts(fn)
    assert inner is _break_body
    assert args == (1, 2)
    assert kwargs == {"flag": False}


def test_break_without_partial_is_refused():
    def replay_fn():
        pass

    with pytest.raises(breakable_ops.UnsupportedBreak, match="not a functools.partial"):
        breakable_ops._break_parts(replay_fn)


def test_encode_decode_round_trip_keeps_modules_scalars_enums():
    model = _Model()
    # A meta tensor stands in for a restored device tensor (no GPU here).
    t = torch.empty(4, device="meta")
    spec = {
        "layer": {"module": "layers.1"},
        "x": {"t": 0},
        "flag": {"v": False},
        "mode": {
            "enum": {"module": __name__, "qualname": "_Color", "name": "RED"},
        },
        "pair": {"tuple": [{"v": 3}, {"v": None}]},
    }
    decoded = breakable_ops._decode(
        {"dict": spec}, [t], dict(model.named_modules(remove_duplicate=False)), "cpu"
    )
    assert decoded["layer"] is model.layers[1]
    assert decoded["x"] is t
    assert decoded["flag"] is False
    assert decoded["mode"] is _Color.RED
    assert decoded["pair"] == (3, None)


def test_encoder_refuses_unknown_objects_and_foreign_modules():
    model = _Model()
    names = {id(m): n for n, m in model.named_modules(remove_duplicate=False)}
    enc = breakable_ops._Encoder(names, [])
    assert enc((1, "a", None, _Color.RED, model.layers[0]), "args") == {
        "tuple": [
            {"v": 1},
            {"v": "a"},
            {"v": None},
            {"enum": {"module": __name__, "qualname": "_Color", "name": "RED"}},
            {"module": "layers.0"},
        ]
    }
    with pytest.raises(breakable_ops.UnsupportedBreak, match="not a submodule"):
        enc(_Layer(), "args[0]")
    with pytest.raises(breakable_ops.UnsupportedBreak, match="archives tensors"):
        enc(object(), "args[0]")
    with pytest.raises(breakable_ops.UnsupportedBreak, match="cpu tensor"):
        enc(torch.zeros(2), "args[0]")


def test_empty_tensor_is_rebuilt_from_its_layout():
    enc = breakable_ops._Encoder({}, [])
    t = torch.empty_strided((0, 4), (4, 1), dtype=torch.float16, device="meta")
    with pytest.raises(breakable_ops.UnsupportedBreak):
        enc(t, "args[0]")  # meta is not cuda; the layout path is checked below
    spec = {"empty": {"shape": [0, 4], "stride": [4, 1], "dtype": "float16"}}
    out = breakable_ops._decode(spec, [], {}, "cpu")
    assert out.shape == (0, 4) and out.stride() == (4, 1) and out.dtype == torch.float16


@pytest.mark.parametrize(
    "module, qualname",
    [
        ("sglang.srt.layers.radix_attention", "RadixAttention._eager_attention"),
        (
            "sglang.srt.layers.radix_linear_attention",
            "RadixLinearAttention._eager_linear_attention",
        ),
    ],
)
def test_sglang_attention_breaks_resolve_by_name(module, qualname):
    """The contract with eager_on_graph: the decorated method resolves by
    module + qualname to the very function the capture's partial closes over
    (the decorator's eager entry)."""
    obj = breakable_ops._resolve_inner(module, qualname)
    assert callable(obj)
    assert breakable_ops._inner_ref(obj) == {"module": module, "qualname": qualname}


def test_bcg_segment_files_have_their_own_manifest_partition():
    from foundry.integration.sglang.graph_ops import graph_partition

    assert graph_partition("graph_7_BCG0_t512_s3_UX_pc0.json") == "breakable:0"
    assert graph_partition("graph_9_BCG1_t512_s3_UX_pc0.json") == "breakable:1"
    assert graph_partition("graph_7_PREFILL_t512_r4_UX_pc0.json") == "prefill"
    assert graph_partition("graph_7_FULL_t8_r8_UX_pcN.json") == "decode"
