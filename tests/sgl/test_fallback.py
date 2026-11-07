# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Fallback contract (API 2.4): outside the supported configurations a
detection point raises ``api.Unsupported``, SGLang's adapter calls
``api.deactivate`` and the engine continues natively; a SAVE that falls back
marks its archive invalid and a LOAD of it falls back too.

The scheduler process is simulated on the CPU: the config, the per-process
state and the workspace live in ``tmp_path``; the region / hook side effects
of the deactivation and the graph capture itself are recorded instead of run.

    pytest tests/sgl/test_fallback.py -v
"""

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")
adapter_module = pytest.importorskip("sglang.srt.utils.foundry_adapter")

import torch  # noqa: E402
from foundry.integration.sglang import api, config, graph_ops, hooks  # noqa: E402
from foundry.integration.sglang import runtime as rt  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher  # noqa: E402


class _Dispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        raise NotImplementedError

    def combine(self, combine_input):
        raise NotImplementedError


class _NoPrebuildDispatcher(_Dispatcher):
    """A new backend that builds its buffer at the first dispatch."""


class _PendingDispatcher(_Dispatcher):
    @property
    def prebuild_pending(self):
        return True

    def prebuild(self):
        pass


class _PrebuiltDispatcher(_Dispatcher):
    def __init__(self):
        super().__init__()
        self.cache = None

    def prebuild(self):
        pass


def _model(dispatcher):
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.dispatcher = dispatcher
    return model


class _Graph:
    resets = 0

    def reset(self):
        self.resets += 1


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """``scheduler(mode, past_setup=True)``: this process as a Foundry
    scheduler; returns (adapter, config, calls of leave_graph_extension)."""
    monkeypatch.delenv(api.UNSUPPORTED_ENV, raising=False)
    monkeypatch.setattr(api, "_unsupported", None)
    monkeypatch.setattr(hooks, "_scope", None)
    left = []
    monkeypatch.setattr(rt, "leave_graph_extension", lambda: left.append(True))

    def make(mode, past_setup=True):
        cfg = config.CUDAGraphExtensionConfig(
            mode=config.CUDAGraphExtensionMode(mode), workspace_root=str(tmp_path / "archive")
        )
        cfg.workspace_dir = str(tmp_path / "archive" / "rank_0")
        os.makedirs(cfg.workspace_dir, exist_ok=True)
        monkeypatch.setattr(config, "_config", cfg)
        state = rt.CUDAGraphExtensionState(layout_started=True) if past_setup else None
        monkeypatch.setattr(rt, "_state", state)
        monkeypatch.setattr(api, "_active", (mode, None))
        return adapter_module._FoundryAdapterReal(api, mode, None), cfg, left

    return make


def _marker(cfg):
    path = os.path.join(cfg.workspace_root, rt._INVALID_FILE)
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(cfg.workspace_dir, rt._INVALID_FILE))
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# First capture scope: the prebuild protocol
# ---------------------------------------------------------------------------


def test_dispatcher_without_prebuild_falls_back_at_the_first_scope(scheduler, monkeypatch):
    """No prebuild: Unsupported before any bootstrap, the capture loop runs
    natively once and the SAVE archive is marked invalid."""
    adapter, cfg, left = scheduler("save")
    rt.get_state().layout_started = False

    def bootstrap():
        raise AssertionError("a bootstrap ran after an unsupported capture state")

    monkeypatch.setattr(graph_ops, "bootstrap_flashinfer_tactics", lambda path: bootstrap())
    model = _model(_NoPrebuildDispatcher())
    loops = []
    with adapter.capture_scope(phase="decode", model=model):
        loops.append(adapter.capture_one(shape_key=None, forward_fn=None, pool=None, stream=None))
    assert loops == [None]
    assert not adapter.enabled
    marker = _marker(cfg)
    assert (marker["step"], marker["reason"]) == (
        "capture state",
        "_NoPrebuildDispatcher has no prebuild",
    )
    assert left == [True]
    assert config.get_graph_extension_mode() == config.CUDAGraphExtensionMode.NONE


def test_capture_state_reasons():
    assert graph_ops.capture_state_unsupported(_model(_PrebuiltDispatcher())) is None
    assert (
        graph_ops.capture_state_unsupported(_model(_PendingDispatcher()))
        == "_PendingDispatcher prebuild left state pending"
    )
    wrapper = _PrebuiltDispatcher()
    wrapper.inners = [_NoPrebuildDispatcher()]
    assert (
        graph_ops.capture_state_unsupported(_model(wrapper))
        == "_NoPrebuildDispatcher has no prebuild"
    )


# ---------------------------------------------------------------------------
# SAVE captures: attributes created inside the capture
# ---------------------------------------------------------------------------


def test_attribute_created_during_capture_discards_the_graph(scheduler, monkeypatch):
    """The capture that creates dispatcher state falls back: its graph is kept
    aside and not saved, the attributes are back to absent / None (the native
    warmup rebuilds them), and the archive is marked invalid."""
    adapter, cfg, _ = scheduler("save")
    graph = _Graph()
    monkeypatch.setattr(graph_ops, "capture_graph", lambda pool, stream, fn: (graph, fn()))

    def not_saved(*args, **kwargs):
        raise AssertionError("a graph that created state was saved")

    monkeypatch.setattr(graph_ops, "save_graph", not_saved)
    dispatcher = _PrebuiltDispatcher()

    def forward():
        dispatcher.cache = torch.zeros(1)
        dispatcher.lazy_buffer = torch.zeros(1)
        return torch.zeros(1)

    with adapter.capture_scope(phase="decode", model=_model(dispatcher)):
        captured = adapter.capture_one(
            shape_key=SimpleNamespace(attention_variant=None),
            forward_fn=forward,
            pool=None,
            stream=None,
        )
    assert captured is None
    # Kept alive, not reset: the pool's only graph would drop its use count to 0.
    assert graph.resets == 0 and graph in hooks._discarded_graphs
    assert dispatcher.cache is None and "lazy_buffer" not in vars(dispatcher)
    marker = _marker(cfg)
    assert marker["step"] == "capture"
    assert "_PrebuiltDispatcher.cache" in marker["reason"]
    assert "_PrebuiltDispatcher.lazy_buffer" in marker["reason"]


def test_capture_that_creates_nothing_is_saved(scheduler, monkeypatch):
    adapter, cfg, _ = scheduler("save")
    monkeypatch.setattr(graph_ops, "capture_graph", lambda pool, stream, fn: (_Graph(), fn()))
    saved = []
    monkeypatch.setattr(graph_ops, "save_graph", lambda *args, **kwargs: saved.append(args))
    monkeypatch.setattr(graph_ops, "save_decode_graph_state", lambda kind: None)
    monkeypatch.setattr(graph_ops, "save_graph_manifest", lambda: None)
    monkeypatch.setattr(graph_ops, "pack_fatbins", lambda: None)
    monkeypatch.setattr(rt, "record_region_layout", lambda: 0)
    monkeypatch.setattr(rt, "log_alloc_offset", lambda label: None)
    with adapter.capture_scope(phase="decode", model=_model(_PrebuiltDispatcher())):
        captured = adapter.capture_one(
            shape_key=SimpleNamespace(attention_variant=None),
            forward_fn=lambda: torch.zeros(1),
            pool=None,
            stream=None,
        )
    assert captured is not None and len(saved) == 1
    assert adapter.enabled
    assert not os.path.exists(os.path.join(cfg.workspace_root, rt._INVALID_FILE))


# ---------------------------------------------------------------------------
# LOAD of an invalid archive
# ---------------------------------------------------------------------------


def test_load_of_an_invalid_archive_falls_back_before_setup(scheduler, monkeypatch):
    adapter, cfg, _ = scheduler("save")
    rt.mark_archive_invalid("capture state", "_NoPrebuildDispatcher has no prebuild")
    adapter, cfg, left = scheduler("load", past_setup=False)
    from sglang.srt import runtime_context

    parallel = SimpleNamespace(
        tp_rank=0, pp_rank=0, dp_rank=None, tp_size=1, pp_size=1, attn_dp_enabled=False
    )
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: parallel)

    def setup(*args, **kwargs):
        raise AssertionError("LOAD set up an invalid archive")

    monkeypatch.setattr(hooks, "_disable_dynamo", setup)
    monkeypatch.setattr(hooks, "_before_distributed_init", setup)
    adapter.before_parallel_init("cuda")
    assert not adapter.enabled
    assert adapter.replay_saved_memory_pool_config() is None
    assert left == []  # nothing was set up


# ---------------------------------------------------------------------------
# After the fallback
# ---------------------------------------------------------------------------


def test_every_call_is_a_no_op_after_deactivate(scheduler, monkeypatch, tmp_path):
    _, cfg, _ = scheduler("save")
    api.deactivate(step="capture state", reason="test")
    assert api.replay_saved_memory_pool_config() is None
    api.record_memory_pool_config(SimpleNamespace())
    assert not os.path.exists(os.path.join(cfg.workspace_dir, "warmup_state.json"))
    loops = []
    with api.capture_scope("decode", model=_model(_NoPrebuildDispatcher())):
        loops.append(1)
    assert loops == [1]
    preload = os.environ.get("LD_PRELOAD")
    with api.configure_subprocess(None):
        assert json.loads(os.environ[api.UNSUPPORTED_ENV]) == ["capture state", "test", False]
        assert os.environ.get("LD_PRELOAD") == preload
    assert api.UNSUPPORTED_ENV not in os.environ


def test_a_child_of_a_fallen_back_launcher_falls_back_at_activation(monkeypatch):
    monkeypatch.setenv(api.UNSUPPORTED_ENV, json.dumps(["server args", "LoRA", False]))
    with pytest.raises(api.Unsupported) as info:
        api.activate(mode="save")
    assert (info.value.step, info.value.reason, info.value.inherited) == (
        "server args",
        "LoRA",
        True,
    )


# ---------------------------------------------------------------------------
# Rank agreement
# ---------------------------------------------------------------------------


def test_every_rank_falls_back_when_one_is_unsupported(monkeypatch):
    """A rank without a reason of its own still raises, naming the other
    rank's reason; with no reason anywhere nothing is gathered."""
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state as ps

    gathered = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        ps, "get_world_group", lambda: SimpleNamespace(cpu_group=None, world_size=2)
    )

    def all_gather_object(out, obj, group=None):
        gathered.append(obj)
        out[:] = [None, "_NoPrebuildDispatcher has no prebuild"]

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: t.fill_(1))
    with pytest.raises(api.Unsupported, match=r"has no prebuild \(found on rank 1\)"):
        graph_ops.agree_unsupported("capture state", None)
    assert gathered == [None]

    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: t.fill_(0))
    gathered.clear()
    graph_ops.agree_unsupported("capture state", None)
    assert gathered == []
