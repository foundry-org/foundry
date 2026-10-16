# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The sglang integration's on-disk runtime state, on a tmp workspace (no sglang,
no GPU): warmup_state.json (per-rank file wins over the workspace-root copy,
which is written once) and the region layout's capture_loop_version check.

    pytest tests/unit/test_sglang_runtime_state.py -q
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from foundry.integration.sglang import runtime
from foundry.integration.sglang.config import CUDAGraphExtensionMode

SAVE, LOAD = CUDAGraphExtensionMode.SAVE, CUDAGraphExtensionMode.LOAD


@pytest.fixture
def as_rank(tmp_path, monkeypatch):
    """as_rank(rank, mode): make this process that rank of a workspace under tmp_path."""

    def use(rank, mode=SAVE):
        cfg = SimpleNamespace(
            mode=mode, workspace_root=str(tmp_path), workspace_dir=str(tmp_path / f"rank_{rank}")
        )
        monkeypatch.setattr(runtime, "get_config", lambda: cfg)
        monkeypatch.setattr(runtime, "_state", SimpleNamespace(rank=rank))
        return cfg

    return use


def _state(tag):
    return runtime.WarmupState(memory_pool_config={"tag": tag}, final_alloc_offset=tag)


def _read(path):
    return json.loads(path.read_text())["memory_pool_config"]["tag"]


def test_save_writes_the_rank_file_and_the_root_copy(as_rank, tmp_path):
    as_rank(0)
    runtime.save_warmup_state(_state(0))
    assert _read(tmp_path / "rank_0" / "warmup_state.json") == 0
    assert _read(tmp_path / "warmup_state.json") == 0


def test_root_copy_is_written_once(as_rank, tmp_path):
    as_rank(0)
    runtime.save_warmup_state(_state(0))
    as_rank(1)
    runtime.save_warmup_state(_state(1))
    assert _read(tmp_path / "rank_1" / "warmup_state.json") == 1
    assert _read(tmp_path / "warmup_state.json") == 0, "a non-zero rank overwrote the root copy"


def test_nonzero_rank_writes_a_missing_root_copy(as_rank, tmp_path):
    as_rank(1)
    runtime.save_warmup_state(_state(1))
    assert _read(tmp_path / "warmup_state.json") == 1


def test_rank_file_wins_over_the_root_copy(as_rank):
    as_rank(0)
    runtime.save_warmup_state(_state(0))
    as_rank(1)
    runtime.save_warmup_state(_state(1))
    state = runtime.load_warmup_state()
    assert state.memory_pool_config == {"tag": 1} and state.final_alloc_offset == 1
    as_rank(2)  # no rank file: the root copy
    assert runtime.load_warmup_state().memory_pool_config == {"tag": 0}


def test_unknown_keys_are_ignored(as_rank, tmp_path):
    as_rank(0)
    (tmp_path / "warmup_state.json").write_text(
        json.dumps({"memory_pool_config": {"tag": 7}, "field_of_a_newer_version": 1})
    )
    assert runtime.load_warmup_state().memory_pool_config == {"tag": 7}


def test_missing_state_raises(as_rank):
    as_rank(0, LOAD)
    with pytest.raises(RuntimeError, match="warmup state file not found"):
        runtime.load_warmup_state()


def test_without_a_config_save_is_a_noop_and_load_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "get_config", lambda: None)
    runtime.save_warmup_state(_state(0))
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="workspace_root is not initialized"):
        runtime.load_warmup_state()


def _layout(cfg, **fields):
    rank_dir = Path(cfg.workspace_dir)
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "region_layout.json").write_text(json.dumps({"live_ranges": [], **fields}))


def test_current_capture_loop_version_loads(as_rank):
    cfg = as_rank(0, LOAD)
    _layout(cfg, capture_loop_version=runtime.CAPTURE_LOOP_VERSION)
    runtime.check_capture_loop_version()


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"capture_loop_version": 1}, "capture_loop_version=1"),
        ({}, "capture_loop_version=None"),
        (None, "region layout not found"),
    ],
    ids=["old", "unmarked", "missing"],
)
def test_other_capture_loop_versions_are_refused(as_rank, fields, match):
    cfg = as_rank(0, LOAD)
    if fields is not None:
        _layout(cfg, **fields)
    with pytest.raises(RuntimeError, match=match):
        runtime.check_capture_loop_version()


def test_capture_loop_version_is_checked_on_load_only(as_rank):
    cfg = as_rank(0, SAVE)
    _layout(cfg, capture_loop_version=1)
    runtime.check_capture_loop_version()
