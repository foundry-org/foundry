# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Fixtures of the SGLang tests. Helpers the test modules import are in
_sglang_helpers.py (pytest puts this directory on sys.path when it loads this
conftest, also for tests/sglang/e2e)."""

import json
import os

import pytest
from _sglang_helpers import MINI_CONFIG


@pytest.fixture
def model_dir(tmp_path):
    """A model directory with only the mini config.json (CPU-only resolution)."""
    (tmp_path / "config.json").write_text(json.dumps(MINI_CONFIG))
    return str(tmp_path)


@pytest.fixture
def restore_environ():
    # Resolution writes environment variables that outlive the record.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def clean_pin_env(restore_environ):
    """No pinned variable inherited from the calling shell."""
    from foundry.integration.sglang import plugin

    names = [pin.name for pin in plugin.ENV_PINS]
    for name in names + [plugin.NCCL_CUMEM_PIN.name, plugin._ENV_PINNED_MARK]:
        os.environ.pop(name, None)
    yield
