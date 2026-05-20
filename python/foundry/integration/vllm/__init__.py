# SPDX-License-Identifier: Apache-2.0
"""vLLM integration for foundry.

Public API consumed by ``vllm/compilation/foundry_shim.py``:

- :class:`CUDAGraphExtensionMode` — mode enum (``NONE``/``SAVE``/``LOAD``).
- :func:`get_graph_extension_mode` — current mode (``NONE`` if not configured).
- :func:`install_hooks` — entry point invoked from ``CompilationConfig.__post_init__``
  when ``graph_extension_config_path`` is set. Loads the TOML and installs
  every runtime monkey-patch listed in ``claude-doc/03-vllm-hook-surface.md``.

See the ``claude-doc/`` design docs for the full architecture.
"""

from foundry.integration.vllm.config import (
    CUDAGraphExtensionMode,
    get_graph_extension_mode,
)
from foundry.integration.vllm.hooks import install_hooks

__all__ = [
    "CUDAGraphExtensionMode",
    "get_graph_extension_mode",
    "install_hooks",
]
