# SPDX-License-Identifier: Apache-2.0
"""Foundry SGLang integration configuration."""

from __future__ import annotations

import enum
import importlib.util
import tomllib
from dataclasses import dataclass
from pathlib import Path


class CUDAGraphExtensionMode(str, enum.Enum):
    NONE = "none"
    SAVE = "save"
    LOAD = "load"


@dataclass
class CUDAGraphExtensionConfig:
    mode: CUDAGraphExtensionMode = CUDAGraphExtensionMode.NONE
    hook_library_path: str | None = None
    nvshmem_host_path: str | None = None
    base_addr: int = 0x600000000000
    region_size: str = "64GB"
    workspace_root: str = "foundry_archive"
    workspace_dir: str | None = None
    scratch_space_size: str = "64MB"

    @classmethod
    def from_toml(cls, path: str | Path) -> "CUDAGraphExtensionConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        base_addr_value = data.get("base_addr", cls.base_addr)
        base_addr = (
            int(base_addr_value, 0)
            if isinstance(base_addr_value, str)
            else base_addr_value
        )

        hook_library_path = data.get("hook_library_path")
        if hook_library_path is None:
            hook_library_path = cls._detect_hook_so_path()

        return cls(
            mode=CUDAGraphExtensionMode(data.get("mode", cls.mode.value)),
            hook_library_path=hook_library_path,
            nvshmem_host_path=data.get("nvshmem_host_path"),
            base_addr=base_addr,
            region_size=data.get("region_size", cls.region_size),
            workspace_root=data.get("workspace_root", cls.workspace_root),
            scratch_space_size=data.get(
                "scratch_space_size", cls.scratch_space_size
            ),
        )

    @staticmethod
    def _detect_hook_so_path() -> str | None:
        spec = importlib.util.find_spec("foundry.ops")
        if spec and spec.origin:
            ops_so_path = Path(spec.origin).resolve()
            hook_so_path = ops_so_path.parent / "libcuda_hook.so"
            if hook_so_path.exists():
                return str(hook_so_path)
        return None


_config: CUDAGraphExtensionConfig | None = None


def load_graph_extension_config(path: str) -> None:
    global _config
    _config = CUDAGraphExtensionConfig.from_toml(path)


def get_config() -> CUDAGraphExtensionConfig | None:
    return _config


def get_graph_extension_mode() -> CUDAGraphExtensionMode:
    if _config is None:
        return CUDAGraphExtensionMode.NONE
    return _config.mode


def get_workspace_root() -> str | None:
    if _config is None:
        return None
    return _config.workspace_root


def get_hook_library_path() -> str | None:
    if _config is None:
        return None
    return _config.hook_library_path


def get_nvshmem_host_path() -> str | None:
    if _config is None:
        return None
    return _config.nvshmem_host_path


def compute_workspace_rank(server_args, tp_rank: int, pp_rank: int, dp_rank: int | None) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        return pp_rank * server_args.tp_size + tp_rank
    dp_index = dp_rank or 0
    return dp_index * server_args.tp_size * server_args.pp_size + pp_rank * server_args.tp_size + tp_rank

