# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Contract test: every sglang symbol Foundry patches or reads at a fixed
signature still exists, on either layout Foundry supports (upstream main:
``bootstrap.init_parallel_runtime``, no ``ModelRunner.ps``; fork base: groups
built in ``ModelRunner.init_torch_distributed``). A rename upstream fails here
instead of at a user's LOAD.

    pytest tests/integration/sglang/test_sglang_hook_points.py -v
"""

import importlib
import inspect
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sglang")


def _resolve(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _params(obj):
    return list(inspect.signature(obj).parameters)


# (module, qualname, parameter names) Foundry wraps on both layouts.
_HOOK_POINTS = [
    (
        "sglang.srt.model_executor.model_runner",
        "ModelRunner.init_torch_distributed",
        ["self"],
    ),
    (
        "sglang.srt.model_executor.model_runner",
        "ModelRunner.alloc_memory_pool",
        ["self", "memory_pool_config"],
    ),
    (
        "sglang.srt.mem_cache.kv_cache_configurator",
        "KVCacheConfigurator._resolve_memory_pool_config",
        ["self", "pre_model_load_memory"],
    ),
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        "DecodeCudaGraphRunner.capture",
        ["self"],
    ),
    (
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        "DecodeCudaGraphRunner._resolve_shared_read_ends",
        ["self", "attn_backend", "forward_mode"],
    ),
    (
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
        "PrefillCudaGraphRunner.capture",
        ["self"],
    ),
    (
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
        "FullCudaGraphBackend.capture_one",
        ["self", "shape_key", "forward_fn", "capture_inputs", "post_warmup_hook"],
    ),
    (
        "sglang.srt.managers.data_parallel_controller",
        "DataParallelController.launch_tensor_parallel_group",
        ["self", "server_args", "port_args", "base_gpu_id", "dp_rank", "worker_ports"],
    ),
    (
        "sglang.srt.entrypoints.engine",
        "Engine._launch_scheduler_processes",
        # bound classmethod: cls is not listed
        ["server_args", "port_args", "run_scheduler_process_func", "placement_group"],
    ),
]


@pytest.mark.parametrize("module, qualname, params", _HOOK_POINTS, ids=lambda x: str(x))
def test_hook_point_signature(module, qualname, params):
    assert _params(_resolve(module, qualname)) == params


def test_distributed_bring_up_site():
    """Upstream main: init_parallel_runtime(*, server_args, device, dist_port)
    and ranks in get_parallel(). Fork base: ModelRunner keeps ``ps``."""
    from sglang.srt.distributed import bootstrap

    if hasattr(bootstrap, "init_parallel_runtime"):
        assert _params(bootstrap.init_parallel_runtime) == ["server_args", "device", "dist_port"]
        from sglang.srt.runtime_context import ParallelContext, SpawnRanks

        assert {"world_rank", "dp_rank", "gpu_id"} <= set(SpawnRanks.__struct_fields__)
        assert ParallelContext is not None
    else:
        from sglang.srt.model_executor.model_runner import ModelRunner

        assert "self.ps" in inspect.getsource(ModelRunner.__init__)


def test_engine_launch_is_a_classmethod():
    from sglang.srt.entrypoints.engine import Engine

    assert isinstance(Engine.__dict__["_launch_scheduler_processes"], classmethod)


def test_foundry_graphs_support_backend_cleanup():
    """Upstream main's FullCudaGraphBackend.cleanup() calls graph.reset() on
    every graph in ``_graphs``; the graphs Foundry places there are
    foundry.ops.CUDAGraph objects (the fork base's cleanup() does not reset)."""
    from foundry import ops

    assert callable(getattr(ops.CUDAGraph, "reset", None))


def test_resolution_hook_sites_are_overridable_and_called():
    from sglang.srt.arg_groups import pipeline, resolution_hooks

    src = inspect.getsource(pipeline.run_resolution_pipeline)
    for name in (
        "apply_inkling_prefill_cuda_graph_default",
        "handle_cuda_graph_config",
        "handle_other_validations",
    ):
        assert name in resolution_hooks._OVERRIDABLE_HOOKS
        assert f"run_hook({name}, server_args)" in src
    # The pins must land before the prefill defaults and the config parse.
    assert src.index("run_hook(apply_inkling_prefill_cuda_graph_default") < src.index(
        "run_hook(handle_cuda_graph_config"
    )


def test_plugin_entry_points_are_called_in_the_processes_foundry_needs():
    """load_plugins() runs in the launcher entries and each scheduler process.
    The DP controller does not load plugins (it inherits LD_PRELOAD from the
    launcher's environment instead)."""
    from sglang.srt.entrypoints import engine
    from sglang.srt.managers import scheduler

    assert "load_plugins()" in inspect.getsource(scheduler.run_scheduler_process)
    assert "load_plugins()" in inspect.getsource(engine.Engine.__init__)


def test_attributes_read_on_the_decode_runner():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    src = inspect.getsource(DecodeCudaGraphRunner)
    for attr in (
        "self.backend",
        "self.attention_graph_variants",
        "self.captured_req_width",
        "self.capture_bs",
        "self.seq_len_fill_value",
        "self.deepep_adapter",
        "self.in_graph_metadata_prep_done",
    ):
        assert attr in src, attr
    for method in ("_make_graph_key", "_capture_graph_size", "warmup", "capture_one_shape"):
        assert hasattr(DecodeCudaGraphRunner, method), method


def test_install_patches_keep_dispatch_shape(tmp_path):
    """After hooks.install (fresh interpreter: the patches are process-global):
    the launch patch is still a classmethod, and the bring-up wrapper sits on
    the site this layout uses."""
    cfg = tmp_path / "foundry.toml"
    cfg.write_text(f'mode = "save"\nworkspace_root = "{tmp_path / "ws"}"\n')
    code = textwrap.dedent(
        f"""
        from foundry.integration.sglang.hooks import install
        install({str(cfg)!r})
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.distributed import bootstrap
        from sglang.srt.model_executor.model_runner import ModelRunner
        assert isinstance(Engine.__dict__["_launch_scheduler_processes"], classmethod)
        if hasattr(bootstrap, "init_parallel_runtime"):
            assert hasattr(bootstrap.init_parallel_runtime, "__wrapped__")
        assert hasattr(ModelRunner.init_torch_distributed, "__wrapped__")
        print("OK")
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "FOUNDRY_GRAPH_EXTENSION_CONFIG"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-4000:]
