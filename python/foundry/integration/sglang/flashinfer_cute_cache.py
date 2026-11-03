# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Persistent on-disk cache for FlashInfer's SM90 GDN prefill CuTe-DSL kernels.

FlashInfer's ``gdn_kernels/delta_rule_dsl`` compiles its SM90 kernels with
``cute.compile`` behind a process-local dict. :func:`install` routes that
compile through FlashInfer's ``build_and_load_cute_dsl_kernel`` (its on-disk
``.o`` cache under ``~/.cache/flashinfer/<version>/<arch>/cached_ops/``), so
one process compiles and every later one loads. Applied on SAVE and LOAD
before the capture loop when SGLang's FlashInfer GDN kernel module is loaded,
**only with ``FOUNDRY_FLASHINFER_GDN_CUTE_CACHE=1``** (default off).

Why off by default (2x H100, FlashInfer 0.7.2, Qwen3.5-2B TP2, 2026-11-03):
with the patch, the eager prefill output of the disk-loaded kernels varied
run to run (first-token logprob off by 0.029 / 0.056 in 4 of 6 LOADs of one
archive, 0 of 9 native runs), with the patch off 6 of 6 LOADs were exact, and
with the patch on plus ``CUDA_LAUNCH_BLOCKING=1`` also 6 of 6. The cache key
is not the problem (FlashInfer's full compile key plus the compile options;
a key holding an address bypasses the disk cache), and FlashInfer already
writes the ``.o`` atomically under a module lock. The kernel reloaded through
``cute.runtime.load_module(..., enable_tvm_ffi=True)`` is most likely not
ordered with the caller's stream the way the in-process ``cute.compile``
result is (inferred from CUDA_LAUNCH_BLOCKING removing the variation; the
stream it launches on was not traced). Until that is understood, the
in-process compile (8-10 s per scheduler at the first extend) is the
default.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

CACHE_MODULE = "flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache"
CACHE_PACKAGE = "flashinfer.gdn_kernels.delta_rule_dsl"
SGLANG_GDN_FLASHINFER = "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer"
# cached_ops/<this>_<arch>_cute_dsl/<kernel name>.o
CACHED_OPS_MODULE = "gdn_delta_rule_dsl"
_UNSTABLE_KEY = re.compile(r"0x[0-9a-fA-F]{6,}")

_installed: str | None = None
OPT_IN_ENV = "FOUNDRY_FLASHINFER_GDN_CUTE_CACHE"


def enabled() -> bool:
    """The patch is opt-in (see the module docstring)."""
    return os.environ.get(OPT_IN_ENV) == "1"


def kernel_name(func: Any, key_text: str) -> str:
    """Per-kernel cache name: the kernel class plus a digest of the compile
    key (dtypes, static shapes, options), the sole per-kernel cache key of
    FlashInfer's ``.o`` cache."""
    return f"{type(func).__qualname__}_{hashlib.sha256(key_text.encode()).hexdigest()[:16]}"


def gdn_flashinfer_in_use() -> bool:
    """SGLang imports its FlashInfer GDN kernel module only when a hybrid GDN
    model runs with the FlashInfer prefill or decode kernel."""
    return SGLANG_GDN_FLASHINFER in sys.modules


def install() -> str | None:
    """Route ``custom_compile_cache.cached_compile`` (and the name as already
    imported by the delta_rule modules) through FlashInfer's persistent
    ``.o`` cache. Returns a status string for the log, or None when this
    FlashInfer has no such module or helper (nothing to do). Idempotent."""
    global _installed
    if _installed is not None:
        return _installed
    try:
        ccc = importlib.import_module(CACHE_MODULE)
        cute_dsl_core = importlib.import_module("flashinfer.jit.cute_dsl_core")
        cute = importlib.import_module("cutlass.cute")
    except Exception as exc:
        logger.debug("[Foundry] FlashInfer CuTe-DSL cache patch not applicable: %r", exc)
        return None
    build_and_load = getattr(cute_dsl_core, "build_and_load_cute_dsl_kernel", None)
    original = getattr(ccc, "cached_compile", None)
    in_mem = getattr(ccc, "_in_mem_compile_cache", None)
    options_key = getattr(ccc, "_compile_options_key", None)
    if build_and_load is None or original is None or in_mem is None or options_key is None:
        logger.debug("[Foundry] FlashInfer CuTe-DSL cache patch not applicable (API)")
        return None
    sources = sorted(glob.glob(os.path.join(os.path.dirname(ccc.__file__), "*.py")))

    def cached_compile(func, *args, compile_options=None, **kwargs):
        key = (func._get_compile_key(), options_key(compile_options))
        compiled = in_mem.get(key)
        if compiled is not None:
            return compiled
        key_text = repr(key)

        def compile_fn():
            return cute.compile[compile_options](func, *args, **kwargs)

        if _UNSTABLE_KEY.search(key_text):
            logger.info(
                "[Foundry] CuTe-DSL cache: %s key holds an address, compiled in process",
                type(func).__qualname__,
            )
            compiled = compile_fn()
        else:
            name = kernel_name(func, key_text)
            logger.info("[Foundry] CuTe-DSL cache: %s <- %s", name, key_text)
            compiled = build_and_load(CACHED_OPS_MODULE, name, compile_fn, sources)
        in_mem[key] = compiled
        return compiled

    ccc.cached_compile = cached_compile
    rebound = []
    for name, module in list(sys.modules.items()):
        if (
            name.startswith(CACHE_PACKAGE + ".")
            and module is not ccc
            and getattr(module, "cached_compile", None) is original
        ):
            module.cached_compile = cached_compile
            rebound.append(name.rsplit(".", 1)[-1])
    _installed = "delta_rule_dsl cached_compile -> build_and_load_cute_dsl_kernel" + (
        f" (rebound in {', '.join(rebound)})" if rebound else ""
    )
    return _installed
