# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""The FlashInfer CuTe-DSL ``.o`` cache patch (flashinfer_cute_cache.py) on
fake flashinfer / cutlass modules. No GPU.

    pytest tests/integration/sglang/test_sglang_flashinfer_cute_cache.py -v
"""

import sys
import types

import pytest

fcc = pytest.importorskip("foundry.integration.sglang.flashinfer_cute_cache")


class _Kernel:
    def __init__(self, key):
        self._key = key

    def _get_compile_key(self):
        return self._key


class _Compile:
    """Stands in for cutlass.cute.compile: ``compile[options](func, ...)``."""

    def __init__(self, log):
        self.log = log

    def __getitem__(self, options):
        def run(func, *args, **kwargs):
            self.log.append(("compile", func, options))
            return ("compiled", func, options)

        return run


@pytest.fixture
def fake_flashinfer(monkeypatch, tmp_path):
    log = []
    ccc = types.ModuleType(fcc.CACHE_MODULE)
    ccc.__file__ = str(tmp_path / "custom_compile_cache.py")
    (tmp_path / "custom_compile_cache.py").write_text("# fake\n")
    (tmp_path / "delta_rule_cp_sm90.py").write_text("# fake\n")
    ccc._in_mem_compile_cache = {}
    ccc._compile_options_key = lambda options: None if options is None else ("opts", options)

    def original(func, *args, compile_options=None, **kwargs):
        raise AssertionError("original cached_compile must not run")

    ccc.cached_compile = original
    sibling = types.ModuleType(fcc.CACHE_PACKAGE + ".delta_rule_cp_sm90")
    sibling.cached_compile = original
    core = types.ModuleType("flashinfer.jit.cute_dsl_core")

    def build_and_load(module_name, name, compile_fn, sources):
        log.append(("build_and_load", module_name, name, tuple(sources)))
        return ("loaded", compile_fn())

    core.build_and_load_cute_dsl_kernel = build_and_load
    cute = types.ModuleType("cutlass.cute")
    cute.compile = _Compile(log)
    for name, module in {
        fcc.CACHE_MODULE: ccc,
        sibling.__name__: sibling,
        "flashinfer.jit.cute_dsl_core": core,
        "cutlass.cute": cute,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(fcc, "_installed", None)
    return types.SimpleNamespace(ccc=ccc, sibling=sibling, log=log, original=original)


def test_stable_key_goes_through_the_o_cache_and_rebinds_siblings(fake_flashinfer):
    f = fake_flashinfer
    status = fcc.install()
    assert "build_and_load_cute_dsl_kernel" in status and "delta_rule_cp_sm90" in status
    assert f.sibling.cached_compile is f.ccc.cached_compile is not f.original
    kernel = _Kernel((("dtype", "bf16"), ("hd", 128)))
    out = f.ccc.cached_compile(kernel, 1, 2, compile_options="sm90")
    assert out == ("loaded", ("compiled", kernel, "sm90"))
    kinds = [e[0] for e in f.log]
    assert kinds == ["build_and_load", "compile"]
    _, module_name, name, sources = f.log[0]
    assert module_name == fcc.CACHED_OPS_MODULE
    assert name.startswith("_Kernel_") and len(name) == len("_Kernel_") + 16
    assert [s.rsplit("/", 1)[-1] for s in sources] == [
        "custom_compile_cache.py",
        "delta_rule_cp_sm90.py",
    ]
    # In-memory hit: nothing compiles or loads again.
    assert f.ccc.cached_compile(kernel, 1, 2, compile_options="sm90") is out
    assert len(f.log) == 2


def test_unstable_key_takes_the_plain_compile(fake_flashinfer):
    f = fake_flashinfer
    fcc.install()
    kernel = _Kernel(("obj", "<object at 0x7f00deadbeef>"))
    out = f.ccc.cached_compile(kernel, compile_options="sm90")
    assert out == ("compiled", kernel, "sm90")
    assert [e[0] for e in f.log] == ["compile"]


def test_same_key_same_name_across_processes(fake_flashinfer):
    fcc.install()
    a = fcc.kernel_name(_Kernel(None), repr((("dtype", "bf16"),)))
    b = fcc.kernel_name(_Kernel(None), repr((("dtype", "bf16"),)))
    c = fcc.kernel_name(_Kernel(None), repr((("dtype", "fp16"),)))
    assert a == b != c


def test_install_is_idempotent_and_noop_without_flashinfer(fake_flashinfer, monkeypatch):
    first = fcc.install()
    assert fcc.install() == first
    monkeypatch.setattr(fcc, "_installed", None)
    monkeypatch.delitem(sys.modules, "flashinfer.jit.cute_dsl_core")
    monkeypatch.setattr(
        fcc.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    assert fcc.install() is None


def test_gdn_flashinfer_detection(monkeypatch):
    monkeypatch.delitem(sys.modules, fcc.SGLANG_GDN_FLASHINFER, raising=False)
    assert not fcc.gdn_flashinfer_in_use()
    monkeypatch.setitem(sys.modules, fcc.SGLANG_GDN_FLASHINFER, types.ModuleType("x"))
    assert fcc.gdn_flashinfer_in_use()
