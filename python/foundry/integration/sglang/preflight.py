# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Pre-launch check that Foundry will actually run inside ``sglang serve``:
the installed SGLang declares Foundry as a dependency
(``sglang.srt.utils.foundry_adapter`` exists; ``--cuda-graph-persistence``),
the adapter and this Foundry speak the same integration API, and the TOML and
the archive are usable. The TOML's ``mode`` is ignored (the flag's mode
wins), so pass ``--save`` / ``--load``.

Run it with the interpreter that runs ``sglang serve``, with ``-P`` (or from a
directory without an ``sglang/`` checkout) so the cwd cannot shadow the
installed package::

    python -P -m foundry.integration.sglang.preflight --toml foundry_save.toml --save
    python -P -m foundry.integration.sglang.preflight --toml foundry_load.toml --load

It prints one ``[Foundry preflight] ok: ...`` line per check, or a single
``[Foundry preflight] FAILED: <reason>`` line on stderr and exits 1. Relative
``workspace_root`` values resolve against the current directory (``--cwd``),
the directory ``sglang serve`` will run in. The end-to-end tests call it;
recipe/sglang/README.md shows how to run it by hand.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = (
    "pip install foundry-core (or pip install -e foundry --no-build-isolation from a "
    "checkout), in the venv that runs sglang serve"
)
PREFIX = "[Foundry preflight]"


class PreflightError(RuntimeError):
    """A check failed; the message is the one-line reason."""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _sglang_commit(pkg_dir: Path, version: str) -> str:
    """An editable install is a git checkout: ask git. Otherwise take the
    ``+g<hash>`` a setuptools_scm version carries, if any."""
    head = _git(pkg_dir, "rev-parse", "--short=10", "HEAD")
    if head is not None and head.returncode == 0:
        return head.stdout.strip()
    m = re.search(r"\+g([0-9a-f]{7,})", version)
    return m.group(1) if m else "unknown"


def check_sglang_origin(sglang_mod, cwd: Path | None = None) -> str | None:
    """``sglang`` is the installed package, not a source directory shadowing it.

    ``python -m`` (and ``-c``) put the current directory first on ``sys.path``;
    run from a workspace that holds the ``sglang/`` checkout, ``import sglang``
    finds that directory (a namespace package, ``__file__`` None: raises) or,
    from inside ``sglang/python``, the source tree through the cwd entry (a
    warning: it is the installed package only if that is the editable
    checkout). ``sglang serve`` itself does not add the cwd."""
    cwd = (cwd or Path.cwd()).resolve()
    hint = (
        "run the preflight from another directory or with PYTHONSAFEPATH=1 (python -P), "
        "as recipe/sglang/README.md shows"
    )
    file = getattr(sglang_mod, "__file__", None)
    if file is None:
        paths = list(dict.fromkeys(str(p) for p in getattr(sglang_mod, "__path__", [])))
        raise PreflightError(
            f"'import sglang' resolved to a namespace package at {paths} (no __init__.py): a "
            f"directory named sglang on sys.path (the source checkout in {cwd}?) shadows the "
            f"installed sglang; {hint}"
        )
    pkg_dir = Path(file).resolve().parent
    if pkg_dir.parent == cwd and sys.path and sys.path[0] in ("", ".", str(cwd)):
        return (
            f"'import sglang' resolved to {pkg_dir} through the current directory on sys.path, "
            f"which sglang serve does not see (fine only if that is the editable install); {hint}"
        )
    return None


def dependency_route_adapter():
    """SGLang's Foundry adapter module when this sglang calls Foundry directly
    (``--cuda-graph-persistence``), else None."""
    import importlib

    try:
        return importlib.import_module("sglang.srt.utils.foundry_adapter")
    except ImportError:
        return None


def check_dependency_route(adapter) -> tuple[str, str | None]:
    """The adapter's integration API major matches this Foundry's api module."""
    import sglang

    from foundry.integration.sglang import api

    origin_warning = check_sglang_origin(sglang)
    want = getattr(adapter, "FOUNDRY_INTEGRATION_API_MAJOR", None)
    have = api.INTEGRATION_API_VERSION
    if want != have[0]:
        raise PreflightError(
            f"sglang's foundry_adapter calls integration API major {want}, this Foundry "
            f"provides {have}: install the Foundry version sglang[foundry] names"
        )
    version = getattr(sglang, "__version__", "unknown")
    pkg_dir = Path(sglang.__file__).resolve().parent
    commit = _sglang_commit(pkg_dir, version)
    info = (
        f"sglang {version} commit {commit} at {pkg_dir} calls Foundry directly "
        f"(--cuda-graph-persistence; integration API {have[0]}.{have[1]})"
    )
    return info, origin_warning


def check_toml(toml: str, expect_mode: str | None, cwd: Path, flag_mode: bool = False) -> str:
    """``flag_mode``: the dependency route, where ``expect_mode`` comes from
    ``--cuda-graph-persistence`` and replaces the TOML's ``mode``."""
    from foundry.integration.sglang.config import CUDAGraphExtensionConfig

    path = Path(toml)
    if not path.is_absolute():
        path = cwd / path
    if not path.is_file():
        raise PreflightError(f"Foundry TOML {path} does not exist")
    try:
        cfg = CUDAGraphExtensionConfig.from_toml(path)
    except Exception as exc:
        raise PreflightError(f"Foundry TOML {path} is invalid: {exc}") from exc
    mode = cfg.mode.value
    if flag_mode:
        if expect_mode is None:
            raise PreflightError(
                "pass --save or --load (the mode of --cuda-graph-persistence); "
                "the TOML's mode is ignored on this sglang"
            )
        mode = expect_mode
    if expect_mode is not None and mode != expect_mode:
        raise PreflightError(f"Foundry TOML {path} has mode = {mode!r}, expected {expect_mode!r}")
    if mode not in ("save", "load"):
        raise PreflightError(f"Foundry TOML {path} has mode = {mode!r}: use 'save' or 'load'")
    if cfg.hook_library_path is None or not Path(cfg.hook_library_path).is_file():
        raise PreflightError(
            "libcuda_hook.so not found next to foundry.ops (the native build is incomplete): "
            f"{INSTALL_HINT}"
        )
    archive = Path(cfg.workspace_root)
    if not archive.is_absolute():
        archive = cwd / archive
    if mode == "load":
        if not archive.is_dir():
            raise PreflightError(
                f"LOAD archive {archive} (workspace_root in {path}) does not exist: "
                "run --save first"
            )
        ranks = sorted(p.name for p in archive.glob("rank_*") if p.is_dir())
        if not ranks:
            raise PreflightError(f"LOAD archive {archive} has no rank_* dirs: run --save first")
        return f"TOML {path} mode=load, archive {archive} ({len(ranks)} rank dirs)"
    note = ""
    if archive.exists():
        note = " (exists: rm -rf it before a SAVE of another model or topology)"
    return f"TOML {path} mode=save, archive {archive}{note}"


def run(
    toml: str,
    expect_mode: str | None = None,
    cwd: str | os.PathLike | None = None,
) -> tuple[list[str], list[str]]:
    """Run every check; returns (info lines, warnings), raises PreflightError."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        import sglang  # noqa: F401
    except Exception as exc:
        raise PreflightError(f"cannot import sglang ({exc!r}): run in the SGLang venv") from exc
    adapter = dependency_route_adapter()
    if adapter is None:
        raise PreflightError(
            "this sglang has no Foundry adapter (sglang.srt.utils.foundry_adapter, "
            "--cuda-graph-persistence): install an sglang that declares sglang[foundry]"
        )
    info, warning = check_dependency_route(adapter)
    lines = [info, check_toml(toml, expect_mode, base, flag_mode=True)]
    return lines, [warning] if warning else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toml", required=True, help="Foundry TOML the engine will be given")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", dest="mode", action="store_const", const="save")
    group.add_argument("--load", dest="mode", action="store_const", const="load")
    parser.add_argument("--cwd", default=None, help="directory sglang serve runs in (default: .)")
    args = parser.parse_args(argv)
    try:
        lines, warnings = run(args.toml, args.mode, args.cwd)
    except PreflightError as exc:
        print(f"{PREFIX} FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"{PREFIX} python {sys.executable}")
    for line in lines:
        print(f"{PREFIX} ok: {line}")
    for warning in warnings:
        print(f"{PREFIX} WARNING: {warning}", file=sys.stderr)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
