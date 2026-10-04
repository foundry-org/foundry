# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Pre-launch check that Foundry will actually run inside ``sglang serve``.

Two routes. On an SGLang that declares Foundry as a dependency
(``sglang.srt.utils.foundry_adapter`` exists; ``--cuda-graph-persistence``),
the check is that the adapter and this Foundry speak the same integration API
and that the TOML and the archive are usable; the TOML's ``mode`` is ignored
there (the flag's mode wins), so pass ``--save`` / ``--load``.

On the plugin route, SGLang falls back to a native run without any error in
two cases a plugin cannot report itself, because its code never runs:

- ``FOUNDRY_GRAPH_EXTENSION_CONFIG`` is set but the ``foundry`` entry point
  (group ``sglang.srt.plugins``) is not registered in the serving environment
  (Foundry not installed there, or a stale install from before the plugin);
- ``SGLANG_PLUGINS`` is set and does not name ``foundry``.

Run it with the interpreter that runs ``sglang serve``, with ``-P`` (or from a
directory without an ``sglang/`` checkout) so the cwd cannot shadow the
installed package::

    python -P -m foundry.integration.sglang.preflight --toml foundry_save.toml --save
    python -P -m foundry.integration.sglang.preflight --toml foundry_load.toml --load

It prints one ``[Foundry preflight] ok: ...`` line per check, or a single
``[Foundry preflight] FAILED: <reason>`` line on stderr and exits 1. Relative
``workspace_root`` values resolve against the current directory (``--cwd``),
the directory ``sglang serve`` will run in. The recipe scripts
(``recipe/sglang/serve_common.sh``) and the end-to-end test call it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

# Upstream sglang main commit the plugin was validated on (fork branch
# ``foundry-plugin``). An older tree that has the plugin framework and the
# three resolution steps (e.g. the fork base) is accepted with a warning.
VALIDATED_SGLANG_COMMIT = "fa090f7755"
ENTRY_POINT_GROUP = "sglang.srt.plugins"
ENTRY_POINT_NAME = "foundry"
ENTRY_POINT_VALUE = "foundry_sglang_plugin:load"
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


def _sglang_commit(pkg_dir: Path, version: str) -> tuple[str, str]:
    """(commit, contains-validated) with contains-validated in yes/no/unknown.

    An editable install is a git checkout: ask git. Otherwise take the
    ``+g<hash>`` a setuptools_scm version carries, if any."""
    head = _git(pkg_dir, "rev-parse", "--short=10", "HEAD")
    if head is not None and head.returncode == 0:
        anc = _git(pkg_dir, "merge-base", "--is-ancestor", VALIDATED_SGLANG_COMMIT, "HEAD")
        rc = anc.returncode if anc is not None else -1
        return head.stdout.strip(), {0: "yes", 1: "no"}.get(rc, "unknown")
    m = re.search(r"\+g([0-9a-f]{7,})", version)
    return (m.group(1) if m else "unknown"), "unknown"


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
        "as recipe/sglang/serve_common.sh does"
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


def check_sglang() -> tuple[str, str | None]:
    """SGLang has the plugin framework and the resolution steps Foundry wraps.

    Returns (info line, warning or None)."""
    try:
        import sglang
    except Exception as exc:
        raise PreflightError(f"cannot import sglang ({exc!r}): run in the SGLang venv") from exc
    origin_warning = check_sglang_origin(sglang)
    try:
        import sglang.srt.plugins as plugins
    except Exception as exc:
        raise PreflightError(
            f"cannot import sglang.srt.plugins ({exc!r}): this sglang has no plugin framework; "
            f"use upstream main >= {VALIDATED_SGLANG_COMMIT} (fork branch foundry-plugin)"
        ) from exc
    if getattr(plugins, "GENERAL_PLUGINS_GROUP", ENTRY_POINT_GROUP) != ENTRY_POINT_GROUP:
        raise PreflightError(
            f"sglang loads general plugins from {plugins.GENERAL_PLUGINS_GROUP!r}, "
            f"not {ENTRY_POINT_GROUP!r}"
        )
    try:
        from sglang.srt.arg_groups import resolution_hooks
    except Exception as exc:
        raise PreflightError(
            f"cannot import sglang.srt.arg_groups.resolution_hooks ({exc!r}): "
            f"use sglang main >= {VALIDATED_SGLANG_COMMIT}"
        ) from exc
    from foundry.integration.sglang.plugin import RESOLUTION_HOOK_STEPS

    overridable = getattr(resolution_hooks, "_OVERRIDABLE_HOOKS", frozenset())
    missing = [name for name in RESOLUTION_HOOK_STEPS if name not in overridable]
    if missing:
        raise PreflightError(
            f"sglang has no overridable resolution step(s) {missing}: the plugin cannot pin the "
            f"graph settings; use sglang main >= {VALIDATED_SGLANG_COMMIT}"
        )

    version = getattr(sglang, "__version__", "unknown")
    pkg_dir = Path(sglang.__file__).resolve().parent
    commit, contains = _sglang_commit(pkg_dir, version)
    info = (
        f"sglang {version} commit {commit} at {pkg_dir} "
        f"(contains validated {VALIDATED_SGLANG_COMMIT}: {contains})"
    )
    warnings = [origin_warning] if origin_warning else []
    if contains == "no":
        warnings.append(
            f"sglang {commit} does not contain the validated commit {VALIDATED_SGLANG_COMMIT}; "
            "it has the plugin surface, so continuing (the fork base is known to work)"
        )
    return info, "; ".join(warnings) or None


def dependency_route_adapter():
    """SGLang's Foundry adapter module when this sglang calls Foundry directly
    (``--cuda-graph-persistence``), else None (plugin route)."""
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
    commit, _ = _sglang_commit(pkg_dir, version)
    info = (
        f"sglang {version} commit {commit} at {pkg_dir} calls Foundry directly "
        f"(--cuda-graph-persistence; integration API {have[0]}.{have[1]})"
    )
    return info, origin_warning


def check_entry_point() -> str:
    from importlib.metadata import entry_points

    eps = [e for e in entry_points(group=ENTRY_POINT_GROUP) if e.name == ENTRY_POINT_NAME]
    if not eps:
        raise PreflightError(
            f"foundry plugin entry point not registered in group {ENTRY_POINT_GROUP!r}: "
            f"{INSTALL_HINT}; SGLang would run natively"
        )
    ep = eps[0]
    if ep.value != ENTRY_POINT_VALUE:
        raise PreflightError(
            f"entry point {ENTRY_POINT_NAME} -> {ep.value!r}, expected {ENTRY_POINT_VALUE!r}: "
            f"stale foundry install; {INSTALL_HINT}"
        )
    try:
        ep.load()
    except Exception as exc:
        raise PreflightError(
            f"entry point {ep.value} does not import ({exc!r}): {INSTALL_HINT}"
        ) from exc
    dist = getattr(ep, "dist", None)
    where = f" (dist {dist.name} {dist.version})" if dist is not None else ""
    return f"entry point {ENTRY_POINT_GROUP}:{ep.name} -> {ep.value}{where}"


def check_plugin_allowlist(env: Mapping[str, str]) -> str:
    """Same parsing as sglang's ``load_plugins_by_group``."""
    raw = env.get("SGLANG_PLUGINS", "")
    if not raw:
        return "SGLANG_PLUGINS unset (every installed plugin loads)"
    allowed = {x.strip() for x in raw.split(",") if x.strip()}
    if ENTRY_POINT_NAME not in allowed:
        raise PreflightError(
            f"SGLANG_PLUGINS={raw!r} does not include {ENTRY_POINT_NAME!r}: sglang would skip the "
            "plugin and run natively; add foundry or unset it"
        )
    return f"SGLANG_PLUGINS={raw!r} includes {ENTRY_POINT_NAME}"


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
    env: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Run every check; returns (info lines, warnings), raises PreflightError."""
    env = os.environ if env is None else env
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        import sglang  # noqa: F401
    except Exception as exc:
        raise PreflightError(f"cannot import sglang ({exc!r}): run in the SGLang venv") from exc
    adapter = dependency_route_adapter()
    if adapter is not None:
        info, warning = check_dependency_route(adapter)
        lines = [info, check_toml(toml, expect_mode, base, flag_mode=True)]
        return lines, [warning] if warning else []
    info, warning = check_sglang()
    lines = [
        info,
        check_entry_point(),
        check_plugin_allowlist(env),
        check_toml(toml, expect_mode, base),
    ]
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
