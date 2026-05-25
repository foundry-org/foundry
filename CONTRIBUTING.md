# Contributing to Foundry

Thanks for your interest in contributing. This page covers the basics: env
setup, the lint / format toolchain, and the commit-time checks.

## Environment

```bash
# 1. Create / activate a venv (Python ≥ 3.10).
conda create -p venv/ python=3.12 && conda activate venv/

# 2. Build Foundry editable.
pip install -e . --no-build-isolation

# 3. Install the pre-commit toolchain.
pip install -r requirements/lint.txt
pre-commit install
```

`pre-commit install` registers both the `pre-commit` and `commit-msg` hook
stages — every `git commit` will then run the linters on the staged files and
auto-append a `Signed-off-by:` trailer (DCO style).

## What runs on commit

The full list lives in `.pre-commit-config.yaml`. Highlights:

| Hook | Scope | What it does |
|---|---|---|
| `ruff-check` / `ruff-format` | Python | lint + autoformat. Config in `pyproject.toml::[tool.ruff]`. |
| `clang-format` | `*.c, *.cc, *.cpp, *.h, *.hpp, *.cu, *.cuh` | C++/CUDA reformat. Style in `.clang-format`. |
| `markdownlint-cli2` | `*.md` | markdown lint (autofix). Config in `.markdownlint.yaml`. |
| `actionlint` | `.github/workflows/*.yml` | GitHub Actions workflow lint. |
| `check-spdx-header` | `*.py` | enforces the two-line SPDX header on new Python files; auto-prepends if missing. |
| `signoff-commit` | commit message | appends `Signed-off-by:` trailer if it isn't there. |
| `check-filenames` | repo-wide | rejects filenames containing spaces. |

## Running manually

```bash
# Run every hook on staged files (what `git commit` does):
pre-commit run

# Run every hook on the whole repo:
pre-commit run --all-files

# Run one specific hook on every file:
pre-commit run ruff-check --all-files

# Skip a particular hook for one commit:
SKIP=clang-format git commit -m "wip"

# Bypass everything (last resort — won't pass CI):
git commit --no-verify -m "wip"
```

## Safety net after a pre-commit run

`ruff` is aggressive about auto-fixing "unused" imports. For re-export modules
(`foundry/__init__.py`, `foundry/allocation_region.py`) that's an *API break*
disguised as a cosmetic cleanup. Two layers protect against this:

1. `pyproject.toml::[tool.ruff.lint].unfixable = ["F401", "F841"]` — ruff
   reports unused imports but does not auto-remove them. You still see the
   warning; you decide whether to delete or to add `__all__`.
2. `tests/test_imports.py` — a smoke test that imports every public Foundry
   API. Run it after any pre-commit-driven cleanup:

   ```bash
   pytest tests/test_imports.py -q
   ```

   If it fails, an import got dropped. Add the missing name to the source
   module's `__all__` (so ruff future-proofs against the same fix) and
   re-import.

## Filing changes

- Each commit must include a `Signed-off-by:` line. The `signoff-commit` hook
  adds it for you using your `git config user.name` / `user.email`. Configure
  those once if you haven't.
- Every Python file should start with:

  ```python
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: Copyright contributors to the Foundry project
  ```

  `check-spdx-header` will add this for you if missing — but it'll fail the
  commit and you'll need to re-stage and re-commit.

## Tests

```bash
pytest tests/
```

C++ side test build comes from CMake; see `CMakeLists.txt` and `README.md`.

## Reporting issues / proposing changes

File an issue at <https://github.com/foundry-org/foundry/issues> describing
the symptom and, where applicable, a minimal repro. For larger design
changes, an issue with a short proposal before opening a PR is appreciated.
