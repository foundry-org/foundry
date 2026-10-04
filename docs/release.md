# Releasing foundry-core

Foundry is published to PyPI as **`foundry-core`** (import name `foundry`).
Releases are built and uploaded by `.github/workflows/release.yml` when a
`v*` tag is pushed. This page covers the one-time setup, the release
checklist and how torch/CUDA pairs map to wheels.

## What a release contains

`foundry.ops` is a torch C++ extension that uses ATen/c10 internals, so a
wheel only works with the torch it was built against (major.minor and CUDA
major), like sglang-kernel and flashinfer. PyPI rejects local version labels
(`+cu130`), so the release follows sglang-kernel's convention:

- **One torch/CUDA pair per release line goes to PyPI** with the plain
  version. It is the first entry of `BUILD_PAIRS` at the top of the workflow.
  Its wheels declare `Requires-Dist: torch==A.B.C`, the exact torch they were built against (SGLang pins torch the same way).
- **Other pairs go to the GitHub Release only**, built with a local version
  `X.Y.Z+cuNNN.torchA.B` (for example `0.1.0+cu128.torch2.12`) so the file
  names do not collide. Users install them by URL.
- The **sdist** goes to PyPI and to the GitHub Release. It contains the native
  sources and the vendored Boost headers.

| Release line | PyPI pair | CPython | Platform |
|---|---|---|---|
| 0.1.x | torch 2.13, cu130 | 3.10-3.13 | manylinux_2_28 x86_64 |

SGLang depends on `foundry-core>=0.1.0,<0.2`. Changing the PyPI pair (for
example to torch 2.14) therefore needs a new minor line (0.2.0) coordinated
with SGLang's own torch bump; 0.1.x patch releases keep torch 2.13 / cu130.
Keep the table above, the README installation table and `RELEASE.md` in sync
with `BUILD_PAIRS`.

## One-time setup

### PyPI trusted publisher

The workflow uploads with [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so no API token is stored in GitHub. Before the first release the
`foundry-core` project does not exist yet, so register a **pending
publisher**: log in to PyPI, open *Your account > Publishing > Add a new
pending publisher > GitHub*, and enter:

| Field | Value |
|---|---|
| PyPI project name | `foundry-core` |
| Owner | `foundry-org` |
| Repository name | `foundry` (the repository the tag is pushed to; use `foundry-priv` if releasing from there) |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

After the first successful upload the pending publisher becomes a normal
publisher of the project (*Manage project > Publishing*). Add other
maintainers as project owners there.

### GitHub environment

In the repository: *Settings > Environments > New environment*, name it
`pypi`. Recommended: add required reviewers (the PyPI upload then waits for
approval) and a deployment branch/tag rule allowing tags `v*`. The release job
creates the GitHub Release with the workflow's `GITHUB_TOKEN`
(`contents: write`, granted in the workflow), so Actions must be allowed to
create releases (the default for a repository).

## Release checklist

1. **Vendor Boost** (once, and again when bumping Boost). On any Linux or macOS
   machine with curl and a C++ compiler:

   ```bash
   tools/release/vendor_boost.sh            # pinned Boost 1.90.0, SHA-256 checked
   git add third_party/boost
   git commit -m "[build] Vendor Boost 1.90.0 headers"
   ```

   The script builds `bcp`, copies the header subset Foundry includes, runs a
   header-only compile-and-link check and rewrites `third_party/boost/README.md`.
   The workflow fails early when `third_party/boost/boost/version.hpp` is
   missing. Another Boost: `BOOST_VERSION=1.92.0 BOOST_SHA256=<sha256> tools/release/vendor_boost.sh`
   (the checksum is published at
   `https://archives.boost.io/release/<ver>/source/boost_<ver_>.tar.bz2.json`).

2. **Bump the version** in `pyproject.toml` (`[project] version`), add the
   `RELEASE.md` section, and update `BUILD_PAIRS` / the pairing tables if the
   torch pair changes.

3. **Validate on a GPU host** before tagging. With docker:

   ```bash
   tools/release/build_wheel.sh --docker --python 3.12 --torch 2.13.0 --cuda cu130
   FOUNDRY_DOCKER_GPUS=all tools/release/build_wheel.sh --docker --python 3.12   # smoke against the real driver
   ```

   The script runs inside `pytorch/manylinux2_28-builder:cuda13.0`, exactly as
   CI does, and leaves the repaired wheel in `wheelhouse/`. Then install that
   wheel into an SGLang venv with the same torch and run a recipe SAVE / LOAD
   (`recipe/sglang/`), which exercises `libcuda_hook.so` preloading from the
   installed package.

4. **Dry run on GitHub** (optional): *Actions > release > Run workflow* on the
   release branch with `publish = no`. It builds the sdist and all wheels and
   uploads them as workflow artifacts without publishing.

5. **Tag and push**:

   ```bash
   git tag v0.1.0
   git push pub v0.1.0      # the remote that holds the trusted-publisher repo
   ```

   The plan job checks that the tag equals `v<pyproject version>`. Then:
   sdist, wheels (each repaired, checked and smoke-tested), the PyPI upload of
   the PyPI-pair wheels + sdist (waits for the `pypi` environment approval if
   configured), and the GitHub Release `v0.1.0` with every wheel and the sdist
   attached.

6. **Verify** in a fresh venv:

   ```bash
   pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
   pip install foundry-core==0.1.0
   python -c "import foundry, foundry.integration.sglang.api as a; print(foundry.__version__, a.INTEGRATION_API_VERSION)"
   ```

A PyPI version can never be re-uploaded. If a published release is broken,
yank it on PyPI and release the next patch version.

## Extra torch/CUDA pairs

- **Permanent:** append `{"torch": "2.12.0", "cuda": "cu128"}` to
  `BUILD_PAIRS`. Every tagged release then also builds those wheels (local
  version, GitHub Release only). The builder image is derived from the CUDA
  tag (`cu128` uses `pytorch/manylinux2_28-builder:cuda12.8`); add
  `"image": "..."` to the entry to override it.
- **One-off:** *Run workflow* with `torch = 2.12.0`, `cuda = cu128`,
  `publish = yes`. Only that pair is built; it skips PyPI and uploads to the
  GitHub Release of `v<pyproject version>` (creating it if needed).

Users install such a wheel by URL, matching their Python:

```bash
pip install "https://github.com/foundry-org/foundry/releases/download/v0.1.0/foundry_core-0.1.0+cu128.torch2.12-cp312-cp312-manylinux_2_28_x86_64.whl"
```

## What the wheel build checks

`tools/release/build_wheel.sh` (CI and local) builds from a clean copy of the
tree with `FOUNDRY_WHEEL_BUILD=1 FOUNDRY_REQUIRE_VENDORED_BOOST=1` and then:

- `auditwheel repair --plat manylinux_2_28_x86_64`, excluding `libtorch*`,
  `libc10*`, `libtorch_python`, `libcuda.so.1`, `libcudart*`, `libnvrtc*` (all
  provided by torch or the driver at runtime) and `libcuda_hook.so`, which must
  stay at `foundry/libcuda_hook.so` under that name because SGLang preloads it
  from there;
- fails if `foundry/libcuda_hook.so` or `foundry/ops.*.so` is missing, if
  auditwheel grafted a renamed hook or a torch/CUDA/driver library, if either
  shared object has a Boost `DT_NEEDED`, if `foundry.ops` does not need
  `libcuda_hook.so` by that name, or if it lost its `$ORIGIN` RPATH;
- installs the wheel and the same torch into a clean venv and runs
  `import foundry, foundry.integration.sglang.api` and `import foundry.ops`.
  GitHub runners have no NVIDIA driver, so `libcuda.so.1` there is the CUDA
  toolkit stub (`lib64/stubs/libcuda.so`). That proves the loader resolves
  every library and the hook's driver-symbol table is complete, but it runs no
  CUDA; step 3 of the checklist covers that.
