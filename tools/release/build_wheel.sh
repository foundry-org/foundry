#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
#
# Build, repair, check and smoke-test one foundry-core wheel, exactly as the
# release workflow (.github/workflows/release.yml) does.
#
# Inside a manylinux builder container (pytorch/manylinux2_28-builder:cudaXY.Z):
#   tools/release/build_wheel.sh --python 3.12 --torch 2.13.0 --cuda cu130
# From a host with docker (runs itself inside the matching builder image):
#   tools/release/build_wheel.sh --docker --python 3.12 --torch 2.13.0 --cuda cu130
#   FOUNDRY_DOCKER_GPUS=all tools/release/build_wheel.sh --docker ...   # smoke with a real driver
#
# Options:
#   --python X.Y        CPython version (manylinux /opt/python/cpXY-cpXY)   [3.12]
#   --torch  X.Y.Z      torch build to compile against                        [2.13.0]
#   --cuda   cuXYZ      torch CUDA flavor (index https://download.pytorch.org/whl/cuXYZ) [cu130]
#   --local-version S   append "+S" to the wheel version (GitHub-Release-only builds
#                       for a non-default torch/CUDA pair; PyPI rejects local versions)
#   --out DIR           output directory for the repaired wheel             [wheelhouse]
#   --no-smoke          skip the install + import check
#   --docker            re-run this script inside pytorch/manylinux2_28-builder:<cuda>
#
# Requires the vendored Boost headers (tools/release/vendor_boost.sh).
set -euo pipefail

PY_VER=3.12
TORCH_VER=2.13.0
CUDA_TAG=cu130
LOCAL_VERSION=""
OUT=wheelhouse
SMOKE=1
DOCKER=0
ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PY_VER="$2"; shift 2 ;;
        --torch) TORCH_VER="$2"; shift 2 ;;
        --cuda) CUDA_TAG="$2"; shift 2 ;;
        --local-version) LOCAL_VERSION="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --no-smoke) SMOKE=0; shift ;;
        --docker) DOCKER=1; shift ;;
        -h | --help) sed -n '5,25p' "$0"; exit 0 ;;
        *) echo "build_wheel: unknown argument $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log() { echo "build_wheel: $*"; }
die() { echo "build_wheel: ERROR: $*" >&2; exit 1; }

# cu130 -> 13.0, cu128 -> 12.8
cuda_dotted() {
    local d="${1#cu}"
    echo "${d%?}.${d: -1}"
}

if [[ "$DOCKER" == 1 ]]; then
    IMAGE="${FOUNDRY_BUILDER_IMAGE:-docker.io/pytorch/manylinux2_28-builder:cuda$(cuda_dotted "$CUDA_TAG")}"
    FWD=()
    for a in "${ARGS[@]}"; do [[ "$a" != --docker ]] && FWD+=("$a"); done
    GPU_ARGS=()
    [[ -n "${FOUNDRY_DOCKER_GPUS:-}" ]] && GPU_ARGS=(--gpus "$FOUNDRY_DOCKER_GPUS")
    log "running inside $IMAGE"
    exec docker run --rm "${GPU_ARGS[@]}" -v "$ROOT:/src" -w /src \
        -e MAX_JOBS -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        "$IMAGE" bash tools/release/build_wheel.sh "${FWD[@]}"
fi

cd "$ROOT"
[[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"

[[ -f third_party/boost/boost/version.hpp ]] || die "third_party/boost is not populated.
  Run tools/release/vendor_boost.sh and commit third_party/boost before building release wheels."

CP="cp${PY_VER/./}"
PYBIN="/opt/python/${CP}-${CP}/bin/python"
[[ -x "$PYBIN" ]] || PYBIN="$(command -v "python${PY_VER}" || true)"
[[ -n "$PYBIN" && -x "$PYBIN" ]] || die "no CPython ${PY_VER} (expected /opt/python/${CP}-${CP}; run inside the manylinux builder or pass --docker)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "no CUDA toolkit at $CUDA_HOME"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

WORK="$(mktemp -d /tmp/foundry-wheel.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

# torch (+ its nvidia-* deps) is downloaded once and reused by the smoke venv.
log "python $("$PYBIN" -V 2>&1), torch ${TORCH_VER} (${CUDA_TAG}), CUDA_HOME=$CUDA_HOME"
"$PYBIN" -m venv "$WORK/build-venv"
BPY="$WORK/build-venv/bin/python"
"$BPY" -m pip install -q -U pip
"$BPY" -m pip download -q -d "$WORK/torch-wheels" --index-url "$TORCH_INDEX" "torch==${TORCH_VER}"
"$BPY" -m pip install -q --no-index --find-links "$WORK/torch-wheels" "torch==${TORCH_VER}"
"$BPY" -m pip install -q "cmake>=4.0" ninja "setuptools>=80" wheel
"$BPY" -c "import torch; print('build torch', torch.__version__, 'cuda', torch.version.cuda)"

# Build from a clean copy of the tree: no stale CMake cache, editable-install
# .so files (package-data "*.so" would ship them) or build info from another run.
mkdir -p "$WORK/src"
tar -C "$ROOT" \
    --exclude=./.git --exclude=./build --exclude=./dist --exclude=./wheelhouse \
    --exclude='*.so' --exclude='*.egg-info' --exclude=__pycache__ \
    --exclude=./python/foundry/_build_info.py \
    -cf - . | tar -C "$WORK/src" -xf -

log "pip wheel"
(
    cd "$WORK/src"
    PATH="$WORK/build-venv/bin:$PATH" \
        FOUNDRY_WHEEL_BUILD=1 FOUNDRY_REQUIRE_VENDORED_BOOST=1 \
        "$BPY" -m pip wheel . --no-build-isolation --no-deps -w "$WORK/raw" -v
)
RAW="$(ls "$WORK"/raw/foundry_core-*.whl)"
[[ "$(echo "$RAW" | wc -l)" == 1 ]] || die "expected one wheel in $WORK/raw"

if [[ -n "$LOCAL_VERSION" ]]; then
    # Retag version X.Y.Z -> X.Y.Z+LOCAL in METADATA, the .dist-info name and the file name.
    log "local version +$LOCAL_VERSION"
    "$BPY" -m wheel unpack -d "$WORK/unpacked" "$RAW"
    UNP="$(ls -d "$WORK"/unpacked/foundry_core-*)"
    DI="$(ls -d "$UNP"/foundry_core-*.dist-info)"
    BASE_VER="$(sed -n 's/^Version: //p' "$DI/METADATA")"
    NEW_VER="${BASE_VER}+${LOCAL_VERSION}"
    sed -i "s/^Version: .*/Version: ${NEW_VER}/" "$DI/METADATA"
    mv "$DI" "$UNP/foundry_core-${NEW_VER}.dist-info"
    rm -f "$RAW"
    "$BPY" -m wheel pack -d "$WORK/raw" "$UNP"
    RAW="$(ls "$WORK"/raw/foundry_core-*.whl)"
fi

# auditwheel: never graft torch, CUDA or the driver (they come from the torch
# install / the host), and never touch libcuda_hook.so: SGLang LD_PRELOADs it
# from foundry/libcuda_hook.so, so it must keep that name and location.
"$PYBIN" -m venv "$WORK/tool-venv"
"$WORK/tool-venv/bin/pip" install -q "auditwheel>=6.1" patchelf
AW="$WORK/tool-venv/bin/auditwheel"
TORCH_LIB="$("$BPY" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
# Without a driver, libcuda.so.1 comes from the toolkit stub: enough for the
# loader, auditwheel and the hook's dlsym table, not for running CUDA.
HAVE_DRIVER=0
ldconfig -p >"$WORK/ldconfig.txt" 2>/dev/null || true
grep -q 'libcuda\.so\.1 ' "$WORK/ldconfig.txt" && HAVE_DRIVER=1
STUB_DIR=""
if [[ "$HAVE_DRIVER" == 0 && -f "$CUDA_HOME/lib64/stubs/libcuda.so" ]]; then
    STUB_DIR="$WORK/stub"
    mkdir -p "$STUB_DIR"
    ln -sf "$CUDA_HOME/lib64/stubs/libcuda.so" "$STUB_DIR/libcuda.so.1"
fi
AW_LD="$TORCH_LIB:$CUDA_HOME/lib64${STUB_DIR:+:$STUB_DIR}:${LD_LIBRARY_PATH:-}"
EXCLUDES=(
    'libtorch*.so*' 'libc10*.so*' 'libtorch_python.so'
    libtorch.so libtorch_cpu.so libtorch_cuda.so libc10.so libc10_cuda.so
    'libcuda.so*' libcuda.so.1 'libcudart*.so*' 'libnvrtc*.so*'
    libcuda_hook.so
)
EXCL_ARGS=()
for e in "${EXCLUDES[@]}"; do EXCL_ARGS+=(--exclude "$e"); done
log "auditwheel repair"
LD_LIBRARY_PATH="$AW_LD" \
    "$AW" repair --plat manylinux_2_28_x86_64 "${EXCL_ARGS[@]}" -w "$WORK/repaired" "$RAW"
WHEEL="$(ls "$WORK"/repaired/foundry_core-*.whl)"
LD_LIBRARY_PATH="$AW_LD" "$AW" show "$WHEEL" || log "WARNING: auditwheel show exited nonzero"

log "checking wheel layout"
# List once into a file: with pipefail, `unzip -l | grep -q` fails when grep exits early (SIGPIPE in unzip).
unzip -l "$WHEEL" | tee "$WORK/wheel.list"
grep -qE ' foundry/libcuda_hook\.so$' "$WORK/wheel.list" || die "foundry/libcuda_hook.so missing from the wheel"
grep -qE ' foundry/ops\..*\.so$' "$WORK/wheel.list" || die "foundry/ops*.so missing from the wheel"
grep -q 'libcuda_hook-' "$WORK/wheel.list" && die "auditwheel grafted a renamed libcuda_hook copy"
grep -qE 'foundry_core\.libs/.*(boost|torch|c10|cudart|libcuda)' "$WORK/wheel.list" \
    && die "auditwheel grafted a library that must come from the host/torch"
mkdir -p "$WORK/x"
unzip -q -o "$WHEEL" -d "$WORK/x"
readelf -d "$WORK/x/foundry/libcuda_hook.so" | tee "$WORK/hook.dyn"
grep -qi boost "$WORK/hook.dyn" && die "libcuda_hook.so has a Boost DT_NEEDED"
OPS_SO="$(ls "$WORK"/x/foundry/ops.*.so)"
readelf -d "$OPS_SO" | tee "$WORK/ops.dyn"
grep -qi boost "$WORK/ops.dyn" && die "foundry.ops has a Boost DT_NEEDED"
grep -q 'Shared library: \[libcuda_hook.so\]' "$WORK/ops.dyn" || die "foundry.ops does not need libcuda_hook.so by that name"
grep -qE 'R(UN)?PATH.*\$ORIGIN' "$WORK/ops.dyn" || die "foundry.ops lost its \$ORIGIN RPATH"
cat "$WORK/x"/foundry_core-*.dist-info/METADATA | grep -E '^(Name|Version|Requires-Dist):'

if [[ "$SMOKE" == 1 ]]; then
    log "smoke test in a clean venv"
    "$PYBIN" -m venv "$WORK/smoke-venv"
    SPY="$WORK/smoke-venv/bin/python"
    "$SPY" -m pip install -q -U pip
    "$SPY" -m pip install -q --no-index --find-links "$WORK/torch-wheels" "torch==${TORCH_VER}"
    "$SPY" -m pip install -q --no-index --find-links "$WORK/torch-wheels" "$WHEEL"
    SMOKE_LD="${STUB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ "$HAVE_DRIVER" == 1 ]]; then
        log "driver libcuda.so.1 present"
    elif [[ -n "$STUB_DIR" ]]; then
        log "no driver: using the CUDA stub libcuda.so.1"
    else
        log "WARNING: no driver and no stub libcuda: import foundry skipped"
    fi
    (
        cd "$WORK"
        "$SPY" - <<'PY'
import importlib.util, pathlib
spec = importlib.util.find_spec("foundry")
pkg = pathlib.Path(spec.origin).parent
hook = pkg / "libcuda_hook.so"
assert hook.is_file(), hook
from importlib.metadata import version, requires
print("foundry-core", version("foundry-core"), "requires", requires("foundry-core"))
print("hook", hook)
PY
        if [[ "$HAVE_DRIVER" == 1 || -n "$STUB_DIR" ]]; then
            LD_LIBRARY_PATH="$SMOKE_LD" "$SPY" -c "import foundry, foundry.integration.sglang.api as a; print(foundry.__version__, a.INTEGRATION_API_VERSION)"
            LD_LIBRARY_PATH="$SMOKE_LD" "$SPY" -c "import foundry.ops; print('foundry.ops', foundry.ops.__file__)"
        fi
    )
fi

cp "$WHEEL" "$OUT/"
if [[ -n "${HOST_UID:-}" ]]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT" 2>/dev/null || true
fi
log "done: $OUT/$(basename "$WHEEL")"
