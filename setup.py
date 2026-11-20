# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import re
import subprocess
from pathlib import Path

import torch
from setuptools import setup

# FOUNDRY_SDIST=1: build only the source distribution. Nothing is compiled, so
# torch.utils.cpp_extension (which needs a CUDA toolkit via CUDA_HOME) is not
# imported and no extension module is declared.
SDIST = os.getenv("FOUNDRY_SDIST") == "1"
if not SDIST:
    from setuptools import Extension
    from setuptools.command.bdist_wheel import bdist_wheel
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, library_paths

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# foundry/libfoundry_ops.so (the native part of foundry.ops) is compiled against the
# LibTorch Stable ABI only, targeting this torch, and has no Python C API: it registers
# torch custom ops and foundry/ops.py loads it with torch.ops.load_library. One build
# loads on every CPython and every torch >= MIN_TORCH (no upper bound; newer-torch behaviour
# Foundry depends on is checked at run time, see foundry/_loader.py).
MIN_TORCH = (2, 14)
# TORCH_TARGET_VERSION encoding of torch/headeronly/version.h: one byte per major/minor.
TORCH_TARGET_VERSION = f"0x{MIN_TORCH[0]:02x}{MIN_TORCH[1]:02x}000000000000"
# The only torch header trees foundry.ops may include (copied into build/ so
# nothing else of torch/include is on the include path).
# The native library's dotted name: built to foundry/libfoundry_ops.so.
OPS_LIBRARY = "foundry.libfoundry_ops"
STABLE_HEADER_TREES = (
    "torch/headeronly",
    "torch/csrc/stable",
    "torch/csrc/inductor/aoti_torch/c",
)

# FOUNDRY_WHEEL_BUILD=1 (set by tools/release/build_wheel.sh and the release
# workflow): build a relocatable wheel: no absolute RPATH into the build machine's
# CUDA toolkit. libfoundry_ops.so never has an RPATH into torch (stable ABI, torch >= 2.14).
WHEEL_BUILD = bool(os.getenv("FOUNDRY_WHEEL_BUILD"))


# Boost is used header-only (Boost.JSON through csrc/boost_json_src.cpp); no
# compiled Boost library is linked. Both shared objects must see the SAME Boost
# headers, so the directory resolved here is also handed to CMake.
VENDORED_BOOST_DIR = os.path.join(ROOT_DIR, "third_party", "boost")
MIN_BOOST_VERSION = 108300  # 1.83: boost::concurrent_flat_map
# Default compiler search dirs: never pass these as -I (it breaks #include_next).
_IMPLICIT_INCLUDE_DIRS = {"/usr/include", "/usr/local/include"}


def _boost_version(include_dir):
    header = Path(include_dir) / "boost" / "version.hpp"
    if not header.is_file():
        return None
    m = re.search(r"^#define\s+BOOST_VERSION\s+(\d+)", header.read_text(), re.MULTILINE)
    return int(m.group(1)) if m else None


def _fmt_boost(v):
    return f"{v // 100000}.{v // 100 % 1000}.{v % 100}"


def resolve_boost_include_dir():
    """Pick the Boost header directory: explicit override, vendored copy, system."""
    explicit = os.getenv("FOUNDRY_BOOST_INCLUDE_DIR")
    if explicit:
        candidates = [("FOUNDRY_BOOST_INCLUDE_DIR", explicit)]
    else:
        candidates = [("vendored", VENDORED_BOOST_DIR)]
        if (
            os.getenv("FOUNDRY_REQUIRE_VENDORED_BOOST")
            and _boost_version(VENDORED_BOOST_DIR) is None
        ):
            raise RuntimeError(
                "FOUNDRY_REQUIRE_VENDORED_BOOST is set but third_party/boost/boost is missing: "
                "run tools/release/vendor_boost.sh and commit third_party/boost"
            )
        if os.getenv("BOOST_INCLUDEDIR"):
            candidates.append(("BOOST_INCLUDEDIR", os.environ["BOOST_INCLUDEDIR"]))
        for env in ("BOOST_ROOT", "CONDA_PREFIX"):
            if os.getenv(env):
                candidates.append((env, os.path.join(os.environ[env], "include")))
        # The prefixes CMake searches for BoostConfig (the CMake fallback) hold the headers too.
        for prefix in os.getenv("CMAKE_PREFIX_PATH", "").split(os.pathsep):
            if prefix:
                candidates.append(("CMAKE_PREFIX_PATH", os.path.join(prefix, "include")))
        candidates += [("system", "/usr/local/include"), ("system", "/usr/include")]

    rejected = []
    for origin, d in candidates:
        v = _boost_version(d)
        if v is None:
            continue
        if v < MIN_BOOST_VERSION:
            rejected.append(f"{d} ({_fmt_boost(v)})")
            continue
        print(f"foundry: Boost {_fmt_boost(v)} headers from {d} ({origin})")
        return os.path.abspath(d)
    raise RuntimeError(
        "Boost >= 1.83 headers not found"
        + (f"; too old: {', '.join(rejected)}" if rejected else "")
        + ". Run tools/release/vendor_boost.sh, install Boost headers "
        "(e.g. apt-get install libboost-dev, conda install -c conda-forge boost-cpp), "
        "or set FOUNDRY_BOOST_INCLUDE_DIR to the directory containing boost/version.hpp."
    )


def get_compile_flags():
    flags = []
    if os.getenv("FOUNDRY_DEBUG"):
        flags.append("-DFOUNDRY_DEBUG")
    flags.append("-fvisibility=hidden")
    return flags


def torch_requirement():
    """foundry.ops uses only the LibTorch Stable ABI (target MIN_TORCH): one build
    loads on any torch >= MIN_TORCH. Never pinned to the build torch."""
    return f"torch>={MIN_TORCH[0]}.{MIN_TORCH[1]}"


def write_build_info(boost_version):
    """Record the build torch/CUDA and the stable-ABI target for foundry._loader."""
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < MIN_TORCH:
        raise RuntimeError(
            f"building foundry.ops needs torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} (its stable-ABI "
            f"headers), found {torch.__version__}"
        )
    pkg = Path(ROOT_DIR) / "python" / "foundry"
    text = (pkg / "_build_info.py.in").read_text()
    for key, value in {
        "@TORCH_VERSION@": torch.__version__,
        "@TORCH_CUDA_VERSION@": torch.version.cuda or "",
        "@STABLE_ABI_TARGET@": f"{MIN_TORCH[0]}.{MIN_TORCH[1]}",
        "@BOOST_VERSION@": boost_version,
    }.items():
        text = text.replace(key, value)
    out = pkg / "_build_info.py"
    if not out.exists() or out.read_text() != text:
        out.write_text(text)


boost_include_dir = resolve_boost_include_dir()
if not SDIST:
    # The sdist ships the placeholder; the real values are written when it is built.
    write_build_info(_fmt_boost(_boost_version(boost_include_dir)))
if SDIST:
    ext_modules = []
    cmdclass = {}
else:
    # Restricted torch include tree: only the stable-ABI headers (see STABLE_HEADER_TREES).
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    stable_include = os.path.join(ROOT_DIR, "build", "stable_include")
    import shutil

    shutil.rmtree(stable_include, ignore_errors=True)
    for tree in STABLE_HEADER_TREES:
        shutil.copytree(os.path.join(torch_include, tree), os.path.join(stable_include, tree))
    cuda_include = os.path.join(CUDA_HOME, "include") if CUDA_HOME else None
    common_include_dirs = [
        stable_include,
        os.path.join(ROOT_DIR, "include"),
    ]
    if cuda_include:
        common_include_dirs.append(cuda_include)
    if boost_include_dir not in _IMPLICIT_INCLUDE_DIRS:
        common_include_dirs.append(boost_include_dir)
    common_library_dirs = library_paths(device_type="cuda")
    # Link -lcuda against the toolkit stub when present (build containers have no
    # driver); the runtime loads the real libcuda.so.1. Not added to the RPATH.
    if CUDA_HOME and os.path.isfile(os.path.join(CUDA_HOME, "lib64", "stubs", "libcuda.so")):
        common_library_dirs.append(os.path.join(CUDA_HOME, "lib64", "stubs"))

    # libcuda_hook.so sits next to libfoundry_ops.so ($ORIGIN). A source build also bakes the
    # CUDA toolkit library dir (cudart). Never torch's lib dir: foundry imports torch first,
    # so libtorch_cpu / libtorch_cuda resolve to whichever torch is installed.
    ops_rpaths = ["$ORIGIN"]
    if not WHEEL_BUILD:
        ops_rpaths += [
            p for p in library_paths(device_type="cuda") if CUDA_HOME and p.startswith(CUDA_HOME)
        ]

    class CustomBuildExt(BuildExtension):
        def build_extensions(self):
            # Build hook library using CMake
            build_dir = Path(self.build_lib) / "foundry"
            build_dir.mkdir(parents=True, exist_ok=True)

            cmake_build_dir = Path(ROOT_DIR) / "build"
            cmake_build_dir.mkdir(parents=True, exist_ok=True)

            # Run CMake
            subprocess.check_call(
                [
                    "cmake",
                    "-S",
                    ROOT_DIR,
                    "-B",
                    str(cmake_build_dir),
                    f"-DCMAKE_INSTALL_PREFIX={build_dir}",
                    f"-DFOUNDRY_BOOST_INCLUDE_DIR={boost_include_dir}",
                ]
                + (["-DCMAKE_CXX_FLAGS=-DFOUNDRY_DEBUG"] if os.getenv("FOUNDRY_DEBUG") else [])
            )

            # Build and install
            subprocess.check_call(["cmake", "--build", str(cmake_build_dir)])
            subprocess.check_call(["cmake", "--install", str(cmake_build_dir)])

            # Update ext_modules to link against the built hook library
            hook_lib_path = build_dir / "libcuda_hook.so"
            for ext in self.extensions:
                if ext.name == OPS_LIBRARY:
                    ext.extra_link_args.append(str(hook_lib_path))

            super().build_extensions()

        def get_ext_filename(self, fullname):
            # A plain shared library (no Python module init, no ABI suffix):
            # foundry/libfoundry_ops.so, loaded by torch.ops.load_library.
            # (called with the dotted name and with its last component alone)
            if fullname.split(".")[-1] == OPS_LIBRARY.split(".")[-1]:
                return os.path.join(*fullname.split(".")) + ".so"
            return super().get_ext_filename(fullname)

        def copy_extensions_to_source(self):
            super().copy_extensions_to_source()
            # Also copy the hook library to the source directory
            build_dir = Path(self.build_lib) / "foundry"
            src_dir = Path(ROOT_DIR) / "python" / "foundry"
            hook_lib = build_dir / "libcuda_hook.so"
            if hook_lib.exists():
                import shutil

                shutil.copy2(str(hook_lib), str(src_dir))

    class PlatformWheel(bdist_wheel):
        """py3-none-<platform>: the wheel's native code uses no Python ABI (torch custom
        ops through the LibTorch Stable ABI), so one wheel serves every CPython."""

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    ext_modules = [
        Extension(
            name=OPS_LIBRARY,
            sources=[
                "csrc/binding.cpp",
                "csrc/CUDAGraph.cpp",
                "csrc/CUDAGraphParallel.cpp",
                "csrc/BinaryGraphIO.cpp",
                "csrc/boost_json_src.cpp",
            ],
            include_dirs=common_include_dirs,
            library_dirs=common_library_dirs,
            language="c++",
            define_macros=[("USE_CUDA", "1"), ("TORCH_TARGET_VERSION", TORCH_TARGET_VERSION)],
            extra_compile_args={"cxx": ["-O3"] + get_compile_flags()},
            # Stable C shims: aoti_torch_*/torch_* in libtorch_cpu, the CUDA ones in libtorch_cuda.
            libraries=["torch_cpu", "torch_cuda", "cudart"],
            extra_link_args=["-lcuda"] + [f"-Wl,-rpath,{p}" for p in ops_rpaths],
        ),
    ]

    cmdclass = {"build_ext": CustomBuildExt, "bdist_wheel": PlatformWheel}

setup(
    # tomli: the integrations read their TOML config with tomllib, stdlib from 3.11.
    install_requires=[torch_requirement(), 'tomli>=1.1; python_version < "3.11"'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
