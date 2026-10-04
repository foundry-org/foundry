# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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


boost_include_dir = resolve_boost_include_dir()
common_include_dirs = include_paths(device_type="cuda") + [os.path.join(ROOT_DIR, "include")]
if boost_include_dir not in _IMPLICIT_INCLUDE_DIRS:
    common_include_dirs.append(boost_include_dir)
common_library_dirs = library_paths(device_type="cuda")


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
            if ext.name == "foundry.ops":
                ext.extra_link_args.append(str(hook_lib_path))

        super().build_extensions()

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        # Also copy the hook library to the source directory
        build_dir = Path(self.build_lib) / "foundry"
        src_dir = Path(ROOT_DIR) / "python" / "foundry"
        hook_lib = build_dir / "libcuda_hook.so"
        if hook_lib.exists():
            import shutil

            shutil.copy2(str(hook_lib), str(src_dir))


ext_modules = [
    CUDAExtension(
        name="foundry.ops",
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
        extra_compile_args={
            "cxx": ["-O3"] + get_compile_flags(),
            "nvcc": ["-O3"] + get_compile_flags(),
        },
        extra_link_args=[
            "-lcuda",
            "-Wl,-rpath,$ORIGIN",
        ]
        + [f"-Wl,-rpath,{p}" for p in library_paths(device_type="cuda")],
    ),
]

setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=ext_modules,
)
