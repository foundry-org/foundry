# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_boost_paths():
    cmake_query_dir = Path(ROOT_DIR) / "build_boost_query"
    cmake_query_dir.mkdir(parents=True, exist_ok=True)

    cmake_script = cmake_query_dir / "CMakeLists.txt"
    cmake_script.write_text("""
cmake_minimum_required(VERSION 4.0)
project(boost_query LANGUAGES CXX)
find_package(Boost 1.83.0 CONFIG REQUIRED COMPONENTS filesystem json)

get_target_property(BOOST_FS_INCLUDE Boost::filesystem INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(BOOST_JSON_INCLUDE Boost::json INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(BOOST_FS_LOCATION Boost::filesystem LOCATION)
get_target_property(BOOST_JSON_LOCATION Boost::json LOCATION)

message("BOOST_INCLUDE_DIRS=${BOOST_FS_INCLUDE}")
get_filename_component(BOOST_LIB_DIR "${BOOST_FS_LOCATION}" DIRECTORY)
message("BOOST_LIBRARY_DIRS=${BOOST_LIB_DIR}")
""")

    result = subprocess.run(
        ["cmake", "-S", str(cmake_query_dir), "-B", str(cmake_query_dir)],
        capture_output=True,
        text=True,
    )

    include_dirs = []
    library_dirs = []

    for line in result.stderr.splitlines():
        if "BOOST_INCLUDE_DIRS=" in line:
            dirs = line.split("=", 1)[1].strip()
            if dirs:
                include_dirs = [d for d in dirs.split(";") if d]
        elif "BOOST_LIBRARY_DIRS=" in line:
            dirs = line.split("=", 1)[1].strip()
            if dirs:
                library_dirs = [d for d in dirs.split(";") if d]

    shutil.rmtree(cmake_query_dir, ignore_errors=True)

    return include_dirs, library_dirs


def get_compile_flags():
    flags = []
    if os.getenv("FOUNDRY_DEBUG"):
        flags.append("-DFOUNDRY_DEBUG")
    flags.append("-fvisibility=hidden")
    return flags


boost_include_dirs, boost_library_dirs = get_boost_paths()
common_include_dirs = (
    include_paths(device_type="cuda") + [os.path.join(ROOT_DIR, "include")] + boost_include_dirs
)
common_library_dirs = library_paths(device_type="cuda") + boost_library_dirs


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
            ]
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
            "-lboost_filesystem",
            "-lboost_json",
            "-Wl,-rpath,$ORIGIN",
        ]
        + [f"-Wl,-rpath,{p}" for p in library_paths(device_type="cuda")],
    ),
]

setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=ext_modules,
)
