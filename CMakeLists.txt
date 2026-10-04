cmake_minimum_required(VERSION 4.0)
project(cuda_hook LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

# Boost is used header-only (unordered, json via csrc/boost_json_src.cpp, crc,
# format); nothing links a compiled Boost library. Header source, in order:
#   1. FOUNDRY_BOOST_INCLUDE_DIR (setup.py passes the directory it resolved, so
#      libcuda_hook.so and foundry.ops always compile against the same Boost);
#   2. the vendored subset in third_party/boost (tools/release/vendor_boost.sh);
#   3. a system Boost >= 1.83 found through its CMake config files.
set(FOUNDRY_BOOST_INCLUDE_DIR "" CACHE PATH "Directory that contains boost/version.hpp")
set(FOUNDRY_VENDORED_BOOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost")
if(FOUNDRY_BOOST_INCLUDE_DIR)
    set(FOUNDRY_BOOST_INCLUDES "${FOUNDRY_BOOST_INCLUDE_DIR}")
    message(STATUS "Boost headers: ${FOUNDRY_BOOST_INCLUDE_DIR} (FOUNDRY_BOOST_INCLUDE_DIR)")
elseif(EXISTS "${FOUNDRY_VENDORED_BOOST_DIR}/boost/version.hpp")
    set(FOUNDRY_BOOST_INCLUDES "${FOUNDRY_VENDORED_BOOST_DIR}")
    message(STATUS "Boost headers: ${FOUNDRY_VENDORED_BOOST_DIR} (vendored)")
else()
    find_package(Boost 1.83.0 CONFIG REQUIRED)
    get_target_property(FOUNDRY_BOOST_INCLUDES Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Boost headers: ${FOUNDRY_BOOST_INCLUDES} (system Boost ${Boost_VERSION})")
endif()

add_library(cuda_hook SHARED
    csrc/hook.cpp
    csrc/boost_json_src.cpp
)

# The Boost.JSON definitions stay local to the preloaded hook (see the file).
set_source_files_properties(csrc/boost_json_src.cpp PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden"
)

target_include_directories(cuda_hook PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_include_directories(cuda_hook SYSTEM PRIVATE ${FOUNDRY_BOOST_INCLUDES})

target_link_libraries(cuda_hook PRIVATE
    # NOTE: cuda_hook should link against
    # neither CUDA driver or CUDA runtime
    # CUDA::cuda_driver
    dl
    pthread
)

if(DEFINED ENV{HOOK_DEBUG})
    message(STATUS "HOOK_DEBUG is defined, enabling debug mode")
    target_compile_definitions(cuda_hook PRIVATE HOOK_DEBUG)
endif()

set_target_properties(cuda_hook PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    OUTPUT_NAME "cuda_hook"
    PREFIX "lib"
    SUFFIX ".so"
)

install(TARGETS cuda_hook
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
