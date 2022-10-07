cmake_minimum_required(VERSION 3.22.1)

if(POLICY CMP0135)
  # make the timestamps of ExternalProject_ADD match the download time
  # https://cmake.org/cmake/help/latest/policy/CMP0135.html
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

# Fetch rapids-cmake
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.02/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
endif()
# Initialize rapids-cmake
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
# utilities for generating export set package metadata
include(rapids-export)
# utilities for finding packages
include(rapids-find)
# utilities for defining project defaults
include(rapids-cmake)
# utilities for using CPM
include(rapids-cpm)

##############################################################################
# - Project definition -------------------------------------------------------

# Define the project and set the version and languages
if(NOT EXISTS ${CMAKE_BINARY_DIR}/execution.bs)
  file(DOWNLOAD "https://raw.githubusercontent.com/brycelelbach/wg21_p2300_execution/main/execution.bs"
      ${CMAKE_BINARY_DIR}/execution.bs)
endif()
file(STRINGS "${CMAKE_BINARY_DIR}/execution.bs" STD_EXECUTION_BS_REVISION_LINE REGEX "Revision: [0-9]+")
string(REGEX REPLACE "Revision: ([0-9]+)" "\\1" STD_EXECUTION_BS_REVISION ${STD_EXECUTION_BS_REVISION_LINE})

# nvc++ isn't supported by (s)ccache yet, so unset these before the `project()`
# call so CMake's CXX compiler detection doesn't throw attempting to use it
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/CompilerHelpers.cmake)

disable_compiler_launcher(LANG CXX VAR _cxx_launcher)
disable_compiler_launcher(LANG CUDA VAR _cuda_launcher)

project(STDEXEC VERSION "0.${STD_EXECUTION_BS_REVISION}.0" LANGUAGES CXX)

enable_compiler_launcher(LANG CXX VAR _cxx_launcher)
enable_compiler_launcher(LANG CUDA VAR _cuda_launcher)

# Print CMake configuration
message(STATUS "System           : ${CMAKE_SYSTEM}")
message(STATUS "System name      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "System ver       : ${CMAKE_SYSTEM_VERSION}")
message(STATUS)

# Set the version and current build date
set(STDEXEC_VERSION "${PROJECT_VERSION}")
set(STDEXEC_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
string(TIMESTAMP STDEXEC_BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP STDEXEC_BUILD_YEAR "%Y")

message(STATUS "Library ver      : ${STDEXEC_VERSION}")
message(STATUS "Build date       : ${STDEXEC_BUILD_DATE}")
message(STATUS "Build year       : ${STDEXEC_BUILD_YEAR}")
message(STATUS)

# Integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Write the version header
rapids_cmake_write_version_file(include/stdexec_version_config.hpp)

# Set CMAKE_BUILD_TYPE=Release the default if none provided
rapids_cmake_build_type(Release)

##############################################################################
# - Dependencies -------------------------------------------------------------

# Initialize CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

# Add Catch2
set(Catch2_VERSION 2.13.6)
# Always download it, don't attempt to do `find_package(Catch2)` first
set(CPM_DOWNLOAD_Catch2 TRUE)
rapids_cpm_find(Catch2 ${Catch2_VERSION}
  GLOBAL_TARGETS Catch2::Catch2
  BUILD_EXPORT_SET stdexec-exports
  CPM_ARGS
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v${Catch2_VERSION}.zip
)

# Add range-v3
set(range-v3_VERSION 0.12.0)
# Always download it, don't attempt to do `find_package(range-v3)` first
set(CPM_DOWNLOAD_range-v3 TRUE)
rapids_cpm_find(range-v3 ${range-v3_VERSION}
  GLOBAL_TARGETS range-v3::range-v3
  BUILD_EXPORT_SET stdexec-exports
  CPM_ARGS
    URL https://github.com/ericniebler/range-v3/archive/refs/tags/${range-v3_VERSION}.zip
)

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
rapids_find_package(Threads REQUIRED
  BUILD_EXPORT_SET stdexec-exports
  INSTALL_EXPORT_SET stdexec-exports
)

##############################################################################
# - Main library targets -----------------------------------------------------

set(stdexec_export_targets)

# Define the main library
add_library(stdexec INTERFACE)

list(APPEND stdexec_export_targets stdexec)

# Set library version
set_target_properties(stdexec PROPERTIES
                      VERSION "${STDEXEC_VERSION}"
                      SOVERSION "${STDEXEC_VERSION_MAJOR}")

# Declare the public include directories
target_include_directories(stdexec INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                           )

target_link_libraries(stdexec INTERFACE Threads::Threads)

# Use C++20 standard
target_compile_features(stdexec INTERFACE cxx_std_20)

# Enable coroutines for GCC
target_compile_options(stdexec INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fcoroutines>
                       )

add_library(STDEXEC::stdexec ALIAS stdexec)

# Don't require building everything when installing
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

# Support target for examples and tests
add_library(stdexec_executable_flags INTERFACE)

# Enable warnings
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -Wall>
                       $<$<CXX_COMPILER_ID:MSVC>:
                       /W4>)

# Increase the error limit with NVC++
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<CXX_COMPILER_ID:NVHPC>:-e1000>
                       )

# Silence warnings with GCC
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>
                       )

# Silence warnings with NVHPC
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<CXX_COMPILER_ID:NVHPC>:--diag_suppress177,550,111,497,554>
                       )

# Template backtrace limit
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
                       -ferror-limit=0
                       -fmacro-backtrace-limit=0
                       -ftemplate-backtrace-limit=0>
                       )

# Always enable colored output
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
                       -fcolor-diagnostics>
                       $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
                       )

# Compilation timing
option(STDEXEC_ENABLE_COMPILATION_TIMING "Enable compilation timing" OFF)
if(STDEXEC_ENABLE_COMPILATION_TIMING)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "/usr/bin/time -f '%e'")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "/usr/bin/time -f '%e'")
endif()

# Clang CUDA options
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
                       -Wno-unknown-cuda-version
                       -Xclang=-fcuda-allow-variadic-functions>
                       )

# Set up nvexec library
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  option(STDEXEC_ENABLE_CUDA "Enable CUDA targets" ON)
  # Unset these if using nvc++
  disable_compiler(LANG CUDA)
  disable_compiler_launcher(LANG CXX)
  disable_compiler_launcher(LANG CUDA)
else()
  option(STDEXEC_ENABLE_CUDA "Enable CUDA targets" OFF)
endif()

if(STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE nvexec_sources include/nvexec/*.cuh)
    add_library(nvexec INTERFACE ${nvexec_sources})
    list(APPEND stdexec_export_targets nvexec)
    add_library(STDEXEC::nvexec ALIAS nvexec)

    target_compile_features(nvexec INTERFACE cuda_std_20)
    target_link_libraries(nvexec INTERFACE STDEXEC::stdexec)

    target_compile_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=managed>)
    target_link_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=managed>)

    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
        include(rapids-cuda)
        # Needs to run before `enable_language(CUDA)`
        rapids_cuda_init_architectures(STDEXEC)
        enable_language(CUDA)
        # Since stdexec only enables CUDA optionally we need to manually include
        # the file that rapids_cuda_init_architectures relies on `project` calling
        if(CMAKE_PROJECT_STDEXEC_INCLUDE)
            include("${CMAKE_PROJECT_STDEXEC_INCLUDE}")
        endif()
        # Must come after enable_language(CUDA)
        # Use `-isystem <path>` instead of `-isystem=<path>`
        # because the former works with clangd intellisense
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

        set_source_files_properties(${nvexec_sources} PROPERTIES LANGUAGE CUDA)

        include(rapids-find)
        rapids_find_package(
            CUDAToolkit REQUIRED
            BUILD_EXPORT_SET stdexec-exports
            INSTALL_EXPORT_SET stdexec-exports
        )

        target_link_libraries(nvexec INTERFACE CUDA::cudart)

        include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
        rapids_cpm_libcudacxx(
            BUILD_EXPORT_SET stdexec-exports
            INSTALL_EXPORT_SET stdexec-exports)

        target_link_libraries(stdexec INTERFACE libcudacxx::libcudacxx)
    endif()
endif()

option(STDEXEC_BUILD_EXAMPLES "Build stdexec examples" ON)
option(STDEXEC_BUILD_TESTS "Build stdexec tests" ON)
option(BUILD_TESTING "" ${STDEXEC_BUILD_TESTS})

# Don't build tests if configuring stdexec as a submodule of another
# CMake project, unless they explicitly set STDEXEC_BUILD_TESTS=TRUE
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) OR STDEXEC_BUILD_TESTS)
    # CTest automatically calls enable_testing() if BUILD_TESTING is ON
    # https://cmake.org/cmake/help/latest/module/CTest.html#module:CTest
    include(CTest)
endif()

# Configure test executables
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

# Configure example executables
if(STDEXEC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

##############################################################################
# Install targets ------------------------------------------------------------

include(CPack)
include(GNUInstallDirs)

install(TARGETS stdexec
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT stdexec-exports)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/stdexec_version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

##############################################################################
# Install exports ------------------------------------------------------------

set(code_string "")

# Install side of the export set
rapids_export(
  INSTALL ${stdexec_export_targets}
  EXPORT_SET stdexec-exports
  GLOBAL_TARGETS ${stdexec_export_targets}
  NAMESPACE STDEXEC::
  FINAL_CODE_BLOCK code_string
)

# Build side of the export set so a user can use the build dir as a CMake package root
rapids_export(
  BUILD ${stdexec_export_targets}
  EXPORT_SET stdexec-exports
  GLOBAL_TARGETS ${stdexec_export_targets}
  NAMESPACE STDEXEC::
  FINAL_CODE_BLOCK code_string
)
