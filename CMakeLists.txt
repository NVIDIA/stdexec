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
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.10/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
endif()
# Initialize rapids-cmake
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
# utilities for generating export set package metadata
include(rapids-export)
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
project(STDEXEC VERSION "0.${STD_EXECUTION_BS_REVISION}.0" LANGUAGES CXX)

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
rapids_cpm_init()

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

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

##############################################################################
# - Main library targets -----------------------------------------------------

# Define the main library
add_library(stdexec INTERFACE)

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
                       $<$<CXX_COMPILER_ID:GNU>:-fcoroutines>
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
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -ftemplate-backtrace-limit=0>
                       )

# Always enable colored output
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
                       -fcolor-diagnostics>
                       $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
                       )

# Set up CUDASchedulers library
option(STDEXEC_ENABLE_CUDA "Enable CUDA targets for non-nvc++ compilers" NO)

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE SCHED_SOURCES include/nvexec/*.cuh)
    add_library(CUDASchedulers INTERFACE ${SCHED_SOURCES})
    target_link_libraries(CUDASchedulers INTERFACE stdexec stdexec_executable_flags)
    target_include_directories(CUDASchedulers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(CUDASchedulers INTERFACE cxx_std_20)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(CUDASchedulers INTERFACE -x cu -Wno-unknown-cuda-version)
      foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers INTERFACE --cuda-gpu-arch=sm_${arch})
      endforeach()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
      target_compile_options(CUDASchedulers INTERFACE -stdpar -gpu=nomanaged)
      target_link_options(CUDASchedulers INTERFACE -stdpar -gpu=nomanaged)
      foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers INTERFACE -gpu=cc${arch})
      endforeach()
    endif()
endif()

# Now, set up test executable
enable_testing()

set(test_sourceFiles
    test/test_main.cpp
    test/stdexec/cpos/test_cpo_bulk.cpp
    test/stdexec/cpos/test_cpo_ensure_started.cpp
    test/stdexec/cpos/test_cpo_receiver.cpp
    test/stdexec/cpos/test_cpo_start.cpp
    test/stdexec/cpos/test_cpo_connect.cpp
    test/stdexec/cpos/test_cpo_schedule.cpp
    test/stdexec/cpos/test_cpo_split.cpp
    test/stdexec/cpos/test_cpo_upon_error.cpp
    test/stdexec/cpos/test_cpo_upon_stopped.cpp
    test/stdexec/concepts/test_concept_scheduler.cpp
    test/stdexec/concepts/test_concepts_receiver.cpp
    test/stdexec/concepts/test_concept_operation_state.cpp
    test/stdexec/concepts/test_concepts_sender.cpp
    test/stdexec/concepts/test_awaitables.cpp
    test/stdexec/algos/factories/test_just.cpp
    test/stdexec/algos/factories/test_transfer_just.cpp
    test/stdexec/algos/factories/test_just_error.cpp
    test/stdexec/algos/factories/test_just_stopped.cpp
    test/stdexec/algos/factories/test_read.cpp
    test/stdexec/algos/adaptors/test_on.cpp
    test/stdexec/algos/adaptors/test_transfer.cpp
    test/stdexec/algos/adaptors/test_schedule_from.cpp
    test/stdexec/algos/adaptors/test_then.cpp
    test/stdexec/algos/adaptors/test_upon_error.cpp
    test/stdexec/algos/adaptors/test_upon_stopped.cpp
    test/stdexec/algos/adaptors/test_let_value.cpp
    test/stdexec/algos/adaptors/test_let_error.cpp
    test/stdexec/algos/adaptors/test_let_stopped.cpp
    test/stdexec/algos/adaptors/test_bulk.cpp
    test/stdexec/algos/adaptors/test_split.cpp
    test/stdexec/algos/adaptors/test_when_all.cpp
    test/stdexec/algos/adaptors/test_transfer_when_all.cpp
    test/stdexec/algos/adaptors/test_into_variant.cpp
    test/stdexec/algos/adaptors/test_stopped_as_optional.cpp
    test/stdexec/algos/adaptors/test_stopped_as_error.cpp
    test/stdexec/algos/adaptors/test_ensure_started.cpp
    test/stdexec/algos/consumers/test_start_detached.cpp
    test/stdexec/algos/consumers/test_sync_wait.cpp
    test/stdexec/algos/other/test_execute.cpp
    test/stdexec/detail/test_completion_signatures.cpp
    test/stdexec/detail/test_utility.cpp
    test/stdexec/queries/test_get_forward_progress_guarantee.cpp
    test/stdexec/queries/test_forwarding_queries.cpp
    test/exec/test_type_async_scope.cpp
    test/exec/test_create.cpp
    test/exec/test_env.cpp
    test/exec/test_on.cpp
    test/exec/test_on2.cpp
    test/exec/test_on3.cpp
    test/exec/async_scope/test_dtor.cpp
    test/exec/async_scope/test_spawn.cpp
    test/exec/async_scope/test_spawn_future.cpp
    test/exec/async_scope/test_empty.cpp
    test/exec/async_scope/test_stop.cpp
    )

add_executable(test.stdexec ${test_sourceFiles})

target_include_directories(test.stdexec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test.stdexec PUBLIC STDEXEC::stdexec stdexec_executable_flags Catch2::Catch2)

# Discover the Catch2 test built by the application
include(CTest)
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)
catch_discover_tests(test.stdexec)

# Set up examples
function(def_example target sourceFile)
    add_executable(${target} ${sourceFile})
    target_link_libraries(${target} PRIVATE STDEXEC::stdexec stdexec_executable_flags)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

def_example(clangd.helper "examples/_clangd_helper_file.cpp")
def_example(example.hello_world "examples/hello_world.cpp")
def_example(example.hello_coro "examples/hello_coro.cpp")
def_example(example.scope "examples/scope.cpp")
def_example(example.retry "examples/retry.cpp")
def_example(example.then "examples/then.cpp")
def_example(example.server_theme.let_value "examples/server_theme/let_value.cpp")
def_example(example.server_theme.on_transfer "examples/server_theme/on_transfer.cpp")
def_example(example.server_theme.then_upon "examples/server_theme/then_upon.cpp")
def_example(example.server_theme.split_bulk "examples/server_theme/split_bulk.cpp")

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
  INSTALL stdexec
  EXPORT_SET stdexec-exports
  GLOBAL_TARGETS stdexec
  NAMESPACE STDEXEC::
  FINAL_CODE_BLOCK code_string
)

# Build side of the export set so a user can use the build dir as a CMake package root
rapids_export(
  BUILD stdexec
  EXPORT_SET stdexec-exports
  GLOBAL_TARGETS stdexec
  NAMESPACE STDEXEC::
  FINAL_CODE_BLOCK code_string
)

# TODO
add_subdirectory(examples/nvexec)
add_subdirectory(test/nvexec)
