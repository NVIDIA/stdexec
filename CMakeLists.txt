cmake_minimum_required(VERSION 3.25.0)

if(POLICY CMP0135)
  # make the timestamps of ExternalProject_ADD match the download time
  # https://cmake.org/cmake/help/latest/policy/CMP0135.html
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

if(POLICY CMP0141)
  # Enables the use of the MSVC_DEBUG_INFORMATION_FORMAT target property
  # https://cmake.org/cmake/help/latest/policy/CMP0141.html
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0141 NEW)
endif()

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

# Fetch rapids-cmake
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.02/RAPIDS.cmake
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
  file(DOWNLOAD "https://raw.githubusercontent.com/cplusplus/sender-receiver/main/execution.bs"
      ${CMAKE_BINARY_DIR}/execution.bs)
endif()
file(STRINGS "${CMAKE_BINARY_DIR}/execution.bs" STD_EXECUTION_BS_REVISION_LINE REGEX "Revision: [0-9]+")
string(REGEX REPLACE "Revision: ([0-9]+)" "\\1" STD_EXECUTION_BS_REVISION ${STD_EXECUTION_BS_REVISION_LINE})

# nvc++ isn't supported by (s)ccache yet, so unset these before the `project()`
# call so CMake's CXX compiler detection doesn't throw attempting to use it
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/CompilerHelpers.cmake)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Don't build tests if configuring stdexec as a submodule of another
# CMake project, unless they explicitly set STDEXEC_BUILD_TESTS=TRUE,
# or they enabled CTest's BUILD_TESTING option
if ((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) OR BUILD_TESTING)
    set(STDEXEC_BUILD_TESTS_DEFAULT ON)
else()
    set(STDEXEC_BUILD_TESTS_DEFAULT OFF)
endif()
option(STDEXEC_BUILD_TESTS "Build stdexec tests" ${STDEXEC_BUILD_TESTS_DEFAULT})

# STDEXEC_BUILD_TESTS is used solely to configure CTest's BUILD_TESTING option,
# which is CMake's preferred option for enabling testing when using CTest.
set(BUILD_TESTING ${STDEXEC_BUILD_TESTS})

if (BUILD_TESTING)
  # CTest automatically calls enable_testing() if BUILD_TESTING is ON
  # https://cmake.org/cmake/help/latest/module/CTest.html#module:CTest
  include(CTest)
endif()

##############################################################################
# - Dependencies -------------------------------------------------------------

# Initialize CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

if (BUILD_TESTING)
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
endif()

# Add ICM
set(icm_VERSION 1.5.0)
# Always download it, don't attempt to do `find_package(ICM)` first
set(CPM_DOWNLOAD_icm TRUE)
rapids_cpm_find(icm ${icm_VERSION}
  CPM_ARGS
    GITHUB_REPOSITORY iboB/icm
    GIT_TAG v${icm_VERSION}
    VERSION ${icm_VERSION}
    PATCH_COMMAND git restore -- . && git apply ${CMAKE_CURRENT_LIST_DIR}/cmake/cpm/patches/icm/regex-build-error.diff
)

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
rapids_find_package(Threads REQUIRED
  BUILD_EXPORT_SET stdexec-exports
  INSTALL_EXPORT_SET stdexec-exports
)

##############################################################################
# - Main library targets -----------------------------------------------------

# Detect the compiler frontend (GNU, Clang, MSVC, etc.)
if(DEFINED CMAKE_CXX_COMPILER_FRONTEND_VARIANT)
  set(stdexec_compiler_frontend ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT})
else()
  set(stdexec_compiler_frontend ${CMAKE_CXX_COMPILER_ID})
endif()

set(stdexec_export_targets)

# Define the main library
add_library(stdexec INTERFACE)

target_sources(stdexec
PUBLIC
  FILE_SET headers
  TYPE HEADERS
  BASE_DIRS include
  FILES
    include/exec/__detail/__atomic_intrusive_queue.hpp
    include/exec/__detail/__atomic_ref.hpp
    include/exec/__detail/__basic_sequence.hpp
    include/exec/__detail/__bit_cast.hpp
    include/exec/__detail/__bwos_lifo_queue.hpp
    include/exec/__detail/__numa.hpp
    include/exec/__detail/__system_context_default_impl.hpp
    include/exec/__detail/__system_context_default_impl_entry.hpp
    include/exec/__detail/__system_context_replaceability_api.hpp
    include/exec/__detail/__xorshift.hpp
    include/exec/__detail/intrusive_heap.hpp
    include/exec/any_sender_of.hpp
    include/exec/async_scope.hpp
    include/exec/at_coroutine_exit.hpp
    include/exec/create.hpp
    include/exec/env.hpp
    include/exec/finally.hpp
    include/exec/inline_scheduler.hpp
    include/exec/into_tuple.hpp
    include/exec/just_from.hpp
    include/exec/libdispatch_queue.hpp
    include/exec/linux/__detail/memory_mapped_region.hpp
    include/exec/linux/__detail/safe_file_descriptor.hpp
    include/exec/linux/io_uring_context.hpp
    include/exec/linux/memory_mapped_region.hpp
    include/exec/linux/safe_file_descriptor.hpp
    include/exec/materialize.hpp
    include/exec/on.hpp
    include/exec/on_coro_disposition.hpp
    include/exec/repeat_effect_until.hpp
    include/exec/repeat_n.hpp
    include/exec/reschedule.hpp
    include/exec/scope.hpp
    include/exec/sequence.hpp
    include/exec/sequence/any_sequence_of.hpp
    include/exec/sequence/empty_sequence.hpp
    include/exec/sequence/ignore_all_values.hpp
    include/exec/sequence/iterate.hpp
    include/exec/sequence/transform_each.hpp
    include/exec/sequence_senders.hpp
    include/exec/single_thread_context.hpp
    include/exec/start_now.hpp
    include/exec/static_thread_pool.hpp
    include/exec/system_context.hpp
    include/exec/task.hpp
    include/exec/timed_scheduler.hpp
    include/exec/timed_thread_scheduler.hpp
    include/exec/trampoline_scheduler.hpp
    include/exec/variant_sender.hpp
    include/exec/when_any.hpp
    include/stdexec/__detail/__as_awaitable.hpp
    include/stdexec/__detail/__awaitable.hpp
    include/stdexec/__detail/__basic_sender.hpp
    include/stdexec/__detail/__bulk.hpp
    include/stdexec/__detail/__completion_signatures.hpp
    include/stdexec/__detail/__concepts.hpp
    include/stdexec/__detail/__config.hpp
    include/stdexec/__detail/__connect_awaitable.hpp
    include/stdexec/__detail/__continues_on.hpp
    include/stdexec/__detail/__cpo.hpp
    include/stdexec/__detail/__debug.hpp
    include/stdexec/__detail/__diagnostics.hpp
    include/stdexec/__detail/__domain.hpp
    include/stdexec/__detail/__ensure_started.hpp
    include/stdexec/__detail/__env.hpp
    include/stdexec/__detail/__execute.hpp
    include/stdexec/__detail/__execution_fwd.hpp
    include/stdexec/__detail/__force_include.hpp
    include/stdexec/__detail/__inline_scheduler.hpp
    include/stdexec/__detail/__into_variant.hpp
    include/stdexec/__detail/__intrusive_mpsc_queue.hpp
    include/stdexec/__detail/__intrusive_ptr.hpp
    include/stdexec/__detail/__intrusive_queue.hpp
    include/stdexec/__detail/__intrusive_slist.hpp
    include/stdexec/__detail/__just.hpp
    include/stdexec/__detail/__let.hpp
    include/stdexec/__detail/__manual_lifetime.hpp
    include/stdexec/__detail/__meta.hpp
    include/stdexec/__detail/__on.hpp
    include/stdexec/__detail/__operation_states.hpp
    include/stdexec/__detail/__optional.hpp
    include/stdexec/__detail/__ranges.hpp
    include/stdexec/__detail/__read_env.hpp
    include/stdexec/__detail/__receiver_adaptor.hpp
    include/stdexec/__detail/__receiver_ref.hpp
    include/stdexec/__detail/__receivers.hpp
    include/stdexec/__detail/__run_loop.hpp
    include/stdexec/__detail/__schedule_from.hpp
    include/stdexec/__detail/__schedulers.hpp
    include/stdexec/__detail/__scope.hpp
    include/stdexec/__detail/__sender_adaptor_closure.hpp
    include/stdexec/__detail/__sender_introspection.hpp
    include/stdexec/__detail/__senders.hpp
    include/stdexec/__detail/__senders_core.hpp
    include/stdexec/__detail/__shared.hpp
    include/stdexec/__detail/__spin_loop_pause.hpp
    include/stdexec/__detail/__split.hpp
    include/stdexec/__detail/__start_detached.hpp
    include/stdexec/__detail/__starts_on.hpp
    include/stdexec/__detail/__stop_token.hpp
    include/stdexec/__detail/__stopped_as_error.hpp
    include/stdexec/__detail/__stopped_as_optional.hpp
    include/stdexec/__detail/__submit.hpp
    include/stdexec/__detail/__sync_wait.hpp
    include/stdexec/__detail/__tag_invoke.hpp
    include/stdexec/__detail/__then.hpp
    include/stdexec/__detail/__transfer_just.hpp
    include/stdexec/__detail/__transform_completion_signatures.hpp
    include/stdexec/__detail/__transform_sender.hpp
    include/stdexec/__detail/__tuple.hpp
    include/stdexec/__detail/__type_traits.hpp
    include/stdexec/__detail/__upon_error.hpp
    include/stdexec/__detail/__upon_stopped.hpp
    include/stdexec/__detail/__utility.hpp
    include/stdexec/__detail/__variant.hpp
    include/stdexec/__detail/__when_all.hpp
    include/stdexec/__detail/__with_awaitable_senders.hpp
    include/stdexec/__detail/__write_env.hpp
    include/stdexec/concepts.hpp
    include/stdexec/coroutine.hpp
    include/stdexec/execution.hpp
    include/stdexec/functional.hpp
    include/stdexec/stop_token.hpp
  # stdexec_version_config.hpp is generated by raipds' script
  FILE_SET version_config
  TYPE HEADERS
  BASE_DIRS ${CMAKE_BINARY_DIR}/include
  FILES
    ${CMAKE_BINARY_DIR}/include/stdexec_version_config.hpp
)
list(APPEND stdexec_export_targets stdexec)

# Set library version
set_target_properties(stdexec PROPERTIES
                      VERSION "${STDEXEC_VERSION}"
                      SOVERSION "${STDEXEC_VERSION_MAJOR}")

# Declare the public include directories
include(GNUInstallDirs)

target_link_libraries(stdexec INTERFACE Threads::Threads)

# Use C++20 standard
target_compile_features(stdexec INTERFACE cxx_std_20)

# # Enable GPU compilation when using NVHPC compiler
# target_compile_options(stdexec INTERFACE
#                        $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=gpu>
#                        )
# target_link_options(stdexec INTERFACE
#                        $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-stdpar=gpu>
#                        )

# Enable coroutines for GCC
target_compile_options(stdexec INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fcoroutines>
                       )

# Increase the concepts diagnostics depth for GCC
target_compile_options(stdexec INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fconcepts-diagnostics-depth=10>
                       )

# Do you want a preprocessor that works? Picky, picky.
target_compile_options(stdexec INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus /Zc:preprocessor>
                       )

option(STDEXEC_ENABLE_EXTRA_TYPE_CHECKING "Enable extra type checking that is costly at compile-time" OFF)

if (STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)
    target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)
endif()

add_library(STDEXEC::stdexec ALIAS stdexec)

# Don't require building everything when installing
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

# Support target for examples and tests
add_library(stdexec_executable_flags INTERFACE)

# Enable warnings
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<STREQUAL:${stdexec_compiler_frontend},GNU>:-Wall>
                       $<$<STREQUAL:${stdexec_compiler_frontend},AppleClang>:-Wall>
                       $<$<STREQUAL:${stdexec_compiler_frontend},MSVC>:/W4>)

# Increase the error limit with NVC++
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<CXX_COMPILER_ID:NVHPC>:-e1000>)

# Silence warnings
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>
                       $<$<CXX_COMPILER_ID:NVHPC>:--diag_suppress177,550,111,497,554>
                       $<$<CXX_COMPILER_ID:MSVC>:/wd4100 /wd4101 /wd4127 /wd4324 /wd4456 /wd4459>)

# Template backtrace limit
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
                          $<$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>:/clang:>-ferror-limit=0
                          $<$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>:/clang:>-fmacro-backtrace-limit=0
                          $<$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>:/clang:>-ftemplate-backtrace-limit=0>
                       $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,23.3.0>>:
                       -ftemplate-backtrace-limit=0>)

# # Always enable colored output
# target_compile_options(stdexec_executable_flags INTERFACE
#                        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
#                        -fcolor-diagnostics>
#                        $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
#                        )

# Clang CUDA options
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
                       -Wno-unknown-cuda-version
                       -Xclang=-fcuda-allow-variadic-functions
                       -D_GLIBCXX_USE_TBB_PAR_BACKEND=0
                       -include stdexec/__detail/__force_include.hpp>
                       )

target_compile_definitions(
  stdexec_executable_flags INTERFACE
  $<$<NOT:$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>>:STDEXEC_ENABLE_EXTRA_TYPE_CHECKING>)

# Support target for examples and tests
add_library(nvexec_executable_flags INTERFACE)

target_compile_options(nvexec_executable_flags INTERFACE
                       $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-gpu=nomanaged>)
target_link_options(nvexec_executable_flags INTERFACE
                    $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-gpu=nomanaged>)

# Set up nvexec library
option(STDEXEC_ENABLE_CUDA "Enable CUDA targets for non-nvc++ compilers" OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(STDEXEC_ENABLE_CUDA ON)
    # Unset these if using nvc++
    disable_compiler(LANG CUDA)
endif()

if(STDEXEC_ENABLE_CUDA)

    add_library(nvexec INTERFACE)
    target_sources(nvexec
    PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS include
      FILES
        include/nvexec/detail/config.cuh
        include/nvexec/detail/cuda_atomic.cuh
        include/nvexec/detail/cuda_fwd.cuh
        include/nvexec/detail/memory.cuh
        include/nvexec/detail/queue.cuh
        include/nvexec/detail/throw_on_cuda_error.cuh
        include/nvexec/detail/variant.cuh
        include/nvexec/multi_gpu_context.cuh
        include/nvexec/nvtx.cuh
        include/nvexec/stream/algorithm_base.cuh
        include/nvexec/stream/bulk.cuh
        include/nvexec/stream/common.cuh
        include/nvexec/stream/continues_on.cuh
        include/nvexec/stream/ensure_started.cuh
        include/nvexec/stream/launch.cuh
        include/nvexec/stream/let_xxx.cuh
        include/nvexec/stream/reduce.cuh
        include/nvexec/stream/schedule_from.cuh
        include/nvexec/stream/split.cuh
        include/nvexec/stream/start_detached.cuh
        include/nvexec/stream/submit.cuh
        include/nvexec/stream/sync_wait.cuh
        include/nvexec/stream/then.cuh
        include/nvexec/stream/upon_error.cuh
        include/nvexec/stream/upon_stopped.cuh
        include/nvexec/stream/when_all.cuh
        include/nvexec/stream_context.cuh
    )
    list(APPEND stdexec_export_targets nvexec)
    add_library(STDEXEC::nvexec ALIAS nvexec)

    target_compile_features(nvexec INTERFACE cuda_std_20)
    target_link_libraries(nvexec INTERFACE STDEXEC::stdexec)

    target_compile_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>)
    target_link_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>)

    install(TARGETS nvexec
        EXPORT stdexec-exports
        FILE_SET headers)

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

        include(${rapids-cmake-dir}/cpm/cccl.cmake)
        rapids_cpm_cccl(
            BUILD_EXPORT_SET stdexec-exports
            INSTALL_EXPORT_SET stdexec-exports
            SYSTEM TRUE)

        target_link_libraries(stdexec INTERFACE CCCL::CCCL)
    endif ()
endif ()


option(STDEXEC_ENABLE_TBB "Enable TBB targets" OFF)

if (STDEXEC_ENABLE_TBB)
    include(rapids-find)
    rapids_find_package(
        TBB REQUIRED
        BUILD_EXPORT_SET stdexec-exports
        INSTALL_EXPORT_SET stdexec-exports
    )

    add_library(tbbexec INTERFACE)
    target_sources(tbbexec
    PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS include
      FILES
        include/tbbexec/tbb_thread_pool.hpp
    )
    list(APPEND stdexec_export_targets tbbexec)
    add_library(STDEXEC::tbbexec ALIAS tbbexec)

    target_link_libraries(tbbexec
        INTERFACE
        STDEXEC::stdexec
        TBB::tbb
        )

    install(TARGETS tbbexec
        EXPORT stdexec-exports
        FILE_SET headers)
endif()

include(CheckIncludeFileCXX)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  CHECK_INCLUDE_FILE_CXX("dispatch/dispatch.h" STDEXEC_FOUND_LIBDISPATCH)
  option(STDEXEC_ENABLE_LIBDISPATCH "Enable the tests for the Grand Central Dispatch scheduler" ${STDEXEC_FOUND_LIBDISPATCH})
endif()

option (STDEXEC_ENABLE_NUMA "Enable NUMA affinity for static_thread_pool" OFF)
if (STDEXEC_ENABLE_NUMA)
  find_package(numa REQUIRED)
  target_link_libraries(stdexec INTERFACE numa::numa)
  target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_NUMA)
endif()

set(SYSTEM_CONTEXT_SOURCES src/system_context/system_context.cpp)
add_library(system_context STATIC ${SYSTEM_CONTEXT_SOURCES})
target_compile_features(system_context PUBLIC cxx_std_20)
set_target_properties(system_context PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_compile_options(system_context PUBLIC
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus /Zc:preprocessor>
    )
add_library(STDEXEC::system_context ALIAS system_context)
target_link_libraries(system_context PUBLIC stdexec)


if(CMAKE_CROSSCOMPILING)
  CHECK_INCLUDE_FILE_CXX("linux/io_uring.h" STDEXEC_FOUND_IO_URING)
else()
  include(CheckCXXSourceRuns)
  CHECK_CXX_SOURCE_RUNS(
    "
    #include <linux/io_uring.h>
    #include <sys/syscall.h>
    #include <unistd.h>

    #include <cstdlib>

    int main()
    {
      io_uring_params a = {};
      return syscall(__NR_io_uring_setup, 1, &a) != -1
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
    }
    "
    STDEXEC_FOUND_IO_URING)
endif()
option(STDEXEC_ENABLE_IO_URING_TESTS "Enable io_uring tests" ${STDEXEC_FOUND_IO_URING})

option(STDEXEC_BUILD_DOCS "Build stdexec documentation" OFF)
option(STDEXEC_BUILD_EXAMPLES "Build stdexec examples" ON)

# Configure documentation
if(STDEXEC_BUILD_DOCS)
    add_subdirectory(docs)
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

install(TARGETS stdexec system_context
	EXPORT stdexec-exports
	FILE_SET headers
	FILE_SET version_config)

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
