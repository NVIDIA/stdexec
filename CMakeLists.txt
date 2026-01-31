cmake_minimum_required(VERSION 3.25.0)

set(ENV{SCCACHE_NO_DIST_COMPILE} "1")

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

if(POLICY CMP0167)
  # Don't try to use old FindBoost.cmake
  # https://cmake.org/cmake/help/latest/policy/CMP0167.html
  cmake_policy(SET CMP0167 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

# Fetch rapids-cmake
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.02/RAPIDS.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
endif()
# Initialize rapids-cmake
include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
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
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/execution.bs)
  file(DOWNLOAD "https://raw.githubusercontent.com/cplusplus/sender-receiver/main/execution.bs"
      ${CMAKE_CURRENT_BINARY_DIR}/execution.bs)
endif()
file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/execution.bs" STD_EXECUTION_BS_REVISION_LINE REGEX "Revision: [0-9]+")
string(REGEX REPLACE "Revision: ([0-9]+)" "\\1" STD_EXECUTION_BS_REVISION ${STD_EXECUTION_BS_REVISION_LINE})

project(STDEXEC VERSION "0.${STD_EXECUTION_BS_REVISION}.0" LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/CompilerHelpers.cmake)

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

file(GLOB_RECURSE exec_headers CONFIGURE_DEPENDS include/exec/*.hpp)
file(GLOB_RECURSE stdexec_headers CONFIGURE_DEPENDS include/stdexec/*.hpp)
target_sources(stdexec
PUBLIC
  FILE_SET headers
  TYPE HEADERS
  BASE_DIRS include
  FILES
    ${exec_headers}
    ${stdexec_headers}
    include/execpools/thread_pool_base.hpp
  # stdexec_version_config.hpp is generated by rapids' script
  FILE_SET version_config
  TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/stdexec_version_config.hpp
)
list(APPEND stdexec_export_targets stdexec)

# Set library version
set_target_properties(stdexec PROPERTIES
                      VERSION "${STDEXEC_VERSION}"
                      SOVERSION "${STDEXEC_VERSION_MAJOR}")

if (BUILD_TESTING)
  # Test that headers are self-contained
  set_target_properties(stdexec PROPERTIES
                        VERIFY_INTERFACE_HEADER_SETS TRUE)
endif()

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

set(STDEXEC_NAMESPACE "stdexec" CACHE STRING "The name of the top-level namespace for stdexec")
if (NOT STDEXEC_NAMESPACE STREQUAL "stdexec")
  target_compile_definitions(stdexec INTERFACE STDEXEC_NAMESPACE=${STDEXEC_NAMESPACE})
endif()

option(STDEXEC_ENABLE_EXTRA_TYPE_CHECKING "Enable extra type checking that is costly at compile-time" OFF)
if (STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)
  target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)
endif()

add_library(STDEXEC::stdexec ALIAS stdexec)

# Support target for examples and tests
add_library(stdexec_executable_flags INTERFACE)

# Enable warnings
target_compile_options(stdexec_executable_flags INTERFACE
                       $<$<STREQUAL:${stdexec_compiler_frontend},GNU>:-Wall;-Werror=unused-parameter>
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

if (STDEXEC_IS_TOP_LEVEL)
  # Integrate with LLVM/clang tooling
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/clangd_compile_info.cmake)
  add_executable(_clangd_helper_file include/._clangd_helper_file.cpp)
  target_link_libraries(_clangd_helper_file PRIVATE STDEXEC::stdexec)
endif()

# Set up nvexec library
option(STDEXEC_ENABLE_CUDA "Enable CUDA targets for non-nvc++ compilers" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(STDEXEC_ENABLE_CUDA ON)
    # Unset these if using nvc++
    disable_compiler(LANG CUDA)

    set(_nvhpc_seperate_memory_flags "-gpu=mem:separate")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 24.5.0)
        set(_nvhpc_seperate_memory_flags "-gpu=nomanaged")
    endif()
endif()

# Support target for examples and tests
add_library(nvexec_executable_flags INTERFACE)

target_compile_options(nvexec_executable_flags INTERFACE
                       $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:${_nvhpc_seperate_memory_flags}>)
target_link_options(nvexec_executable_flags INTERFACE
                    $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:${_nvhpc_seperate_memory_flags}>)

if(STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE nvexec_headers CONFIGURE_DEPENDS include/nvexec/*.cuh)
    add_library(nvexec INTERFACE)
    target_sources(nvexec
    PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS include
      FILES ${nvexec_headers}
    )
    list(APPEND stdexec_export_targets nvexec)
    add_library(STDEXEC::nvexec ALIAS nvexec)

    target_compile_features(nvexec INTERFACE cuda_std_20)
    target_link_libraries(nvexec INTERFACE STDEXEC::stdexec)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    set(_gpus)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
      if(_arch MATCHES "-real")
        string(REPLACE "-real" "" _arch "${_arch}")
        string(PREPEND _arch "cc")
      elseif(_arch MATCHES "-virtual")
        string(REPLACE "-virtual" "" _arch "${_arch}")
        string(PREPEND _arch "cc")
      elseif(_arch MATCHES "^[0-9]+$")
        string(PREPEND _arch "cc")
      elseif(_arch MATCHES "^native$")
        string(PREPEND _arch "cc")
      endif()
      list(APPEND _gpus "${_arch}")
    endforeach()
    list(JOIN _gpus "," _gpus)

    target_compile_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=${_gpus}>)
    target_link_options(nvexec INTERFACE
      $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=${_gpus}>)

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

include(ConfigureTBB)
include(ConfigureTaskflow)
include(ConfigureASIO)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  check_include_file_cxx("dispatch/dispatch.h" STDEXEC_FOUND_LIBDISPATCH)
  option(STDEXEC_ENABLE_LIBDISPATCH "Enable use of the Grand Central Dispatch scheduler" ${STDEXEC_FOUND_LIBDISPATCH})
  target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_LIBDISPATCH)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  check_include_file_cxx("windows.h" STDEXEC_FOUND_WINDOWS_HEADER)
  option(STDEXEC_ENABLE_WINDOWS_THREAD_POOL "Enable use of the Windows Thread Pool scheduler" ${STDEXEC_FOUND_WINDOWS_HEADER})
  target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_WINDOWS_THREAD_POOL)
endif()

option (STDEXEC_ENABLE_NUMA "Enable NUMA affinity for static_thread_pool" OFF)
if (STDEXEC_ENABLE_NUMA)
  find_package(numa REQUIRED)
  target_link_libraries(stdexec INTERFACE numa::numa)
  target_compile_definitions(stdexec INTERFACE STDEXEC_ENABLE_NUMA)
endif()

set(SYSTEM_CONTEXT_SOURCES src/system_context/system_context.cpp)
add_library(system_context ${SYSTEM_CONTEXT_SOURCES})
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


option(STDEXEC_ENABLE_IO_URING "Enable the use of the io_uring scheduler on Linux" OFF)

if(STDEXEC_ENABLE_IO_URING)
  if(CMAKE_CROSSCOMPILING)
    check_include_file_cxx("linux/io_uring.h" STDEXEC_ENABLE_IO_URING)
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
      STDEXEC_ENABLE_IO_URING)
  endif()
endif()

option(STDEXEC_BUILD_DOCS "Build stdexec documentation" OFF)
option(STDEXEC_BUILD_EXAMPLES "Build stdexec examples" ON)

# Configure documentation
if(STDEXEC_BUILD_DOCS)
    add_subdirectory(docs EXCLUDE_FROM_ALL)
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
