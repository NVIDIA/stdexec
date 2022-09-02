cmake_minimum_required(VERSION 3.23.1)

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
project(P2300 VERSION 0.6.0 LANGUAGES CXX)

# Print CMake configuration
message(STATUS "System           : ${CMAKE_SYSTEM}")
message(STATUS "System name      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "System ver       : ${CMAKE_SYSTEM_VERSION}")
message(STATUS)

# Set the version and current build date
set(P2300_VERSION "${PROJECT_VERSION}")
set(P2300_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
string(TIMESTAMP P2300_BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP P2300_BUILD_YEAR "%Y")

message(STATUS "Library ver      : ${P2300_VERSION}")
message(STATUS "Build date       : ${P2300_BUILD_DATE}")
message(STATUS "Build year       : ${P2300_BUILD_YEAR}")
message(STATUS)

# Integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Write the version header
rapids_cmake_write_version_file(include/p2300_version_config.hpp)

# Set CMAKE_BUILD_TYPE=Release the default if none provided
rapids_cmake_build_type(Release)

# Set CMAKE_CXX_STANDARD=20 if none provided
if(NOT CMAKE_CXX_STANDARD)
  message(VERBOSE "Setting CXX standard to '20' since none specified.")
  set(CMAKE_CXX_STANDARD 20 CACHE STRING "Choose the C++ standard." FORCE)
endif()

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
  BUILD_EXPORT_SET p2300-exports
  CPM_ARGS
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v${Catch2_VERSION}.zip
)

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

##############################################################################
# - Main library targets -----------------------------------------------------

# Define the main library
add_library(p2300 INTERFACE)

# Set library version
set_target_properties(p2300 PROPERTIES
                      VERSION "${P2300_VERSION}"
                      SOVERSION "${P2300_VERSION_MAJOR}")

# Declare the public include directories
target_include_directories(p2300 INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>
                           )

target_link_libraries(p2300 INTERFACE Threads::Threads)

# Use C++20 standard
target_compile_features(p2300 INTERFACE cxx_std_20)

# Turn all warnings
target_compile_options(p2300 INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -Wall>
                       $<$<CXX_COMPILER_ID:MSVC>:
                       /W4>)
# template backtrace limit
target_compile_options(p2300 INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -ftemplate-backtrace-limit=0>
                       )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
    add_compile_options(-Wno-non-template-friend)
    add_compile_options(-fcoroutines)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
endif ()

add_library(P2300::p2300 ALIAS p2300)

# Now, set up test executable
enable_testing()

set(test_sourceFiles
    test/test_main.cpp
    test/cpos/test_cpo_bulk.cpp
    test/cpos/test_cpo_ensure_started.cpp
    test/cpos/test_cpo_receiver.cpp
    test/cpos/test_cpo_start.cpp
    test/cpos/test_cpo_connect.cpp
    test/cpos/test_cpo_schedule.cpp
    test/cpos/test_cpo_split.cpp
    test/cpos/test_cpo_upon_error.cpp
    test/cpos/test_cpo_upon_stopped.cpp
    test/concepts/test_concept_scheduler.cpp
    test/concepts/test_concepts_receiver.cpp
    test/concepts/test_concept_operation_state.cpp
    test/concepts/test_concepts_sender.cpp
    test/types/test_type_async_scope.cpp
    test/algos/factories/test_just.cpp
    test/algos/factories/test_transfer_just.cpp
    test/algos/factories/test_just_error.cpp
    test/algos/factories/test_just_stopped.cpp
    test/algos/adaptors/test_on.cpp
    test/algos/adaptors/test_transfer.cpp
    test/algos/adaptors/test_schedule_from.cpp
    test/algos/adaptors/test_then.cpp
    test/algos/adaptors/test_upon_error.cpp
    test/algos/adaptors/test_upon_stopped.cpp
    test/algos/adaptors/test_let_value.cpp
    test/algos/adaptors/test_let_error.cpp
    test/algos/adaptors/test_let_stopped.cpp
    test/algos/adaptors/test_bulk.cpp
    test/algos/adaptors/test_split.cpp
    test/algos/adaptors/test_when_all.cpp
    test/algos/adaptors/test_transfer_when_all.cpp
    test/algos/adaptors/test_into_variant.cpp
    test/algos/adaptors/test_stopped_as_optional.cpp
    test/algos/adaptors/test_stopped_as_error.cpp
    test/algos/adaptors/test_ensure_started.cpp
    test/algos/consumers/test_start_detached.cpp
    test/algos/consumers/test_sync_wait.cpp
    test/algos/other/test_execute.cpp
    test/detail/test_completion_signatures.cpp
    test/detail/test_create.cpp
    test/detail/test_utility.cpp
    test/queries/test_env.cpp
    test/queries/test_get_forward_progress_guarantee.cpp
    test/async_scope/test_dtor.cpp
    test/async_scope/test_spawn.cpp
    test/async_scope/test_spawn_future.cpp
    test/async_scope/test_empty.cpp
    test/async_scope/test_stop.cpp
    )

add_executable(test.P2300 ${test_sourceFiles})

target_include_directories(test.P2300 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test.P2300 PUBLIC P2300::p2300 Catch2::Catch2)

# Discover the Catch2 test built by the application
include(CTest)
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)
catch_discover_tests(test.P2300)

# Set up examples
function(def_example target sourceFile)
    add_executable(${target} ${sourceFile})
    target_link_libraries(${target} PRIVATE P2300::p2300)
endfunction()

def_example(clangd.helper "examples/_clangd_helper_file.cpp")
def_example(example.hello_world "examples/hello_world.cpp")
def_example(example.hello_coro "examples/hello_coro.cpp")
def_example(example.scope "examples/scope.cpp")
def_example(example.retry "examples/retry.cpp")
def_example(example.then "examples/then.cpp")
def_example(example.server_theme.let_value "examples/server_theme/let_value.cpp")
def_example(example.server_theme.on_transfer "examples/server_theme/on_transfer.cpp")

##############################################################################
# Install targets ------------------------------------------------------------

include(CPack)
include(GNUInstallDirs)

install(TARGETS p2300
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT p2300-exports)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/p2300_version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

##############################################################################
# Install exports ------------------------------------------------------------

set(code_string "")

# Install side of the export set
rapids_export(
  INSTALL p2300
  EXPORT_SET p2300-exports
  GLOBAL_TARGETS p2300
  NAMESPACE P2300::
  FINAL_CODE_BLOCK code_string
)

# Build side of the export set so a user can use the build dir as a CMake package root
rapids_export(
  BUILD p2300
  EXPORT_SET p2300-exports
  GLOBAL_TARGETS p2300
  NAMESPACE P2300::
  FINAL_CODE_BLOCK code_string
)
