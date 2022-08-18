cmake_minimum_required(VERSION 3.17.0)

# Define the project
project(P2300-dev LANGUAGES CXX)

if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else ()
    message(ERROR "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

# Set the version of the project
project(P2300-dev VERSION "${CONAN_PACKAGE_VERSION}")

# Print CMake configuration
message(STATUS "System           : ${CMAKE_SYSTEM}")
message(STATUS "System name      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "System ver       : ${CMAKE_SYSTEM_VERSION}")
message(STATUS "Conan compiler   : ${CONAN_SETTINGS_COMPILER} ${CONAN_SETTINGS_COMPILER_VERSION}")
message(STATUS "Conan libcxx     : ${CONAN_SETTINGS_COMPILER_LIBCXX}")
message(STATUS)

# The main library
add_library(P2300 INTERFACE)

# Set the version and current build date
set(P2300_VERSION "${PROJECT_VERSION}")
set(P2300_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
string(TIMESTAMP P2300_BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP P2300_BUILD_YEAR "%Y")

message(STATUS "Library ver      : ${P2300_VERSION}")
message(STATUS "Build date       : ${P2300_BUILD_DATE}")
message(STATUS "Build year       : ${P2300_BUILD_YEAR}")
message(STATUS)

# Set library version
set_target_properties(P2300 PROPERTIES
                      VERSION "${P2300_VERSION}"
                      SOVERSION "${P2300_VERSION_MAJOR}")

# Declare the public include directories
target_include_directories(P2300 INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>
                           )

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)
target_link_libraries(P2300 INTERFACE Threads::Threads)

# Use C++20 standard
target_compile_features(P2300 INTERFACE cxx_std_20)

# Turn all warnings
target_compile_options(P2300 INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -Wall>
                       $<$<CXX_COMPILER_ID:MSVC>:
                       /W4>)
# template backtrace limit
target_compile_options(P2300 INTERFACE
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -ftemplate-backtrace-limit=0>
                       )
# Proper use of libc++
if (CONAN_SETTINGS_COMPILER_LIBCXX STREQUAL "libc++")
    target_compile_options(P2300 INTERFACE -stdlib=libc++)
    target_link_options(P2300 INTERFACE -lc++)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
    add_compile_options(-Wno-non-template-friend)
    add_compile_options(-fcoroutines)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
endif ()

add_library(P2300::p2300 ALIAS P2300)


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
target_link_libraries(test.P2300 P2300)
target_link_libraries(test.P2300 CONAN_PKG::catch2)

# Discover the Catch2 test built by the application
include(CTest)
include(Catch)
catch_discover_tests(test.P2300)


# Set up examples
function(def_example target sourceFile)
    add_executable(${target} ${sourceFile})
    target_link_libraries(${target} P2300)
endfunction()

def_example(example.hello_world "examples/hello_world.cpp")
def_example(example.hello_coro "examples/hello_coro.cpp")
def_example(example.scope "examples/scope.cpp")
def_example(example.retry "examples/retry.cpp")
def_example(example.then "examples/then.cpp")
def_example(example.server_theme.let_value "examples/server_theme/let_value.cpp")
def_example(example.server_theme.on_transfer "examples/server_theme/on_transfer.cpp")
