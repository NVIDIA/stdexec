#=============================================================================
# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

set(stdexec_test_sources
    test_main.cpp
    stdexec/cpos/test_cpo_bulk.cpp
    stdexec/cpos/test_cpo_ensure_started.cpp
    stdexec/cpos/test_cpo_receiver.cpp
    stdexec/cpos/test_cpo_start.cpp
    stdexec/cpos/test_cpo_connect.cpp
    stdexec/cpos/test_cpo_schedule.cpp
    stdexec/cpos/test_cpo_split.cpp
    stdexec/cpos/test_cpo_upon_error.cpp
    stdexec/cpos/test_cpo_upon_stopped.cpp
    stdexec/concepts/test_concept_scheduler.cpp
    stdexec/concepts/test_concepts_receiver.cpp
    stdexec/concepts/test_concept_operation_state.cpp
    stdexec/concepts/test_concepts_sender.cpp
    stdexec/concepts/test_concepts_stop_tokens.cpp
    stdexec/concepts/test_concepts_scope_association.cpp
    stdexec/concepts/test_concepts_scope_token.cpp
    stdexec/concepts/test_awaitables.cpp
    stdexec/algos/factories/test_just.cpp
    stdexec/algos/factories/test_transfer_just.cpp
    stdexec/algos/factories/test_just_error.cpp
    stdexec/algos/factories/test_just_stopped.cpp
    stdexec/algos/factories/test_read.cpp
    stdexec/algos/factories/test_schedule.cpp
    stdexec/algos/adaptors/test_associate.cpp
    stdexec/algos/adaptors/test_starts_on.cpp
    stdexec/algos/adaptors/test_on.cpp
    stdexec/algos/adaptors/test_on2.cpp
    stdexec/algos/adaptors/test_on3.cpp
    stdexec/algos/adaptors/test_continues_on.cpp
    stdexec/algos/adaptors/test_then.cpp
    stdexec/algos/adaptors/test_upon_error.cpp
    stdexec/algos/adaptors/test_upon_stopped.cpp
    stdexec/algos/adaptors/test_let_value.cpp
    stdexec/algos/adaptors/test_let_error.cpp
    stdexec/algos/adaptors/test_let_stopped.cpp
    stdexec/algos/adaptors/test_bulk.cpp
    stdexec/algos/adaptors/test_split.cpp
    stdexec/algos/adaptors/test_when_all.cpp
    stdexec/algos/adaptors/test_transfer_when_all.cpp
    stdexec/algos/adaptors/test_into_variant.cpp
    stdexec/algos/adaptors/test_stopped_as_optional.cpp
    stdexec/algos/adaptors/test_stopped_as_error.cpp
    stdexec/algos/adaptors/test_ensure_started.cpp
    stdexec/algos/adaptors/test_write_env.cpp
    stdexec/algos/adaptors/test_stop_when.cpp
    stdexec/algos/adaptors/test_spawn_future.cpp
    stdexec/algos/consumers/test_start_detached.cpp
    stdexec/algos/consumers/test_sync_wait.cpp
    stdexec/algos/consumers/test_spawn.cpp
    stdexec/algos/other/test_execute.cpp
    stdexec/detail/test_any.cpp
    stdexec/detail/test_completion_signatures.cpp
    stdexec/detail/test_utility.cpp
    stdexec/schedulers/test_task_scheduler.cpp
    stdexec/queries/test_env.cpp
    stdexec/queries/test_get_forward_progress_guarantee.cpp
    stdexec/queries/test_forwarding_queries.cpp
    stdexec/types/test_task.cpp
    stdexec/types/test_counting_scopes.cpp
    )

add_library(common_test_settings INTERFACE)
set_target_properties(common_test_settings PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    MSVC_DEBUG_INFORMATION_FORMAT Embedded
    )
target_include_directories(common_test_settings INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(common_test_settings INTERFACE STDEXEC_NAMESPACE=std::execution)
# target_compile_definitions(
#     common_test_settings INTERFACE
#     $<$<NOT:$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>>:STDEXEC_ENABLE_EXTRA_TYPE_CHECKING>)

add_executable(test.stdexec ${stdexec_test_sources})
target_link_libraries(test.stdexec
    PUBLIC
    STDEXEC::stdexec
    stdexec_executable_flags
    Catch2::Catch2
    PRIVATE
    common_test_settings)

add_executable(test.scratch test_main.cpp test_scratch.cpp)
target_link_libraries(test.scratch
    PUBLIC
    STDEXEC::stdexec
    $<TARGET_NAME_IF_EXISTS:STDEXEC::tbbpool>
    stdexec_executable_flags
    Catch2::Catch2
    PRIVATE
    common_test_settings)

# Discover the Catch2 test built by the application
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)

# For testing that builds fail as expected
include(${icm_SOURCE_DIR}/icm_build_failure_testing.cmake)

catch_discover_tests(test.stdexec)
catch_discover_tests(test.scratch)

add_subdirectory(exec)

if(STDEXEC_ENABLE_CUDA)
    add_subdirectory(nvexec)
endif()

if(STDEXEC_ENABLE_ASIO)
    add_subdirectory(asioexec)
endif()

# build failure tests: tests which parse a source file for an expected error
# message
icm_add_build_failure_test(
    NAME test_let_fail1
    TARGET test_let_fail1
    SOURCES PARSE stdexec/algos/adaptors/test_let_fail1.cpp
    LIBRARIES stdexec
    FOLDER test
)

icm_add_build_failure_test(
    NAME test_then_fail1
    TARGET test_then_fail1
    SOURCES PARSE stdexec/algos/adaptors/test_then_fail1.cpp
    LIBRARIES stdexec
    FOLDER test
)

# # Adding multiple tests with a glob
# icm_glob_build_failure_tests(
#     PATTERN *_fail*.cpp
#     LIBRARIES stdexec
#     PREFIX stdexec
#     FOLDER test/stdexec/algos/adaptors
# )
