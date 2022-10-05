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
    stdexec/concepts/test_awaitables.cpp
    stdexec/algos/factories/test_just.cpp
    stdexec/algos/factories/test_transfer_just.cpp
    stdexec/algos/factories/test_just_error.cpp
    stdexec/algos/factories/test_just_stopped.cpp
    stdexec/algos/factories/test_read.cpp
    stdexec/algos/adaptors/test_on.cpp
    stdexec/algos/adaptors/test_transfer.cpp
    stdexec/algos/adaptors/test_schedule_from.cpp
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
    stdexec/algos/consumers/test_start_detached.cpp
    stdexec/algos/consumers/test_sync_wait.cpp
    stdexec/algos/other/test_execute.cpp
    stdexec/detail/test_completion_signatures.cpp
    stdexec/detail/test_utility.cpp
    stdexec/queries/test_get_forward_progress_guarantee.cpp
    stdexec/queries/test_forwarding_queries.cpp
    exec/test_any_sender.cpp
    exec/test_type_async_scope.cpp
    exec/test_create.cpp
    exec/test_env.cpp
    exec/test_on.cpp
    exec/test_on2.cpp
    exec/test_on3.cpp
    exec/async_scope/test_dtor.cpp
    exec/async_scope/test_spawn.cpp
    exec/async_scope/test_spawn_future.cpp
    exec/async_scope/test_empty.cpp
    exec/async_scope/test_stop.cpp
    exec/test_when_any.cpp
    exec/test_at_coroutine_exit.cpp
    exec/test_materialize.cpp
    )

add_executable(test.stdexec ${stdexec_test_sources})

target_include_directories(test.stdexec PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(test.stdexec PUBLIC STDEXEC::stdexec stdexec_executable_flags Catch2::Catch2 range-v3::range-v3)

# Discover the Catch2 test built by the application
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)

catch_discover_tests(test.stdexec)

if(STDEXEC_ENABLE_CUDA)
    add_subdirectory(nvexec)
endif()
