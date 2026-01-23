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

set(exec_test_sources
    ../test_main.cpp
    test_bwos_lifo_queue.cpp
    test_any_sender.cpp
    test_task.cpp
    test_timed_thread_scheduler.cpp
    test_variant_sender.cpp
    test_type_async_scope.cpp
    test_completion_signatures.cpp
    test_create.cpp
    test_env.cpp
    test_finally.cpp
    test_into_tuple.cpp
    test_repeat_until.cpp
    test_repeat_n.cpp
    async_scope/test_dtor.cpp
    async_scope/test_spawn.cpp
    async_scope/test_spawn_future.cpp
    async_scope/test_start_now.cpp
    async_scope/test_empty.cpp
    async_scope/test_stop.cpp
    test_when_any.cpp
    test_at_coroutine_exit.cpp
    test_materialize.cpp
    $<$<BOOL:${STDEXEC_ENABLE_IO_URING}>:test_io_uring_context.cpp>
    $<$<BOOL:${STDEXEC_ENABLE_WINDOWS_THREAD_POOL}>:test_windows_thread_pool_context.cpp>
    test_trampoline_scheduler.cpp
    test_sequence_senders.cpp
    test_sequence.cpp
    test_static_thread_pool.cpp
    test_just_from.cpp
    test_fork.cpp
    sequence/test_any_sequence_of.cpp
    sequence/test_empty_sequence.cpp
    sequence/test_ignore_all_values.cpp
    sequence/test_iterate.cpp
    sequence/test_transform_each.cpp
    sequence/test_merge.cpp
    # These tests cause Microsoft's compiler to run out of memory
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:sequence/test_merge_each.cpp>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:sequence/test_merge_each_threaded.cpp>
    $<$<BOOL:${STDEXEC_ENABLE_TBB}>:../execpools/test_tbb_thread_pool.cpp>
    $<$<BOOL:${STDEXEC_ENABLE_TASKFLOW}>:../execpools/test_taskflow_thread_pool.cpp>
    $<$<BOOL:${STDEXEC_ENABLE_ASIO}>:../execpools/test_asio_thread_pool.cpp>
    test_system_context.cpp
    $<$<BOOL:${STDEXEC_ENABLE_LIBDISPATCH}>:test_libdispatch.cpp>
    test_unless_stop_requested.cpp
    )

add_executable(test.exec ${exec_test_sources})
target_link_libraries(test.exec
    PUBLIC
    STDEXEC::stdexec
    $<TARGET_NAME_IF_EXISTS:STDEXEC::tbbpool>
    $<TARGET_NAME_IF_EXISTS:STDEXEC::taskflow_pool>
    $<TARGET_NAME_IF_EXISTS:STDEXEC::asio_pool>
    stdexec_executable_flags
    Catch2::Catch2
    PRIVATE
    common_test_settings)

add_executable(test.system_context_replaceability ../test_main.cpp test_system_context_replaceability.cpp)
target_link_libraries(test.system_context_replaceability
    PUBLIC
    STDEXEC::stdexec
    STDEXEC::system_context
    stdexec_executable_flags
    Catch2::Catch2
    PRIVATE
    common_test_settings)

# Discover the Catch2 test built by the application
catch_discover_tests(test.exec)
if(NOT STDEXEC_ENABLE_CUDA)
    catch_discover_tests(test.system_context_replaceability)
endif()

# Test which parses a source file for an expected error message
icm_add_build_failure_test(
    NAME test_repeat_until_fail
    TARGET test_repeat_until_fail
    SOURCES PARSE test_repeat_until_fail.cpp
    LIBRARIES stdexec
    FOLDER test
)

icm_add_build_failure_test(
    NAME test_repeat_n_fail
    TARGET test_repeat_n_fail
    SOURCES PARSE test_repeat_n_fail.cpp
    LIBRARIES stdexec
    FOLDER test
)
