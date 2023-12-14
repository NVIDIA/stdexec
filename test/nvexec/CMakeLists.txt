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

set(nvexec_test_sources
    bulk.cpp
    ensure_started.cpp
    start_detached.cpp
    variant.cpp
    monotonic_buffer.cpp
    synchronized_pool.cpp
    split.cpp
    upon_stopped.cpp
    transfer.cpp
    launch.cpp
    let_error.cpp
    let_stopped.cpp
    let_value.cpp
    test_main.cpp
    then.cpp
    reduce.cpp
    split.cpp
    start_detached.cpp
    transfer.cpp
    transfer_when_all.cpp
    upon_error.cpp
    upon_stopped.cpp
    variant.cpp
    when_all.cpp
)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
    set_source_files_properties(${nvexec_test_sources} PROPERTIES LANGUAGE CUDA)
endif()

add_executable(test.nvexec ${nvexec_test_sources})
set_target_properties(test.nvexec PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_include_directories(test.nvexec PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(test.nvexec
                      STDEXEC::nvexec
                      stdexec_executable_flags
                      Catch2::Catch2
                      nvexec_executable_flags)

# if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
#     target_compile_options(test.nvexec PRIVATE "-ftemplate-backtrace-limit=9999")
# endif()

catch_discover_tests(test.nvexec)
