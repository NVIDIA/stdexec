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
include(CheckCompilerFlag)
check_compiler_flag(CXX -Wno-c2y-extensions HAVE_C2Y_EXTENSIONS_WARNING)

function(add_compile_diagnostics TARGET)
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic -Werror $<$<BOOL:${HAVE_C2Y_EXTENSIONS_WARNING}>:-Wno-c2y-extensions>)
    endif()
endfunction()

set(stdexec_test_sources
    stdexec/algos/adaptors/test_write_env.cpp
    )

add_library(common_test_settings INTERFACE)
set_target_properties(common_test_settings PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    MSVC_DEBUG_INFORMATION_FORMAT Embedded
    )
target_compile_definitions(common_test_settings INTERFACE
    STDEXEC_NAMESPACE=std::execution
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    )
target_compile_options(common_test_settings INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4714> # function marked as __forceinline not inlined
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized> # warnings being emitted from stdlib headers, why?
    $<$<CXX_COMPILER_ID:Clang>:-Wno-gnu-line-marker>
    )

add_executable(test.stdexec ${stdexec_test_sources})
add_compile_diagnostics(test.stdexec)
target_link_libraries(test.stdexec
    PUBLIC
    STDEXEC::stdexec
    stdexec_executable_flags
    PRIVATE
    common_test_settings)
