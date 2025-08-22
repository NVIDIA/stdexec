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

function(split pair name_out path_out)
    string(STRIP "${pair}" pair)
    string(REPLACE ":" ";" pair "${pair}")
    list(POP_FRONT pair _name)
    list(POP_FRONT pair _path)
    string(STRIP "${_name}" _name)
    string(STRIP "${_path}" _path)
    set(${name_out} "${_name}" PARENT_SCOPE)
    set(${path_out} "${_path}" PARENT_SCOPE)
endfunction()

set(_lang_cxx CXX)
set(_lang_gpu CXX)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
    set(_lang_gpu CUDA)
endif()

###############################################################################
# Define targets for common properties
###############################################################################

add_library(nvexec_example INTERFACE)
target_include_directories(nvexec_example
    INTERFACE ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(nvexec_example
    INTERFACE nvexec_executable_flags
)

add_library(stdpar_multicore INTERFACE)
target_include_directories(stdpar_multicore
    SYSTEM
    INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:${CUDAToolkit_INCLUDE_DIRS}>
)

target_compile_options(stdpar_multicore
    INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
              $<$<COMPILE_LANG_AND_ID:CXX,Clang>:
                # libc++ doesn't include stdpar yet
                -stdlib=libstdc++
                # Silence Thrust -Wdeprecated-builtins warnings
                -Wno-deprecated-builtins>
)

target_link_options(stdpar_multicore
    INTERFACE $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
)

function(set_executable_output_name target)
    string(LENGTH "${target}" str_len)
    string(FIND "${target}" "." last_dot REVERSE)
    math(EXPR last_dot "${last_dot} + 1")
    math(EXPR str_len "${str_len} - ${last_dot}")
    string(SUBSTRING "${target}" ${last_dot} ${str_len} exe_name)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${exe_name})
endfunction()

###############################################################################
# CPU examples
###############################################################################
set(nvexec_cpu_examples
    "example.nvexec.maxwell_cpu_mt : maxwell_cpu_mt.cpp"
    "example.nvexec.maxwell_cpu_st : maxwell_cpu_st.cpp"
)

function(def_cpu_example example)
    split(${example} target source)
    add_executable(${target} ${source})
    set_executable_output_name(${target})
    target_link_libraries(${target}
        PRIVATE stdpar_multicore
                STDEXEC::stdexec
                stdexec_executable_flags
    )
    set_source_files_properties(${source} PROPERTIES LANGUAGE ${_lang_cxx})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_link_options(${target} PRIVATE -lc++abi -ltbb)
    endif()
endfunction()

foreach(example ${nvexec_cpu_examples})
    def_cpu_example(${example})
endforeach()

###############################################################################
# GPU examples
###############################################################################
set(nvexec_gpu_examples
    "         example.nvexec.bulk : bulk.cpp"
    "       example.nvexec.reduce : reduce.cpp"
    "        example.nvexec.split : split.cpp"
    "         example.nvexec.nvtx : nvtx.cpp"
    "       example.nvexec.launch : launch.cpp"
    "example.nvexec.maxwell_gpu_s : maxwell_gpu_s.cpp"
    "example.nvexec.maxwell_gpu_m : maxwell_gpu_m.cpp"
)

function(def_gpu_example example)
    split(${example} target source)
    add_executable(${target} ${source})
    set_executable_output_name(${target})
    target_link_libraries(${target}
        PRIVATE nvexec_example
                STDEXEC::nvexec
                stdexec_executable_flags
    )
    set_source_files_properties(${source} PROPERTIES LANGUAGE ${_lang_gpu})
endfunction()

foreach(example ${nvexec_gpu_examples})
    def_gpu_example(${example})
endforeach()

###############################################################################
# MPI examples
###############################################################################

set(compiler_name "")
get_filename_component(compiler_name "${CMAKE_CXX_COMPILER}" NAME)

if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") AND (compiler_name STREQUAL "mpic++"))

    function(def_mpi_example target source)
        add_executable(${target} ${source})
        set_executable_output_name(${target})
        target_link_libraries(${target}
            PRIVATE nvexec_example
                    STDEXEC::nvexec
                    stdexec_executable_flags
        )
        target_compile_definitions(${target} PRIVATE ${ARGN})
        set_source_files_properties(${source} PROPERTIES LANGUAGE ${_lang_gpu})
    endfunction()

    def_mpi_example(example.nvexec.maxwell_distributed    maxwell_distributed.cpp)
    def_mpi_example(example.nvexec.maxwell_distributed_ov maxwell_distributed.cpp OVERLAP)
endif()
