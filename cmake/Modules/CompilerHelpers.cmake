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

include_guard(GLOBAL)

# nvhpc_find_libcudacxx_include_dir(<out_var>)
# Invokes nvc++ -stdpar to locate the libcudacxx include root (the directory
# that contains cuda/std/bit) and stores it in <out_var>.
function(nvhpc_find_libcudacxx_include_dir out_var)
  set(_src "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/find_libcudacxx.cu")
  file(WRITE "${_src}" "#include <cuda/std/bit>\n")
  execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -stdpar -E -M "${_src}"
      OUTPUT_VARIABLE _output
      ERROR_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "(/[^\n]*/cuda/std/bit) " _match "${_output}")
  if(CMAKE_MATCH_1)
    string(REGEX REPLACE "/cuda/std/bit$" "" _dir "${CMAKE_MATCH_1}")
    set(${out_var} "${_dir}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

# cuda_archs_to_gpu_list(<archs> <out_var>) Converts a list of CUDA architectures (e.g.
# 80, 86, 90) to a comma-separated list of GPU names (e.g. cc80, cc86, cc90) and stores it
# in <out_var>. This is useful for passing to the -gpu option of nvc++.
function(cuda_archs_to_gpu_list archs out_var)
  set(_gpus)
  foreach(_arch IN LISTS archs)
    string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
    string(PREPEND _arch "cc")
    list(APPEND _gpus "${_arch}")
  endforeach()
  list(JOIN _gpus "," _gpus)
  set(${out_var} "${_gpus}" PARENT_SCOPE)
endfunction()

function(disable_compiler)
  cmake_parse_arguments("" "" "LANG;VAR" "" ${ARGN})
  set(_val)

  if(DEFINED ENV{CMAKE_${_LANG}_COMPILER})
    set(_val "$ENV{CMAKE_${_LANG}_COMPILER}")
    unset(ENV{CMAKE_${_LANG}_COMPILER})
  endif()

  if(CMAKE_${_LANG}_COMPILER)
    set(_val "${CMAKE_${_LANG}_COMPILER}")
    unset(CMAKE_${_LANG}_COMPILER PARENT_SCOPE)
    unset(CMAKE_${_LANG}_COMPILER CACHE)
  endif()

  if(_VAR)
    set(${_VAR} "${_val}" PARENT_SCOPE)
  endif()
endfunction()
