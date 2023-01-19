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

function(disable_compiler_launcher)
  cmake_parse_arguments("" "" "LANG;VAR" "" ${ARGN})
  set(_val)

  if(DEFINED ENV{CMAKE_${_LANG}_COMPILER_LAUNCHER})
    set(_val "$ENV{CMAKE_${_LANG}_COMPILER_LAUNCHER}")
    unset(ENV{CMAKE_${_LANG}_COMPILER_LAUNCHER})
  endif()

  if(CMAKE_${_LANG}_COMPILER_LAUNCHER)
    set(_val "${CMAKE_${_LANG}_COMPILER_LAUNCHER}")
    unset(CMAKE_${_LANG}_COMPILER_LAUNCHER PARENT_SCOPE)
    unset(CMAKE_${_LANG}_COMPILER_LAUNCHER CACHE)
  endif()

  if(_VAR)
    set(${_VAR} "${_val}" PARENT_SCOPE)
  endif()
endfunction()

function(enable_compiler_launcher)
  cmake_parse_arguments("" "" "LANG;VAR" "" ${ARGN})
  if(${_VAR})
    set(CMAKE_${_LANG}_COMPILER_LAUNCHER "${${_VAR}}" PARENT_SCOPE)
    set(CMAKE_${_LANG}_COMPILER_LAUNCHER "${${_VAR}}" CACHE STRING "" FORCE)
    unset(${_VAR} PARENT_SCOPE)
  endif()
endfunction()
