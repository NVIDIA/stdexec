#
# Copyright (c) 2023 Maikel Nadolski
# Copyright (c) 2023 NVIDIA Corporation
#
# Licensed under the Apache License Version 2.0 with LLVM Exceptions
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#   https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#[=======================================================================[.rst:
Findnuma
-------

Finds the numa library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``numa::numa``
  The numa library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``numa_FOUND``
  True if the system has the Foo library.
``numa_VERSION``
  The version of the Foo library which was found.
``numa_INCLUDE_DIRS``
  Include directories needed to use Foo.
``numa_LIBRARIES``
  Libraries needed to link to Foo.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``numa_INCLUDE_DIR``
  The directory containing ``numa.h``.
``numa_LIBRARY``
  The path to the Foo library.

#]=======================================================================]

find_path(numa_INCLUDE_DIR
  NAMES numa.h
  PATHS ${PC_Foo_INCLUDE_DIRS}
  PATH_SUFFIXES numa
)
find_library(numa_LIBRARY
  NAMES numa
  PATHS ${PC_Foo_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(numa
  FOUND_VAR numa_FOUND
  REQUIRED_VARS
    numa_LIBRARY
    numa_INCLUDE_DIR
  VERSION_VAR numa_VERSION
)

if(numa_FOUND)
  set(numa_LIBRARIES ${numa_LIBRARY})
  set(numa_INCLUDE_DIRS ${numa_INCLUDE_DIR})
  set(numa_DEFINITIONS ${PC_numa_CFLAGS_OTHER})
endif()

if(numa_FOUND AND NOT TARGET numa::numa)
  add_library(numa::numa UNKNOWN IMPORTED)
  set_target_properties(numa::numa PROPERTIES
    IMPORTED_LOCATION "${numa_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${PC_numa_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${numa_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  numa_INCLUDE_DIR
  numa_LIBRARY
)
