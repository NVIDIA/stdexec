#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#
# This is the preferred entry point for projects using rapids-cmake
#

# Allow users to control which version is used
if(NOT rapids-cmake-version)
  # Define a default version if the user doesn't set one
  set(rapids-cmake-version 24.02)
endif()

# Allow users to control which GitHub repo is fetched
if(NOT rapids-cmake-repo)
  # Define a default repo if the user doesn't set one
  set(rapids-cmake-repo rapidsai/rapids-cmake)
endif()

# Allow users to control which branch is fetched
if(NOT rapids-cmake-branch)
  # Define a default branch if the user doesn't set one
  set(rapids-cmake-branch "branch-${rapids-cmake-version}")
endif()

# Allow users to control the exact URL passed to FetchContent
if(NOT rapids-cmake-url)
  # Construct a default URL if the user doesn't set one
  set(rapids-cmake-url "https://github.com/${rapids-cmake-repo}/")
  # In order of specificity
  if(rapids-cmake-sha)
    # An exact git SHA takes precedence over anything
    string(APPEND rapids-cmake-url "archive/${rapids-cmake-sha}.zip")
  elseif(rapids-cmake-tag)
    # Followed by a git tag name
    string(APPEND rapids-cmake-url "archive/refs/tags/${rapids-cmake-tag}.zip")
  else()
    # Or if neither of the above two were defined, use a branch
    string(APPEND rapids-cmake-url "archive/refs/heads/${rapids-cmake-branch}.zip")
  endif()
endif()

if(POLICY CMP0135)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)
FetchContent_Declare(rapids-cmake URL "${rapids-cmake-url}")
if(POLICY CMP0135)
  cmake_policy(POP)
endif()
FetchContent_GetProperties(rapids-cmake)
if(rapids-cmake_POPULATED)
  # Something else has already populated rapids-cmake, only thing
  # we need to do is setup the CMAKE_MODULE_PATH
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
else()
  FetchContent_MakeAvailable(rapids-cmake)
endif()
