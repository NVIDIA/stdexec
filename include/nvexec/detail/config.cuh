/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format Language: Cpp
// IWYU pragma: always_keep

#pragma once

#include "../../stdexec/__detail/__config.hpp" // IWYU pragma: export

#if !defined(_NVHPC_CUDA) && !defined(__CUDACC__)
#  error The NVIDIA schedulers and utilities require CUDA support
#endif

namespace nvexec::_strm {
  using namespace stdexec;
} // namespace nvexec::_strm
