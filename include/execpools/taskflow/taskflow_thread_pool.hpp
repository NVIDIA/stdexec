/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#pragma once

#include "../../stdexec/__detail/__config.hpp"

#if STDEXEC_MSVC()
#  pragma message(                                                                                 \
    "WARNING: The header <execpools/taskflow/taskflow_thread_pool.hpp> is deprecated. Please include <exec/taskflow/taskflow_thread_pool.hpp> instead.")
#else
#  warning                                                                                         \
    "The header <execpools/taskflow/taskflow_thread_pool.hpp> is deprecated. Please include <exec/taskflow/taskflow_thread_pool.hpp> instead."
#endif

#include "../../exec/taskflow/taskflow_thread_pool.hpp"  // IWYU pragma: export

namespace execpools
{

  using taskflow_thread_pool [[deprecated(
    "execpools::taskflow_thread_pool has been renamed to "
    "exec::taskflow::taskflow_thread_pool instead.")]] = exec::taskflow::taskflow_thread_pool;

}  // namespace execpools
