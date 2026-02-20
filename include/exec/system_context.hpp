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

#include "../stdexec/__detail/__execution_fwd.hpp"

#include "../stdexec/__detail/__parallel_scheduler.hpp" // IWYU pragma: export
#include "detail/system_context_replaceability_api.hpp" // IWYU pragma: export

#if STDEXEC_MSVC()
#  pragma message(                                                                                 \
    "WARNING: The header <exec/system_context.hpp> is deprecated. Please include <stdexec/execution.hpp> instead.")
#else
#  warning                                                                                         \
    "The header <exec/system_context.hpp> is deprecated. Please include <stdexec/execution.hpp> instead."
#endif

namespace experimental::execution {
  using parallel_scheduler
    [[deprecated("Please use stdexec::parallel_scheduler instead")]] = STDEXEC::parallel_scheduler;

  [[deprecated("Please use stdexec::get_parallel_scheduler instead")]]
  inline auto get_parallel_scheduler() noexcept -> STDEXEC::parallel_scheduler {
    return STDEXEC::get_parallel_scheduler();
  }
} // namespace experimental::execution

namespace exec = experimental::execution;

