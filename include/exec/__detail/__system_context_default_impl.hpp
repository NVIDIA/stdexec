/*
 * Copyright (c) 2023 Lee Howes, Lucian Radu Teodorescu
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

#include "__system_context_backend.hpp"
#include "exec/static_thread_pool.hpp"

namespace exec::__system_context_default_impl {
  using system_context_replaceability::system_scheduler;

  struct __system_scheduler_impl
    : __system_context_backend::__system_scheduler_impl<exec::static_thread_pool> {
    __system_scheduler_impl() = default;

    uint32_t max_concurrency() noexcept override {
      uint32_t n = std::thread::hardware_concurrency();
      return n == 0 ? 1 : n;
    }

    stdexec::forward_progress_guarantee get_forward_progress_guarantee() noexcept override {
      return stdexec::forward_progress_guarantee::parallel;
    }
  };

} // namespace exec::__system_context_default_impl
