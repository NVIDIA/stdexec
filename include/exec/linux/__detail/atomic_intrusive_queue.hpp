/*
 * Copyright (C) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../../../stdexec/__detail/__intrusive_queue.hpp"

namespace exec {
  template <auto _NextPtr>
  class __atomic_intrusive_queue;

  template <class _T, _T* _T::*_NextPtr>
  class __atomic_intrusive_queue<_NextPtr> {
   public:
    using __node_pointer = _T*;
    using __atomic_node_pointer = std::atomic<_T*>;

    [[nodiscard]] bool empty() const noexcept {
      return __head_.load(std::memory_order_relaxed) == nullptr;
    }

    void push_front(__node_pointer t) noexcept {
      __node_pointer __old_head = __head_.load(std::memory_order_relaxed);
      do {
        t->*_NextPtr = __old_head;
      } while (!__head_.compare_exchange_weak(__old_head, t, std::memory_order_acq_rel));
    }

    stdexec::__intrusive_queue<_NextPtr> pop_all() noexcept {
      return stdexec::__intrusive_queue<_NextPtr>::make_reversed(
        __head_.exchange(nullptr, std::memory_order_acq_rel));
    }

   private:
    __atomic_node_pointer __head_{nullptr};
  };
}