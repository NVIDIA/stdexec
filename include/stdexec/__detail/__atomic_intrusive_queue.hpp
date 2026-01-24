/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "__config.hpp"

//
#include "__atomic.hpp"
#include "__intrusive_queue.hpp"

#include <cassert>
#include <cstddef>

namespace STDEXEC {

  // An atomic queue that supports multiple producers and a single consumer.
  template <auto _NextPtr>
  class __atomic_intrusive_queue;

  template <class _Tp, _Tp* _Tp::* _NextPtr>
  class alignas(64) __atomic_intrusive_queue<_NextPtr> {
   public:
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto push(_Tp* __node) noexcept -> bool {
      STDEXEC_ASSERT(__node != nullptr);
      _Tp* __old_head = __head_.load(__std::memory_order_relaxed);
      do {
        __node->*_NextPtr = __old_head;
      } while (!__head_.compare_exchange_weak(__old_head, __node, __std::memory_order_acq_rel));

      // If the queue was empty before, we notify the consumer thread that there is now an
      // item available. If the queue was not empty, we do not notify, because the consumer
      // thread has already been notified.
      if (__old_head != nullptr) {
        return false;
      }

      // There can be only one consumer thread, so we can use notify_one here instead of
      // notify_all:
      __head_.notify_one();
      return true;
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr void wait_for_item() noexcept {
      // Wait until the queue has an item in it:
      __head_.wait(nullptr);
    }

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto pop_all() noexcept -> __intrusive_queue<_NextPtr> {
      auto* const __list = __head_.exchange(nullptr, __std::memory_order_acquire);
      return __intrusive_queue<_NextPtr>::make_reversed(__list);
    }

   private:
    __std::atomic<_Tp*> __head_{nullptr};
  };

} // namespace STDEXEC
