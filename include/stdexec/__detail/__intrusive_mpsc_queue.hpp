/*
 * Copyright (c) Dmitiy V'jukov
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

// general design of this MPSC queue is taken from
// https://www.1024cores.net/home/lock-free-algorithms/queues/intrusive-mpsc-node-based-queue

#pragma once


#include <atomic>

#include "./__spin_loop_pause.hpp"

namespace stdexec {
  template <auto _Ptr>
  class __intrusive_mpsc_queue;

  template <class _Node, std::atomic<void*> _Node::* _Next>
  class __intrusive_mpsc_queue<_Next> {
    std::atomic<void*> __back_{&__nil_};
    void* __front_{&__nil_};
    std::atomic<_Node*> __nil_ = nullptr;

    void push_back_nil() {
      __nil_.store(nullptr, std::memory_order_relaxed);
      auto* __prev = static_cast<_Node*>(__back_.exchange(&__nil_, std::memory_order_acq_rel));
      (__prev->*_Next).store(&__nil_, std::memory_order_release);
    }

   public:
    auto push_back(_Node* __new_node) noexcept -> bool {
      (__new_node->*_Next).store(nullptr, std::memory_order_relaxed);
      void* __prev_back = __back_.exchange(__new_node, std::memory_order_acq_rel);
      bool __is_nil = __prev_back == static_cast<void*>(&__nil_);
      if (__is_nil) {
        __nil_.store(__new_node, std::memory_order_release);
      } else {
        (static_cast<_Node*>(__prev_back)->*_Next).store(__new_node, std::memory_order_release);
      }
      return __is_nil;
    }

    auto pop_front() noexcept -> _Node* {
      if (__front_ == static_cast<void*>(&__nil_)) {
        _Node* __next = __nil_.load(std::memory_order_acquire);
        if (!__next) {
          return nullptr;
        }
        __front_ = __next;
      }
      auto* __front = static_cast<_Node*>(__front_);
      void* __next = (__front->*_Next).load(std::memory_order_acquire);
      if (__next) {
        __front_ = __next;
        return __front;
      }
      STDEXEC_ASSERT(!__next);
      push_back_nil();
      do {
        __spin_loop_pause();
        __next = (__front->*_Next).load(std::memory_order_acquire);
      } while (!__next);
      __front_ = __next;
      return __front;
    }
  };
} // namespace stdexec