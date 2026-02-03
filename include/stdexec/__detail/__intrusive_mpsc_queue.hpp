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


#include "__atomic.hpp"

#include "./__spin_loop_pause.hpp"

namespace STDEXEC {
  template <auto _Ptr>
  class __intrusive_mpsc_queue;

  // _Node must be default_initializable only for the queue to construct an
  // internal "stub" node - only the _Next data element is accessed internally.
  template <class _Node, __std::atomic<void*> _Node::* _Next>
    requires __std::default_initializable<_Node>
  class __intrusive_mpsc_queue<_Next> {

    __std::atomic<void*> __back_{&__stub_};
    __std::atomic<void*> __head_{&__stub_};
    _Node __stub_;

   public:

    __intrusive_mpsc_queue() {
      (__stub_.*_Next).store(nullptr, __std::memory_order_release);
    }

    constexpr auto push_back(_Node* __new_node) noexcept -> bool {
      (__new_node->*_Next).store(nullptr, __std::memory_order_release);
      _Node* __prev = static_cast<_Node*>(
          __head_.exchange(static_cast<void*>(__new_node), __std::memory_order_acq_rel)
      );
      bool was_stub = __prev == &__stub_;
      (__prev->*_Next).store(static_cast<void*>(__new_node), __std::memory_order_release);
      return was_stub;
    }

    constexpr auto pop_front() noexcept -> _Node* {
      _Node* __back = static_cast<_Node*>(__back_.load(__std::memory_order_relaxed));
      _Node* __next = static_cast<_Node*>((__back->*_Next).load(__std::memory_order_acquire));
      if (__back == &__stub_) {
        if (nullptr == __next)
          return nullptr;
        __back_.store(static_cast<void*>(__next), __std::memory_order_relaxed);
        __back = __next;
        __next = static_cast<_Node*>((__next->*_Next).load(__std::memory_order_acquire));
      }
      if (__next) {
        __back_.store(static_cast<void*>(__next), __std::memory_order_relaxed);
        return __back;
      }
      _Node* __head = static_cast<_Node*>(__head_.load(__std::memory_order_relaxed));
      if (__back != __head)
        return nullptr;
      push_back(&__stub_);
      __next = static_cast<_Node*>((__back->*_Next).load(__std::memory_order_acquire));
      if (__next) {
        __back_.store(static_cast<void*>(__next), __std::memory_order_relaxed);
        return __back;
      }
      return nullptr;
    }
  };

} // namespace STDEXEC
