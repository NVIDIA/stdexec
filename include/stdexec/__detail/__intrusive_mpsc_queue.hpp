/*
 * Copyright (c) Dmitry V'jukov
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

#include "stdexec/__detail/__config.hpp"

namespace STDEXEC
{
  template <auto _Ptr>
  class __intrusive_mpsc_queue;

  // _Node must be default_initializable only for the queue to construct an
  // internal "stub" node - only the _Next data element is accessed internally.
  template <class _Node, __std::atomic<_Node*> _Node::* _Next>
    requires __std::default_initializable<_Node>
  class __intrusive_mpsc_queue<_Next>
  {
    __std::atomic<_Node*> __head_{&__stub_};
    _Node*                __tail_{&__stub_};
    _Node                 __stub_{};

   public:
    __intrusive_mpsc_queue()
    {
      (__stub_.*_Next).store(nullptr, __std::memory_order_release);
    }

    constexpr auto push_back(_Node* __new_node) noexcept -> bool
    {
      (__new_node->*_Next).store(nullptr, __std::memory_order_relaxed);
      _Node* __prev = __head_.exchange(__new_node, __std::memory_order_acq_rel);
      (__prev->*_Next).store(__new_node, __std::memory_order_release);
      return __prev == &__stub_;
    }

    constexpr auto pop_front() noexcept -> _Node*
    {
      _Node* __tail = this->__tail_;
      STDEXEC_ASSERT(__tail != nullptr);
      _Node* __next = (__tail->*_Next).load(__std::memory_order_acquire);
      // If tail is pointing to the stub node we need to advance it once more
      if (&__stub_ == __tail)
      {
        if (nullptr == __next)
        {
          return nullptr;
        }
        this->__tail_ = __next;
        __tail        = __next;
        __next        = (__next->*_Next).load(__std::memory_order_acquire);
      }
      // Normal case: there is a next node and we can just advance the tail
      if (nullptr != __next)
      {
        this->__tail_ = __next;
        return __tail;
      }
      // Next is nullptr here means that either:
      // 1) There are no more nodes in the queue
      // 2) A producer is in the middle of adding a new node
      _Node const * __head = this->__head_.load(__std::memory_order_acquire);
      // A producer is in the middle of adding a new node
      // we cannot return tail as we cannot link the next node yet
      if (__tail != __head)
      {
        return nullptr;
      }
      // No more nodes in the queue - we need to insert a stub node
      // to be able to link to an eventual empty state (or new nodes)
      push_back(&__stub_);
      // Now re-attempt to load next
      __next = (__tail->*_Next).load(__std::memory_order_acquire);
      if (nullptr != __next)
      {
        // Successfully linked either a new node or the stub node
        this->__tail_ = __next;
        return __tail;
      }
      // A producer is in the middle of adding a new node since next is still nullptr
      // and not our stub node, thus we cannot link the next node yet
      return nullptr;
    }
  };

}  // namespace STDEXEC
