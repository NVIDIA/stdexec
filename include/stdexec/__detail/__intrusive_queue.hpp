/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <cassert>
#include <tuple>
#include <utility>

#include "__config.hpp"

namespace stdexec {
  namespace __queue {
    template <auto _Next>
      class __intrusive_queue;

    template <class _Item, _Item* _Item::*_Next>
      class __intrusive_queue<_Next> {
      public:
        __intrusive_queue() noexcept = default;

        __intrusive_queue(__intrusive_queue&& __other) noexcept
          : __head_(std::exchange(__other.__head_, nullptr))
          , __tail_(std::exchange(__other.__tail_, nullptr)) {}

        __intrusive_queue& operator=(__intrusive_queue __other) noexcept {
          std::swap(__head_, __other.__head_);
          std::swap(__tail_, __other.__tail_);
          return *this;
        }

        ~__intrusive_queue() {
          STDEXEC_ASSERT(empty());
        }

        static __intrusive_queue make_reversed(_Item* __list) noexcept {
          _Item* __new_head = nullptr;
          _Item* __new_tail = __list;
          while (__list != nullptr) {
            _Item* __next = __list->*_Next;
            __list->*_Next = __new_head;
            __new_head = __list;
            __list = __next;
          }

          __intrusive_queue __result;
          __result.__head_ = __new_head;
          __result.__tail_ = __new_tail;
          return __result;
        }

        [[nodiscard]] bool empty() const noexcept {
          return __head_ == nullptr;
        }

        [[nodiscard]] _Item* pop_front() noexcept {
          STDEXEC_ASSERT(!empty());
          _Item* __item = std::exchange(__head_, __head_->*_Next);
          if (__head_ == nullptr) {
            __tail_ = nullptr;
          }
          return __item;
        }

        void push_front(_Item* __item) noexcept {
          STDEXEC_ASSERT(__item != nullptr);
          __item->*_Next = __head_;
          __head_ = __item;
          if (__tail_ == nullptr) {
            __tail_ = __item;
          }
        }

        void push_back(_Item* __item) noexcept {
          STDEXEC_ASSERT(__item != nullptr);
          __item->*_Next = nullptr;
          if (__tail_ == nullptr) {
            __head_ = __item;
          } else {
            __tail_->*_Next = __item;
          }
          __tail_ = __item;
        }

        void append(__intrusive_queue __other) noexcept {
          if (__other.empty())
            return;
          auto* __other_head = std::exchange(__other.__head_, nullptr);
          if (empty()) {
            __head_ = __other_head;
          } else {
            __tail_->*_Next = __other_head;
          }
          __tail_ = std::exchange(__other.__tail_, nullptr);
        }

        void prepend(__intrusive_queue __other) noexcept {
          if (__other.empty())
            return;

          __other.__tail_->*_Next = __head_;
          __head_ = __other.__head_;
          if (__tail_ == nullptr) {
            __tail_ = __other.__tail_;
          }

          __other.__tail_ = nullptr;
          __other.__head_ = nullptr;
        }

      private:
        _Item* __head_ = nullptr;
        _Item* __tail_ = nullptr;
      };
  } // namespace __queue

  using __queue::__intrusive_queue;

} // namespace stdexec
