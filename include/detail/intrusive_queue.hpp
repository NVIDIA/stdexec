/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) NVIDIA
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

namespace std::execution::__detail {

template <auto _Next>
  class __intrusive_queue;

template <class _Item, _Item* _Item::*_Next>
  class __intrusive_queue<_Next> {
   public:
    __intrusive_queue() noexcept = default;

    __intrusive_queue(__intrusive_queue&& __other) noexcept
      : __head_(std::exchange(__other.head_, nullptr))
      , __tail_(std::exchange(__other.tail_, nullptr)) {}

    __intrusive_queue& operator=(__intrusive_queue __other) noexcept {
      std::swap(__head_, __other.__head_);
      std::swap(__tail_, __other.__tail_);
      return *this;
    }

    ~__intrusive_queue() {
      assert(__empty());
    }

    static __intrusive_queue __make_reversed(_Item* __list) noexcept {
      _Item* __newHead = nullptr;
      _Item* __newTail = __list;
      while (__list != nullptr) {
        _Item* next = __list->*_Next;
        __list->*_Next = __newHead;
        __newHead = __list;
        __list = next;
      }

      __intrusive_queue __result;
      __result.head_ = __newHead;
      __result.tail_ = __newTail;
      return __result;
    }

    [[nodiscard]] bool __empty() const noexcept {
      return __head_ == nullptr;
    }

    [[nodiscard]] _Item* __pop_front() noexcept {
      assert(!__empty());
      _Item* __item = std::exchange(__head_, __head_->*_Next);
      if (__head_ == nullptr) {
        __tail_ = nullptr;
      }
      return __item;
    }

    void __push_front(_Item* __item) noexcept {
      assert(__item != nullptr);
      __item->*_Next = __head_;
      __head_ = __item;
      if (__tail_ == nullptr) {
        __tail_ = __item;
      }
    }

    void __push_back(_Item* __item) noexcept {
      assert(__item != nullptr);
      __item->*_Next = nullptr;
      if (__tail_ == nullptr) {
        __head_ = __item;
      } else {
        __tail_->*_Next = __item;
      }
      __tail_ = __item;
    }

    void __append(__intrusive_queue __other) noexcept {
      if (__other.__empty())
        return;
      auto* otherHead = std::exchange(__other.__head_, nullptr);
      if (__empty()) {
        __head_ = otherHead;
      } else {
        __tail_->*_Next = otherHead;
      }
      __tail_ = std::exchange(__other.tail_, nullptr);
    }

    void __prepend(__intrusive_queue __other) noexcept {
      if (__other.__empty())
        return;

      __other.__tail_->*_Next = __head_;
      __head_ = __other.head_;
      if (__tail_ == nullptr) {
        __tail_ = __other.tail_;
      }

      __other.__tail_ = nullptr;
      __other.__head_ = nullptr;
    }

   private:
    _Item* __head_ = nullptr;
    _Item* __tail_ = nullptr;
  };

} // namespace std::execution
