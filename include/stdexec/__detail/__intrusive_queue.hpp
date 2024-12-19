/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include <cstddef>
#include <cassert>
#include <utility>

namespace stdexec {
  namespace __queue {
    template <auto _Next>
    class __intrusive_queue;

    template <class _Item, _Item* _Item::* _Next>
    class __intrusive_queue<_Next> {
     public:
      __intrusive_queue() noexcept = default;

      __intrusive_queue(__intrusive_queue&& __other) noexcept
        : __head_(std::exchange(__other.__head_, nullptr))
        , __tail_(std::exchange(__other.__tail_, nullptr)) {
      }

      __intrusive_queue(_Item* __head, _Item* __tail) noexcept
        : __head_(__head)
        , __tail_(__tail) {
      }

      auto operator=(__intrusive_queue __other) noexcept -> __intrusive_queue& {
        std::swap(__head_, __other.__head_);
        std::swap(__tail_, __other.__tail_);
        return *this;
      }

      ~__intrusive_queue() {
        STDEXEC_ASSERT(empty());
      }

      static auto make_reversed(_Item* __list) noexcept -> __intrusive_queue {
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

      static auto make(_Item* __list) noexcept -> __intrusive_queue {
        __intrusive_queue __result{};
        __result.__head_ = __list;
        __result.__tail_ = __list;
        if (__list == nullptr) {
          return __result;
        }
        while (__result.__tail_->*_Next != nullptr) {
          __result.__tail_ = __result.__tail_->*_Next;
        }
        return __result;
      }

      [[nodiscard]]
      auto empty() const noexcept -> bool {
        return __head_ == nullptr;
      }

      void clear() noexcept {
        __head_ = nullptr;
        __tail_ = nullptr;
      }

      [[nodiscard]]
      auto pop_front() noexcept -> _Item* {
        STDEXEC_ASSERT(!empty());
        _Item* __item = std::exchange(__head_, __head_->*_Next);
        // This should test if __head_ == nullptr, but due to a bug in
        // nvc++'s optimization, `__head_` isn't assigned until later.
        // Filed as NVBug#3952534.
        if (__item->*_Next == nullptr) {
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

      struct iterator {
        using difference_type = std::ptrdiff_t;
        using value_type = _Item*;

        _Item* __predecessor_ = nullptr;
        _Item* __item_ = nullptr;

        iterator() noexcept = default;

        explicit iterator(_Item* __pred, _Item* __item) noexcept
          : __predecessor_(__pred)
          , __item_(__item) {
        }

        [[nodiscard]]
        auto operator*() const noexcept -> _Item* {
          STDEXEC_ASSERT(__item_ != nullptr);
          return __item_;
        }

        [[nodiscard]]
        auto operator->() const noexcept -> _Item** {
          STDEXEC_ASSERT(__item_ != nullptr);
          return &__item_;
        }

        auto operator++() noexcept -> iterator& {
          __predecessor_ = __item_;
          if (__item_) {
            __item_ = __item_->*_Next;
          }
          return *this;
        }

        auto operator++(int) noexcept -> iterator {
          iterator __result = *this;
          ++*this;
          return __result;
        }

        friend auto operator==(const iterator&, const iterator&) noexcept -> bool = default;
      };

      [[nodiscard]]
      auto begin() const noexcept -> iterator {
        return iterator(nullptr, __head_);
      }

      [[nodiscard]]
      auto end() const noexcept -> iterator {
        return iterator(__tail_, nullptr);
      }

      void splice(iterator pos, __intrusive_queue& other, iterator first, iterator last) noexcept {
        if (first == last) {
          return;
        }
        STDEXEC_ASSERT(first.__item_ != nullptr);
        STDEXEC_ASSERT(last.__predecessor_ != nullptr);
        if (other.__head_ == first.__item_) {
          other.__head_ = last.__item_;
          if (other.__head_ == nullptr) {
            other.__tail_ = nullptr;
          }
        } else {
          STDEXEC_ASSERT(first.__predecessor_ != nullptr);
          first.__predecessor_->*_Next = last.__item_;
          last.__predecessor_->*_Next = pos.__item_;
        }
        if (empty()) {
          __head_ = first.__item_;
          __tail_ = last.__predecessor_;
        } else {
          pos.__predecessor_->*_Next = first.__item_;
          if (pos.__item_ == nullptr) {
            __tail_ = last.__predecessor_;
          }
        }
      }

      auto front() const noexcept -> _Item* {
        return __head_;
      }

      auto back() const noexcept -> _Item* {
        return __tail_;
      }

     private:
      _Item* __head_ = nullptr;
      _Item* __tail_ = nullptr;
    };
  } // namespace __queue

  using __queue::__intrusive_queue;

} // namespace stdexec
