/*
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
#pragma once

#include "__config.hpp"

#include <cstddef>
#include <cassert>
#include <iterator>
#include <utility>

namespace stdexec {
  namespace __slist {
    template <auto _Next>
    class __intrusive_slist;

    template <class _Item, _Item* _Item::* _Next>
    class __intrusive_slist<_Next> {
     public:
      __intrusive_slist() noexcept = default;

      __intrusive_slist(__intrusive_slist&& __other) noexcept
        : __head_(std::exchange(__other.__head_, nullptr)) {
      }

      __intrusive_slist(_Item* __head) noexcept
        : __head_(__head) {
      }

      auto swap(__intrusive_slist& __other) noexcept -> void {
        std::swap(__head_, __other.__head_);
      }

      auto operator=(__intrusive_slist __other) noexcept -> __intrusive_slist& {
        swap(__other);
        return *this;
      }

      [[nodiscard]]
      auto empty() const noexcept -> bool {
        return __head_ == nullptr;
      }

      auto front() const noexcept -> _Item* {
        return __head_;
      }

      void clear() noexcept {
        __head_ = nullptr;
      }

      [[nodiscard]]
      auto pop_front() noexcept -> _Item* {
        STDEXEC_ASSERT(!empty());
        return std::exchange(__head_, __head_->*_Next);
      }

      void push_front(_Item* __item) noexcept {
        STDEXEC_ASSERT(__item != nullptr);
        __item->*_Next = std::exchange(__head_, __item);
      }

      [[nodiscard]]
      auto remove(_Item* __item) noexcept -> _Item* {
        STDEXEC_ASSERT(__item != nullptr);
        if (__head_ == __item) {
          return pop_front();
        }

        for (_Item* __current: *this) {
          if (__current->*_Next == __item) {
            __current->*_Next = __item->*_Next;
            return __item;
          }
        }

        return nullptr;
      }

      struct iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = _Item*;
        using reference = _Item*;
        using pointer = _Item**;

        _Item* __item_ = nullptr;

        iterator() noexcept = default;

        explicit iterator(_Item* __item) noexcept
          : __item_(__item) {
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
          STDEXEC_ASSERT(__item_ != nullptr);
          __item_ = __item_->*_Next;
          return *this;
        }

        auto operator++(int) noexcept -> iterator {
          iterator __result = *this;
          ++*this;
          return __result;
        }

        auto operator==(const iterator&) const noexcept -> bool = default;
      };

      [[nodiscard]]
      auto begin() const noexcept -> iterator {
        return iterator(__head_);
      }

      [[nodiscard]]
      auto end() const noexcept -> iterator {
        return iterator(nullptr);
      }

     private:
      _Item* __head_ = nullptr;
    };
  } // namespace __slist

  using __slist::__intrusive_slist;

} // namespace stdexec
