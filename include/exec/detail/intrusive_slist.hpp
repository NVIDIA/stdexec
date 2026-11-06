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

#include "../../stdexec/__detail/__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  if !STDEXEC_USE_MODULES()
#    include <cassert>
#    include <cstddef>
#    include <iterator>
#    include <utility>
#  endif

#  include "../../stdexec/__detail/__prologue.hpp"

namespace experimental::execution::detail
{
  template <auto Next>
  class intrusive_slist;

  template <class Item, Item* Item::* Next>
  class intrusive_slist<Next>
  {
   public:
    constexpr intrusive_slist() noexcept = default;

    constexpr intrusive_slist(intrusive_slist&& other) noexcept
      : head_(std::exchange(other.head_, nullptr))
    {}

    constexpr intrusive_slist(Item* __head) noexcept
      : head_(__head)
    {}

    constexpr auto swap(intrusive_slist& other) noexcept -> void
    {
      std::swap(head_, other.head_);
    }

    constexpr auto operator=(intrusive_slist other) noexcept -> intrusive_slist&
    {
      swap(other);
      return *this;
    }

    [[nodiscard]]
    constexpr auto empty() const noexcept -> bool
    {
      return head_ == nullptr;
    }

    constexpr auto front() const noexcept -> Item*
    {
      return head_;
    }

    constexpr void clear() noexcept
    {
      head_ = nullptr;
    }

    [[nodiscard]]
    constexpr auto pop_front() noexcept -> Item*
    {
      STDEXEC_ASSERT(!empty());
      return std::exchange(head_, head_->*Next);
    }

    constexpr void push_front(Item* item) noexcept
    {
      STDEXEC_ASSERT(item != nullptr);
      item->*Next = std::exchange(head_, item);
    }

    [[nodiscard]]
    constexpr auto remove(Item* item) noexcept -> Item*
    {
      STDEXEC_ASSERT(item != nullptr);
      if (head_ == item)
      {
        return pop_front();
      }

      for (Item* current: *this)
      {
        if (current->*Next == item)
        {
          current->*Next = item->*Next;
          return item;
        }
      }

      return nullptr;
    }

    struct iterator
    {
      using iterator_category = std::forward_iterator_tag;
      using difference_type   = std::ptrdiff_t;
      using value_type        = Item*;
      using reference         = Item*;
      using pointer           = Item**;

      Item* item_ = nullptr;

      constexpr iterator() noexcept = default;

      constexpr explicit iterator(Item* item) noexcept
        : item_(item)
      {}

      [[nodiscard]]
      constexpr auto operator*() const noexcept -> Item*
      {
        STDEXEC_ASSERT(item_ != nullptr);
        return item_;
      }

      [[nodiscard]]
      constexpr auto operator->() const noexcept -> Item**
      {
        STDEXEC_ASSERT(item_ != nullptr);
        return &item_;
      }

      constexpr auto operator++() noexcept -> iterator&
      {
        STDEXEC_ASSERT(item_ != nullptr);
        item_ = item_->*Next;
        return *this;
      }

      constexpr auto operator++(int) noexcept -> iterator
      {
        iterator result = *this;
        ++*this;
        return result;
      }

      constexpr auto operator==(iterator const &) const noexcept -> bool = default;
    };

    [[nodiscard]]
    constexpr auto begin() const noexcept -> iterator
    {
      return iterator(head_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> iterator
    {
      return iterator(nullptr);
    }

   private:
    Item* head_ = nullptr;
  };
}  // namespace experimental::execution::detail

namespace exec = experimental::execution;

#  include "../../stdexec/__detail/__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
