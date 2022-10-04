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

namespace _P2300::execution::detail {

template <auto Next>
  class __intrusive_queue;

template <class Item, Item* Item::*Next>
  class __intrusive_queue<Next> {
   public:
    __intrusive_queue() noexcept = default;

    __intrusive_queue(__intrusive_queue&& other) noexcept
      : head_(std::exchange(other.head_, nullptr))
      , tail_(std::exchange(other.tail_, nullptr)) {}

    __intrusive_queue& operator=(__intrusive_queue other) noexcept {
      std::swap(head_, other.head_);
      std::swap(tail_, other.tail_);
      return *this;
    }

    ~__intrusive_queue() {
      assert(empty());
    }

    static __intrusive_queue make_reversed(Item* list) noexcept {
      Item* newHead = nullptr;
      Item* newTail = list;
      while (list != nullptr) {
        Item* next = list->*Next;
        list->*Next = newHead;
        newHead = list;
        list = next;
      }

      __intrusive_queue result;
      result.head_ = newHead;
      result.tail_ = newTail;
      return result;
    }

    [[nodiscard]] bool empty() const noexcept {
      return head_ == nullptr;
    }

    [[nodiscard]] Item* pop_front() noexcept {
      assert(!empty());
      Item* item = std::exchange(head_, head_->*Next);
      if (head_ == nullptr) {
        tail_ = nullptr;
      }
      return item;
    }

    void push_front(Item* item) noexcept {
      assert(item != nullptr);
      item->*Next = head_;
      head_ = item;
      if (tail_ == nullptr) {
        tail_ = item;
      }
    }

    void push_back(Item* item) noexcept {
      assert(item != nullptr);
      item->*Next = nullptr;
      if (tail_ == nullptr) {
        head_ = item;
      } else {
        tail_->*Next = item;
      }
      tail_ = item;
    }

    void append(__intrusive_queue other) noexcept {
      if (other.empty())
        return;
      auto* otherHead = std::exchange(other.head_, nullptr);
      if (empty()) {
        head_ = otherHead;
      } else {
        tail_->*Next = otherHead;
      }
      tail_ = std::exchange(other.tail_, nullptr);
    }

    void prepend(__intrusive_queue other) noexcept {
      if (other.empty())
        return;

      other.tail_->*Next = head_;
      head_ = other.head_;
      if (tail_ == nullptr) {
        tail_ = other.tail_;
      }

      other.tail_ = nullptr;
      other.head_ = nullptr;
    }

   private:
    Item* head_ = nullptr;
    Item* tail_ = nullptr;
  };

} // namespace detail
