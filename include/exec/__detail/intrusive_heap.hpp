/*
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
#pragma once

#include "../../stdexec/__detail/__config.hpp"

#include <cstddef>
#include <bit>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_partial_spec_arg_list)

namespace exec {
#if defined(__cpp_lib_int_pow2) && __cpp_lib_int_pow2 >= 202002L
  namespace detail {
    using std::bit_ceil;
  } // namespace detail
#else
#  if defined(__has_builtin) && __has_builtin(__builtin_clzll)
  namespace detail {
    inline std::size_t bit_ceil(std::size_t n) noexcept {
      const int leading_zeros = __builtin_clzll(n);
      const std::size_t p = std::size_t{1} << (sizeof(std::size_t) * 8 - leading_zeros);
      return p < n ? p << 1 : p;
    }
  } // namespace detail
#  else
  namespace detail {
    inline std::size_t bit_ceil(std::size_t n) noexcept {
      std::size_t m = n - 1;
      m |= m >> 1;
      m |= m >> 2;
      m |= m >> 4;
      m |= m >> 8;
      m |= m >> 16;
      m |= m >> 32;
      return m + 1;
    }
  } // namespace detail
#  endif
#endif

  template <class Node, class KeyT, auto Key, auto Prev, auto Left, auto Right>
  class intrusive_heap;

  template <
    class Node,
    class KeyT,
    KeyT Node::* Key,
    Node* Node::* Prev,
    Node* Node::* Left,
    Node* Node::* Right
  >
  class intrusive_heap<Node, KeyT, Key, Prev, Left, Right> {
   public:
    void insert(Node* node) noexcept {
      node->*Prev = nullptr;
      node->*Left = nullptr;
      node->*Right = nullptr;
      if (root_ == nullptr) {
        root_ = node;
        size_ = 1;
        return;
      }
      Node* parent = iterate_to_parent_of_end();
      STDEXEC_ASSERT(parent);
      STDEXEC_ASSERT(parent->*Left == nullptr || parent->*Right == nullptr);
      if (parent->*Left == nullptr) {
        parent->*Left = node;
      } else {
        STDEXEC_ASSERT(parent->*Right == nullptr);
        parent->*Right = node;
      }
      node->*Prev = parent;
      size_ += 1;
      bottom_up_heapify(node);
    }

    void pop_front() noexcept {
      if (size_ == 0) {
        return;
      }
      if (size_ == 1) {
        root_ = nullptr;
        size_ = 0;
        return;
      }
      Node* leaf = iterate_to_back();
      STDEXEC_ASSERT(leaf);
      STDEXEC_ASSERT(leaf->*Left == nullptr && leaf->*Right == nullptr);

      if (leaf->*Prev->*Left == leaf) {
        STDEXEC_ASSERT(leaf->*Prev->*Right == nullptr);
        leaf->*Prev->*Left = nullptr;
      } else {
        STDEXEC_ASSERT(leaf->*Prev->*Left != nullptr);
        leaf->*Prev->*Right = nullptr;
      }
      size_ -= 1;
      leaf->*Prev = nullptr;
      leaf->*Left = std::exchange(root_->*Left, nullptr);
      leaf->*Right = std::exchange(root_->*Right, nullptr);
      if (leaf->*Left) {
        leaf->*Left->*Prev = leaf;
      }
      if (leaf->*Right) {
        leaf->*Right->*Prev = leaf;
      }
      STDEXEC_ASSERT(root_->*Prev == nullptr);
      root_ = leaf;
      top_down_heapify(root_);
    }

    auto front() const noexcept -> Node* {
      return root_;
    }

    auto erase(Node* node) noexcept -> bool {
      if (node->*Prev == nullptr && node != root_) {
        // node is not in the heap
        return false;
      }
      node->*Key = KeyT{}; // TODO: set min value
      bottom_up_heapify(node);
      STDEXEC_ASSERT(node == front());
      pop_front();
      return true;
    }

   private:
    Node* root_ = nullptr;
    std::size_t size_ = 0;

    void swap_parent_child(Node* parent, Node* child) {
      Node* grand_parent = parent->*Prev;
      if (grand_parent) {
        if (grand_parent->*Left == parent) {
          grand_parent->*Left = child;
        } else {
          grand_parent->*Right = child;
        }
      } else {
        root_ = child;
      }
      child->*Prev = grand_parent;
      if (parent->*Right == child) {
        parent->*Right = std::exchange(child->*Right, parent);
        std::swap(parent->*Left, child->*Left);
      } else {
        parent->*Left = std::exchange(child->*Left, parent);
        std::swap(parent->*Right, child->*Right);
      }
      if (parent->*Left) {
        parent->*Left->*Prev = parent;
      }
      if (parent->*Right) {
        parent->*Right->*Prev = parent;
      }
      if (child->*Left) {
        child->*Left->*Prev = child;
      }
      if (child->*Right) {
        child->*Right->*Prev = child;
      }
    }

    void bottom_up_heapify(Node* node) noexcept {
      while (node->*Prev && !(node->*Prev->*Key < node->*Key)) {
        swap_parent_child(node->*Prev, node);
      }
    }

    void top_down_heapify(Node* parent) noexcept {
      while (parent->*Left) {
        Node* left = parent->*Left;
        Node* right = parent->*Right;
        Node* child = left;
        if (right && right->*Key < left->*Key) {
          child = right;
        }
        if (child->*Key < parent->*Key) {
          swap_parent_child(parent, child);
        } else {
          break;
        }
      }
    }

    auto iterate_to_parent_of(std::size_t pos) noexcept -> Node* {
      std::size_t index = detail::bit_ceil(pos);
      if (index > pos) {
        index /= 4;
      } else {
        index /= 2;
      }
      Node* node = root_;
      while (index > 1) {
        if (pos & index) {
          node = node->*Right;
        } else {
          node = node->*Left;
        }
        index /= 2;
      }
      return node;
    }

    auto iterate_to_parent_of_end() noexcept -> Node* {
      return iterate_to_parent_of(size_ + 1);
    }

    auto iterate_to_back() noexcept -> Node* {
      Node* parent = iterate_to_parent_of(size_);
      STDEXEC_ASSERT(parent->*Left != nullptr);
      if (parent->*Right) {
        return parent->*Right;
      }
      return parent->*Left;
    }
  };
} // namespace exec

STDEXEC_PRAGMA_POP()
