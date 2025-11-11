/*
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2025 TypeCombinator
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

#include <bit>
#include <cstddef>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_partial_spec_arg_list)

namespace exec {
#if defined(__cpp_lib_int_pow2) && __cpp_lib_int_pow2 >= 2020'02L
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

  namespace detail {
    static inline std::size_t path_bit_mask(std::size_t n) noexcept {
      return detail::bit_ceil(n + 1) >> 2;
    }
  } // namespace detail

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
      size_++;
      Node** cur_ptr = &root_;
      Node* cur = root_;
      Node* parent = nullptr;

      auto path = size_;
      auto path_mask = detail::path_bit_mask(size_);
      while (path_mask) {
        if (node->*Key < cur->*Key) {
          node->*Prev = parent;
          cur->*Prev = node;
          *cur_ptr = node;
          do { // Replace one by one until reaching the bottom.
            if (path_mask & path) {
              node->*Right = cur;
              node->*Left = cur->*Left;
              cur->*Left->*Prev = node;
              node = cur;
              cur = cur->*Right;
            } else {
              node->*Right = cur->*Right;
              node->*Left = cur;
              if (cur->*Right != nullptr) [[likely]] {
                cur->*Right->*Prev = node;
              }
              node = cur;
              cur = cur->*Left;
            }
            path_mask >>= 1;
          } while (path_mask);

          // The right child of the last node on the path is always null.
          STDEXEC_ASSERT(node->*Right == nullptr);
          node->*Left = nullptr;
          return;
        }
        if (path_mask & path) {
          cur_ptr = &(cur->*Right);
        } else {
          cur_ptr = &(cur->*Left);
        }
        parent = cur;
        cur = *cur_ptr;
        path_mask >>= 1;
      }

      *cur_ptr = node;
      node->*Prev = parent;
      node->*Right = nullptr;
      node->*Left = nullptr;
    }

    void pop_front() noexcept {
      if (size_ <= 1) [[unlikely]] {
        root_ = nullptr;
        size_ = 0;
        return;
      }
      STDEXEC_ASSERT(root_ != nullptr);
      Node* cur = remove_last_leaf(&root_, size_);
      Node* top = root_;
      // Replace the top.
      cur->*Right = top->*Right;
      cur->*Left = top->*Left;
      root_ = cur;

      size_--;
      sift_down(root_, &root_, nullptr);
      root_->*Prev = nullptr;
      return;
    }

    auto front() const noexcept -> Node* {
      return root_;
    }

    auto erase(Node* node) noexcept -> bool {
      if (node->*Prev == nullptr && node != root_) {
        // node is not in the heap
        return false;
      }
      STDEXEC_ASSERT(root_ != nullptr);
      STDEXEC_ASSERT(size_ != 0);
      Node* cur = remove_last_leaf(&root_, size_);
      size_--;
      if (cur == node) [[unlikely]] { // Remove the last leaf?
        node->*Prev = nullptr;        // Make node erasure-aware.
        return true;
      }
      // Replace the node.
      cur->*Right = node->*Right;
      cur->*Left = node->*Left;
      Node** cur_ptr;
      Node* parent = node->*Prev;
      if (parent != nullptr) {
        if (parent->*Right == node) {
          cur_ptr = &(parent->*Right);
        } else {
          cur_ptr = &(parent->*Left);
        }
        *cur_ptr = cur;
        if (cur->*Key < parent->*Key) {
          sift_up(cur, parent);
          node->*Prev = nullptr; // Make node erasure-aware.
          return true;
        }
      } else {
        cur_ptr = &root_;
        *cur_ptr = cur;
      }
      sift_down(cur, cur_ptr, parent);
      (*cur_ptr)->*Prev = parent;
      node->*Prev = nullptr; // Make node erasure-aware.
      return true;
    }

   private:
    Node* root_ = nullptr;
    std::size_t size_ = 0;

    static inline void swap_with_right_child(Node* parent, Node* child) noexcept {
      parent->*Right = child->*Right;
      child->*Right = parent;

      Node* t = parent->*Left;
      parent->*Left = child->*Left;
      child->*Left = t;
    }

    static inline void swap_with_left_child(Node* parent, Node* child) noexcept {
      Node* t = parent->*Right;
      parent->*Right = child->*Right;
      child->*Right = t;

      parent->*Left = child->*Left;
      child->*Left = parent;
    }

    static inline void sift_up(Node* cur, Node* parent) noexcept {
      Node* sentinel = nullptr;
      Node** child0_parent_ptr = (cur->*Left != nullptr) ? &(cur->*Left->*Prev) : &sentinel;
      Node** child1_parent_ptr = (cur->*Right != nullptr) ? &(cur->*Right->*Prev) : &sentinel;

      *child1_parent_ptr = parent;
      Node* grand_parent = parent->*Prev;
      do {
        *child0_parent_ptr = parent;
        if (parent->*Right == cur) {
          child0_parent_ptr = &(parent->*Left->*Prev);
          swap_with_right_child(parent, cur);
        } else {
          child0_parent_ptr = &(parent->*Right->*Prev);
          swap_with_left_child(parent, cur);
        }
        child1_parent_ptr = &(parent->*Prev);
        // The last leaf node won't percolate up beyond the top node，so the grand_parent won't be null.
        if (grand_parent->*Right == parent) {
          grand_parent->*Right = cur;
        } else {
          grand_parent->*Left = cur;
        }
        parent = grand_parent;
        grand_parent = parent->*Prev;
      } while (cur->*Key < parent->*Key);
      *child0_parent_ptr = cur;
      *child1_parent_ptr = cur;
      cur->*Prev = parent;
    }

    static inline void sift_down(Node* cur, Node** cur_ptr, Node* parent) noexcept {
      Node* right = cur->*Right;
      Node* left = cur->*Left;
      while (left != nullptr) {
        if ((right != nullptr) && (right->*Key < left->*Key)) {
          if (right->*Key < cur->*Key) {
            swap_with_right_child(cur, right);
            parent = right;
            left->*Prev = right;
            *cur_ptr = right;
            cur_ptr = &(right->*Right);
          } else {
            right->*Prev = cur;
            left->*Prev = cur;
            break;
          }
        } else {
          if (left->*Key < cur->*Key) {
            swap_with_left_child(cur, left);
            parent = left;
            if (right != nullptr) [[likely]] {
              right->*Prev = left;
            }
            *cur_ptr = left;
            cur_ptr = &(left->*Left);
          } else {
            if (right != nullptr) [[likely]] {
              right->*Prev = cur;
            }
            left->*Prev = cur;
            break;
          }
        }
        left = cur->*Left;
        right = cur->*Right;
      }
      cur->*Prev = parent;
    }

    static inline Node* remove_last_leaf(Node** cur_ptr, std::size_t path) noexcept {
      auto path_mask = detail::path_bit_mask(path);
      Node* cur = *cur_ptr;
      while (path_mask) {
        if (path_mask & path) {
          cur_ptr = &(cur->*Right);
        } else {
          cur_ptr = &(cur->*Left);
        }
        cur = *cur_ptr;
        path_mask >>= 1;
      }
      *cur_ptr = nullptr;
      return cur;
    }
  };
} // namespace exec

STDEXEC_PRAGMA_POP()
