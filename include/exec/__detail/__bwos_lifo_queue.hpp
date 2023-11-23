/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <atomic>
#include <bit>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

// The below code for spin_loop_pause is taken from https://github.com/max0x7ba/atomic_queue/blob/master/include/atomic_queue/defs.h
// Copyright (c) 2019 Maxim Egorushkin. MIT License.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if STDEXEC_MSVC()
#include <intrin.h>
#endif
namespace exec::bwos {
  static inline void spin_loop_pause() noexcept {
#if STDEXEC_MSVC()
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
  }
}
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
namespace exec::bwos {
  static inline void spin_loop_pause() noexcept {
#if ( \
  defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__) \
  || defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) \
  || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__) \
  || defined(__ARM_ARCH_8A__) || defined(__aarch64__))
    asm volatile("yield" ::: "memory");
#elif defined(_M_ARM64)
    __yield();
#else
    asm volatile("nop" ::: "memory");
#endif
  }
}
#else
namespace exec::bwos {
  static inline void spin_loop_pause() noexcept {
  }
}
#endif

/** 
 * This is an implementation of the BWOS queue as described in
 * BWoS: Formally Verified Block-based Work Stealing for Parallel Processing (Wang et al. 2023)
 */

namespace exec::bwos {
  inline constexpr std::size_t hardware_destructive_interference_size = 64;
  inline constexpr std::size_t hardware_constructive_interference_size = 64;

  enum class lifo_queue_error_code {
    success,
    done,
    empty,
    full,
    conflict,
  };

  template <class Tp>
  struct fetch_result {
    lifo_queue_error_code status;
    Tp value;
  };

  struct takeover_result {
    std::size_t front;
    std::size_t back;
  };

  template <class Tp, class Allocator = std::allocator<Tp>>
  class lifo_queue {
   public:
    explicit lifo_queue(
      std::size_t num_blocks,
      std::size_t block_size,
      Allocator allocator = Allocator());

    Tp pop_back() noexcept;

    Tp steal_front() noexcept;

    bool push_back(Tp value) noexcept;

    template <class Iterator, class Sentinel>
    Iterator push_back(Iterator first, Sentinel last) noexcept;

    std::size_t get_available_capacity() const noexcept;
    std::size_t get_free_capacity() const noexcept;

    std::size_t block_size() const noexcept;
    std::size_t num_blocks() const noexcept;

   private:
    template <class Sp>
    using allocator_of_t = typename std::allocator_traits<Allocator>::template rebind_alloc<Sp>;

    struct block_type {
      explicit block_type(std::size_t block_size, Allocator allocator = Allocator());

      block_type(const block_type &);
      block_type &operator=(const block_type &);

      block_type(block_type &&) noexcept;
      block_type &operator=(block_type &&) noexcept;

      lifo_queue_error_code put(Tp value) noexcept;

      template <class Iterator, class Sentinel>
      Iterator bulk_put(Iterator first, Sentinel last) noexcept;

      fetch_result<Tp> get() noexcept;

      fetch_result<Tp> steal() noexcept;

      takeover_result takeover() noexcept;
      bool is_writable() const noexcept;

      std::size_t free_capacity() const noexcept;

      void grant() noexcept;

      bool reclaim() noexcept;

      bool is_stealable() const noexcept;

      std::size_t block_size() const noexcept;

      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> tail_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_tail_{};
      std::vector<Tp, Allocator> ring_buffer_{};
    };

    bool advance_get_index() noexcept;
    bool advance_steal_index(std::size_t expected_thief_counter) noexcept;
    bool advance_put_index() noexcept;

    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> owner_block_{1};
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> thief_block_{0};
    std::vector<block_type, allocator_of_t<block_type>> blocks_{};
    std::size_t mask_{};
  };

  /////////////////////////////////////////////////////////////////////////////
  // Implementation of lifo_queue member methods

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::lifo_queue(
    std::size_t num_blocks,
    std::size_t block_size,
    Allocator allocator)
    : blocks_(
      std::max(static_cast<size_t>(2), std::bit_ceil(num_blocks)),
      block_type(block_size, allocator),
      allocator_of_t<block_type>(allocator))
    , mask_(blocks_.size() - 1) {
    blocks_[owner_block_].reclaim();
  }

  template <class Tp, class Allocator>
  Tp lifo_queue<Tp, Allocator>::pop_back() noexcept {
    do {
      std::size_t owner_index = owner_block_.load(std::memory_order_relaxed) & mask_;
      block_type &current_block = blocks_[owner_index];
      auto [ec, value] = current_block.get();
      if (ec == lifo_queue_error_code::success) {
        return value;
      }
      if (ec == lifo_queue_error_code::done) {
        return Tp{};
      }
    } while (advance_get_index());
    return Tp{};
  }

  template <class Tp, class Allocator>
  Tp lifo_queue<Tp, Allocator>::steal_front() noexcept {
    std::size_t thief = 0;
    do {
      thief = thief_block_.load(std::memory_order_relaxed);
      std::size_t thief_index = thief & mask_;
      block_type &block = blocks_[thief_index];
      fetch_result result = block.steal();
      while (result.status != lifo_queue_error_code::done) {
        if (result.status == lifo_queue_error_code::success) {
          return result.value;
        }
        if (result.status == lifo_queue_error_code::empty) {
          return Tp{};
        }
        result = block.steal();
      }
    } while (advance_steal_index(thief));
    return Tp{};
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::push_back(Tp value) noexcept {
    do {
      std::size_t owner_index = owner_block_.load(std::memory_order_relaxed) & mask_;
      block_type &current_block = blocks_[owner_index];
      auto ec = current_block.put(value);
      if (ec == lifo_queue_error_code::success) {
        return true;
      }
    } while (advance_put_index());
    return false;
  }

  template <class Tp, class Allocator>
  template <class Iterator, class Sentinel>
  Iterator lifo_queue<Tp, Allocator>::push_back(Iterator first, Sentinel last) noexcept {
    do {
      std::size_t owner_index = owner_block_.load(std::memory_order_relaxed) & mask_;
      block_type &current_block = blocks_[owner_index];
      first = current_block.bulk_put(first, last);
    } while (first != last && advance_put_index());
    return first;
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::get_free_capacity() const noexcept {
    std::size_t owner_counter = owner_block_.load(std::memory_order_relaxed);
    std::size_t owner_index = owner_counter & mask_;
    std::size_t local_capacity = blocks_[owner_index].free_capacity();
    std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
    std::size_t diff = owner_counter - thief_counter;
    std::size_t rest = blocks_.size() - diff - 1;
    return local_capacity + rest * block_size();
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::get_available_capacity() const noexcept {
    return num_blocks() * block_size();
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::block_size() const noexcept {
    return blocks_[0].block_size();
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::num_blocks() const noexcept {
    return blocks_.size();
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::advance_get_index() noexcept {
    std::size_t owner_counter = owner_block_.load(std::memory_order_relaxed);
    std::size_t predecessor = owner_counter - 1ul;
    std::size_t predecessor_index = predecessor & mask_;
    block_type &previous_block = blocks_[predecessor_index];
    takeover_result result = previous_block.takeover();
    if (result.front != result.back) {
      std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
      if (thief_counter == predecessor) {
        predecessor += blocks_.size();
        thief_counter += blocks_.size() - 1ul;
        thief_block_.store(thief_counter, std::memory_order_relaxed);
      }
      owner_block_.store(predecessor, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::advance_put_index() noexcept {
    std::size_t owner_counter = owner_block_.load(std::memory_order_relaxed);
    std::size_t next_counter = owner_counter + 1ul;
    std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
    STDEXEC_ASSERT(thief_counter < next_counter);
    if (next_counter - thief_counter >= blocks_.size()) {
      return false;
    }
    std::size_t next_index = next_counter & mask_;
    block_type &next_block = blocks_[next_index];
    if (!next_block.is_writable()) [[unlikely]] {
      return false;
    }
    std::size_t owner_index = owner_counter & mask_;
    block_type &current_block = blocks_[owner_index];
    current_block.grant();
    owner_block_.store(next_counter, std::memory_order_relaxed);
    next_block.reclaim();
    return true;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::advance_steal_index(std::size_t expected_thief_counter) noexcept {
    std::size_t thief_counter = expected_thief_counter;
    std::size_t next_counter = thief_counter + 1;
    std::size_t next_index = next_counter & mask_;
    block_type &next_block = blocks_[next_index];
    if (next_block.is_stealable()) {
      thief_block_.compare_exchange_strong(thief_counter, next_counter, std::memory_order_relaxed);
      return true;
    }
    return thief_block_.load(std::memory_order_relaxed) != thief_counter;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Implementation of lifo_queue::block_type member methods

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(std::size_t block_size, Allocator allocator)
    : head_{0}
    , tail_{0}
    , steal_head_{0}
    , steal_tail_{block_size}
    , ring_buffer_(block_size, allocator) {
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(const block_type &other) {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = other.ring_buffer_;
  }

  template <class Tp, class Allocator>
  typename lifo_queue<Tp, Allocator>::block_type &
    lifo_queue<Tp, Allocator>::block_type::operator=(const block_type &other) {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = other.ring_buffer_;
    return *this;
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(block_type &&other) noexcept {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = std::exchange(std::move(other.ring_buffer_), {});
  }

  template <class Tp, class Allocator>
  typename lifo_queue<Tp, Allocator>::block_type &
    lifo_queue<Tp, Allocator>::block_type::operator=(block_type &&other) noexcept {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = std::exchange(std::move(other.ring_buffer_), {});
    return *this;
  }

  template <class Tp, class Allocator>
  lifo_queue_error_code lifo_queue<Tp, Allocator>::block_type::put(Tp value) noexcept {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (back < block_size()) [[likely]] {
      ring_buffer_[back] = static_cast<Tp &&>(value);
      tail_.store(back + 1, std::memory_order_release);
      return lifo_queue_error_code::success;
    }
    return lifo_queue_error_code::full;
  }

  template <class Tp, class Allocator>
  template <class Iterator, class Sentinel>
  Iterator lifo_queue<Tp, Allocator>::block_type::bulk_put(Iterator first, Sentinel last) noexcept {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    while (first != last && back < block_size()) {
      ring_buffer_[back] = static_cast<Tp &&>(*first);
      ++back;
      ++first;
    }
    tail_.store(back, std::memory_order_release);
    return first;
  }

  template <class Tp, class Allocator>
  fetch_result<Tp> lifo_queue<Tp, Allocator>::block_type::get() noexcept {
    std::uint64_t front = head_.load(std::memory_order_relaxed);
    if (front == block_size()) [[unlikely]] {
      return {lifo_queue_error_code::done, nullptr};
    }
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (front == back) [[unlikely]] {
      return {lifo_queue_error_code::empty, nullptr};
    }
    Tp value = static_cast<Tp &&>(ring_buffer_[back - 1]);
    tail_.store(back - 1, std::memory_order_release);
    return {lifo_queue_error_code::success, value};
  }

  template <class Tp, class Allocator>
  fetch_result<Tp> lifo_queue<Tp, Allocator>::block_type::steal() noexcept {
    std::uint64_t spos = steal_tail_.load(std::memory_order_relaxed);
    fetch_result<Tp> result{};
    if (spos == block_size()) [[unlikely]] {
      result.status = lifo_queue_error_code::done;
      return result;
    }
    std::uint64_t back = tail_.load(std::memory_order_acquire);
    if (spos == back) [[unlikely]] {
      result.status = lifo_queue_error_code::empty;
      return result;
    }
    if (!steal_tail_.compare_exchange_strong(spos, spos + 1, std::memory_order_relaxed)) {
      result.status = lifo_queue_error_code::conflict;
      return result;
    }
    result.value = static_cast<Tp &&>(ring_buffer_[spos]);
    steal_head_.fetch_add(1, std::memory_order_release);
    result.status = lifo_queue_error_code::success;
    return result;
  }

  template <class Tp, class Allocator>
  takeover_result lifo_queue<Tp, Allocator>::block_type::takeover() noexcept {
    std::uint64_t spos = steal_tail_.exchange(block_size(), std::memory_order_relaxed);
    if (spos == block_size()) [[unlikely]] {
      return {head_.load(std::memory_order_relaxed), tail_.load(std::memory_order_relaxed)};
    }
    head_.store(spos, std::memory_order_relaxed);
    return {spos, tail_.load(std::memory_order_relaxed)};
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::block_type::is_writable() const noexcept {
    std::uint64_t expected_steal = block_size();
    std::uint64_t spos = steal_tail_.load(std::memory_order_relaxed);
    return spos == expected_steal;
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::block_type::free_capacity() const noexcept {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    return block_size() - back;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::block_type::reclaim() noexcept {
    std::uint64_t expected_steal_head_ = tail_.load(std::memory_order_relaxed);
    while (steal_head_.load(std::memory_order_acquire) != expected_steal_head_) {
      spin_loop_pause();
    }
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    steal_tail_.store(block_size(), std::memory_order_relaxed);
    steal_head_.store(0, std::memory_order_relaxed);
    return false;
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::block_type::block_size() const noexcept {
    return ring_buffer_.size();
  }

  template <class Tp, class Allocator>
  void lifo_queue<Tp, Allocator>::block_type::grant() noexcept {
    std::uint64_t old_head = head_.exchange(block_size(), std::memory_order_relaxed);
    steal_tail_.store(old_head, std::memory_order_release);
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::block_type::is_stealable() const noexcept {
    return steal_tail_.load(std::memory_order_acquire) != block_size();
  }
}