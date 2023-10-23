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

#include <atomic>
#include <bit>
#include <memory>
#include <new>
#include <utility>
#include <vector>

// The below code for spin_loop_pause is taken from https://github.com/max0x7ba/atomic_queue/blob/master/include/atomic_queue/defs.h
// Copyright (c) 2019 Maxim Egorushkin. MIT License.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>

namespace exec::bwos {
  static inline void spin_loop_pause() noexcept {
    _mm_pause();
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
    asm volatile ("yield" ::: "memory");
#elif defined(_M_ARM64)
    __yield();
#else
    asm volatile ("nop" ::: "memory");
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

#if __cpp_lib_hardware_interference_size >= 201603
  inline constexpr std::size_t hardware_destructive_interference_size =
    std::hardware_destructive_interference_size;
  inline constexpr std::size_t hardware_constructive_interference_size =
    std::hardware_constructive_interference_size;
#else
  inline constexpr std::size_t hardware_destructive_interference_size = 64;
  inline constexpr std::size_t hardware_constructive_interference_size = 64;
#endif

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

    Tp get() noexcept;

    Tp steal() noexcept;

    bool put(Tp value) noexcept;

    std::size_t get_available_capacity() const noexcept;

    std::size_t get_block_size() const noexcept;

   private:
    template <class Sp>
    using allocator_of_t = typename std::allocator_traits<Allocator>::template rebind_alloc<Sp>;

    struct block_type {
      struct alloc_deleter {
        [[no_unique_address]] Allocator allocator_;
        std::size_t block_size_;

        void operator()(Tp *ptr) noexcept {
          std::allocator_traits<Allocator>::deallocate(allocator_, ptr, block_size_);
        }
      };

      explicit block_type(std::size_t block_size, Allocator allocator = Allocator());

      block_type(const block_type &) = delete;
      block_type &operator=(const block_type &) = delete;

      block_type(block_type &&) noexcept;
      block_type &operator=(block_type &&) noexcept;

      lifo_queue_error_code put(Tp value) noexcept;

      fetch_result<Tp> get() noexcept;

      fetch_result<Tp> steal() noexcept;

      takeover_result takeover() noexcept;

      std::size_t free_capacity() const noexcept;

      void grant() noexcept;

      bool reclaim(std::size_t expectedPos) noexcept;

      bool is_stealable() const noexcept;

      std::size_t block_size() const noexcept;

      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> tail_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_tail_{};
      std::unique_ptr<Tp[], alloc_deleter> ring_buffer_{};
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
    : blocks_(std::bit_ceil(num_blocks), block_type(block_size, allocator)) 
    , mask_(blocks_.size() - 1)
    {
    }

  template <class Tp, class Allocator>
  Tp lifo_queue<Tp, Allocator>::get() noexcept {
    do {
      std::size_t owner_index = owner_block_.load(std::memory_order_relaxed) & mask_;
      block_type &current_block = blocks_[owner_index];
      auto [ec, value] = current_block.get();
      if (ec == lifo_queue_error_code::success) {
        return value;
      }
      if (ec == lifo_queue_error_code::done) {
        return nullptr;
      }
    } while (advance_get_index());
    return nullptr;
  }

  template <class Tp, class Allocator>
  Tp lifo_queue<Tp, Allocator>::steal() noexcept {
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
          return nullptr;
        }
        result = block.steal();
      }
    } while (advance_steal_index(thief));
    return nullptr;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::put(Tp value) noexcept {
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
  std::size_t lifo_queue<Tp, Allocator>::get_available_capacity() const noexcept {
    std::size_t block_size = blocks_[0].block_size_;
    std::size_t owner_counter = owner_block_.load(std::memory_order_relaxed);
    std::size_t owner_index = owner_counter & mask_;
    std::size_t local_capacity = blocks_[owner_index].GetAvailableCapacity();
    std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
    std::size_t diff = owner_counter - thief_counter;
    std::size_t rest = blocks_.size() - diff - 1;
    return local_capacity + rest * block_size;
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::get_block_size() const noexcept {
    return blocks_[0].block_size_;
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
    std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
    std::size_t next_counter = owner_counter + 1ul;
    if (next_counter == thief_counter + blocks_.size()) [[unlikely]] {
      return false;
    }
    std::size_t owner_index = owner_counter & mask_;
    std::size_t next_index = next_counter & mask_;
    block_type &currentBlock = blocks_[owner_index];
    block_type &nextBlock = blocks_[next_index];
    takeover_result result = nextBlock.takeover();
    currentBlock.grant();
    owner_block_.store(next_counter, std::memory_order_relaxed);
    while (!nextBlock.reclaim(result.front)) {
      spin_loop_pause();
    }
    return true;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::advance_steal_index(std::size_t expected_thief_counter) noexcept {
    std::size_t thief_counter = expected_thief_counter;
    std::size_t next_counter = thief_counter + 1;
    std::size_t next_index = next_counter & mask_;
    block_type &nextBlock = blocks_[next_index];
    if (nextBlock.is_stealable()) {
      thief_block_.compare_exchange_strong(thief_counter, next_counter, std::memory_order_relaxed);
      return true;
    }
    return thief_block_.load(std::memory_order_relaxed) != thief_counter;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Implementation of lifo_queue::block_type member methods

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(
    std::size_t block_size, Allocator allocator)
    : steal_tail_{block_size}
    , ring_buffer_{std::make_unique<Tp[]>(block_size, alloc_deleter{allocator, block_size})} {
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(block_type &&other) noexcept {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(
      other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = std::exchange(other.ring_buffer_, nullptr);
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type &
  lifo_queue<Tp, Allocator>::block_type::operator=(block_type &&other) noexcept {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(
      other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = std::exchange(other.ring_buffer_, nullptr);
    return *this;
  }

  template <class Tp, class Allocator>
  lifo_queue_error_code lifo_queue<Tp, Allocator>::block_type::put(Tp value) noexcept {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (back == block_size()) {
      return lifo_queue_error_code::full;
    }
    ring_buffer_[back] = value;
    tail_.store(back + 1, std::memory_order_release);
    return lifo_queue_error_code::success;
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
    tail_.store(back - 1, std::memory_order_relaxed);
    return {lifo_queue_error_code::success, ring_buffer_[back - 1]};
  }

  template <class Tp, class Allocator>
  fetch_result<Tp> lifo_queue<Tp, Allocator>::block_type::steal() noexcept {
    std::uint64_t steal = steal_tail_.load(std::memory_order_relaxed);
    if (steal == block_size()) {
      return {lifo_queue_error_code::done, nullptr};
    }
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (back == steal) {
      return {lifo_queue_error_code::empty, nullptr};
    }
    if (!steal_tail_.compare_exchange_strong(steal, steal + 1, std::memory_order_relaxed)) {
      return {lifo_queue_error_code::conflict, nullptr};
    }
    void *value = ring_buffer_[steal];
    steal_head_.fetch_add(1, std::memory_order_relaxed);
    return {lifo_queue_error_code::success, value};
  }

  template <class Tp, class Allocator>
  takeover_result lifo_queue<Tp, Allocator>::block_type::takeover() noexcept {
    std::uint64_t sPos = steal_tail_.exchange(block_size(), std::memory_order_relaxed);
    if (sPos == block_size()) [[unlikely]] {
      return {head_.load(std::memory_order_relaxed), tail_.load(std::memory_order_relaxed)};
    }
    head_.store(sPos, std::memory_order_relaxed);
    return {sPos, tail_.load(std::memory_order_relaxed)};
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::block_type::free_capacity() const noexcept {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    return block_size() - back;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::block_type::reclaim(std::size_t expectedPos) noexcept {
    std::uint64_t sCnt = steal_head_.load(std::memory_order_relaxed);
    if (expectedPos == sCnt) {
      head_.store(0, std::memory_order_relaxed);
      tail_.store(0, std::memory_order_relaxed);
      steal_tail_.store(block_size(), std::memory_order_relaxed);
      steal_head_.store(0, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

  template <class Tp, class Allocator>
  std::size_t lifo_queue<Tp, Allocator>::block_type::block_size() const noexcept {
    return ring_buffer_.get_deleter().block_size_;
  }

  template <class Tp, class Allocator>
  void lifo_queue<Tp, Allocator>::block_type::grant() noexcept {
    std::uint64_t fPos = head_.exchange(block_size(), std::memory_order_relaxed);
    steal_tail_.store(fPos, std::memory_order_release);
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::block_type::is_stealable() const noexcept {
    return steal_tail_.load(std::memory_order_acquire) != block_size();
  }
}