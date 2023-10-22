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
#include <new>
#include <vector>

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
    Tp *value;
  };

  struct takeover_result {
    std::size_t front;
    std::size_t back;
  };

  template <class Tp, class Allocator = std::allocator<Tp *>>
  class lifo_queue {
   public:
    explicit lifo_queue(
      std::size_t num_blocks,
      std::size_t block_size,
      Allocator allocator = Allocator());
    ~lifo_queue();

    Tp *get() noexcept;

    Tp *steal() noexcept;

    bool put(Tp *value) noexcept;

    std::size_t get_available_capacity() const noexcept;

    std::size_t get_block_size() const noexcept;

   private:
    template <class Sp>
    using allocator_of_t = typename std::allocator_traits<Allocator>::template rebind_alloc<Sp>;

    struct block_type {
      explicit block_type(std::size_t block_size, Allocator allocator = Allocator());
      ~block_type();

      block_type(const block_type &) = delete;
      block_type &operator=(const block_type &) = delete;

      block_type(block_type &&) noexcept;
      block_type &operator=(block_type &&) noexcept;

      lifo_queue_error_code put(Tp *value) noexcept;

      fetch_result get() noexcept;

      fetch_result steal() noexcept;

      takeover_result takeover() noexcept;

      std::size_t free_capacity() const noexcept;

      void grant() noexcept;

      bool reclaim(std::size_t expectedPos) noexcept;

      bool is_stealable() const noexcept;

      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> tail_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_tail_{};
      Tp **ring_buffer_{};
      std::size_t block_size_;
      Allocator allocator_;
    };

    bool advance_get_index() noexcept;
    bool advance_steal_index(std::size_t expectedThiefCounter) noexcept;
    bool advance_put_index() noexcept;

    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> owner_block_{1};
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> thief_block_{0};
    block_type *blocks_{nullptr};
    std::size_t mask_{};
    allocator_of_t<block_type> allocator_;
  };

  // Implementation

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::lifo_queue(
    std::size_t num_blocks,
    std::size_t block_size,
    Allocator allocator)
    : mask_(std::bit_ceil(num_blocks) - 1)
    , allocator_(allocator) {
    std::size_t size = mask_ + 1;
    blocks_ = std::allocator_traits<allocator_of_t<block_type>>::allocate(allocator_, size);
    for (std::size_t i = 0; i < size; ++i) {
      try {
        std::allocator_traits<allocator_of_t<block_type>>::construct(
          allocator_, blocks_ + i, block_size, allocator);
      } catch (...) {
        for (std::size_t j = 0; j < i; ++j) {
          std::allocator_traits<allocator_of_t<block_type>>::destroy(allocator_, blocks_ + j);
        }
        std::allocator_traits<allocator_of_t<block_type>>::deallocate(allocator_, blocks_, size);
        throw;
      }
    }
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::~lifo_queue() {
    std::size_t size = mask_ + 1;
    for (std::size_t i = 0; i < size; ++i) {
      std::allocator_traits<allocator_of_t<block_type>>::destroy(allocator_, blocks_ + i);
    }
    std::allocator_traits<allocator_of_t<block_type>>::deallocate(allocator_, blocks_, size);
  }

  template <class Tp, class Allocator>
  Tp *lifo_queue<Tp, Allocator>::get() noexcept {
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
  Tp *lifo_queue<Tp, Allocator>::steal() noexcept {
    std::size_t thief = 0;
    do {
      thief = thief_block_.load(std::memory_order_relaxed);
      std::size_t thief_index = thief & mask_;
      block_type &block = blocks_[thiefIndex];
      fetch_result result = block.steal();
      while (result.status != LifoQueueErrorCode::done) {
        if (result.status == LifoQueueErrorCode::success) {
          return result.value;
        }
        if (result.status == LifoQueueErrorCode::empty) {
          return nullptr;
        }
        result = block.steal();
      }
    } while (advance_steal_index(thief));
    return nullptr;
  }

  template <class Tp, class Allocator>
  bool lifo_queue<Tp, Allocator>::put(Tp *value) noexcept {
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
}