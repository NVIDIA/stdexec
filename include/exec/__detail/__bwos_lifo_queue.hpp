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
#include "../../stdexec/__detail/__spin_loop_pause.hpp"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

    auto pop_back() noexcept -> Tp;

    auto steal_front() noexcept -> Tp;

    auto push_back(Tp value) noexcept -> bool;

    template <class Iterator, class Sentinel>
    auto push_back(Iterator first, Sentinel last) noexcept -> Iterator;

    [[nodiscard]]
    auto get_available_capacity() const noexcept -> std::size_t;
    [[nodiscard]]
    auto get_free_capacity() const noexcept -> std::size_t;

    [[nodiscard]]
    auto block_size() const noexcept -> std::size_t;
    [[nodiscard]]
    auto num_blocks() const noexcept -> std::size_t;

   private:
    template <class Sp>
    using allocator_of_t = typename std::allocator_traits<Allocator>::template rebind_alloc<Sp>;

    struct block_type {
      explicit block_type(std::size_t block_size, Allocator allocator = Allocator());

      block_type(const block_type &);
      auto operator=(const block_type &) -> block_type &;

      block_type(block_type &&) noexcept;
      auto operator=(block_type &&) noexcept -> block_type &;

      auto put(Tp value) noexcept -> lifo_queue_error_code;

      template <class Iterator, class Sentinel>
      auto bulk_put(Iterator first, Sentinel last) noexcept -> Iterator;

      auto get() noexcept -> fetch_result<Tp>;

      auto steal() noexcept -> fetch_result<Tp>;

      auto takeover() noexcept -> takeover_result;
      [[nodiscard]]
      auto is_writable() const noexcept -> bool;

      [[nodiscard]]
      auto free_capacity() const noexcept -> std::size_t;

      void grant() noexcept;

      auto reclaim() noexcept -> bool;

      [[nodiscard]]
      auto is_stealable() const noexcept -> bool;

      [[nodiscard]]
      auto block_size() const noexcept -> std::size_t;

      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> tail_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_head_{};
      alignas(hardware_destructive_interference_size) std::atomic<std::uint64_t> steal_tail_{};
      std::vector<Tp, Allocator> ring_buffer_;
    };

    auto advance_get_index() noexcept -> bool;
    auto advance_steal_index(std::size_t expected_thief_counter) noexcept -> bool;
    auto advance_put_index() noexcept -> bool;

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
    blocks_[owner_block_.load()].reclaim();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::pop_back() noexcept -> Tp {
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
  auto lifo_queue<Tp, Allocator>::steal_front() noexcept -> Tp {
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
  auto lifo_queue<Tp, Allocator>::push_back(Tp value) noexcept -> bool {
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
  auto lifo_queue<Tp, Allocator>::push_back(Iterator first, Sentinel last) noexcept -> Iterator {
    do {
      std::size_t owner_index = owner_block_.load(std::memory_order_relaxed) & mask_;
      block_type &current_block = blocks_[owner_index];
      first = current_block.bulk_put(first, last);
    } while (first != last && advance_put_index());
    return first;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::get_free_capacity() const noexcept -> std::size_t {
    std::size_t owner_counter = owner_block_.load(std::memory_order_relaxed);
    std::size_t owner_index = owner_counter & mask_;
    std::size_t local_capacity = blocks_[owner_index].free_capacity();
    std::size_t thief_counter = thief_block_.load(std::memory_order_relaxed);
    std::size_t diff = owner_counter - thief_counter;
    std::size_t rest = blocks_.size() - diff - 1;
    return local_capacity + rest * block_size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::get_available_capacity() const noexcept -> std::size_t {
    return num_blocks() * block_size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_size() const noexcept -> std::size_t {
    return blocks_[0].block_size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::num_blocks() const noexcept -> std::size_t {
    return blocks_.size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::advance_get_index() noexcept -> bool {
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
  auto lifo_queue<Tp, Allocator>::advance_put_index() noexcept -> bool {
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
  auto lifo_queue<Tp, Allocator>::advance_steal_index(std::size_t expected_thief_counter) noexcept
    -> bool {
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
  lifo_queue<Tp, Allocator>::block_type::block_type(const block_type &other)
    : ring_buffer_(other.ring_buffer_) {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::operator=(const block_type &other) ->
    typename lifo_queue<Tp, Allocator>::block_type & {
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
  auto lifo_queue<Tp, Allocator>::block_type::operator=(block_type &&other) noexcept ->
    typename lifo_queue<Tp, Allocator>::block_type & {
    head_.store(other.head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    tail_.store(other.tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    steal_head_.store(other.steal_head_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ring_buffer_ = std::exchange(std::move(other.ring_buffer_), {});
    return *this;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::put(Tp value) noexcept -> lifo_queue_error_code {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (back < block_size()) [[likely]] {
      ring_buffer_[static_cast<std::size_t>(back)] = static_cast<Tp &&>(value);
      tail_.store(back + 1, std::memory_order_release);
      return lifo_queue_error_code::success;
    }
    return lifo_queue_error_code::full;
  }

  template <class Tp, class Allocator>
  template <class Iterator, class Sentinel>
  auto lifo_queue<Tp, Allocator>::block_type::bulk_put(Iterator first, Sentinel last) noexcept
    -> Iterator {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    while (first != last && back < block_size()) {
      ring_buffer_[static_cast<std::size_t>(back)] = static_cast<Tp &&>(*first);
      ++back;
      ++first;
    }
    tail_.store(back, std::memory_order_release);
    return first;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::get() noexcept -> fetch_result<Tp> {
    std::uint64_t front = head_.load(std::memory_order_relaxed);
    if (front == block_size()) [[unlikely]] {
      return {lifo_queue_error_code::done, nullptr};
    }
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    if (front == back) [[unlikely]] {
      return {lifo_queue_error_code::empty, nullptr};
    }
    Tp value = static_cast<Tp &&>(ring_buffer_[static_cast<std::size_t>(back - 1)]);
    tail_.store(back - 1, std::memory_order_release);
    return {lifo_queue_error_code::success, value};
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::steal() noexcept -> fetch_result<Tp> {
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
    result.value = static_cast<Tp &&>(ring_buffer_[static_cast<std::size_t>(spos)]);
    steal_head_.fetch_add(1, std::memory_order_release);
    result.status = lifo_queue_error_code::success;
    return result;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::takeover() noexcept -> takeover_result {
    std::uint64_t spos = steal_tail_.exchange(block_size(), std::memory_order_relaxed);
    if (spos == block_size()) [[unlikely]] {
      return {
        .front = static_cast<std::size_t>(head_.load(std::memory_order_relaxed)),
        .back = static_cast<std::size_t>(tail_.load(std::memory_order_relaxed))};
    }
    head_.store(spos, std::memory_order_relaxed);
    return {
      .front = static_cast<std::size_t>(spos),
      .back = static_cast<std::size_t>(tail_.load(std::memory_order_relaxed))};
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::is_writable() const noexcept -> bool {
    std::uint64_t expected_steal = block_size();
    std::uint64_t spos = steal_tail_.load(std::memory_order_relaxed);
    return spos == expected_steal;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::free_capacity() const noexcept -> std::size_t {
    std::uint64_t back = tail_.load(std::memory_order_relaxed);
    return block_size() - back;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::reclaim() noexcept -> bool {
    std::uint64_t expected_steal_head_ = tail_.load(std::memory_order_relaxed);
    while (steal_head_.load(std::memory_order_acquire) != expected_steal_head_) {
      stdexec::__spin_loop_pause();
    }
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    steal_tail_.store(block_size(), std::memory_order_relaxed);
    steal_head_.store(0, std::memory_order_relaxed);
    return false;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::block_size() const noexcept -> std::size_t {
    return ring_buffer_.size();
  }

  template <class Tp, class Allocator>
  void lifo_queue<Tp, Allocator>::block_type::grant() noexcept {
    std::uint64_t old_head = head_.exchange(block_size(), std::memory_order_relaxed);
    steal_tail_.store(old_head, std::memory_order_release);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::is_stealable() const noexcept -> bool {
    return steal_tail_.load(std::memory_order_acquire) != block_size();
  }
} // namespace exec::bwos
