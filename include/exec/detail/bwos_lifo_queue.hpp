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

#include "../../stdexec/__detail/__atomic.hpp"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

/**
 * This is an implementation of the BWOS queue as described in
 * BWoS: Formally Verified Block-based Work Stealing for Parallel Processing (Wang et al. 2023)
 *
 * BLOCK COUNTER ENCODING:
 * ----------------------
 * Block counters (last_block_, start_block_, and internal counters) are 64-bit values with:
 *   - Bits [63:32]: Round number (32-bit unsigned, wraps on overflow)
 *   - Bits [31:0]:  Block index within the circular block array (masked by mask_)
 *
 * The round number increments each time the index wraps around from mask_ to 0.
 * This encoding allows:
 *   - Direct comparison of counters to determine relative order
 *   - ABA problem prevention through round tracking
 *   - Efficient atomic operations on a single 64-bit value
 *
 * TWO-LEVEL ROUND SYSTEM:
 * ----------------------
 * 1. Queue-level rounds: Tracked in last_block_/start_block_ upper 32 bits
 *    - Increments when the owner wraps around the circular block array
 *    - Passed to blocks during reclaim() to initialize block-level rounds
 *
 * 2. Block-level rounds: Stored in head_/steal_tail_ upper 32 bits
 *    - Prevents ABA problems when blocks are reused
 *    - Synchronized through takeover()/grant() atomic swaps
 *    - Verified by is_writable() before block reuse
 *
 * THREAD SAFETY:
 * -------------
 * - Owner thread: Exclusively calls push_back() and pop_back()
 * - Thief threads: Multiple threads can concurrently call steal_front()
 * - last_block_: Modified only by owner (relaxed ordering)
 * - start_block_: Modified only by owner when reclaiming blocks
 * - Block synchronization via head_/steal_tail_ swaps and acquire/release ordering
 */

namespace experimental::execution::bwos
{
  inline constexpr std::size_t hardware_destructive_interference_size  = 64;
  inline constexpr std::size_t hardware_constructive_interference_size = 64;

  enum class lifo_queue_error_code
  {
    success,
    done,
    empty,
    full,
    conflict,
  };

  template <class Tp>
  struct fetch_result
  {
    lifo_queue_error_code status;
    Tp                    value;
  };

  template <class Tp, class Allocator = std::allocator<Tp>>
  class lifo_queue
  {
   public:
    explicit lifo_queue(std::size_t num_blocks,
                        std::size_t block_size,
                        Allocator   allocator = Allocator());

    auto pop_back() noexcept -> Tp;

    auto steal_front() noexcept -> Tp;

    auto push_back(Tp value) noexcept -> bool;

    template <class Iterator, class Sentinel>
    auto push_back(Iterator first, Sentinel last) noexcept -> Iterator;

    [[nodiscard]]
    auto block_size() const noexcept -> std::size_t;
    [[nodiscard]]
    auto num_blocks() const noexcept -> std::size_t;

   private:
    template <class Sp>
    using allocator_of_t = std::allocator_traits<Allocator>::template rebind_alloc<Sp>;

    struct block_type
    {
      explicit block_type(std::size_t block_size, Allocator allocator = Allocator());

      block_type(block_type const &);
      auto operator=(block_type const &) -> block_type&;

      block_type(block_type&&) noexcept;
      auto operator=(block_type&&) noexcept -> block_type&;

      auto put(Tp value) noexcept -> lifo_queue_error_code;

      template <class Iterator, class Sentinel>
      auto bulk_put(Iterator first, Sentinel last) noexcept -> Iterator;

      auto get() noexcept -> fetch_result<Tp>;

      auto steal(std::uint32_t round) noexcept -> fetch_result<Tp>;

      auto takeover() noexcept -> void;

      [[nodiscard]]
      auto is_writable(std::uint32_t round) const noexcept -> bool;

      void grant() noexcept;

      auto reclaim(std::uint32_t round) noexcept -> void;

      [[nodiscard]]
      auto block_size() const noexcept -> std::size_t;

      auto reduce_round() noexcept -> void;

      // Block-level synchronization state (each on separate cache line to avoid false sharing)

      // Combined round (upper 32 bits) and index (lower 32 bits) for owner-side access.
      // Swapped with steal_tail_ during takeover()/grant() operations.
      alignas(hardware_destructive_interference_size) STDEXEC::__std::atomic<std::uint64_t> head_{};

      // Current tail position for owner push/pop operations (plain index, no round).
      // Modified only by owner thread.
      alignas(hardware_destructive_interference_size) STDEXEC::__std::atomic<std::uint64_t> tail_{};

      // Count of completed steal operations. Used to synchronize with thieves during reclaim().
      // Incremented by thieves after successful steal.
      alignas(hardware_destructive_interference_size)
        STDEXEC::__std::atomic<std::uint64_t> steal_count_{};
      // Combined round (upper 32 bits) and steal position (lower 32 bits) for thief access.
      // When lower bits == block_size(), the block is exhausted for stealing.
      // Swapped with head_ during takeover()/grant() operations.
      alignas(
        hardware_destructive_interference_size) STDEXEC::__std::atomic<std::uint64_t> steal_tail_{};

      std::vector<Tp, Allocator> ring_buffer_;
    };

    auto advance_get_index(std::size_t& owner, std::size_t owner_index) noexcept -> bool;
    auto advance_steal_index(std::size_t& thief) noexcept -> bool;
    auto advance_put_index(std::size_t& owner) noexcept -> bool;

    auto increase_block_counter(std::size_t counter) const noexcept -> std::size_t;
    auto decrease_block_counter(std::size_t counter) const noexcept -> std::size_t;

    // Block counter (round in upper 32 bits, index in lower bits) for the most recent
    // block owned by push/pop operations. Modified only by the owner thread.
    alignas(hardware_destructive_interference_size) STDEXEC::__std::atomic<std::size_t> last_block_{
      0};

    // Block counter for the oldest block available for stealing.
    // Modified only by the owner thread when it advances to a new block and needs to
    // reclaim the block at start_block_ position.
    alignas(
      hardware_destructive_interference_size) STDEXEC::__std::atomic<std::size_t> start_block_{0};

    // Circular array of blocks. Size is always a power of 2 for efficient masking.
    std::vector<block_type, allocator_of_t<block_type>> blocks_{};

    // Bitmask (blocks_.size() - 1) for extracting block index from counters.
    std::size_t mask_{};
  };

  /////////////////////////////////////////////////////////////////////////////
  // Implementation of lifo_queue member methods

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::lifo_queue(std::size_t num_blocks,
                                        std::size_t block_size,
                                        Allocator   allocator)
    : blocks_(std::bit_ceil(num_blocks),
              block_type(block_size, allocator),
              allocator_of_t<block_type>(allocator))
    , mask_(blocks_.size() - 1)
  {
    blocks_[0].reclaim(0);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::pop_back() noexcept -> Tp
  {
    std::size_t owner = last_block_.load(STDEXEC::__std::memory_order_relaxed);
    std::size_t owner_index{};
    do
    {
      owner_index               = owner & mask_;
      block_type& current_block = blocks_[owner_index];
      auto [ec, value]          = current_block.get();
      if (ec == lifo_queue_error_code::success)
      {
        return value;
      }
      if (ec == lifo_queue_error_code::done)
      {
        return Tp{};
      }
      assert(ec == lifo_queue_error_code::empty);
    }
    while (advance_get_index(owner, owner_index));
    return Tp{};
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::steal_front() noexcept -> Tp
  {
    std::size_t thief = start_block_.load(STDEXEC::__std::memory_order_relaxed);
    do
    {
      auto const        thief_round = static_cast<std::uint32_t>(thief >> 32);
      std::size_t const thief_index = thief & mask_;
      block_type&       block       = blocks_[thief_index];
      fetch_result      result      = block.steal(thief_round);
      while (result.status != lifo_queue_error_code::done)
      {
        if (result.status == lifo_queue_error_code::success)
        {
          return result.value;
        }
        if (result.status == lifo_queue_error_code::empty)
        {
          return Tp{};
        }
        assert(result.status == lifo_queue_error_code::conflict);
        result = block.steal(thief_round);
      }
    }
    while (advance_steal_index(thief));
    return Tp{};
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::push_back(Tp value) noexcept -> bool
  {
    std::size_t owner = last_block_.load(STDEXEC::__std::memory_order_relaxed);
    do
    {
      std::size_t owner_index   = owner & mask_;
      block_type& current_block = blocks_[owner_index];
      auto        ec            = current_block.put(value);
      if (ec == lifo_queue_error_code::success)
      {
        return true;
      }
      assert(ec == lifo_queue_error_code::full);
    }
    while (advance_put_index(owner));
    return false;
  }

  template <class Tp, class Allocator>
  template <class Iterator, class Sentinel>
  auto lifo_queue<Tp, Allocator>::push_back(Iterator first, Sentinel last) noexcept -> Iterator
  {
    std::size_t owner = last_block_.load(STDEXEC::__std::memory_order_relaxed);
    do
    {
      std::size_t owner_index   = owner & mask_;
      block_type& current_block = blocks_[owner_index];
      first                     = current_block.bulk_put(first, last);
    }
    while (first != last && advance_put_index(owner));
    return first;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_size() const noexcept -> std::size_t
  {
    return blocks_[0].block_size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::num_blocks() const noexcept -> std::size_t
  {
    return blocks_.size();
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::increase_block_counter(std::size_t counter) const noexcept
    -> std::size_t
  {
    std::uint32_t round      = static_cast<std::uint32_t>(counter >> 32);
    std::size_t   index      = counter & mask_;
    std::size_t   next_index = (index + 1) & mask_;
    std::uint32_t next_round = round + static_cast<std::uint32_t>(next_index == 0);
    return (static_cast<std::size_t>(next_round) << 32) | next_index;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::decrease_block_counter(std::size_t counter) const noexcept
    -> std::size_t
  {
    std::uint32_t round      = static_cast<std::uint32_t>(counter >> 32);
    std::size_t   index      = counter & mask_;
    std::size_t   prev_index = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(index) - 1)
                           & mask_;
    std::uint32_t prev_round = round - static_cast<std::uint32_t>(index == 0);
    return (static_cast<std::size_t>(prev_round) << 32) | prev_index;
  }

  template <class Tp, class Allocator>
  auto
  lifo_queue<Tp, Allocator>::advance_get_index(std::size_t& owner, std::size_t owner_index) noexcept
    -> bool
  {
    std::size_t start = start_block_.load(STDEXEC::__std::memory_order_relaxed);
    if (start == owner)
    {
      // Cannot move backward past start_block_ - queue is empty
      return false;
    }
    // Move to predecessor block, properly decrementing round if wrapping backward
    std::size_t predecessor       = decrease_block_counter(owner);
    std::size_t predecessor_index = predecessor & mask_;
    block_type& previous_block    = blocks_[predecessor_index];
    block_type& current_block     = blocks_[owner_index];
    current_block.reduce_round();
    previous_block.takeover();
    last_block_.store(predecessor, STDEXEC::__std::memory_order_relaxed);
    owner = predecessor;
    return true;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::advance_put_index(std::size_t& owner) noexcept -> bool
  {
    std::size_t next_index  = (owner + 1ul) & mask_;
    std::size_t owner_index = owner & mask_;
    if (next_index == owner_index)
    {
      // Wrap-around would hit the same block - queue is full
      return false;
    }
    // Calculate next round, incrementing when wrapping to index 0
    std::uint32_t next_round = static_cast<std::uint32_t>(owner >> 32)
                             + static_cast<std::uint32_t>(next_index == 0);
    block_type& next_block = blocks_[next_index];
    if (!next_block.is_writable(next_round))
    {
      // Block is not yet safe to reuse (thieves still accessing previous generation)
      return false;
    }
    std::size_t first       = start_block_.load(STDEXEC::__std::memory_order_relaxed);
    std::size_t first_index = first & mask_;
    if (next_index == first_index)
    {
      // About to reuse the block at start_block_, so advance start_block_ forward
      start_block_.store(increase_block_counter(first), STDEXEC::__std::memory_order_relaxed);
    }
    block_type& current_block = blocks_[owner_index];
    current_block.grant();
    owner = (static_cast<std::size_t>(next_round) << 32) | next_index;
    next_block.reclaim(next_round);
    last_block_.store(owner, STDEXEC::__std::memory_order_relaxed);
    return true;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::advance_steal_index(std::size_t& thief) noexcept -> bool
  {
    thief = increase_block_counter(thief);
    return thief < last_block_.load(STDEXEC::__std::memory_order_relaxed);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Implementation of lifo_queue::block_type member methods

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(std::size_t block_size, Allocator allocator)
    : head_{0xFFFF'FFFF'0000'0000 | block_size}
    , tail_{block_size}
    , steal_count_{block_size}
    , steal_tail_{0xFFFF'FFFF'0000'0000 | block_size}
    , ring_buffer_(block_size, allocator)
  {}

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(block_type const & other)
    : ring_buffer_(other.ring_buffer_)
  {
    head_.store(other.head_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    tail_.store(other.tail_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(STDEXEC::__std::memory_order_relaxed),
                      STDEXEC::__std::memory_order_relaxed);
    steal_count_.store(other.steal_count_.load(STDEXEC::__std::memory_order_relaxed),
                       STDEXEC::__std::memory_order_relaxed);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::operator=(block_type const & other)
    -> lifo_queue<Tp, Allocator>::block_type&
  {
    head_.store(other.head_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    tail_.store(other.tail_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(STDEXEC::__std::memory_order_relaxed),
                      STDEXEC::__std::memory_order_relaxed);
    steal_count_.store(other.steal_count_.load(STDEXEC::__std::memory_order_relaxed),
                       STDEXEC::__std::memory_order_relaxed);
    ring_buffer_ = other.ring_buffer_;
    return *this;
  }

  template <class Tp, class Allocator>
  lifo_queue<Tp, Allocator>::block_type::block_type(block_type&& other) noexcept
  {
    head_.store(other.head_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    tail_.store(other.tail_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(STDEXEC::__std::memory_order_relaxed),
                      STDEXEC::__std::memory_order_relaxed);
    steal_count_.store(other.steal_count_.load(STDEXEC::__std::memory_order_relaxed),
                       STDEXEC::__std::memory_order_relaxed);
    ring_buffer_ = std::exchange(std::move(other.ring_buffer_), {});
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::operator=(block_type&& other) noexcept
    -> lifo_queue<Tp, Allocator>::block_type&
  {
    head_.store(other.head_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    tail_.store(other.tail_.load(STDEXEC::__std::memory_order_relaxed),
                STDEXEC::__std::memory_order_relaxed);
    steal_tail_.store(other.steal_tail_.load(STDEXEC::__std::memory_order_relaxed),
                      STDEXEC::__std::memory_order_relaxed);
    steal_count_.store(other.steal_count_.load(STDEXEC::__std::memory_order_relaxed),
                       STDEXEC::__std::memory_order_relaxed);
    ring_buffer_ = std::exchange(std::move(other.ring_buffer_), {});
    return *this;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::put(Tp value) noexcept -> lifo_queue_error_code
  {
    std::uint64_t back     = tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t back_idx = back & 0xFFFF'FFFFu;
    if (back_idx < block_size()) [[likely]]
    {
      ring_buffer_[static_cast<std::size_t>(back_idx)] = static_cast<Tp&&>(value);
      tail_.store(back + 1, STDEXEC::__std::memory_order_release);
      return lifo_queue_error_code::success;
    }
    return lifo_queue_error_code::full;
  }

  template <class Tp, class Allocator>
  template <class Iterator, class Sentinel>
  auto lifo_queue<Tp, Allocator>::block_type::bulk_put(Iterator first, Sentinel last) noexcept
    -> Iterator
  {
    std::uint64_t back     = tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t back_idx = back & 0xFFFF'FFFFu;
    while (first != last && back_idx < block_size())
    {
      ring_buffer_[static_cast<std::size_t>(back_idx)] = static_cast<Tp&&>(*first);
      ++back;
      ++back_idx;
      ++first;
    }
    tail_.store(back, STDEXEC::__std::memory_order_release);
    return first;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::get() noexcept -> fetch_result<Tp>
  {
    std::uint64_t back     = tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t back_idx = back & 0xFFFF'FFFFu;
    if (back_idx == 0) [[unlikely]]
    {
      return {lifo_queue_error_code::empty, Tp{}};
    }
    // Extract index from head_ (which contains round in upper 32 bits)
    std::uint64_t front     = head_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t front_idx = front & 0xFFFF'FFFFu;
    if (front_idx == back_idx) [[unlikely]]
    {
      // Block is empty (head and tail indices match)
      return {lifo_queue_error_code::empty, Tp{}};
    }
    Tp value = static_cast<Tp&&>(ring_buffer_[static_cast<std::size_t>(back_idx - 1)]);
    tail_.store(back - 1, STDEXEC::__std::memory_order_release);
    return {lifo_queue_error_code::success, value};
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::steal(std::uint32_t thief_round) noexcept
    -> fetch_result<Tp>
  {
    std::uint64_t    spos  = steal_tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t    sidx  = spos & 0xFFFF'FFFFu;
    std::uint64_t    round = spos >> 32;
    fetch_result<Tp> result{};
    if (sidx == block_size())
    {
      // Block is marked as exhausted for stealing (steal_tail index == block_size)
      // Check round to distinguish between:
      //   - done: This is the correct generation (thief_round matches)
      //   - empty: This is a stale/future generation (round mismatch)
      result.status = thief_round == round ? lifo_queue_error_code::done
                                           : lifo_queue_error_code::empty;
      return result;
    }
    // Acquire ordering ensures we see items written by owner's release in put()
    std::uint64_t back = tail_.load(STDEXEC::__std::memory_order_acquire);
    if (sidx == back)
    {
      // No items available between steal_tail and tail
      result.status = lifo_queue_error_code::empty;
      return result;
    }
    // Try to claim the item at spos by atomically incrementing steal_tail_
    if (!steal_tail_.compare_exchange_strong(spos, spos + 1, STDEXEC::__std::memory_order_relaxed))
    {
      // Another thief claimed this item, retry
      result.status = lifo_queue_error_code::conflict;
      return result;
    }
    // Successfully claimed the item
    result.value = static_cast<Tp&&>(ring_buffer_[static_cast<std::size_t>(sidx)]);
    // Release ordering ensures reclaim() sees this increment after we've read the value
    steal_count_.fetch_add(1, STDEXEC::__std::memory_order_release);
    result.status = lifo_queue_error_code::success;
    return result;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::reduce_round() noexcept -> void
  {
    // Decrement the round in steal_tail_ when moving backward in the block array.
    // Called by advance_get_index() when the owner retreats to a previous block.
    std::uint64_t steal_tail     = steal_tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint32_t round          = static_cast<std::uint32_t>(steal_tail >> 32);
    std::uint64_t steal_index    = steal_tail & 0xFFFF'FFFFu;
    std::uint64_t new_steal_tail = (static_cast<std::uint64_t>(round - 1) << 32) | steal_index;
    steal_tail_.store(new_steal_tail, STDEXEC::__std::memory_order_relaxed);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::takeover() noexcept -> void
  {
    // Called when the owner moves backward to this block.
    // Swaps head_ and steal_tail_ to establish new boundaries.
    // The old steal_tail_ becomes the new head_ (start of owner's range).
    std::uint64_t head = head_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t spos = steal_tail_.exchange(head, STDEXEC::__std::memory_order_relaxed);
    head_.store(spos, STDEXEC::__std::memory_order_relaxed);
  }

  template <class Tp, class Allocator>
  auto
  lifo_queue<Tp, Allocator>::block_type::is_writable(std::uint32_t round) const noexcept -> bool
  {
    // Check if this block can be safely reused for the given round.
    // The block is writable if steal_tail_ shows it's exhausted (index == block_size)
    // and the round is from the previous generation (round - 1).
    // This prevents reusing a block while thieves might still be accessing it.
    std::uint64_t expanded_old_round = static_cast<std::uint64_t>(round - 1) << 32;
    std::uint64_t writeable_spos     = expanded_old_round | block_size();
    std::uint64_t spos               = steal_tail_.load(STDEXEC::__std::memory_order_relaxed);
    return spos == writeable_spos;
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::reclaim(std::uint32_t round) noexcept -> void
  {
    // Reset this block for reuse with a new round number.
    // Must wait for all thieves to finish accessing this block from previous generation.

    // Expected steal_count is the index from head_ (number of items that were available to steal)
    std::uint64_t expected_steal_count_ = head_.load(STDEXEC::__std::memory_order_relaxed)
                                        & 0xFFFF'FFFFu;
    // Spin until all thieves have reported completion via steal_count_
    // Acquire ordering ensures we see all thief modifications before proceeding
    while (steal_count_.load(STDEXEC::__std::memory_order_acquire) != expected_steal_count_)
    {
      STDEXEC::__spin_loop_pause();
    }
    // All thieves have finished - safe to reset the block
    std::uint64_t expanded_round = static_cast<std::uint64_t>(round) << 32;
    head_.store(expanded_round, STDEXEC::__std::memory_order_relaxed);
    tail_.store(0, STDEXEC::__std::memory_order_relaxed);
    steal_tail_.store(expanded_round | block_size(), STDEXEC::__std::memory_order_relaxed);
    steal_count_.store(0, STDEXEC::__std::memory_order_relaxed);
  }

  template <class Tp, class Allocator>
  auto lifo_queue<Tp, Allocator>::block_type::block_size() const noexcept -> std::size_t
  {
    return ring_buffer_.size();
  }

  template <class Tp, class Allocator>
  void lifo_queue<Tp, Allocator>::block_type::grant() noexcept
  {
    // Called when the owner moves forward to a new block.
    // Makes the current block fully available for stealing by swapping head_ and steal_tail_.
    // The old head_ becomes steal_tail_ (starting point for thieves).
    // The old steal_tail_ (at block_size()) becomes head_ (marking end of owner's range).
    std::uint64_t block_end = steal_tail_.load(STDEXEC::__std::memory_order_relaxed);
    std::uint64_t old_head  = head_.exchange(block_end, STDEXEC::__std::memory_order_relaxed);
    // Release ordering ensures thieves see all items we wrote before starting to steal
    steal_tail_.store(old_head, STDEXEC::__std::memory_order_release);
  }
}  // namespace experimental::execution::bwos

namespace exec = experimental::execution;
