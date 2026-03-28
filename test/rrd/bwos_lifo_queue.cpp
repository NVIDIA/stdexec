/*
 * Copyright (c) 2026 Maikel Nadolski
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <stdexec_relacy.hpp>

#include <exec/detail/bwos_lifo_queue.hpp>

#include <optional>

// Test 1: Owner pushes items, single thief steals.
// Invariant: every pushed item is either popped by owner or stolen by thief exactly once.
struct bwos_push_steal_no_loss : rl::test_suite<bwos_push_steal_no_loss, 2>
{
  static constexpr std::size_t num_blocks = 2;
  static constexpr std::size_t block_size = 2;
  static constexpr std::size_t num_items  = 4;

  std::optional<exec::bwos::lifo_queue<int>> queue{};
  int                                        owner_sum{0};
  int                                        thief_sum{0};

  void before()
  {
    queue.emplace(num_blocks, block_size);
    owner_sum = 0;
    thief_sum = 0;
  }

  void after()
  {
    // Sum of 1..num_items = num_items*(num_items+1)/2
    int expected = static_cast<int>(num_items * (num_items + 1) / 2);
    RL_ASSERT(owner_sum + thief_sum == expected);
    queue.reset();
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Owner: push all items then pop remaining
      for (int i = 1; i <= static_cast<int>(num_items); ++i)
      {
        while (!queue->push_back(i))
        {
          int val = queue->pop_back();
          if (val != 0)
          {
            owner_sum += val;
          }
        }
      }
      // Drain remaining
      while (true)
      {
        int val = queue->pop_back();
        if (val == 0)
        {
          break;
        }
        owner_sum += val;
      }
    }
    else
    {
      // Thief: steal until no more items
      for (std::size_t attempt = 0; attempt < num_items * 4; ++attempt)
      {
        int val = queue->steal_front();
        if (val != 0)
        {
          thief_sum += val;
        }
      }
    }
  }
};

// Test 2: Owner fills a block, grants it, thief steals while owner pushes to next block.
// This exercises the grant()/steal() race on block boundaries.
struct bwos_grant_steal_race : rl::test_suite<bwos_grant_steal_race, 2>
{
  static constexpr std::size_t num_blocks = 4;
  static constexpr std::size_t block_size = 2;

  std::optional<exec::bwos::lifo_queue<int>> queue{};
  int                                        owner_count{0};
  int                                        thief_count{0};
  bool                                       seen[7]{};

  void before()
  {
    queue.emplace(num_blocks, block_size);
    owner_count = 0;
    thief_count = 0;
    for (auto& s: seen)
    {
      s = false;
    }
  }

  void after()
  {
    RL_ASSERT(owner_count + thief_count == 6);
    for (int i = 1; i <= 6; ++i)
    {
      RL_ASSERT(seen[i]);
    }
    queue.reset();
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Push enough to force block transitions (grant)
      for (int i = 1; i <= 6; ++i)
      {
        while (!queue->push_back(i))
        {
          int val = queue->pop_back();
          if (val != 0)
          {
            RL_ASSERT(val >= 1 && val <= 6);
            RL_ASSERT(!seen[val]);
            seen[val] = true;
            owner_count++;
          }
        }
      }
      // Drain remaining
      while (true)
      {
        int val = queue->pop_back();
        if (val == 0)
        {
          break;
        }
        RL_ASSERT(val >= 1 && val <= 6);
        RL_ASSERT(!seen[val]);
        seen[val] = true;
        owner_count++;
      }
    }
    else
    {
      // Thief steals concurrently
      for (std::size_t attempt = 0; attempt < 24; ++attempt)
      {
        int val = queue->steal_front();
        if (val != 0)
        {
          RL_ASSERT(val >= 1 && val <= 6);
          RL_ASSERT(!seen[val]);
          seen[val] = true;
          thief_count++;
        }
      }
    }
  }
};

// Test 3: Owner pushes, pops (triggering takeover), while thief steals.
// This exercises the takeover()/steal() race which was the original bug.
struct bwos_takeover_steal_race : rl::test_suite<bwos_takeover_steal_race, 2>
{
  static constexpr std::size_t num_blocks = 4;
  static constexpr std::size_t block_size = 2;

  std::optional<exec::bwos::lifo_queue<int>> queue{};
  int                                        owner_count{0};
  int                                        thief_count{0};

  void before()
  {
    queue.emplace(num_blocks, block_size);
    owner_count = 0;
    thief_count = 0;
  }

  void after()
  {
    RL_ASSERT(owner_count + thief_count == 3);
    queue.reset();
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Push items to fill first block, then push more to advance blocks
      queue->push_back(1);
      queue->push_back(2);
      queue->push_back(3);  // forces advance to block 1

      // Now pop: 3 from block 1, then 2 triggers takeover back to block 0
      int val = queue->pop_back();
      if (val != 0)
      {
        owner_count++;
      }
      val = queue->pop_back();
      if (val != 0)
      {
        owner_count++;
      }
      val = queue->pop_back();
      if (val != 0)
      {
        owner_count++;
      }
    }
    else
    {
      // Thief steals concurrently during takeover
      for (std::size_t attempt = 0; attempt < 12; ++attempt)
      {
        int val = queue->steal_front();
        if (val != 0)
        {
          thief_count++;
        }
      }
    }
  }
};

// Test 4: Two thieves competing to steal from the same block.
// Tests the CAS contention in steal().
struct bwos_two_thieves : rl::test_suite<bwos_two_thieves, 3>
{
  static constexpr std::size_t num_blocks = 4;
  static constexpr std::size_t block_size = 2;
  static constexpr int         num_items  = 4;

  std::optional<exec::bwos::lifo_queue<int>> queue{};
  int                                        owner_count{0};
  int                                        thief1_count{0};
  int                                        thief2_count{0};
  bool                                       seen[5]{};

  void before()
  {
    queue.emplace(num_blocks, block_size);
    owner_count  = 0;
    thief1_count = 0;
    thief2_count = 0;
    for (auto& s: seen)
    {
      s = false;
    }
  }

  void after()
  {
    RL_ASSERT(owner_count + thief1_count + thief2_count == num_items);
    for (int i = 1; i <= num_items; ++i)
    {
      RL_ASSERT(seen[i]);
    }
    queue.reset();
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Owner pushes items across blocks, then drains
      for (int i = 1; i <= num_items; ++i)
      {
        while (!queue->push_back(i))
        {
          int val = queue->pop_back();
          if (val != 0)
          {
            RL_ASSERT(!seen[val]);
            seen[val] = true;
            owner_count++;
          }
        }
      }
      while (true)
      {
        int val = queue->pop_back();
        if (val == 0)
        {
          break;
        }
        RL_ASSERT(!seen[val]);
        seen[val] = true;
        owner_count++;
      }
    }
    else
    {
      int& my_count = (thread_id == 1) ? thief1_count : thief2_count;
      for (std::size_t attempt = 0; attempt < num_items * 4; ++attempt)
      {
        int val = queue->steal_front();
        if (val != 0)
        {
          RL_ASSERT(val >= 1 && val <= num_items);
          RL_ASSERT(!seen[val]);
          seen[val] = true;
          my_count++;
        }
      }
    }
  }
};

// Test 5: Owner does push-pop-push-pop cycles forcing block wraparound with reclaim.
// Thief steals concurrently. Tests reclaim()/steal_count synchronization.
struct bwos_reclaim_race : rl::test_suite<bwos_reclaim_race, 2>
{
  static constexpr std::size_t num_blocks = 2;
  static constexpr std::size_t block_size = 2;

  std::optional<exec::bwos::lifo_queue<int>> queue{};
  int                                        total_owner{0};
  int                                        total_thief{0};
  int                                        total_pushed{0};

  void before()
  {
    queue.emplace(num_blocks, block_size);
    total_owner  = 0;
    total_thief  = 0;
    total_pushed = 0;
  }

  void after()
  {
    RL_ASSERT(total_owner + total_thief == total_pushed);
    queue.reset();
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Cycle 1: push 2 items (fill block 0), push 1 more (advance to block 1)
      queue->push_back(1);
      total_pushed++;
      queue->push_back(2);
      total_pushed++;
      if (queue->push_back(3))
      {
        total_pushed++;
      }

      // Pop everything to retreat
      while (true)
      {
        int val = queue->pop_back();
        if (val == 0)
        {
          break;
        }
        total_owner++;
      }

      // Cycle 2: push again - this forces block reuse after reclaim
      if (queue->push_back(10))
      {
        total_pushed++;
      }
      if (queue->push_back(20))
      {
        total_pushed++;
      }

      // Drain
      while (true)
      {
        int val = queue->pop_back();
        if (val == 0)
        {
          break;
        }
        total_owner++;
      }
    }
    else
    {
      for (std::size_t attempt = 0; attempt < 20; ++attempt)
      {
        int val = queue->steal_front();
        if (val != 0)
        {
          total_thief++;
        }
      }
    }
  }
};

auto main() -> int
{
  rl::test_params p;
  p.iteration_count       = 100000;
  p.execution_depth_limit = 10000;
  p.search_type           = rl::random_scheduler_type;

  rl::simulate<bwos_push_steal_no_loss>(p);
  rl::simulate<bwos_grant_steal_race>(p);
  rl::simulate<bwos_takeover_steal_race>(p);
  rl::simulate<bwos_two_thieves>(p);
  rl::simulate<bwos_reclaim_race>(p);
  return 0;
}
