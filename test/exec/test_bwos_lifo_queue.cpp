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
#include "exec/detail/bwos_lifo_queue.hpp"

#include <algorithm>
#include <atomic>
#include <ranges>
#include <thread>
#include <vector>

#include <catch2/catch.hpp>

TEST_CASE("exec::bwos::lifo_queue - ", "[bwos]")
{
  exec::bwos::lifo_queue<int*> queue(8, 2);
  int                          x = 1;
  int                          y = 2;
  SECTION("Observers")
  {
    CHECK(queue.block_size() == 2);
    CHECK(queue.num_blocks() == 8);
  }
  SECTION("Empty Get")
  {
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Empty Steal")
  {
    CHECK(queue.steal_front() == nullptr);
  }
  SECTION("Put one, get one")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put one, steal none")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
  }
  SECTION("Put one, get one, put one, get one")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put two, get two")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put three, Steal two")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.steal_front() == &y);
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put 4, Steal 1, Get 3")
  {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
}

TEST_CASE("exec::bwos::lifo_queue - size one block", "[bwos]")
{
  exec::bwos::lifo_queue<int*> queue(1, 1);
  int                          x = 1;
  int                          y = 2;
  CHECK(queue.push_back(&x));
  CHECK_FALSE(queue.push_back(&y));
  CHECK(queue.steal_front() == nullptr);
  CHECK(queue.pop_back() == &x);
  CHECK(queue.pop_back() == nullptr);
}

TEST_CASE("exec::bwos::lifo_queue - two blocks of size one", "[bwos]")
{
  exec::bwos::lifo_queue<int*> queue(2, 1);
  int                          x = 1;
  int                          y = 2;

  // First cycle: push, no steal possible from current block, pop
  CHECK(queue.push_back(&x));
  CHECK(queue.steal_front() == nullptr);
  CHECK(queue.pop_back() == &x);
  CHECK(queue.pop_back() == nullptr);

  // Second cycle: push two items across blocks
  CHECK(queue.push_back(&x));
  CHECK(queue.push_back(&y));
  CHECK_FALSE(queue.push_back(&x));  // queue full
  CHECK(queue.steal_front() == &x);  // steal from first granted block
  CHECK(queue.steal_front() == nullptr);
  CHECK(queue.pop_back() == &y);
  CHECK(queue.pop_back() == nullptr);

  // Third cycle: push and pop again after wraparound
  CHECK(queue.push_back(&x));
  CHECK(queue.pop_back() == &x);
  CHECK(queue.pop_back() == nullptr);
}

TEST_CASE("exec::bwos::lifo_queue - round counter wraparound", "[bwos]")
{
  constexpr std::size_t numBlocks = 4;
  constexpr std::size_t blockSize = 2;
  constexpr std::size_t numItems  = numBlocks * blockSize * 3;  // 3 full rounds

  exec::bwos::lifo_queue<int> queue(numBlocks, blockSize);
  std::vector<int>            stolenItems;

  for (int val = 1; val <= static_cast<int>(numItems); ++val)
  {
    if (!queue.push_back(val))
    {
      // Queue full, steal some items to make space
      while (true)
      {
        int stolen = queue.steal_front();
        if (stolen == 0)
        {
          break;
        }
        stolenItems.push_back(stolen);
      }
      REQUIRE(queue.push_back(val));
    }
  }

  // Stolen items should be a prefix of the values in order
  auto prefix = std::views::iota(1, static_cast<int>(stolenItems.size()) + 1);
  CHECK(std::ranges::equal(stolenItems, prefix));

  std::vector<int> remainingItems;
  while (true)
  {
    int item = queue.pop_back();
    if (item == 0)
    {
      break;
    }
    remainingItems.push_back(item);
  }
  CHECK(std::ranges::is_sorted(remainingItems.rbegin(), remainingItems.rend()));
  CHECK(stolenItems.size() + remainingItems.size() == numItems);
  CHECK(queue.pop_back() == 0);
}

TEST_CASE("exec::bwos::lifo_queue - concurrent stealing", "[bwos]")
{
  constexpr std::size_t numItems   = 2000;
  constexpr std::size_t numThieves = 4;

  exec::bwos::lifo_queue<std::size_t> queue(32, 64);

  for (std::size_t i = 1; i <= numItems; ++i)
  {
    REQUIRE(queue.push_back(i));
  }

  std::vector<std::vector<std::size_t>> stolen(numThieves);
  std::vector<std::thread>              thieves;
  thieves.reserve(numThieves);
  for (std::size_t t = 0; t < numThieves; ++t)
  {
    thieves.emplace_back(
      [&queue, &stolen, t]()
      {
        while (true)
        {
          std::size_t item = queue.steal_front();
          if (item == 0)
          {
            break;
          }
          stolen[t].push_back(item);
        }
      });
  }

  for (auto& thief: thieves)
  {
    thief.join();
  }

  std::vector<std::size_t> allStolen;
  for (auto const & vec: stolen)
  {
    allStolen.insert(allStolen.end(), vec.begin(), vec.end());
  }

  std::ranges::sort(allStolen);
  auto [first, last] = std::ranges::unique(allStolen);
  CHECK(first == last);  // no duplicates
  CHECK(allStolen.size() <= numItems);
}

TEST_CASE("exec::bwos::lifo_queue - concurrent owner and thieves", "[bwos]")
{
  constexpr std::size_t numItems   = 10000;
  constexpr std::size_t numThieves = 2;

  exec::bwos::lifo_queue<std::size_t> queue(16, 32);
  std::atomic<bool>                   done{false};
  std::atomic<std::size_t>            ownerPopped{0};
  std::atomic<std::size_t>            totalStolen{0};

  std::thread owner(
    [&]()
    {
      for (std::size_t i = 1; i <= numItems; ++i)
      {
        while (!queue.push_back(i))
        {
          if (queue.pop_back() != 0)
          {
            ownerPopped++;
          }
        }
        if (i % 100 == 0)
        {
          if (queue.pop_back() != 0)
          {
            ownerPopped++;
          }
        }
      }
      done = true;
    });

  std::vector<std::thread> thieves;
  thieves.reserve(numThieves);
  for (std::size_t t = 0; t < numThieves; ++t)
  {
    thieves.emplace_back(
      [&]()
      {
        while (!done.load(std::memory_order_relaxed))
        {
          if (queue.steal_front() != 0)
          {
            totalStolen++;
          }
        }
        while (queue.steal_front() != 0)
        {
          totalStolen++;
        }
      });
  }

  owner.join();
  for (auto& thief: thieves)
  {
    thief.join();
  }

  std::size_t remaining = 0;
  while (queue.pop_back() != 0)
  {
    remaining++;
  }

  CHECK(ownerPopped + totalStolen + remaining == numItems);
}

TEST_CASE("exec::bwos::lifo_queue - block wraparound with stealing", "[bwos]")
{
  constexpr std::size_t numBlocks     = 4;
  constexpr std::size_t blockSize     = 8;
  constexpr std::size_t rounds        = 5;
  constexpr std::size_t itemsPerRound = numBlocks * blockSize;

  exec::bwos::lifo_queue<std::size_t> queue(numBlocks, blockSize);
  std::atomic<std::size_t>            stolenCount{0};
  std::atomic<bool>                   thiefActive{true};

  std::thread thief(
    [&]()
    {
      while (thiefActive)
      {
        if (queue.steal_front())
        {
          stolenCount++;
        }
      }
    });

  for (std::size_t round = 0; round < rounds; ++round)
  {
    for (std::size_t i = 0; i < itemsPerRound; ++i)
    {
      std::size_t value = (round * itemsPerRound) + i + 1;
      while (!queue.push_back(value))
      {
        queue.pop_back();
      }
    }
  }

  thiefActive = false;
  thief.join();
}

TEST_CASE("exec::bwos::lifo_queue - high contention stress", "[bwos]")
{
  constexpr std::size_t numItems   = 50000;
  constexpr std::size_t numThieves = 8;
  constexpr std::size_t numBlocks  = 16;
  constexpr std::size_t blockSize  = 32;

  exec::bwos::lifo_queue<std::size_t> queue(numBlocks, blockSize);
  std::atomic<std::size_t>            totalStolen{0};
  std::atomic<std::size_t>            pushCount{0};
  std::atomic<bool>                   done{false};

  std::thread owner(
    [&]()
    {
      for (std::size_t i = 1; i <= numItems; ++i)
      {
        while (!queue.push_back(i))
        {
          std::this_thread::yield();
        }
        pushCount++;
      }
      done = true;
    });

  std::vector<std::thread>              thieves;
  std::vector<std::atomic<std::size_t>> thiefCounts(numThieves);

  thieves.reserve(numThieves);
  for (std::size_t t = 0; t < numThieves; ++t)
  {
    thieves.emplace_back(
      [&, t]()
      {
        std::size_t localCount = 0;
        while (true)
        {
          auto val = queue.steal_front();
          if (val != 0)
          {
            localCount++;
          }
          else if (done)
          {
            break;
          }
        }
        thiefCounts[t] = localCount;
      });
  }

  owner.join();
  for (auto& thief: thieves)
  {
    thief.join();
  }

  for (auto const & count: thiefCounts)
  {
    totalStolen += count.load();
  }

  std::size_t remaining = 0;
  while (queue.pop_back() != 0)
  {
    remaining++;
  }

  CHECK(pushCount == numItems);
  CHECK(totalStolen + remaining == numItems);
}

TEST_CASE("exec::bwos::lifo_queue - steal during wraparound", "[bwos]")
{
  constexpr std::size_t numBlocks = 4;
  constexpr std::size_t blockSize = 4;

  exec::bwos::lifo_queue<std::size_t> queue(numBlocks, blockSize);
  std::atomic<bool>                   ownerDone{false};
  std::atomic<std::size_t>            stolen{0};

  std::thread thief(
    [&]()
    {
      while (!ownerDone.load(std::memory_order_relaxed))
      {
        if (queue.steal_front() != 0)
        {
          stolen++;
        }
      }
      // Drain remaining
      while (queue.steal_front() != 0)
      {
        stolen++;
      }
    });

  for (std::size_t round = 0; round < 3; ++round)
  {
    for (std::size_t i = 0; i < numBlocks * blockSize; ++i)
    {
      while (!queue.push_back((round * 100) + i + 1))
      {
        std::this_thread::yield();
      }
    }
  }

  ownerDone = true;
  thief.join();
}

TEST_CASE("exec::bwos::lifo_queue - takeover grant synchronization", "[bwos]")
{
  constexpr std::size_t numBlocks  = 4;
  constexpr std::size_t blockSize  = 16;
  constexpr std::size_t iterations = 1000;

  exec::bwos::lifo_queue<std::size_t> queue(numBlocks, blockSize);
  std::atomic<std::size_t>            totalStolen{0};
  std::atomic<bool>                   ownerDone{false};

  std::thread owner(
    [&]()
    {
      for (std::size_t iter = 0; iter < iterations; ++iter)
      {
        for (std::size_t i = 0; i < blockSize; ++i)
        {
          while (!queue.push_back((iter * blockSize) + i + 1))
          {
            std::this_thread::yield();
          }
        }
        // Pop some back (triggers takeover when moving backward across blocks)
        for (std::size_t i = 0; i < blockSize / 2; ++i)
        {
          queue.pop_back();
        }
      }
      ownerDone = true;
    });

  std::thread thief(
    [&]()
    {
      while (!ownerDone.load(std::memory_order_relaxed))
      {
        if (queue.steal_front() != 0)
        {
          totalStolen++;
        }
      }
    });

  owner.join();
  thief.join();
}