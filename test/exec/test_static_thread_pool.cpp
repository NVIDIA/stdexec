/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include <exec/sequence/ignore_all_values.hpp>
#include <exec/sequence/transform_each.hpp>
#include <exec/start_detached.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>  // IWYU pragma: keep

#include <atomic>
#include <chrono>
#include <exception>
#include <latch>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>
namespace ex = STDEXEC;

namespace
{
  thread_local int current_numa_node = -1;

  struct two_node_numa_policy
  {
    [[nodiscard]]
    constexpr auto num_nodes() const noexcept -> std::size_t
    {
      return 2;
    }

    [[nodiscard]]
    constexpr auto num_cpus(int) const noexcept -> std::size_t
    {
      return 2;
    }

    auto bind_to_node(int node) const noexcept -> int
    {
      current_numa_node = node;
      return 0;
    }

    [[nodiscard]]
    constexpr auto thread_index_to_node(std::size_t index) const noexcept -> int
    {
      return index < 2 ? 1 : 0;
    }
  };

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct throwing_set_next_receiver
  {
    using receiver_concept = ex::receiver_tag;

    bool&               set_value_called_;
    bool&               set_stopped_called_;
    std::exception_ptr& error_;

    template <class Item>
    auto set_next(Item&&) -> decltype(ex::just())
    {
      throw std::runtime_error{"set_next failed"};
    }

    void set_value() noexcept
    {
      set_value_called_ = true;
    }

    void set_stopped() noexcept
    {
      set_stopped_called_ = true;
    }

    void set_error(std::exception_ptr error) noexcept
    {
      error_ = error;
    }

    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }
  };
#endif
}  // namespace

TEST_CASE("constrained static_thread_pool scheduler selects eligible workers",
          "[types][static_thread_pool]")
{
  constexpr std::size_t const num_of_threads = 4;
  exec::static_thread_pool    pool{num_of_threads, {}, exec::numa_policy{two_node_numa_policy{}}};
  exec::nodemask              constraints{};
  constraints.set(0);
  auto scheduler = pool.get_constrained_scheduler(&constraints);

  for (std::size_t i = 0; i < num_of_threads; ++i)
  {
    auto [node] = ex::sync_wait(ex::schedule(scheduler)
                                | ex::then([]() noexcept { return current_numa_node; }))
                    .value();
    CHECK(node == 0);
  }
}

TEST_CASE("static_thread_pool::get_scheduler_on_thread Test start on a specific thread",
          "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::unordered_set<std::thread::id> thread_ids;
  for (size_t i = 0; i < num_of_threads; ++i)
  {
    auto sender = ex::schedule(pool.get_scheduler_on_thread(i))
                | ex::then([&]() -> void { thread_ids.insert(std::this_thread::get_id()); });
    ex::sync_wait(std::move(sender));
  }
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE("bulk on static_thread_pool executes on multiple threads", "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex                          mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto                                sender = ex::starts_on(pool.get_scheduler(),
                              ex::just()
                                | ex::bulk(ex::par_unseq,
                                           num_of_threads,
                                           [&](size_t) -> void
                                           {
                                             std::this_thread::sleep_for(
                                               std::chrono::milliseconds(100));
                                             std::lock_guard lock(mtx);
                                             thread_ids.insert(std::this_thread::get_id());
                                           }));
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE("schedule_all on static_thread_pool handles empty ranges", "[types][static_thread_pool]")
{
  auto pool   = exec::static_thread_pool{2};
  auto sender = exec::schedule_all(pool, std::views::iota(size_t{0}, size_t{0}))
              | exec::ignore_all_values();

  CHECK(ex::sync_wait(std::move(sender)));
}

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
TEST_CASE("schedule_all on static_thread_pool sends errors from set_next",
          "[types][static_thread_pool]")
{
  exec::static_thread_pool pool{1};
  bool                     set_value_called   = false;
  bool                     set_stopped_called = false;
  std::exception_ptr       error;

  auto op =
    exec::subscribe(exec::schedule_all(pool, std::views::iota(0, 1)),
                    throwing_set_next_receiver{set_value_called, set_stopped_called, error});

  ex::start(op);

  CHECK_FALSE(set_value_called);
  CHECK_FALSE(set_stopped_called);
  REQUIRE(error);
  CHECK_THROWS_AS(std::rethrow_exception(error), std::runtime_error);
}
#endif

TEST_CASE("schedule_all on static_thread_pool handles ranges smaller than available parallelism",
          "[types][static_thread_pool]")
{
  constexpr size_t const num_of_threads = 5;
  constexpr int const    range_size     = 3;

  exec::static_thread_pool pool{num_of_threads};
  REQUIRE(range_size < pool.available_parallelism());

  std::atomic<int> count{0};
  std::atomic<int> sum{0};
  auto             sender = exec::schedule_all(pool, std::views::iota(0, range_size))
              | exec::transform_each(ex::then(
                [&](int x) noexcept
                {
                  count.fetch_add(1, std::memory_order_relaxed);
                  sum.fetch_add(x, std::memory_order_relaxed);
                }))
              | exec::ignore_all_values();

  CHECK(ex::sync_wait(std::move(sender)));
  CHECK(count.load(std::memory_order_relaxed) == range_size);
  CHECK(sum.load(std::memory_order_relaxed) == 3);
}
TEST_CASE("bulk on static_thread_pool executes on multiple threads, take 2",
          "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex                          mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto                                sender = ex::schedule(pool.get_scheduler())
              | ex::bulk(ex::par_unseq,
                         num_of_threads,
                         [&](size_t) -> void
                         {
                           std::this_thread::sleep_for(std::chrono::milliseconds(100));
                           std::lock_guard lock(mtx);
                           thread_ids.insert(std::this_thread::get_id());
                         });
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE("static_thread_pool drains remote work after idle transitions",
          "[types][static_thread_pool][stress]")
{
  constexpr std::size_t num_producers = 4;
  constexpr std::size_t rounds        = 10'000;

  std::latch                            ready{num_producers};
  std::atomic<bool>                     start{false};
  std::atomic<bool>                     stop{false};
  std::vector<std::atomic<std::size_t>> completed(num_producers);
  std::vector<std::thread>              producers;
  producers.reserve(num_producers);
  for (auto& count: completed)
  {
    count.store(0, std::memory_order_relaxed);
  }

  exec::static_thread_pool pool{1};
  auto                     scheduler = pool.get_scheduler();

  for (std::size_t producer = 0; producer < num_producers; ++producer)
  {
    producers.emplace_back(
      [&, producer]
      {
        ready.count_down();
        while (!start.load(std::memory_order_acquire))
        {
          std::this_thread::yield();
        }

        auto* const producer_completed = &completed[producer];
        std::size_t expected           = 0;
        for (std::size_t round = 0; round < rounds && !stop.load(std::memory_order_relaxed);
             ++round)
        {
          std::size_t const batch_size = (round % 4 == 0) ? 2 : 1;
          expected += batch_size;
          for (std::size_t i = 0; i < batch_size; ++i)
          {
            exec::start_detached(
              ex::schedule(scheduler)
              | ex::then([producer_completed]
                         { producer_completed->fetch_add(1, std::memory_order_relaxed); }));
          }

          while (!stop.load(std::memory_order_relaxed)
                 && producer_completed->load(std::memory_order_relaxed) < expected)
          {
            std::this_thread::yield();
          }
          std::this_thread::yield();
        }
      });
  }

  ready.wait();
  start.store(true, std::memory_order_release);

  auto const expected        = num_producers * rounds + num_producers * ((rounds + 3) / 4);
  auto const deadline        = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  auto       completed_total = [&]
  {
    std::size_t result = 0;
    for (auto const & count: completed)
    {
      result += count.load(std::memory_order_relaxed);
    }
    return result;
  };

  while (completed_total() < expected && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::yield();
  }
  stop.store(true, std::memory_order_release);
  for (auto& producer: producers)
  {
    producer.join();
  }

  CHECK(completed_total() == expected);
}
