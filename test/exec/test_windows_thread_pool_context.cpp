/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <test_common/catch2.hpp>
#include <test_common/type_helpers.hpp>

#include <stdexec/execution.hpp>

#include <exec/repeat_until.hpp>
#include <exec/windows/windows_thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("windows_thread_pool scheduler provides scheduler_concept",
          "[types][windows_thread_pool][schedulers]")
{
  // regression guard for issue #2134: per [exec.sched], schedulers must
  // provide the scheduler_concept nested alias
  STATIC_REQUIRE(
    std::same_as<exec::windows_thread_pool::scheduler::scheduler_concept, STDEXEC::scheduler_tag>);
}

TEST_CASE("windows_thread_pool: construct_destruct", "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp;
}

TEST_CASE("windows_thread_pool: custom_thread_pool", "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp{2, 4};
  auto                      s = tp.get_scheduler();

  std::atomic<int> count = 0;

  auto incrementCountOnTp = STDEXEC::then(STDEXEC::schedule(s), [&] { ++count; });

  STDEXEC::sync_wait(STDEXEC::when_all(incrementCountOnTp,
                                       incrementCountOnTp,
                                       incrementCountOnTp,
                                       incrementCountOnTp));

  REQUIRE(count.load() == 4);
}

TEST_CASE("windows_thread_pool: schedule", "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp;
  STDEXEC::sync_wait(STDEXEC::schedule(tp.get_scheduler()));
}

TEST_CASE("windows_thread_pool: schedule_completes_on_a_different_thread",
          "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp;
  auto const                mainThreadId = std::this_thread::get_id();
  auto [workThreadId] = STDEXEC::sync_wait(STDEXEC::then(STDEXEC::schedule(tp.get_scheduler()),
                                                         [&]() noexcept
                                                         { return std::this_thread::get_id(); }))
                          .value();
  REQUIRE_FALSE(workThreadId == mainThreadId);
}

// TEST_CASE("windows_thread_pool: schedule_multiple_in_parallel", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto sch = tp.get_scheduler();

//   STDEXEC::sync_wait(STDEXEC::then(
//       STDEXEC::when_all(
//           STDEXEC::schedule(sch), STDEXEC::schedule(sch), STDEXEC::schedule(sch)),
//       [](auto&&...) noexcept { return 0; }));
// }

// TEST_CASE("windows_thread_pool: schedule_cancellation_thread_safety", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto sch = tp.get_scheduler();

//   STDEXEC::sync_wait(exec::repeat_until(
//       STDEXEC::let_stopped(
//           STDEXEC::stop_when(
//               exec::repeat(STDEXEC::schedule(sch)),
//               STDEXEC::schedule(sch)),
//           [] { return STDEXEC::just(); }),
//       [n = 0]() mutable noexcept { return n++ == 1000; }));
// }

TEST_CASE("windows_thread_pool: schedule_after", "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  auto startTime = exec::now(s);

  STDEXEC::sync_wait(exec::schedule_after(s, 50ms));

  auto duration = exec::now(s) - startTime;

  REQUIRE(duration > 40ms);
  REQUIRE(duration < 100ms);
}

// TEST_CASE("windows_thread_pool: schedule_after_cancellation", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto s = tp.get_scheduler();

//   auto startTime = exec::now(s);

//   bool ranWork = false;

//   STDEXEC::sync_wait(STDEXEC::let_stopped(
//       STDEXEC::stop_when(
//           STDEXEC::then(exec::schedule_after(s, 5s), [&] { ranWork = true; }),
//           exec::schedule_after(s, 5ms)),
//       [] { return STDEXEC::just(); }));

//   auto duration = exec::now(s) - startTime;

//   // Work should have been cancelled.
//   REQUIRE_FALSE(ranWork);
//   REQUIRE(duration < 1s);
// }

TEST_CASE("windows_thread_pool: schedule_at", "[types][windows_thread_pool][schedulers]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  auto startTime = exec::now(s);

  STDEXEC::sync_wait(exec::schedule_at(s, startTime + 100ms));

  auto endTime = exec::now(s);
  REQUIRE(endTime >= (startTime + 100ms));
  REQUIRE(endTime < (startTime + 150ms));
}

// TEST_CASE("windows_thread_pool: schedule_at_cancellation", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto s = tp.get_scheduler();

//   auto startTime = exec::now(s);

//   bool ranWork = false;

//   STDEXEC::sync_wait(STDEXEC::let_stopped(
//       STDEXEC::stop_when(
//           STDEXEC::then(
//               exec::schedule_at(s, startTime + 5s), [&] { ranWork = true; }),
//           STDEXEC::schedule_at(s, startTime + 5ms)),
//       [] { return STDEXEC::just(); }));

//   auto duration = exec::now(s) - startTime;

//   // Work should have been cancelled.
//   REQUIRE_FALSE(ranWork);
//   REQUIRE(duration < 1s);
// }

TEST_CASE("windows_thread_pool: scheduler uses a custom domain for bulk",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      sch = tp.get_scheduler();
  CHECK(STDEXEC::get_forward_progress_guarantee(sch)
        == STDEXEC::forward_progress_guarantee::parallel);

  auto domain1 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(sch);
  auto domain2 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(
    STDEXEC::get_env(STDEXEC::schedule(sch)));
  auto domain3 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(
    STDEXEC::get_env(exec::schedule_at(sch, sch.now())));
  auto domain4 = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(
    STDEXEC::get_env(exec::schedule_after(sch, sch.now() - sch.now())));
  STATIC_REQUIRE(std::same_as<decltype(domain1), exec::windows_thread_pool::domain>);
  STATIC_REQUIRE(std::same_as<decltype(domain2), exec::windows_thread_pool::domain>);
  STATIC_REQUIRE(std::same_as<decltype(domain3), exec::windows_thread_pool::domain>);
  STATIC_REQUIRE(std::same_as<decltype(domain4), exec::windows_thread_pool::domain>);
}

TEST_CASE("windows_thread_pool: bulk calls the function with all indices",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  std::vector<int> data{1, 2, 3, 4, 5};
  auto             size  = data.size();
  auto             twice = [](auto i, auto &data)
  {
    data[i] = 2 * data[i];
  };
  auto add = [](auto const &data)
  {
    return std::accumulate(std::begin(data), std::end(data), 0);
  };
  auto sndr = STDEXEC::just(std::move(data)) | STDEXEC::continues_on(s)
            | STDEXEC::bulk(STDEXEC::par, size, twice) | STDEXEC::then(add);

  CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr)) == s);
  auto [res] = STDEXEC::sync_wait(std::move(sndr)).value();
  CHECK(res == 30);
}

TEST_CASE("windows_thread_pool: parallel bulk_chunked splits the shape between the threads",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  constexpr int             shape = 1000;
  exec::windows_thread_pool tp{2, 4};
  auto                      s = tp.get_scheduler();

  auto const       mainThreadId = std::this_thread::get_id();
  std::vector<int> visited(shape, 0);
  std::atomic<int> chunks{0};
  std::atomic<int> chunksOnMainThread{0};

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk_chunked(STDEXEC::par,
                                    shape,
                                    [&](int begin, int end) noexcept
                                    {
                                      ++chunks;
                                      if (std::this_thread::get_id() == mainThreadId)
                                        ++chunksOnMainThread;
                                      for (; begin != end; ++begin)
                                        ++visited[begin];
                                    });

  REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());

  // One chunk per thread of the pool (bounded by the number of cores)
  auto const expectedChunks = (std::clamp) (std::thread::hardware_concurrency(), 1u, 4u);
  CHECK(chunks.load() == static_cast<int>(expectedChunks));
  CHECK(chunksOnMainThread.load() == 0);
  CHECK(std::ranges::all_of(visited, [](int v) { return v == 1; }));
}

TEST_CASE("windows_thread_pool: parallel bulk_chunked never creates more chunks than indices",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  std::mutex                       mtx;
  std::vector<std::pair<int, int>> bounds;

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk_chunked(STDEXEC::par,
                                    1,
                                    [&](int begin, int end)
                                    {
                                      std::lock_guard lock{mtx};
                                      bounds.emplace_back(begin, end);
                                    });

  REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());
  REQUIRE(bounds.size() == 1);
  CHECK(bounds[0] == std::pair{0, 1});
}

TEST_CASE("windows_thread_pool: sequenced bulk_chunked runs a single chunk",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  std::vector<int> bounds;

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk_chunked(STDEXEC::seq,
                                    5,
                                    [&](int begin, int end)
                                    {
                                      bounds.push_back(begin);
                                      bounds.push_back(end);
                                    });

  REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());
  CHECK(bounds == std::vector<int>{0, 5});
}

TEST_CASE("windows_thread_pool: parallel bulk_unchunked calls the function once per index",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  constexpr int             shape = 1000;
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  std::vector<std::atomic<int>> visited(shape);

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk_unchunked(STDEXEC::par, shape, [&](int i) noexcept { ++visited[i]; });

  REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());
  CHECK(std::ranges::all_of(visited, [](auto const &v) { return v.load() == 1; }));
}

TEST_CASE("windows_thread_pool: sequenced bulk_unchunked runs the indices in order",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  std::vector<int> indices;

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk_unchunked(STDEXEC::seq, 4, [&](int i) { indices.push_back(i); });

  REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());
  CHECK(indices == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("windows_thread_pool: bulk with an empty shape forwards the values",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  bool called = false;
  auto sndr   = STDEXEC::just(42) | STDEXEC::continues_on(s)
            | STDEXEC::bulk(STDEXEC::par, 0, [&](int, int) noexcept { called = true; });

  auto [res] = STDEXEC::sync_wait(std::move(sndr)).value();
  CHECK(res == 42);
  CHECK_FALSE(called);
}

TEST_CASE("windows_thread_pool: bulk can be run many times in a row",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  for (int n = 0; n < 200; ++n)
  {
    std::atomic<int> sum{0};
    auto             sndr = STDEXEC::schedule(s)
              | STDEXEC::bulk(STDEXEC::par, n, [&](int i) noexcept { sum += i; });
    REQUIRE(STDEXEC::sync_wait(std::move(sndr)).has_value());
    REQUIRE(sum.load() == n * (n - 1) / 2);
  }
}

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
TEST_CASE("windows_thread_pool: bulk propagates exceptions of the function",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  auto sndr = STDEXEC::schedule(s)
            | STDEXEC::bulk(STDEXEC::par,
                            100,
                            [](int i)
                            {
                              if (i == 42)
                                throw 999;
                            });

  STATIC_REQUIRE(
    set_equivalent<STDEXEC::completion_signatures_of_t<decltype(sndr), STDEXEC::env<>>,
                   STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                                  STDEXEC::set_error_t(std::exception_ptr),
                                                  STDEXEC::set_stopped_t()>>);

  STDEXEC_TRY
  {
    STDEXEC::sync_wait(std::move(sndr));
    CHECK(false);
  }
  STDEXEC_CATCH(int e)
  {
    CHECK(e == 999);
  }
  STDEXEC_CATCH_ALL
  {
    FAIL("invalid exception caught");
  }
}

TEST_CASE("windows_thread_pool: bulk completes with an error when copying the values throws",
          "[types][windows_thread_pool][schedulers][bulk]")
{
  struct value_capture_error
  {};

  struct throwing_value
  {
    throwing_value() = default;

    throwing_value(throwing_value const &)
    {
      throw value_capture_error{};
    }

    throwing_value(throwing_value &&)
    {
      throw value_capture_error{};
    }
  };

  exec::windows_thread_pool tp;
  auto                      s = tp.get_scheduler();

  auto sndr = STDEXEC::schedule(s) | STDEXEC::then([]() noexcept { return throwing_value{}; })
            | STDEXEC::bulk(STDEXEC::par, 0, [](int, throwing_value &) noexcept {});

  STATIC_REQUIRE(
    set_equivalent<STDEXEC::completion_signatures_of_t<decltype(sndr), STDEXEC::env<>>,
                   STDEXEC::completion_signatures<STDEXEC::set_value_t(throwing_value),
                                                  STDEXEC::set_error_t(std::exception_ptr),
                                                  STDEXEC::set_stopped_t()>>);

  STDEXEC_TRY
  {
    STDEXEC::sync_wait(std::move(sndr));
    CHECK(false);
  }
  STDEXEC_CATCH(value_capture_error const &)
  {
  }
  STDEXEC_CATCH_ALL
  {
    FAIL("invalid exception caught");
  }
}
#endif
