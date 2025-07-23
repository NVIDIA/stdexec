/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <catch2/catch.hpp>

#include <stdexec/execution.hpp>

#include <exec/windows/windows_thread_pool.hpp>
#include <exec/repeat_effect_until.hpp>

using namespace std::chrono_literals;

TEST_CASE("windows_thread_pool: construct_destruct", "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp;
}

TEST_CASE("windows_thread_pool: custom_thread_pool", "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp{2, 4};
  auto s = tp.get_scheduler();

  std::atomic<int> count = 0;

  auto incrementCountOnTp = stdexec::then(stdexec::schedule(s), [&] { ++count; });

  stdexec::sync_wait(
    stdexec::when_all(
      incrementCountOnTp, incrementCountOnTp, incrementCountOnTp, incrementCountOnTp));

  REQUIRE(count.load() == 4);
}

TEST_CASE("windows_thread_pool: schedule", "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp;
  stdexec::sync_wait(stdexec::schedule(tp.get_scheduler()));
}

TEST_CASE(
  "windows_thread_pool: schedule_completes_on_a_different_thread",
  "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp;
  const auto mainThreadId = std::this_thread::get_id();
  auto [workThreadId] = stdexec::sync_wait(
                          stdexec::then(
                            stdexec::schedule(tp.get_scheduler()),
                            [&]() noexcept { return std::this_thread::get_id(); }))
                          .value();
  REQUIRE_FALSE(workThreadId == mainThreadId);
}

// TEST_CASE("windows_thread_pool: schedule_multiple_in_parallel", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto sch = tp.get_scheduler();

//   stdexec::sync_wait(stdexec::then(
//       stdexec::when_all(
//           stdexec::schedule(sch), stdexec::schedule(sch), stdexec::schedule(sch)),
//       [](auto&&...) noexcept { return 0; }));
// }

// TEST_CASE("windows_thread_pool: schedule_cancellation_thread_safety", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto sch = tp.get_scheduler();

//   stdexec::sync_wait(exec::repeat_effect_until(
//       stdexec::let_stopped(
//           stdexec::stop_when(
//               exec::repeat_effect(stdexec::schedule(sch)),
//               stdexec::schedule(sch)),
//           [] { return stdexec::just(); }),
//       [n = 0]() mutable noexcept { return n++ == 1000; }));
// }

TEST_CASE("windows_thread_pool: schedule_after", "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp;
  auto s = tp.get_scheduler();

  auto startTime = exec::now(s);

  stdexec::sync_wait(exec::schedule_after(s, 50ms));

  auto duration = exec::now(s) - startTime;

  REQUIRE(duration > 40ms);
  REQUIRE(duration < 100ms);
}

// TEST_CASE("windows_thread_pool: schedule_after_cancellation", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto s = tp.get_scheduler();

//   auto startTime = exec::now(s);

//   bool ranWork = false;

//   stdexec::sync_wait(stdexec::let_stopped(
//       stdexec::stop_when(
//           stdexec::then(exec::schedule_after(s, 5s), [&] { ranWork = true; }),
//           exec::schedule_after(s, 5ms)),
//       [] { return stdexec::just(); }));

//   auto duration = exec::now(s) - startTime;

//   // Work should have been cancelled.
//   REQUIRE_FALSE(ranWork);
//   REQUIRE(duration < 1s);
// }

TEST_CASE("windows_thread_pool: schedule_at", "[types][windows_thread_pool][schedulers]") {
  exec::windows_thread_pool tp;
  auto s = tp.get_scheduler();

  auto startTime = exec::now(s);

  stdexec::sync_wait(exec::schedule_at(s, startTime + 100ms));

  auto endTime = exec::now(s);
  REQUIRE(endTime >= (startTime + 100ms));
  REQUIRE(endTime < (startTime + 150ms));
}

// TEST_CASE("windows_thread_pool: schedule_at_cancellation", "[types][windows_thread_pool][schedulers]") {
//   exec::windows_thread_pool tp;
//   auto s = tp.get_scheduler();

//   auto startTime = exec::now(s);

//   bool ranWork = false;

//   stdexec::sync_wait(stdexec::let_stopped(
//       stdexec::stop_when(
//           stdexec::then(
//               exec::schedule_at(s, startTime + 5s), [&] { ranWork = true; }),
//           stdexec::schedule_at(s, startTime + 5ms)),
//       [] { return stdexec::just(); }));

//   auto duration = exec::now(s) - startTime;

//   // Work should have been cancelled.
//   REQUIRE_FALSE(ranWork);
//   REQUIRE(duration < 1s);
// }
