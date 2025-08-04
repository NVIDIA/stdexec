/*
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

#include <exec/timed_thread_scheduler.hpp>

#include "catch2/catch.hpp"

#include <exec/async_scope.hpp>
#include <exec/when_any.hpp>

// Avoid a TSAN bug in GCC 11 and earlier
#if STDEXEC_GCC() && STDEXEC_GCC_VERSION < 12'00 && defined(__SANITIZE_THREAD__)
// nothing
#else
namespace {
  TEST_CASE(
    "timed_thread_scheduler - unused context",
    "[types][timed_thread_scheduler][schedulers]") {
    static_assert(exec::__timed_scheduler<exec::timed_thread_scheduler>);
    exec::timed_thread_context context;
  }

  TEST_CASE("timed_thread_scheduler - now", "[timed_thread_scheduler][now]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto tp = exec::now(scheduler);
    REQUIRE(tp.time_since_epoch().count() > 0);
  }

  TEST_CASE("timed_thread_scheduler - schedule", "[timed_thread_scheduler][schedule]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    CHECK(stdexec::sync_wait(stdexec::schedule(scheduler)));
  }

  TEST_CASE("timed_thread_scheduler - schedule_at", "[timed_thread_scheduler][schedule_at]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto now = exec::now(scheduler);
    auto tp = now + std::chrono::milliseconds(10);
    CHECK(stdexec::sync_wait(exec::schedule_at(scheduler, tp)));
  }

  TEST_CASE("timed_thread_scheduler - schedule_after", "[timed_thread_scheduler][schedule_at]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto duration = std::chrono::milliseconds(10);
    CHECK(stdexec::sync_wait(exec::schedule_after(scheduler, duration)));
  }

  TEST_CASE("timed_thread_scheduler - when_any", "[timed_thread_scheduler][when_any]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto duration1 = std::chrono::milliseconds(10);
    auto duration2 = std::chrono::seconds(5);
    auto shorter = exec::when_any(
      exec::schedule_after(scheduler, duration1) | stdexec::then([] { return 1; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 2; }));
    auto t0 = std::chrono::steady_clock::now();
    auto [n] = stdexec::sync_wait(std::move(shorter)).value();
    auto t1 = std::chrono::steady_clock::now();
    auto duration = t1 - t0;
    CHECK(duration1 <= duration);
    CHECK(n == 1);
  }

  TEST_CASE("timed_thread_scheduler - more when_any", "[timed_thread_scheduler][when_any]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    auto duration1 = std::chrono::milliseconds(10);
    auto duration2 = std::chrono::seconds(5);
    auto shorter = exec::when_any(
      exec::schedule_after(scheduler, duration1) | stdexec::then([] { return 1; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 2; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 3; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 4; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 5; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 6; }),
      exec::schedule_after(scheduler, duration2) | stdexec::then([] { return 7; }));
    auto t0 = std::chrono::steady_clock::now();
    auto [n] = stdexec::sync_wait(std::move(shorter)).value();
    auto t1 = std::chrono::steady_clock::now();
    auto duration = t1 - t0;
    CHECK(duration1 <= duration);
    CHECK(n == 1);
  }

  TEST_CASE(
    "timed_thread_scheduler - many timers with async scope",
    "[timed_thread_scheduler][async_scope]") {
    exec::timed_thread_context context;
    exec::timed_thread_scheduler scheduler = context.get_scheduler();
    exec::async_scope scope;
    int counter = 0;
    int ntimers = 1'000;
    auto now = exec::now(scheduler);
    auto deadline = now + std::chrono::milliseconds(100);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ntimers; ++i) {
      scope
        .spawn(exec::schedule_at(scheduler, deadline) | stdexec::then([&counter] { ++counter; }));
    }
    CHECK(stdexec::sync_wait(scope.on_empty()));
    auto t1 = std::chrono::steady_clock::now();
    CHECK(counter == ntimers);
    auto duration = t1 - t0;
    CHECK(duration > std::chrono::milliseconds(100));
  }
} // namespace
#endif