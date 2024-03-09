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
} // namespace