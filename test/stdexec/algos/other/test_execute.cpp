/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#include <atomic>
#include <catch2/catch.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>

#include <chrono>

namespace ex = STDEXEC;

using namespace std::chrono_literals;

// Trying to test `execute` with error flows will result in calling `std::terminate()`.
// We don't want that

namespace {

  TEST_CASE("execute works with inline scheduler", "[other][execute]") {
    inline_scheduler sched;
    bool called{false};
    ex::execute(sched, [&] { called = true; });
    // The function should already have been called
    CHECK(called);
  }

  TEST_CASE(
    "execute works with schedulers that need to be triggered manually",
    "[other][execute]") {
    impulse_scheduler sched;
    bool called{false};
    ex::execute(sched, [&] { called = true; });
    // The function has not yet been called
    CHECK_FALSE(called);
    // After an impulse to the scheduler, the function should have been called
    sched.start_next();
    CHECK(called);
  }

  TEST_CASE("execute works on a thread pool", "[other][execute]") {
    exec::static_thread_pool pool{2};
    std::atomic<bool> called{false};
    {
      // launch some work on the thread pool
      ex::execute(pool.get_scheduler(), [&] { called.store(true, std::memory_order_relaxed); });
    }
    // wait for the work to be executed, with timeout
    // perform a poor-man's sync
    // NOTE: it's a shame that the `join` method in static_thread_pool is not public
    for (int i = 0; i < 1000 && !called.load(std::memory_order_relaxed); i++) {
      std::this_thread::sleep_for(1ms);
    }
    // the work should be executed
    REQUIRE(called);
  }
} // namespace
