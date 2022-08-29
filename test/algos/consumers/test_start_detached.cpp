/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <examples/schedulers/static_thread_pool.hpp>

#include <chrono>

namespace ex = std::execution;

using namespace std::chrono_literals;

TEST_CASE("start_detached simple test", "[consumers][start_detached]") {
  bool called{false};
  ex::start_detached(ex::just() | ex::then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("start_detached works with void senders", "[consumers][start_detached]") {
  ex::start_detached(ex::just());
}

TEST_CASE("start_detached works with multi-value senders", "[consumers][start_detached]") {
  ex::start_detached(ex::just(3, 0.1415));
}

// Trying to test `start_detached` with error flows will result in calling `std::terminate()`.
// We don't want that

TEST_CASE(
    "start_detached works with sender ending with `set_stopped`", "[consumers][start_detached]") {
  ex::start_detached(ex::just_stopped());
}

TEST_CASE("start_detached works with senders that do not complete immediately",
    "[consumers][start_detached]") {
  impulse_scheduler sched;
  bool called{false};
  // Start the sender
  ex::start_detached(ex::transfer_just(sched) | ex::then([&] { called = true; }));
  // The `then` function is not yet called
  CHECK_FALSE(called);
  // After an impulse to the scheduler, the function would complete
  sched.start_next();
  CHECK(called);
}

TEST_CASE("start_detached works when changing threads", "[consumers][start_detached]") {
  example::static_thread_pool pool{2};
  bool called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::transfer_just(pool.get_scheduler()) //
                          | ex::then([&] { called = true; });
    ex::start_detached(std::move(snd));
  }
  // wait for the work to be executed, with timeout
  // perform a poor-man's sync
  // NOTE: it's a shame that the `join` method in static_thread_pool is not public
  for (int i = 0; i < 1000 && !called; i++)
    std::this_thread::sleep_for(1ms);
  // the work should be executed
  REQUIRE(called);
}
