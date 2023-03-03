/*
 * Copyright (c) MAikel Nadolski
 * Copyright (c) 2021-2023 NVIDIA Corporation
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

#include <stdexec/coroutine.hpp>

#if !_STD_NO_COROUTINES_
#include <exec/task.hpp>
#include <exec/single_thread_context.hpp>

#include <catch2/catch.hpp>

#include <thread>

using namespace exec;
using namespace stdexec;

namespace {
  task<void> test_stickiness_for_two_single_thread_contexts_nested(
    scheduler auto scheduler1,
    scheduler auto scheduler2,
    auto id1,
    auto id2) {
    CHECK(std::this_thread::get_id() == id2); // This task is started in context2
    co_await schedule(scheduler1);            // Try to schedule in context1
    CHECK(std::this_thread::get_id() == id2); // But this task is still in context2
    co_await complete_inline(scheduler1);     // Transition to context1
    CHECK(std::this_thread::get_id() == id1); // Now we are in context1
  }                                           // Reschedules back to context2

  task<void> test_stickiness_for_two_single_thread_contexts_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    CHECK(std::this_thread::get_id() == id1); // This task is start in context1
    co_await schedule(scheduler2);            // Try to schedule in context2
    CHECK(std::this_thread::get_id() == id1); // But this task is still in context1
    co_await (schedule(scheduler1) | then([&] { CHECK(std::this_thread::get_id() == id1); }));
    co_await (schedule(scheduler2) | then([&] { CHECK(std::this_thread::get_id() == id2); }));
    CHECK(std::this_thread::get_id() == id1);
    co_await complete_inline(scheduler2);     // Transition to context2
    CHECK(std::this_thread::get_id() == id2); // Now we are in context2
    // Child task inherits context2
    co_await test_stickiness_for_two_single_thread_contexts_nested(
      scheduler1, scheduler2, id1, id2);
    CHECK(std::this_thread::get_id() == id2); // Child task is done, we are still in context2
  }                                           // Reschedules back to context1

  task<void> test_stickiness_for_two_single_thread_contexts(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    co_await complete_inline(scheduler1);
    CHECK(std::this_thread::get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2);
    CHECK(std::this_thread::get_id() == id1);
  }
}

TEST_CASE("Test stickiness with two single threads", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  auto id1 = context1.get_thread_id();
  auto id2 = context2.get_thread_id();
  auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
  sync_wait(std::move(t));
}

TEST_CASE("Use two inline schedulers", "[types][sticky][task]") {
  scheduler auto scheduler1 = exec::inline_scheduler{};
  scheduler auto scheduler2 = exec::inline_scheduler{};
  auto id1 = std::this_thread::get_id();
  auto id2 = std::this_thread::get_id();
  auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
  sync_wait(std::move(t));
}

namespace {
  task<void> test_old_behaviour_nested(
    scheduler auto sched1,
    scheduler auto sched2,
    auto id_main_thread,
    auto id1,
    auto id2) {
    CHECK(std::this_thread::get_id() == id2);
    co_await schedule(sched1);
    CHECK(std::this_thread::get_id() == id1);
  }

  task<void> test_old_behaviour(
    scheduler auto sched1,
    scheduler auto sched2,
    auto id_main_thread,
    auto id1,
    auto id2) {
    CHECK(std::this_thread::get_id() == id_main_thread);
    co_await schedule(sched1);
    CHECK(std::this_thread::get_id() == id1); // We changed from main thread to context1
    co_await schedule(sched2);                // Try to schedule in context2
    CHECK(std::this_thread::get_id() == id2); // We changed from context1 to context2
    co_await test_old_behaviour_nested(sched1, sched2, id_main_thread, id1, id2);
    CHECK(std::this_thread::get_id() == id1); // We changed from context2 to context1
  }                                           // completes on id1
}

TEST_CASE("Show old behaviour with one single thread scheduler", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  auto id_main_thread = std::this_thread::get_id();
  auto id1 = context1.get_thread_id();
  auto id2 = context2.get_thread_id();
  auto t = test_old_behaviour(scheduler1, scheduler2, id_main_thread, id1, id2);
  sync_wait(std::move(t));
}

#endif
