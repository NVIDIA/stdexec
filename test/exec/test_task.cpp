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

  // This is a work-around for clang-12 bugs in Release mode
  thread_local int __thread_id = 0;

  int get_id() {
    return __thread_id;
  }

  task<void> test_stickiness_for_two_single_thread_contexts_nested(
    scheduler auto scheduler1,
    scheduler auto scheduler2,
    auto id1,
    auto id2) {
    CHECK(get_id() == id2);                       // This task is started in context2
    co_await schedule(scheduler1);                // Try to schedule in context1
    CHECK(get_id() == id2);                       // But this task is still in context2
    co_await reschedule_coroutine_on(scheduler1); // Transition to context1
    CHECK(get_id() == id1);                       // Now we are in context1
  }                                               // Reschedules back to context2

  task<void> test_stickiness_for_two_single_thread_contexts_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    CHECK(get_id() == id1);        // This task is start in context1
    co_await schedule(scheduler2); // Try to schedule in context2
    CHECK(get_id() == id1);        // But this task is still in context1
    co_await (schedule(scheduler1) | then([&] { CHECK(get_id() == id1); }));
    co_await (schedule(scheduler2) | then([&] { CHECK(get_id() == id2); }));
    CHECK(get_id() == id1);
    co_await reschedule_coroutine_on(scheduler2); // Transition to context2
    CHECK(get_id() == id2);                       // Now we are in context2
    // Child task inherits context2
    co_await test_stickiness_for_two_single_thread_contexts_nested(
      scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id2); // Child task is done, we are still in context2
  }                         // Reschedules back to context1

  task<void> test_stickiness_for_two_single_thread_contexts_with_sender_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    co_await reschedule_coroutine_on(scheduler2); // Transition to context2
    CHECK(get_id() == id2);                       // Now we are in context2
    // Child task inherits context2
    co_await (
      test_stickiness_for_two_single_thread_contexts_nested(scheduler1, scheduler2, id1, id2)
      | then([&] { CHECK(get_id() == id2); }));
  }

  task<void> test_stickiness_for_two_single_thread_contexts(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    co_await reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
  }

  task<void> test_stickiness_for_two_single_thread_contexts_with_sender(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) {
    co_await reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_with_sender_(
      scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
  }
}

TEST_CASE("Test stickiness with two single threads", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 1; }),
    schedule(scheduler2) | then([] { __thread_id = 2; })));
  auto id1 = 1;
  auto id2 = 2;
  auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
  sync_wait(std::move(t));
}

TEST_CASE("Test stickiness with two single threads with on", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 1; }),
    schedule(scheduler2) | then([] { __thread_id = 2; })));
  auto id1 = 1;
  auto id2 = 2;
  auto t = on(
    scheduler1, test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2));
  sync_wait(std::move(t) | then([&] { CHECK(get_id() == id1); }));
}

TEST_CASE("Test stickiness with two single threads with sender", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 1; }),
    schedule(scheduler2) | then([] { __thread_id = 2; })));
  auto id1 = 1;
  auto id2 = 2;
  auto t = test_stickiness_for_two_single_thread_contexts_with_sender(
    scheduler1, scheduler2, id1, id2);
  sync_wait(std::move(t));
}

TEST_CASE("Test stickiness with two single threads with sender with on", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 1; }),
    schedule(scheduler2) | then([] { __thread_id = 2; })));
  auto id1 = 1;
  auto id2 = 2;
  auto t = on(
    scheduler1,
    test_stickiness_for_two_single_thread_contexts_with_sender_(scheduler1, scheduler2, id1, id2));
  sync_wait(std::move(t) | then([&] { CHECK(get_id() == id1); }));
}

TEST_CASE("Use two inline schedulers", "[types][sticky][task]") {
  scheduler auto scheduler1 = exec::inline_scheduler{};
  scheduler auto scheduler2 = exec::inline_scheduler{};
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 0; }),
    schedule(scheduler2) | then([] { __thread_id = 0; })));
  auto id1 = 0;
  auto id2 = 0;
  auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
  sync_wait(std::move(t));
}

namespace {
  task<void> test_stick_on_main_nested(
    scheduler auto sched1,
    scheduler auto sched2,
    auto id_main_thread,
    [[maybe_unused]] auto id1,
    [[maybe_unused]] auto id2) {
    CHECK(get_id() == id_main_thread);
    co_await schedule(sched1);
    CHECK(get_id() == id_main_thread);
  }

  task<void> test_stick_on_main(
    scheduler auto sched1,
    scheduler auto sched2,
    auto id_main_thread,
    [[maybe_unused]] auto id1,
    [[maybe_unused]] auto id2) {
    CHECK(get_id() == id_main_thread);
    co_await schedule(sched1);
    CHECK(get_id() == id_main_thread);
    co_await schedule(sched2);
    CHECK(get_id() == id_main_thread);
    co_await test_stick_on_main_nested(sched1, sched2, id_main_thread, id1, id2);
    CHECK(get_id() == id_main_thread);
  }
}

TEST_CASE("Stick on main thread if completes_inline is not used", "[types][sticky][task]") {
  single_thread_context context1;
  single_thread_context context2;
  scheduler auto scheduler1 = context1.get_scheduler();
  scheduler auto scheduler2 = context2.get_scheduler();
  sync_wait(when_all(
    schedule(scheduler1) | then([] { __thread_id = 1; }),
    schedule(scheduler2) | then([] { __thread_id = 2; })));
  auto id1 = 1;
  auto id2 = 2;
  auto id_main_thread = 0;
  auto t = test_stick_on_main(scheduler1, scheduler2, id_main_thread, id1, id2);
  sync_wait(std::move(t));
}


#endif
