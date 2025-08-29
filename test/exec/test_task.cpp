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

#if !STDEXEC_STD_NO_COROUTINES()
#  include <exec/task.hpp>
#  include <exec/single_thread_context.hpp>
#  include <exec/async_scope.hpp>

#  include <test_common/schedulers.hpp>

#  include <catch2/catch.hpp>

#  include <string>

using namespace exec;
using namespace stdexec;
using namespace std::string_literals;

namespace {

  // This is a work-around for clang-12 bugs in Release mode
  thread_local int __thread_id = 0;

  static_assert(stdexec::sender<exec::task<void>>);

  // This is a work-around for apple clang bugs in Release mode
  STDEXEC_WHEN(STDEXEC_APPLE_CLANG(), [[clang::optnone]]) auto get_id() -> int {
    return __thread_id;
  }

  auto test_stickiness_for_two_single_thread_contexts_nested(
    scheduler auto scheduler1,
    scheduler auto,
    auto id1,
    auto id2) -> task<void> {
    CHECK(get_id() == id2);                       // This task is started in context2
    co_await schedule(scheduler1);                // Try to schedule in context1
    CHECK(get_id() == id2);                       // But this task is still in context2
    co_await reschedule_coroutine_on(scheduler1); // Transition to context1
    CHECK(get_id() == id1);                       // Now we are in context1
  } // Reschedules back to context2

  auto test_stickiness_for_two_single_thread_contexts_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> task<void> {
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
  } // Reschedules back to context1

  auto test_stickiness_for_two_single_thread_contexts_with_sender_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> task<void> {
    co_await reschedule_coroutine_on(scheduler2); // Transition to context2
    CHECK(get_id() == id2);                       // Now we are in context2
    // Child task inherits context2
    co_await (
      test_stickiness_for_two_single_thread_contexts_nested(scheduler1, scheduler2, id1, id2)
      | then([&] { CHECK(get_id() == id2); }));
  }

  auto test_stickiness_for_two_single_thread_contexts(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> task<void> {
    co_await reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
  }

  auto test_stickiness_for_two_single_thread_contexts_with_sender(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> task<void> {
    co_await reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_with_sender_(
      scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
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
    auto t = starts_on(
      scheduler1,
      test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2));
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
    auto t =
      test_stickiness_for_two_single_thread_contexts_with_sender(scheduler1, scheduler2, id1, id2);
    sync_wait(std::move(t));
  }

  TEST_CASE(
    "Test stickiness with two single threads with sender with starts_on",
    "[types][sticky][task]") {
    single_thread_context context1;
    single_thread_context context2;
    scheduler auto scheduler1 = context1.get_scheduler();
    scheduler auto scheduler2 = context2.get_scheduler();
    sync_wait(when_all(
      schedule(scheduler1) | then([] { __thread_id = 1; }),
      schedule(scheduler2) | then([] { __thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto t = starts_on(
      scheduler1,
      test_stickiness_for_two_single_thread_contexts_with_sender_(
        scheduler1, scheduler2, id1, id2));
    sync_wait(std::move(t) | then([&] { CHECK(get_id() == id1); }));
  }

  TEST_CASE("Use two inline schedulers", "[types][sticky][task]") {
    scheduler auto scheduler1 = stdexec::inline_scheduler{};
    scheduler auto scheduler2 = stdexec::inline_scheduler{};
    sync_wait(when_all(
      schedule(scheduler1) | then([] { __thread_id = 0; }),
      schedule(scheduler2) | then([] { __thread_id = 0; })));
    auto id1 = 0;
    auto id2 = 0;
    auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
    sync_wait(std::move(t));
  }

  namespace {
    auto test_stick_on_main_nested(
      scheduler auto sched1,
      scheduler auto,
      auto id_main_thread,
      [[maybe_unused]] auto id1,
      [[maybe_unused]] auto id2) -> task<void> {
      CHECK(get_id() == id_main_thread);
      co_await schedule(sched1);
      CHECK(get_id() == id_main_thread);
    }

    auto test_stick_on_main(
      scheduler auto sched1,
      scheduler auto sched2,
      auto id_main_thread,
      [[maybe_unused]] auto id1,
      [[maybe_unused]] auto id2) -> task<void> {
      CHECK(get_id() == id_main_thread);
      co_await schedule(sched1);
      CHECK(get_id() == id_main_thread);
      co_await schedule(sched2);
      CHECK(get_id() == id_main_thread);
      co_await test_stick_on_main_nested(sched1, sched2, id_main_thread, id1, id2);
      CHECK(get_id() == id_main_thread);
    }
  } // namespace

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

  auto check_stop_possible() -> exec::task<void> {
    auto stop_token = co_await stdexec::get_stop_token();
    CHECK(stop_token.stop_possible());
  }

  TEST_CASE("task - stop token is forwarded", "[types][task]") {
    single_thread_context context{};
    exec::async_scope scope;
    scope.spawn(stdexec::starts_on(context.get_scheduler(), check_stop_possible()));
    CHECK(stdexec::sync_wait(scope.on_empty()));
  }

  TEST_CASE("task - can stop early", "[types][task]") {
    int count = 0;
    auto work = [](int& count) -> exec::task<void> {
      count += 1;
      co_await [](int& count) -> exec::task<void> {
        count += 2;
        co_await stdexec::just_stopped();
        count += 4;
      }(count);
      count += 8;
    }(count);

    auto res = stdexec::sync_wait(std::move(work));
    CHECK(!res.has_value());
    CHECK(count == 3);
  }

  TEST_CASE("task - can co_await task wrapped in write_env", "[types][task]") {
    stdexec::sync_wait([]() -> exec::task<void> {
      co_await stdexec::write_env(
          []() -> exec::task<void> {
            auto token = co_await stdexec::get_stop_token();
            assert(!token.stop_possible());
          }(),
          stdexec::prop{stdexec::get_stop_token, stdexec::never_stop_token{}});
    }());
  }

  struct test_domain {
    template <sender_expr_for<then_t> _Sender>
    static constexpr auto transform_sender(_Sender&&) noexcept {
      return just("goodbye"s);
    }
  };

  struct test_task_context {
    constexpr test_task_context(auto&&...) noexcept {
    }

    template <class _ThisPromise>
    using promise_context_t = test_task_context;

    template <class, class>
    using awaiter_context_t = test_task_context;

    static constexpr auto query(get_scheduler_t) noexcept {
      return basic_inline_scheduler<test_domain>{};
    }
  };

  template <class T>
  using test_task = exec::basic_task<T, test_task_context>;

  TEST_CASE("task - can co_await a sender adaptor closure object", "[types][task]") {
    auto salutation = []() -> test_task<std::string> {
      co_return co_await then([] { return "hello"s; });
    }();
    auto [msg] = stdexec::sync_wait(std::move(salutation)).value();
    CHECK(msg == "goodbye"s);
  }

#  if !STDEXEC_STD_NO_EXCEPTIONS()
  TEST_CASE("task - can error early", "[types][task]") {
    int count = 0;
    auto work = [](int& count) -> exec::task<void> {
      count += 1;
      co_await [](int& count) -> exec::task<void> {
        count += 2;
        co_await stdexec::just_error(std::runtime_error("on noes"));
        count += 4;
      }(count);
      count += 8;
    }(count);

    try {
      stdexec::sync_wait(std::move(work));
      CHECK(false);
    } catch (const std::runtime_error& e) {
      CHECK(std::string_view(e.what()) == "on noes");
    }
    CHECK(count == 3);
  }
#  endif // !STDEXEC_STD_NO_EXCEPTIONS()
} // namespace

#endif
