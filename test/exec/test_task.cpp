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

#if !STDEXEC_NO_STD_COROUTINES()
#  include <exec/async_scope.hpp>
#  include <exec/single_thread_context.hpp>
#  include <exec/task.hpp>

#  include <test_common/schedulers.hpp>

#  include <catch2/catch.hpp>

#  include <string>

namespace ex = STDEXEC;
using namespace std::string_literals;

namespace {

  // This is a work-around for clang-12 bugs in Release mode
  thread_local constinit int thread_id = 0;

  static_assert(STDEXEC::sender<exec::task<void>>);

  // This is a work-around for apple clang bugs in Release mode
  STDEXEC_PP_WHEN(STDEXEC_APPLE_CLANG(), [[clang::optnone]]) auto get_id() -> int {
    return thread_id;
  }

  auto test_stickiness_for_two_single_thread_contexts_nested(
    ex::scheduler auto scheduler1,
    ex::scheduler auto,
    auto id1,
    auto id2) -> exec::task<void> {
    CHECK(get_id() == id2);                       // This task is started in context2
    co_await ex::schedule(scheduler1);                // Try to ex::schedule in context1
    CHECK(get_id() == id2);                       // But this task is still in context2
    co_await exec::reschedule_coroutine_on(scheduler1); // Transition to context1
    CHECK(get_id() == id1);                       // Now we are in context1
  } // Reschedules back to context2

  auto test_stickiness_for_two_single_thread_contexts_(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> exec::task<void> {
    CHECK(get_id() == id1);        // This task is start in context1
    co_await ex::schedule(scheduler2); // Try to ex::schedule in context2
    CHECK(get_id() == id1);        // But this task is still in context1
    co_await (ex::schedule(scheduler1) | ex::then([&] { CHECK(get_id() == id1); }));
    co_await (ex::schedule(scheduler2) | ex::then([&] { CHECK(get_id() == id2); }));
    CHECK(get_id() == id1);
    co_await exec::reschedule_coroutine_on(scheduler2); // Transition to context2
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
    auto id2) -> exec::task<void> {
    co_await exec::reschedule_coroutine_on(scheduler2); // Transition to context2
    CHECK(get_id() == id2);                       // Now we are in context2
    // Child task inherits context2
    co_await (
      test_stickiness_for_two_single_thread_contexts_nested(scheduler1, scheduler2, id1, id2)
      | ex::then([&] { CHECK(get_id() == id2); }));
  }

  auto test_stickiness_for_two_single_thread_contexts(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> exec::task<void> {
    co_await exec::reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
  }

  auto test_stickiness_for_two_single_thread_contexts_with_sender(
    auto scheduler1,
    auto scheduler2,
    auto id1,
    auto id2) -> exec::task<void> {
    co_await exec::reschedule_coroutine_on(scheduler1);
    CHECK(get_id() == id1);
    co_await test_stickiness_for_two_single_thread_contexts_with_sender_(
      scheduler1, scheduler2, id1, id2);
    CHECK(get_id() == id1);
  }

  TEST_CASE("Test stickiness with two single threads", "[types][sticky][task]") {
    exec::single_thread_context context1;
    exec::single_thread_context context2;
    ex::scheduler auto scheduler1 = context1.get_scheduler();
    ex::scheduler auto scheduler2 = context2.get_scheduler();
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 1; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
    ex::sync_wait(std::move(t));
  }

  TEST_CASE("Test stickiness with two single threads with on", "[types][sticky][task]") {
    exec::single_thread_context context1;
    exec::single_thread_context context2;
    ex::scheduler auto scheduler1 = context1.get_scheduler();
    ex::scheduler auto scheduler2 = context2.get_scheduler();
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 1; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto t = ex::starts_on(
      scheduler1,
      test_stickiness_for_two_single_thread_contexts_(scheduler1, scheduler2, id1, id2));
    ex::sync_wait(std::move(t) | ex::then([&] { CHECK(get_id() == id1); }));
  }

  TEST_CASE("Test stickiness with two single threads with sender", "[types][sticky][task]") {
    exec::single_thread_context context1;
    exec::single_thread_context context2;
    ex::scheduler auto scheduler1 = context1.get_scheduler();
    ex::scheduler auto scheduler2 = context2.get_scheduler();
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 1; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto t =
      test_stickiness_for_two_single_thread_contexts_with_sender(scheduler1, scheduler2, id1, id2);
    ex::sync_wait(std::move(t));
  }

  TEST_CASE(
    "Test stickiness with two single threads with sender with starts_on",
    "[types][sticky][task]") {
    exec::single_thread_context context1;
    exec::single_thread_context context2;
    ex::scheduler auto scheduler1 = context1.get_scheduler();
    ex::scheduler auto scheduler2 = context2.get_scheduler();
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 1; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto t = ex::starts_on(
      scheduler1,
      test_stickiness_for_two_single_thread_contexts_with_sender_(
        scheduler1, scheduler2, id1, id2));
    ex::sync_wait(std::move(t) | ex::then([&] { CHECK(get_id() == id1); }));
  }

  TEST_CASE("Use two inline schedulers", "[types][sticky][task]") {
    ex::scheduler auto scheduler1 = STDEXEC::inline_scheduler{};
    ex::scheduler auto scheduler2 = STDEXEC::inline_scheduler{};
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 0; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 0; })));
    auto id1 = 0;
    auto id2 = 0;
    auto t = test_stickiness_for_two_single_thread_contexts(scheduler1, scheduler2, id1, id2);
    ex::sync_wait(std::move(t));
  }

  auto test_stick_on_main_nested(
    ex::scheduler auto sched1,
    ex::scheduler auto,
    auto id_main_thread,
    [[maybe_unused]] auto id1,
    [[maybe_unused]] auto id2) -> exec::task<void> {
    CHECK(get_id() == id_main_thread);
    co_await ex::schedule(sched1);
    CHECK(get_id() == id_main_thread);
  }

  auto test_stick_on_main(
    ex::scheduler auto sched1,
    ex::scheduler auto sched2,
    auto id_main_thread,
    [[maybe_unused]] auto id1,
    [[maybe_unused]] auto id2) -> exec::task<void> {
    CHECK(get_id() == id_main_thread);
    co_await ex::schedule(sched1);
    CHECK(get_id() == id_main_thread);
    co_await ex::schedule(sched2);
    CHECK(get_id() == id_main_thread);
    co_await test_stick_on_main_nested(sched1, sched2, id_main_thread, id1, id2);
    CHECK(get_id() == id_main_thread);
  }

  TEST_CASE("Stick on main thread if completes_inline is not used", "[types][sticky][task]") {
    exec::single_thread_context context1;
    exec::single_thread_context context2;
    ex::scheduler auto scheduler1 = context1.get_scheduler();
    ex::scheduler auto scheduler2 = context2.get_scheduler();
    ex::sync_wait(ex::when_all(
      ex::schedule(scheduler1) | ex::then([] { thread_id = 1; }),
      ex::schedule(scheduler2) | ex::then([] { thread_id = 2; })));
    auto id1 = 1;
    auto id2 = 2;
    auto id_main_thread = 0;
    auto t = test_stick_on_main(scheduler1, scheduler2, id_main_thread, id1, id2);
    ex::sync_wait(std::move(t));
  }

  auto check_stop_possible() -> exec::task<void> {
    auto stop_token = co_await STDEXEC::get_stop_token();
    CHECK(stop_token.stop_possible());
  }

  TEST_CASE("task - stop token is forwarded", "[types][task]") {
    exec::single_thread_context context{};
    exec::async_scope scope;
    scope.spawn(STDEXEC::starts_on(context.get_scheduler(), check_stop_possible()));
    CHECK(STDEXEC::sync_wait(scope.on_empty()));
  }

  TEST_CASE("task - can stop early", "[types][task]") {
    int count = 0;
    auto work = [](int& count) -> exec::task<void> {
      count += 1;
      co_await [](int& count) -> exec::task<void> {
        count += 2;
        co_await STDEXEC::just_stopped();
        count += 4;
      }(count);
      count += 8;
    }(count);

    auto res = STDEXEC::sync_wait(std::move(work));
    CHECK(!res.has_value());
    CHECK(count == 3);
  }

  TEST_CASE("task - can co_await task wrapped in write_env", "[types][task]") {
    STDEXEC::sync_wait([]() -> exec::task<void> {
      co_await STDEXEC::write_env(
        []() -> exec::task<void> {
          auto token = co_await STDEXEC::get_stop_token();
          assert(!token.stop_possible());
          (void) token;
        }(),
        STDEXEC::prop{STDEXEC::get_stop_token, STDEXEC::never_stop_token{}});
    }());
  }

  struct test_domain {
    template <ex::sender_expr_for<ex::then_t> _Sender>
    static constexpr auto transform_sender(STDEXEC::set_value_t, _Sender&&, auto&&...) noexcept {
      return ex::just("goodbye"s);
    }
  };

  struct test_task_context {
    constexpr test_task_context(auto&&...) noexcept {
    }

    template <class _ThisPromise>
    using promise_context_t = test_task_context;

    template <class, class>
    using awaiter_context_t = test_task_context;

    static constexpr auto query(ex::get_scheduler_t) noexcept {
      return basic_inline_scheduler<test_domain>{};
    }
  };

  template <class T>
  using test_task = exec::basic_task<T, test_task_context>;

  TEST_CASE("task - can co_await a sender adaptor closure object", "[types][task]") {
    auto salutation = []() -> test_task<std::string> {
      co_return co_await ex::then([] { return "hello"s; });
    }();
    auto [msg] = STDEXEC::sync_wait(std::move(salutation)).value();
    CHECK(msg == "goodbye"s);
  }

#  if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("task - can error early", "[types][task]") {
    int count = 0;
    auto work = [](int& count) -> exec::task<void> {
      count += 1;
      co_await [](int& count) -> exec::task<void> {
        count += 2;
        co_await STDEXEC::just_error(std::runtime_error("on noes"));
        count += 4;
      }(count);
      count += 8;
    }(count);

    try {
      STDEXEC::sync_wait(std::move(work));
      CHECK(false);
    } catch (const std::runtime_error& e) {
      CHECK(std::string_view(e.what()) == "on noes");
    }
    CHECK(count == 3);
  }
#  endif // !STDEXEC_NO_STD_EXCEPTIONS()
} // namespace

#endif
