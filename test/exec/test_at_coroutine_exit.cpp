/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#if !STDEXEC_STD_NO_COROUTINES() && !STDEXEC_STD_NO_EXCEPTIONS()
#  include <exec/at_coroutine_exit.hpp>
#  include <exec/on_coro_disposition.hpp>
#  include <catch2/catch.hpp>

#  include "../test_common/require_terminate.hpp"
#  include "../test_common/schedulers.hpp"

using namespace exec;
using stdexec::sync_wait;

namespace {
  auto stop() {
    static stopped_scheduler scheduler{};
    auto stop = stdexec::schedule(scheduler);
    return stop;
  }

  auto test_one_cleanup_action(int& result) -> task<void> {
    ++result;
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= 2;
      co_return;
    });
    ++result;
  }

  auto test_two_cleanup_actions(int& result) -> task<void> {
    ++result;
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= 2;
      co_return;
    });
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= result;
      co_return;
    });
    ++result;
  }

  auto test_on_stopped_two_cleanup_actions_with_stop(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_stopped([&result]() -> task<void> {
      result *= 2;
      co_return;
    });
    co_await on_coroutine_stopped([&result]() -> task<void> {
      result *= result;
      co_return;
    });
    ++result;
    co_await stop();
  }

  auto test_one_cleanup_action_with_stop(int& result) -> task<void> {
    ++result;
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= 2;
      co_return;
    });
    co_await stop();
    ++result;
  }

  auto test_on_succeeded_one_cleanup_action(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_succeeded([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    ++result;
  }

  auto test_on_succeeded_one_cleanup_action_with_stop(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_succeeded([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    co_await stop();
    ++result;
  }

  auto test_on_succeeded_one_cleanup_action_with_error(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_succeeded([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    throw 42;
    ++result;
  }

  auto test_on_stopped_one_cleanup_action(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_stopped([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    ++result;
  }

  auto test_on_stopped_one_cleanup_action_with_stop(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_stopped([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    co_await stop();
    ++result;
  }

  auto test_on_stopped_one_cleanup_action_with_error(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_stopped([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    throw 42;
    ++result;
  }

  auto test_on_failed_one_cleanup_action(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_failed([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    ++result;
  }

  auto test_on_failed_one_cleanup_action_with_stop(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_failed([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    co_await stop();
    ++result;
  }

  auto test_on_failed_one_cleanup_action_with_error(int& result) -> task<void> {
    ++result;
    co_await on_coroutine_failed([&result]() -> task<void> {
      result *= 3;
      co_return;
    });
    throw 42;
    ++result;
  }

  auto test_two_cleanup_actions_with_stop(int& result) -> task<void> {
    ++result;
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= 2;
      co_return;
    });
    co_await at_coroutine_exit([&result]() -> task<void> {
      result *= result;
      co_return;
    });
    co_await stop();
    ++result;
  }

  auto test_sender_cleanup_action(int& result) -> task<void> {
    co_await at_coroutine_exit(
      [&result] { return stdexec::just() | stdexec::then([&result] { ++result; }); });
  }

  auto test_stateful_cleanup_action(int& result, int arg) -> task<void> {
    co_await at_coroutine_exit([arg, &result] {
      return stdexec::just() | stdexec::then([arg, &result] { result += arg; });
    });
  }

  auto test_mutable_stateful_cleanup_action(int& result) -> task<void> {
    auto&& [i] = co_await at_coroutine_exit(
      [&result](int&& i) -> task<void> {
        result += i;
        co_return;
      },
      3);
    ++result;
    i *= i;
  }

  auto test_on_succeeded_mutable_stateful_cleanup_action(int& result) -> task<void> {
    auto&& [i] = co_await on_coroutine_succeeded(
      [&result](int&& i) -> task<void> {
        result += i;
        co_return;
      },
      3);
    ++result;
    i *= i;
  }

  auto with_continuation(int& result, task<void> next) -> task<void> {
    co_await std::move(next);
    result *= 3;
  }

#  ifdef REQUIRE_TERMINATE

  void test_cancel_in_cleanup_action_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> { co_await stop(); });
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

  void test_cancel_during_cancellation_unwind_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> {
        co_await stop(); // BOOM
      });
      co_await stop();
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

  void test_throw_in_cleanup_action_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> { throw 42; });
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

  void test_throw_in_cleanup_action_during_exception_unwind_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> { throw 42; });
      throw 42;
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

  void test_cancel_in_cleanup_action_during_exception_unwind_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> { co_await stop(); });
      throw 42;
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

  void test_throw_in_cleanup_action_during_cancellation_unwind_causes_death(int&) {
    task<void> t = []() -> task<void> {
      co_await at_coroutine_exit([]() -> task<void> { throw 42; });
      co_await stop();
    }();
    REQUIRE_TERMINATE([&] { sync_wait(std::move(t)); });
  }

#  endif // REQUIRE_TERMINATE

  TEST_CASE("OneCleanupAction", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_one_cleanup_action(result));
    REQUIRE(result == 4);
  }

  TEST_CASE("TwoCleanupActions", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_two_cleanup_actions(result));
    REQUIRE(result == 8);
  }

  TEST_CASE("OnStoppedTwoCleanupActions", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_stopped_two_cleanup_actions_with_stop(result));
    REQUIRE(result == 8);
  }

  TEST_CASE("OneCleanupActionWithContinuation", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(with_continuation(result, test_one_cleanup_action(result)));
    REQUIRE(result == 12);
  }

  TEST_CASE("TwoCleanupActionsWithContinuation", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(with_continuation(result, test_two_cleanup_actions(result)));
    REQUIRE(result == 24);
  }

  TEST_CASE("CleanupActionWithSender", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_sender_cleanup_action(result));
    REQUIRE(result == 1);
  }

  TEST_CASE("OneCleanupActionWithStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_one_cleanup_action_with_stop(result));
    REQUIRE(result == 2);
  }

  TEST_CASE("OnSucceededOneCleanupAction", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_succeeded_one_cleanup_action(result));
    REQUIRE(result == 6);
  }

  TEST_CASE("OnSucceededOneCleanupActionWithStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_succeeded_one_cleanup_action_with_stop(result));
    REQUIRE(result == 1);
  }

  TEST_CASE("OnSucceededOneCleanupActionWithError", "[task][at_coroutine_exit]") {
    int result = 0;
    CHECK_THROWS_AS(
      stdexec::sync_wait(test_on_succeeded_one_cleanup_action_with_error(result)), int);
    REQUIRE(result == 1);
  }

  TEST_CASE("OnStoppedOneCleanupActionSuccess", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_stopped_one_cleanup_action(result));
    REQUIRE(result == 2);
  }

  TEST_CASE("OnStoppedOneCleanupActionWithStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_stopped_one_cleanup_action_with_stop(result));
    REQUIRE(result == 3);
  }

  TEST_CASE("OnStoppedOneCleanupActionWithError", "[task][at_coroutine_exit]") {
    int result = 0;
    CHECK_THROWS_AS(stdexec::sync_wait(test_on_stopped_one_cleanup_action_with_error(result)), int);
    REQUIRE(result == 1);
  }

  TEST_CASE("OnFailedOneCleanupActionSuccess", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_failed_one_cleanup_action(result));
    REQUIRE(result == 2);
  }

  TEST_CASE("OnFailedOneCleanupActionWithStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_failed_one_cleanup_action_with_stop(result));
    REQUIRE(result == 1);
  }

  TEST_CASE("OnFailedOneCleanupActionWithError", "[task][at_coroutine_exit]") {
    int result = 0;
    CHECK_THROWS_AS(stdexec::sync_wait(test_on_failed_one_cleanup_action_with_error(result)), int);
    REQUIRE(result == 3);
  }

  TEST_CASE("TwoCleanupActionsWithStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_two_cleanup_actions_with_stop(result));
    REQUIRE(result == 2);
  }

  TEST_CASE("OneCleanupActionWithContinuationAndStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(with_continuation(result, test_one_cleanup_action_with_stop(result)));
    REQUIRE(result == 2);
  }

  TEST_CASE("TwoCleanupActionsWithContinuationAndStop", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(with_continuation(result, test_two_cleanup_actions_with_stop(result)));
    REQUIRE(result == 2);
  }

  TEST_CASE("CleanupActionWithStatefulSender", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_stateful_cleanup_action(result, 42));
    REQUIRE(result == 42);
  }

  TEST_CASE("CleanupActionWithMutableStateful", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_mutable_stateful_cleanup_action(result));
    REQUIRE(result == 10);
  }

  TEST_CASE("OnSuccessCleanupActionWithMutableStateful", "[task][at_coroutine_exit]") {
    int result = 0;
    stdexec::sync_wait(test_on_succeeded_mutable_stateful_cleanup_action(result));
    REQUIRE(result == 10);
  }

#  ifdef REQUIRE_TERMINATE

  TEST_CASE("CancelInCleanupActionCallsTerminate", "[task][at_coroutine_exit]") {
    int result = 0;
    test_cancel_in_cleanup_action_causes_death(result);
  }

  TEST_CASE("CancelDuringCancellationUnwindCallsTerminate", "[task][at_coroutine_exit]") {
    int result = 0;
    test_cancel_during_cancellation_unwind_causes_death(result);
  }

  TEST_CASE("ThrowInCleanupActionCallsTerminate", "[task][at_coroutine_exit]") {
    int result = 0;
    test_throw_in_cleanup_action_causes_death(result);
  }

  TEST_CASE(
    "ThrowInCleanupActionDuringExceptionUnwindCallsTerminate",
    "[task][at_coroutine_exit]") {
    int result = 0;
    test_throw_in_cleanup_action_during_exception_unwind_causes_death(result);
  }

  TEST_CASE(
    "CancelInCleanupActionDuringExceptionUnwindCallsTerminate",
    "[task][at_coroutine_exit]") {
    int result = 0;
    test_cancel_in_cleanup_action_during_exception_unwind_causes_death(result);
  }

  TEST_CASE(
    "ThrowInCleanupActionDuringCancellationUnwindCallsTerminate",
    "[task][at_coroutine_exit]") {
    int result = 0;
    test_throw_in_cleanup_action_during_cancellation_unwind_causes_death(result);
  }

#  endif // REQUIRE_TERMINATE

} // unnamed namespace

#endif // !STDEXEC_STD_NO_COROUTINES() && !STDEXEC_STD_NO_EXCEPTIONS()
