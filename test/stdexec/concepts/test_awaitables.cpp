/*
 * Copyright (c) 2022 Shreyas Atre
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

#include <stdexec/coroutine.hpp>
#include <tuple>
#include <variant>

#if !_STD_NO_COROUTINES_

namespace ex = stdexec;

template <typename Awaiter>
struct promise {
  __coro::coroutine_handle<promise> get_return_object() {
    return {__coro::coroutine_handle<promise>::from_promise(*this)};
  }
  __coro::suspend_always initial_suspend() noexcept { return {}; }
  __coro::suspend_always final_suspend() noexcept { return {}; }
  void return_void() {}
  void unhandled_exception() {}

  template <typename... T>
  auto await_transform(T&&...) noexcept {
    return Awaiter{};
  }
};

struct awaiter {
  bool await_ready() { return true; }
  bool await_suspend(__coro::coroutine_handle<>) { return false; }
  bool await_resume() { return false; }
};

using dependent = ex::dependent_completion_signatures<ex::no_env>;

template <typename Awaiter>
struct awaitable_sender_1 {
  Awaiter operator co_await();
};

struct awaitable_sender_2 {
  using promise_type = promise<__coro::suspend_always>;
private:
  friend dependent operator co_await(awaitable_sender_2);
};

struct awaitable_sender_3 {
  using promise_type = promise<awaiter>;
private:
  friend dependent operator co_await(awaitable_sender_3);
};

template <typename Signatures, typename Awaiter>
void test_awaitable_sender1(Signatures&&, Awaiter&&) {
  static_assert(ex::sender<awaitable_sender_1<Awaiter>>);
  static_assert(stdexec::__awaitable<awaitable_sender_1<Awaiter>>);

  static_assert(
    !stdexec::__get_completion_signatures::__with_member_alias<awaitable_sender_1<Awaiter>>);
  static_assert(
      std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_1<Awaiter>>, Signatures>);
}

template <typename Signatures>
void test_awaitable_sender2(Signatures) {
  static_assert(ex::sender<awaitable_sender_2>);
  static_assert(ex::sender<awaitable_sender_2, promise<__coro::suspend_always>>);

  static_assert(stdexec::__awaitable<awaitable_sender_2>);
  static_assert(stdexec::__awaitable<awaitable_sender_2, promise<__coro::suspend_always>>);

  static_assert(
    !stdexec::__get_completion_signatures::__with_member_alias<awaitable_sender_2>);

  static_assert(std::is_same_v<
      ex::completion_signatures_of_t<awaitable_sender_2>,
      dependent>);
  static_assert(std::is_same_v<
      ex::completion_signatures_of_t<awaitable_sender_2, promise<__coro::suspend_always>>,
      Signatures>);
}

template <typename Signatures>
void test_awaitable_sender3(Signatures) {
  static_assert(ex::sender<awaitable_sender_3>);
  static_assert(ex::sender<awaitable_sender_3, promise<awaiter>>);

  static_assert(stdexec::__awaiter<awaiter>);
  static_assert(stdexec::__awaitable<awaitable_sender_3>);
  static_assert(stdexec::__awaitable<awaitable_sender_3, promise<awaiter>>);

  static_assert(
    !stdexec::__get_completion_signatures::__with_member_alias<awaitable_sender_3>);

  static_assert(std::is_same_v<
      ex::completion_signatures_of_t<awaitable_sender_3>,
      dependent>);
  static_assert(std::is_same_v<
      ex::completion_signatures_of_t<awaitable_sender_3, promise<awaiter>>,
      Signatures>);
}

template <typename Error, typename... Values>
auto signature_error_values(Error, Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...), ex::set_error_t(Error)> {
  return {};
}

TEST_CASE("get completion_signatures for awaitables", "[sndtraits][awaitables]") {
  test_awaitable_sender1(
    signature_error_values(std::exception_ptr()), __coro::suspend_always{});
  test_awaitable_sender1(
    signature_error_values(
      std::exception_ptr(),
      stdexec::__await_result_t<awaitable_sender_1<awaiter>>()),
    awaiter{});

  test_awaitable_sender2(
    signature_error_values(
      std::exception_ptr()));

  test_awaitable_sender3(
    signature_error_values(
      std::exception_ptr(),
      stdexec::__await_result_t<awaitable_sender_3, promise<awaiter>>()));
}

#endif // !_STD_NO_COROUTINES_
