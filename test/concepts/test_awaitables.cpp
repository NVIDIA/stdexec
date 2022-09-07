/*
 * Copyright (c) Shreyas Atre
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

#include <coroutine.hpp>
#include <tuple>
#include <variant>

namespace ex = std::execution;

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

template <typename Awaiter>
struct awaitable_sender_1 {
  Awaiter operator co_await();
};

struct awaitable_sender_2 {
  using promise_type = promise<__coro::suspend_always>;
};

struct awaitable_sender_3 {
  using promise_type = promise<awaiter>;
};

template <typename Signatures, typename Awaiter>
void test_awaitable_sender1(Signatures&&, Awaiter&&) {
  static_assert(ex::sender<awaitable_sender_1<Awaiter>>);
  static_assert(ex::__awaitable<awaitable_sender_1<Awaiter>>);

  awaitable_sender_1<Awaiter> s;
  static_assert(!ex::__get_completion_signatures::__with_member_alias<awaitable_sender_1<Awaiter>>);
  static_assert(std::is_same_v<decltype(ex::get_completion_signatures(s)), Signatures>);
  static_assert(
      std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_1<Awaiter>>, Signatures>);

  using value_types_of = ex::value_types_of_t<awaitable_sender_1<Awaiter>>;
  using value_types = typename Signatures::template __gather_sigs<ex::set_value_t,
      ex::__q<ex::__decayed_tuple>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<value_types_of, value_types>);

  using error_types_of = ex::error_types_of_t<awaitable_sender_1<Awaiter>>;
  using error_types = typename Signatures::template __gather_sigs<ex::set_error_t,
      ex::__q1<ex::__id>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Signatures>
void test_awaitable_sender2(Signatures) {
  // is_sender_v relies on get_completion_signatures and is not true
  // even if a sender is an awaitable if it's promise type is not
  // used to evaluate the concept awaitable
  static_assert(ex::sender<awaitable_sender_2, promise<__coro::suspend_always>>);

  static_assert(ex::__awaitable<awaitable_sender_2, promise<__coro::suspend_always>>);

  awaitable_sender_2 s;
  promise<__coro::suspend_always> p;
  static_assert(!ex::__get_completion_signatures::__with_member_alias<awaitable_sender_2>);

  static_assert(std::is_same_v<decltype(ex::get_completion_signatures(s, p)), Signatures>);
  static_assert(
      ex::__awaitable<awaitable_sender_2, ex::__env_or_void<promise<__coro::suspend_always>>>);
  static_assert(std::is_same_v<
      ex::completion_signatures_of_t<awaitable_sender_2, promise<__coro::suspend_always>>,
      Signatures>);

  using value_types_of = ex::value_types_of_t<awaitable_sender_2, promise<__coro::suspend_always>>;
  using value_types = typename Signatures::template __gather_sigs<ex::set_value_t,
      ex::__q<ex::__decayed_tuple>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<value_types_of, value_types>);

  using error_types_of = ex::error_types_of_t<awaitable_sender_2, promise<__coro::suspend_always>>;
  using error_types = typename Signatures::template __gather_sigs<ex::set_error_t,
      ex::__q1<ex::__id>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Signatures>
void test_awaitable_sender3(Signatures) {
  // is_sender_v relies on get_completion_signatures and is not true
  // even if a sender is an awaitable if it's promise type is not
  // used to evaluate the concept awaitable
  static_assert(ex::sender<awaitable_sender_3, promise<awaiter>>);

  static_assert(ex::__awaiter<awaiter>);
  static_assert(ex::__awaitable<awaitable_sender_3, promise<awaiter>>);

  awaitable_sender_3 s;
  promise<awaiter> p;
  static_assert(!ex::__get_completion_signatures::__with_member_alias<awaitable_sender_3>);
  static_assert(std::is_same_v<decltype(ex::get_completion_signatures(s, p)), Signatures>);
  static_assert(std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_3, promise<awaiter>>,
      Signatures>);

  using value_types_of = ex::value_types_of_t<awaitable_sender_3, promise<awaiter>>;
  using value_types = typename Signatures::template __gather_sigs<ex::set_value_t,
      ex::__q<ex::__decayed_tuple>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<value_types_of, value_types>);

  using error_types_of = ex::error_types_of_t<awaitable_sender_3, promise<awaiter>>;
  using error_types = typename Signatures::template __gather_sigs<ex::set_error_t,
      ex::__q1<ex::__id>, ex::__q<ex::__variant>>;

  static_assert(std::is_same_v<error_types_of, error_types>);
}

template <typename Error, typename... Values>
auto signature_error_values(Error, Values...)
    -> ex::completion_signatures<ex::set_value_t(Values...), ex::set_error_t(Error)> {
  return {};
}

TEST_CASE("get completion_signatures for awaitables", "[sndtraits][awaitables]") {
  test_awaitable_sender1(signature_error_values(std::exception_ptr()), __coro::suspend_always{});
  test_awaitable_sender1(signature_error_values(std::exception_ptr(),
                             ex::__await_result_t<awaitable_sender_1<awaiter>>()),
      awaiter{});
  test_awaitable_sender2(signature_error_values(std::exception_ptr()));
  test_awaitable_sender3(signature_error_values(
      std::exception_ptr(), ex::__await_result_t<awaitable_sender_3, promise<awaiter>>()));
}