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

#include <exec/static_thread_pool.hpp>
#include <stdexec/coroutine.hpp>
#include <tuple>
#include <variant>

#include <test_common/type_helpers.hpp>

#if !STDEXEC_STD_NO_COROUTINES_

namespace ex = stdexec;

template <class Sender>
concept sender_with_env =     //
  ex::sender<Sender> &&       //
  requires(const Sender& s) { //
    ex::get_env(s);
  };

template <typename Awaiter>
struct promise {
  __coro::coroutine_handle<promise> get_return_object() {
    return {__coro::coroutine_handle<promise>::from_promise(*this)};
  }

  __coro::suspend_always initial_suspend() noexcept {
    return {};
  }

  __coro::suspend_always final_suspend() noexcept {
    return {};
  }

  void return_void() {
  }

  void unhandled_exception() {
  }

  template <typename... T>
  auto await_transform(T&&...) noexcept {
    return Awaiter{};
  }
};

struct awaiter {
  bool await_ready() {
    return true;
  }

  bool await_suspend(__coro::coroutine_handle<>) {
    return false;
  }

  bool await_resume() {
    return false;
  }
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

struct awaitable_sender_4 {
  using promise_type = promise<__coro::suspend_always>;

 private:
  template <class Promise>
  friend awaiter tag_invoke(ex::as_awaitable_t, awaitable_sender_4, Promise&) {
    return {};
  }

  friend dependent tag_invoke(ex::as_awaitable_t, awaitable_sender_4, ex::no_env_promise&) {
    return {};
  }
};

struct awaitable_sender_5 {
 private:
  template <class Promise>
  friend awaiter tag_invoke(ex::as_awaitable_t, awaitable_sender_5, Promise&) {
    return {};
  }
};

template <typename Signatures, typename Awaiter>
void test_awaitable_sender1(Signatures*, Awaiter&&) {
  static_assert(ex::sender<awaitable_sender_1<Awaiter>>);
  static_assert(sender_with_env<awaitable_sender_1<Awaiter>>);
  static_assert(ex::__awaitable<awaitable_sender_1<Awaiter>>);

  static_assert(!ex::__get_completion_signatures::
                  __with_member_alias<awaitable_sender_1<Awaiter>, ex::empty_env>);
  static_assert(
    std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_1<Awaiter>>, Signatures>);
}

void test_awaitable_sender2() {
  static_assert(ex::sender<awaitable_sender_2>);
  static_assert(sender_with_env<awaitable_sender_2>);
  static_assert(!ex::sender_in<awaitable_sender_2, ex::empty_env>);

  static_assert(ex::__awaitable<awaitable_sender_2>);
  static_assert(ex::__awaitable<awaitable_sender_2, promise<__coro::suspend_always>>);

  static_assert(
    !ex::__get_completion_signatures::__with_member_alias<awaitable_sender_2, ex::empty_env>);

#if STDEXEC_LEGACY_R5_CONCEPTS()
  static_assert(std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_2>, dependent>);
#endif
}

void test_awaitable_sender3() {
  static_assert(ex::sender<awaitable_sender_3>);
  static_assert(sender_with_env<awaitable_sender_3>);
  static_assert(!ex::sender_in<awaitable_sender_3, ex::empty_env>);

  static_assert(ex::__awaiter<awaiter>);
  static_assert(ex::__awaitable<awaitable_sender_3>);
  static_assert(ex::__awaitable<awaitable_sender_3, promise<awaiter>>);

  static_assert(
    !ex::__get_completion_signatures::__with_member_alias<awaitable_sender_3, ex::empty_env>);

#if STDEXEC_LEGACY_R5_CONCEPTS()
  static_assert(std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_3>, dependent>);
#endif
}

template <class Signatures>
void test_awaitable_sender4(Signatures*) {
  static_assert(ex::sender<awaitable_sender_4>);
  static_assert(sender_with_env<awaitable_sender_4>);
  static_assert(ex::sender_in<awaitable_sender_4, ex::empty_env>);

  static_assert(ex::__awaiter<awaiter>);
  static_assert(!ex::__awaitable<awaitable_sender_4>);
  static_assert(ex::__awaitable<awaitable_sender_4, promise<awaiter>>);
  static_assert(ex::__awaitable<awaitable_sender_4, ex::no_env_promise>);
  static_assert(ex::__awaitable<awaitable_sender_4, ex::__env_promise<ex::empty_env>>);

  static_assert(
    !ex::__get_completion_signatures::__with_member_alias<awaitable_sender_4, ex::empty_env>);

#if STDEXEC_LEGACY_R5_CONCEPTS()
  static_assert(std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_4>, dependent>);
#endif
  static_assert(
    !ex::__get_completion_signatures::__with_member_alias<awaitable_sender_4, ex::empty_env>);

  static_assert(
    std::is_same_v< ex::completion_signatures_of_t<awaitable_sender_4, ex::empty_env>, Signatures>);
}

struct connect_awaitable_promise : ex::with_awaitable_senders<connect_awaitable_promise> { };

template <class Signatures>
void test_awaitable_sender5(Signatures*) {
  static_assert(ex::sender<awaitable_sender_5>);
  static_assert(sender_with_env<awaitable_sender_5>);
  static_assert(ex::sender_in<awaitable_sender_5, ex::empty_env>);

  static_assert(ex::__awaiter<awaiter>);
  static_assert(!ex::__awaitable<awaitable_sender_5>);
  static_assert(ex::__awaitable<awaitable_sender_5, promise<awaiter>>);
  static_assert(ex::__awaitable<awaitable_sender_5, ex::no_env_promise>);
  static_assert(ex::__awaitable<awaitable_sender_5, ex::__env_promise<ex::empty_env>>);

  static_assert(
    !ex::__get_completion_signatures::__with_member_alias<awaitable_sender_5, ex::empty_env>);

  static_assert(std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_5>, Signatures>);
  static_assert(
    std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_5, ex::empty_env>, Signatures>);
}

template <typename Error, typename... Values>
auto signature_error_values(Error, Values...) -> ex::
  completion_signatures<ex::set_value_t(Values...), ex::set_error_t(Error), ex::set_stopped_t()>* {
  return {};
}

TEST_CASE("get completion_signatures for awaitables", "[sndtraits][awaitables]") {
  ::test_awaitable_sender1(signature_error_values(std::exception_ptr()), __coro::suspend_always{});
  ::test_awaitable_sender1(
    signature_error_values(
      std::exception_ptr(), ex::__await_result_t<awaitable_sender_1<awaiter>>()),
    awaiter{});

  ::test_awaitable_sender2();

  ::test_awaitable_sender3();

  ::test_awaitable_sender4(signature_error_values(
    std::exception_ptr(), ex::__await_result_t<awaitable_sender_4, promise<awaiter>>()));

  ::test_awaitable_sender5(signature_error_values(
    std::exception_ptr(), ex::__await_result_t<awaitable_sender_5, connect_awaitable_promise>()));
}

struct awaitable_env { };

template <typename Awaiter>
struct awaitable_with_get_env {
  Awaiter operator co_await();

  friend awaitable_env tag_invoke(ex::get_env_t, const awaitable_with_get_env&) noexcept {
    return {};
  }
};

TEST_CASE("get_env for awaitables", "[sndtraits][awaitables]") {
  check_env_type<ex::empty_env>(awaitable_sender_1<awaiter>{});
  check_env_type<ex::empty_env>(awaitable_sender_2{});
  check_env_type<ex::empty_env>(awaitable_sender_3{});
  check_env_type<awaitable_env>(awaitable_with_get_env<awaiter>{});
}

TEST_CASE("env_promise bug when CWG 2369 is fixed", "[sndtraits][awaitables]") {
  exec::static_thread_pool ctx{1};
  ex::scheduler auto sch = ctx.get_scheduler();
  ex::sender auto snd = ex::when_all(ex::then(ex::schedule(sch), []() {}));

  using _Awaitable = decltype(snd);
  using _Promise = ex::__env_promise<ex::empty_env>;
  static_assert(!ex::__awaitable<_Awaitable, _Promise>);
}

#endif // !STDEXEC_STD_NO_COROUTINES_
