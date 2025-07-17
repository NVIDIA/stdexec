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

#include <test_common/type_helpers.hpp>

#if !STDEXEC_STD_NO_COROUTINES()

namespace ex = stdexec;

namespace {

  template <class Sender>
  concept sender_with_env = ex::sender<Sender> && requires(const Sender& s) { ex::get_env(s); };

  template <typename Awaiter>
  struct promise {
    auto get_return_object() -> __coro::coroutine_handle<promise> {
      return {__coro::coroutine_handle<promise>::from_promise(*this)};
    }

    auto initial_suspend() noexcept -> __coro::suspend_always {
      return {};
    }

    auto final_suspend() noexcept -> __coro::suspend_always {
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
    auto await_ready() -> bool {
      return true;
    }

    auto await_suspend(__coro::coroutine_handle<>) -> bool {
      return false;
    }

    auto await_resume() -> bool {
      return false;
    }
  };

  struct invalid_awaiter {
    auto await_ready() -> bool;
    auto await_suspend(__coro::coroutine_handle<>) -> bool;
    //void await_resume();
  };

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wundefined-internal")

  template <typename Awaiter>
  struct awaitable_sender_1 {
    auto operator co_await() -> Awaiter {
      return {};
    }
  };

  struct awaitable_sender_2 {
    using promise_type = promise<__coro::suspend_always>;

   private:
    friend auto operator co_await(awaitable_sender_2) -> invalid_awaiter {
      return {};
    }
  };

  struct awaitable_sender_3 {
    using promise_type = promise<awaiter>;

   private:
    friend auto operator co_await(awaitable_sender_3) -> invalid_awaiter {
      return {};
    }
  };

  STDEXEC_PRAGMA_POP()

  struct awaitable_sender_4 {
    using promise_type = promise<__coro::suspend_always>;

   private:
    template <class Promise>
    friend auto tag_invoke(ex::as_awaitable_t, awaitable_sender_4, Promise&) -> awaiter {
      return {};
    }
  };

  struct awaitable_sender_5 {
   private:
    template <class Promise>
    friend auto tag_invoke(ex::as_awaitable_t, awaitable_sender_5, Promise&) -> awaiter {
      return {};
    }
  };

  template <typename Signatures, typename Awaiter>
  void test_awaitable_sender1(Signatures*, Awaiter&&) {
    static_assert(ex::sender<awaitable_sender_1<Awaiter>>);
    static_assert(sender_with_env<awaitable_sender_1<Awaiter>>);
    static_assert(ex::__awaitable<awaitable_sender_1<Awaiter>>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_1<Awaiter>>);
    static_assert(
      std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_1<Awaiter>>, Signatures>);
  }

  void test_awaitable_sender2() {
    static_assert(!ex::sender<awaitable_sender_2>);
    static_assert(!sender_with_env<awaitable_sender_2>);
    static_assert(!ex::sender_in<awaitable_sender_2, ex::env<>>);

    static_assert(!ex::__awaitable<awaitable_sender_2>);
    static_assert(ex::__awaitable<awaitable_sender_2, promise<__coro::suspend_always>>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_2>);
  }

  void test_awaitable_sender3() {
    static_assert(!ex::sender<awaitable_sender_3>);
    static_assert(!sender_with_env<awaitable_sender_3>);
    static_assert(!ex::sender_in<awaitable_sender_3, ex::env<>>);

    static_assert(ex::__awaiter<awaiter>);
    static_assert(!ex::__awaitable<awaitable_sender_3>);
    static_assert(ex::__awaitable<awaitable_sender_3, promise<awaiter>>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_3>);
  }

  template <class Signatures>
  void test_awaitable_sender4(Signatures*) {
    static_assert(ex::sender<awaitable_sender_4>);
    static_assert(sender_with_env<awaitable_sender_4>);
    static_assert(ex::sender_in<awaitable_sender_4, ex::env<>>);

    static_assert(ex::__awaiter<awaiter>);
    static_assert(!ex::__awaitable<awaitable_sender_4>);
    static_assert(ex::__awaitable<awaitable_sender_4, promise<awaiter>>);
    static_assert(ex::__awaitable<awaitable_sender_4, ex::__env::__promise<ex::env<>>>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_4>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_4>);

    static_assert(
      std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_4, ex::env<>>, Signatures>);
  }

  struct connect_awaitable_promise : ex::with_awaitable_senders<connect_awaitable_promise> { };

  template <class Signatures>
  void test_awaitable_sender5(Signatures*) {
    static_assert(ex::sender<awaitable_sender_5>);
    static_assert(sender_with_env<awaitable_sender_5>);
    static_assert(!ex::sender_in<awaitable_sender_5>);
    static_assert(ex::sender_in<awaitable_sender_5, ex::env<>>);

    static_assert(ex::__awaiter<awaiter>);
    static_assert(!ex::__awaitable<awaitable_sender_5>);
    static_assert(ex::__awaitable<awaitable_sender_5, promise<awaiter>>);
    static_assert(ex::__awaitable<awaitable_sender_5, ex::__env::__promise<ex::env<>>>);

    static_assert(!ex::__sigs::__with_member_alias<awaitable_sender_5>);

    static_assert(
      std::is_same_v<ex::completion_signatures_of_t<awaitable_sender_5, ex::env<>>, Signatures>);
  }

  template <typename Error, typename... Values>
  auto signature_error_values(Error, Values...) -> ex::completion_signatures<
    ex::set_value_t(Values...),
    ex::set_error_t(Error),
    ex::set_stopped_t()
  >* {
    return {};
  }

  TEST_CASE("get completion_signatures for awaitables", "[sndtraits][awaitables]") {
    ::test_awaitable_sender1(
      signature_error_values(std::exception_ptr()), __coro::suspend_always{});
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
    auto operator co_await() -> Awaiter;

    [[nodiscard]]
    auto get_env() const noexcept -> awaitable_env {
      return {};
    }
  };

  TEST_CASE("get_env for awaitables", "[sndtraits][awaitables]") {
    check_env_type<ex::env<>>(awaitable_sender_1<awaiter>{});
    check_env_type<ex::env<>>(awaitable_sender_2{});
    check_env_type<ex::env<>>(awaitable_sender_3{});
    check_env_type<awaitable_env>(awaitable_with_get_env<awaiter>{});
  }

  TEST_CASE("env_promise bug when CWG 2369 is fixed", "[sndtraits][awaitables]") {
    exec::static_thread_pool ctx{1};
    ex::scheduler auto sch = ctx.get_scheduler();
    ex::sender auto snd = ex::when_all(ex::then(ex::schedule(sch), []() { }));

    using _Awaitable = decltype(snd);
    using _Promise = ex::__env::__promise<ex::env<>>;
    static_assert(!ex::__awaitable<_Awaitable, _Promise>);
  }
} // namespace

#endif // STDEXEC_STD_NO_COROUTINES()
