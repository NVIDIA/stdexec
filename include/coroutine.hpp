/*
 * Copyright (c) NVIDIA
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
#pragma once

#include <concepts.hpp>

#if __has_include(<coroutine>)
#include <coroutine>
namespace coro = std;
#else
#include <experimental/coroutine>
namespace coro = std::experimental;
#endif

namespace std {
  // Defined some concepts and utilities for working with awaitables
  template <class P, class A>
  decltype(auto) __await_suspend(A& a) {
    if constexpr (!same_as<P, void>) {
      return a.await_suspend(coro::coroutine_handle<P>{});
    }
  }

  template <class, template <class...> class>
    constexpr bool __is_instance_of = false;
  template <class... As, template <class...> class T>
    constexpr bool __is_instance_of<T<As...>, T> = true;

  template <class T>
    concept __await_suspend_result =
      __one_of<T, void, bool> || __is_instance_of<T, coro::coroutine_handle>;

  template <class A, class P = void>
    concept __awaiter =
      requires (A& a) {
        a.await_ready() ? 1 : 0;
        {std::__await_suspend<P>(a)} -> __await_suspend_result;
        a.await_resume();
      };

  template <class T>
  decltype(auto) __get_awaiter(T&& t, void*) {
    if constexpr (requires { ((T&&) t).operator co_await(); }) {
      return ((T&&) t).operator co_await();
    } else if constexpr (requires { operator co_await((T&&) t); }) {
      return operator co_await((T&&) t);
    } else {
      return (T&&) t;
    }
  }

  template <class T, class P>
  decltype(auto) __get_awaiter(T&& t, P* p)
      requires requires { p->await_transform((T&&) t);} {
    if constexpr (requires { p->await_transform((T&&) t).operator co_await(); }) {
      return p->await_transform((T&&) t).operator co_await();
    } else if constexpr (requires { operator co_await(p->await_transform((T&&) t)); }) {
      return operator co_await(p->await_transform((T&&) t));
    } else {
      return p->await_transform((T&&) t);
    }
  }

  template <class A, class P = void>
    concept __awaitable =
      requires (A&& a, P* p) {
        {std::__get_awaiter((A&&) a, p)} -> __awaiter<P>;
      };

  template <class T>
    T& __as_lvalue(T&&);

  template <class A, class P = void>
      requires __awaitable<A, P>
    using __await_result_t = decltype(std::__as_lvalue(
        std::__get_awaiter(declval<A>(), (P*) nullptr)).await_resume());
}
