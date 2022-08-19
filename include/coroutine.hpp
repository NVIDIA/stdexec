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

#include <version>
#if __cpp_impl_coroutine >= 201902 && __cpp_lib_coroutine	>= 201902
#include <coroutine>
namespace __coro = std;
#elif defined(__cpp_coroutines) && __has_include(<experimental/coroutine>)
#include <experimental/coroutine>
namespace __coro = std::experimental;
#else
#error No coroutine support found
#endif

namespace _P2300 {
  // Defined some concepts and utilities for working with awaitables
  template <class _Promise, class _Awaiter>
  decltype(auto) __await_suspend(_Awaiter& __await) {
    if constexpr (!same_as<_Promise, void>) {
      return __await.await_suspend(__coro::coroutine_handle<_Promise>{});
    }
  }

  template <class _T>
    concept __await_suspend_result =
      __one_of<_T, void, bool> || __is_instance_of<_T, __coro::coroutine_handle>;

  template <class _Awaiter, class _Promise = void>
    concept __awaiter =
      requires (_Awaiter& __await) {
        __await.await_ready() ? 1 : 0;
        { (__await_suspend<_Promise>)(__await) } -> __await_suspend_result;
        __await.await_resume();
      };

  template <class _Awaitable>
  decltype(auto) __get_awaiter(_Awaitable&& __await, void*) {
    if constexpr (requires { ((_Awaitable&&) __await).operator co_await(); }) {
      return ((_Awaitable&&) __await).operator co_await();
    } else if constexpr (requires { operator co_await((_Awaitable&&) __await); }) {
      return operator co_await((_Awaitable&&) __await);
    } else {
      return (_Awaitable&&) __await;
    }
  }

  template <class _Awaitable, class _Promise>
  decltype(auto) __get_awaiter(_Awaitable&& __await, _Promise* __promise)
      requires requires { __promise->await_transform((_Awaitable&&) __await);} {
    if constexpr (requires { __promise->await_transform((_Awaitable&&) __await).operator co_await(); }) {
      return __promise->await_transform((_Awaitable&&) __await).operator co_await();
    } else if constexpr (requires { operator co_await(__promise->await_transform((_Awaitable&&) __await)); }) {
      return operator co_await(__promise->await_transform((_Awaitable&&) __await));
    } else {
      return __promise->await_transform((_Awaitable&&) __await);
    }
  }

  template <class _Awaitable, class _Promise = void>
    concept __awaitable =
      requires (_Awaitable&& __await, _Promise* __promise) {
        { (__get_awaiter)((_Awaitable&&) __await, __promise) } -> __awaiter<_Promise>;
      };

  template <class _T>
    _T& __as_lvalue(_T&&);

  template <class _Awaitable, class _Promise = void>
      requires __awaitable<_Awaitable, _Promise>
    using __await_result_t = decltype((__as_lvalue)(
        (__get_awaiter)(std::declval<_Awaitable>(), (_Promise*) nullptr)).await_resume());
}
