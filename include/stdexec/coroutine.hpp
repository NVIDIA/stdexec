/*
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
#pragma once

#include "concepts.hpp"

#include <version>
#if __cpp_impl_coroutine >= 201902 && __cpp_lib_coroutine >= 201902
#include <coroutine>
namespace __coro = std;
#elif defined(__cpp_coroutines) && __has_include(<experimental/coroutine>)
#include <experimental/coroutine>
namespace __coro = std::experimental;
#else
#define STDEXEC_STD_NO_COROUTINES_ 1
#endif

namespace stdexec {
#if !STDEXEC_STD_NO_COROUTINES_
  // Define some concepts and utilities for working with awaitables
  template <class _Tp>
  concept __await_suspend_result =
    __one_of<_Tp, void, bool> || __is_instance_of<_Tp, __coro::coroutine_handle>;

  template <class _Awaiter, class _Promise>
  concept __with_await_suspend =
    same_as<_Promise, void> || //
    requires(_Awaiter& __awaiter, __coro::coroutine_handle<_Promise> __h) {
      { __awaiter.await_suspend(__h) } -> __await_suspend_result;
    };

  template <class _Awaiter, class _Promise = void>
  concept __awaiter = //
    requires(_Awaiter& __awaiter) {
      __awaiter.await_ready() ? 1 : 0;
      __awaiter.await_resume();
    } && //
    __with_await_suspend<_Awaiter, _Promise>;

#if STDEXEC_MSVC()
  // MSVCBUG https://developercommunity.visualstudio.com/t/operator-co_await-not-found-in-requires/10452721

  template <class _Awaitable>
  void __co_await_constraint(_Awaitable&& __awaitable)
    requires requires { operator co_await((_Awaitable&&) __awaitable); };
#endif

  template <class _Awaitable>
  decltype(auto) __get_awaiter(_Awaitable&& __awaitable, void*) {
    if constexpr (requires { ((_Awaitable&&) __awaitable).operator co_await(); }) {
      return ((_Awaitable&&) __awaitable).operator co_await();
    } else if constexpr (requires {
#if STDEXEC_MSVC()
                           __co_await_constraint((_Awaitable&&) __awaitable);
#else
        operator co_await((_Awaitable&&) __awaitable);
#endif
                         }) {
      return operator co_await((_Awaitable&&) __awaitable);
    } else {
      return (_Awaitable&&) __awaitable;
    }
  }

  template <class _Awaitable, class _Promise>
  decltype(auto) __get_awaiter(_Awaitable&& __awaitable, _Promise* __promise)
    requires requires { __promise->await_transform((_Awaitable&&) __awaitable); }
  {
    if constexpr (
      requires { __promise->await_transform((_Awaitable&&) __awaitable).operator co_await(); }) {
      return __promise->await_transform((_Awaitable&&) __awaitable).operator co_await();
    } else if constexpr (
      requires {
#if STDEXEC_MSVC()
        __co_await_constraint(__promise->await_transform((_Awaitable&&) __awaitable));
#else
        operator co_await(__promise->await_transform((_Awaitable&&) __awaitable));
#endif
      }) {
      return operator co_await(__promise->await_transform((_Awaitable&&) __awaitable));
    } else {
      return __promise->await_transform((_Awaitable&&) __awaitable);
    }
  }

  template <class _Awaitable, class _Promise = void>
  concept __awaitable = //
    requires(_Awaitable&& __awaitable, _Promise* __promise) {
      { stdexec::__get_awaiter((_Awaitable&&) __awaitable, __promise) } -> __awaiter<_Promise>;
    };

  template <class _Tp>
  _Tp& __as_lvalue(_Tp&&);

  template <class _Awaitable, class _Promise = void>
    requires __awaitable<_Awaitable, _Promise>
  using __await_result_t =
    decltype(stdexec::__as_lvalue(
               stdexec::__get_awaiter(std::declval<_Awaitable>(), (_Promise*) nullptr))
               .await_resume());

#else

  template <class _Awaitable, class _Promise = void>
  concept __awaitable = false;

  template <class _Awaitable, class _Promise = void>
    requires __awaitable<_Awaitable, _Promise>
  using __await_result_t = void;

#endif
} // namespace stdexec
