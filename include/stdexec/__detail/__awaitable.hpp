/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__config.hpp"
#include "__concepts.hpp"
#include "__utility.hpp"

namespace stdexec {
#if !STDEXEC_STD_NO_COROUTINES()
  // Define some concepts and utilities for working with awaitables
  template <class _Tp>
  concept __await_suspend_result = __one_of<_Tp, void, bool>
                                || __is_instance_of<_Tp, __coro::coroutine_handle>;

  template <class _Awaiter, class... _Promise>
  concept __awaiter = requires(_Awaiter& __awaiter, __coro::coroutine_handle<_Promise...> __h) {
    __awaiter.await_ready() ? 1 : 0;
    { __awaiter.await_suspend(__h) } -> __await_suspend_result;
    __awaiter.await_resume();
  };

#  if STDEXEC_MSVC()
  // MSVCBUG https://developercommunity.visualstudio.com/t/operator-co_await-not-found-in-requires/10452721

  template <class _Awaitable>
  void __co_await_constraint(_Awaitable&& __awaitable)
    requires requires { operator co_await(static_cast<_Awaitable &&>(__awaitable)); };
#  endif

  template <class _Awaitable>
  auto __get_awaiter(_Awaitable&& __awaitable, __ignore = {}) -> decltype(auto) {
    if constexpr (requires { static_cast<_Awaitable &&>(__awaitable).operator co_await(); }) {
      return static_cast<_Awaitable&&>(__awaitable).operator co_await();
    } else if constexpr (requires {
#  if STDEXEC_MSVC()
                           __co_await_constraint(static_cast<_Awaitable &&>(__awaitable));
#  else
        operator co_await(static_cast<_Awaitable&&>(__awaitable));
#  endif
                         }) {
      return operator co_await(static_cast<_Awaitable&&>(__awaitable));
    } else {
      return static_cast<_Awaitable&&>(__awaitable);
    }
  }

  template <class _Awaitable, class _Promise>
  auto __get_awaiter(_Awaitable&& __awaitable, _Promise* __promise) -> decltype(auto)
    requires requires { __promise->await_transform(static_cast<_Awaitable &&>(__awaitable)); }
  {
    if constexpr (requires {
                    __promise->await_transform(static_cast<_Awaitable &&>(__awaitable))
                      .operator co_await();
                  }) {
      return __promise->await_transform(static_cast<_Awaitable&&>(__awaitable)).operator co_await();
    } else if constexpr (requires {
#  if STDEXEC_MSVC()
                           __co_await_constraint(
                             __promise->await_transform(static_cast<_Awaitable &&>(__awaitable)));
#  else
        operator co_await(__promise->await_transform(static_cast<_Awaitable&&>(__awaitable)));
#  endif
                         }) {
      return operator co_await(__promise->await_transform(static_cast<_Awaitable&&>(__awaitable)));
    } else {
      return __promise->await_transform(static_cast<_Awaitable&&>(__awaitable));
    }
  }

  template <class _Awaitable, class... _Promise>
  concept __awaitable = requires(_Awaitable&& __awaitable, _Promise*... __promise) {
    {
      stdexec::__get_awaiter(static_cast<_Awaitable &&>(__awaitable), __promise...)
    } -> __awaiter<_Promise...>;
  };

  template <class _Tp>
  auto __as_lvalue(_Tp&&) -> _Tp&;

  template <class _Awaitable, class... _Promise>
    requires __awaitable<_Awaitable, _Promise...>
  using __await_result_t = decltype(stdexec::__as_lvalue(
                                      stdexec::__get_awaiter(
                                        std::declval<_Awaitable>(),
                                        static_cast<_Promise*>(nullptr)...))
                                      .await_resume());

#else

  template <class _Awaitable, class... _Promise>
  concept __awaitable = false;

  template <class _Awaitable, class... _Promise>
    requires __awaitable<_Awaitable, _Promise...>
  using __await_result_t = void;

#endif
} // namespace stdexec
