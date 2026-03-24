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

#include "__concepts.hpp"
#include "__config.hpp"
#include "__meta.hpp"

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  // Define some concepts and utilities for working with awaitables
  template <class _Tp>
  concept __await_suspend_result = __one_of<_Tp, void, bool>
                                || __is_instance_of<_Tp, __std::coroutine_handle>;

  template <class _Awaiter, class... _Promise>
  concept __awaiter = requires(_Awaiter &__awaiter, __std::coroutine_handle<_Promise...> __h) {
    __awaiter.await_ready() ? 1 : 0;
    { __awaiter.await_suspend(__h) } -> __await_suspend_result;
    __awaiter.await_resume();
  };

  template <class _Awaitable, class _Promise>
  concept __has_await_transform = requires(_Awaitable &&__awaitable, _Promise &__promise) {
    __promise.await_transform(static_cast<_Awaitable &&>(__awaitable));
  };

  template <class _Awaitable>
  constexpr auto __get_awaitable(_Awaitable &&__awaitable, __ignore = {}) -> decltype(auto)
  {
    return static_cast<_Awaitable &&>(__awaitable);
  }

  template <class _Promise, __has_await_transform<_Promise> _Awaitable>
  constexpr auto __get_awaitable(_Awaitable &&__awaitable, _Promise &__promise) -> decltype(auto)
  {
    return __promise.await_transform(static_cast<_Awaitable &&>(__awaitable));
  }

  template <class _Awaitable>
  constexpr auto __get_awaiter(_Awaitable &&__awaitable) -> decltype(auto)
  {
    if constexpr (requires { __declval<_Awaitable>().operator co_await(); })
    {
      return static_cast<_Awaitable &&>(__awaitable).operator co_await();
    }
    else if constexpr (requires { operator co_await(__declval<_Awaitable>()); })
    {
      return operator co_await(static_cast<_Awaitable &&>(__awaitable));
    }
    else
    {
      return static_cast<_Awaitable &&>(__awaitable);
    }
  }

  template <class _Awaitable, class... _Promise>
  concept __awaitable = requires(_Awaitable &&__awaitable, _Promise &...__promise) {
    {
      STDEXEC::__get_awaiter(
        STDEXEC::__get_awaitable(static_cast<_Awaitable &&>(__awaitable), __promise...))
    } -> __awaiter<_Promise...>;
  };

  template <class _Tp>
  constexpr auto __as_lvalue(_Tp &&) -> _Tp &;

  template <class _Awaitable, class... _Promise>
    requires __awaitable<_Awaitable, _Promise...>
  using __await_result_t = decltype(STDEXEC::__as_lvalue(
                                      STDEXEC::__get_awaiter(
                                        STDEXEC::__get_awaitable(__declval<_Awaitable>(),
                                                                 __declval<_Promise &>()...)))
                                      .await_resume());

#else

  template <class _Awaitable, class... _Promise>
  concept __awaitable = false;

  template <class _Awaitable, class... _Promise>
    requires __awaitable<_Awaitable, _Promise...>
  using __await_result_t = void;

#endif
}  // namespace STDEXEC
