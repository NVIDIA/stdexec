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

#include "__execution_fwd.hpp"

#include "../coroutine.hpp"  // IWYU pragma: keep for __coroutine_handle
#include "__as_awaitable.hpp"
#include "__concepts.hpp"

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  template <class _Promise>
  struct with_awaitable_senders;

  namespace __detail
  {
    struct __with_awaitable_senders
    {
      template <class _OtherPromise>
      constexpr void set_continuation(__std::coroutine_handle<_OtherPromise> __hcoro) noexcept
      {
        static_assert(!__same_as<_OtherPromise, void>);
        __continuation_ = __hcoro;
      }

      constexpr void set_continuation(__coroutine_handle<> __continuation) noexcept
      {
        __continuation_ = __continuation;
      }

      [[nodiscard]]
      constexpr auto continuation() const noexcept -> __coroutine_handle<>
      {
        return __continuation_;
      }

      [[nodiscard]]
      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
      {
        return __continuation_.unhandled_stopped();
      }

     private:
      template <class>
      friend struct STDEXEC::with_awaitable_senders;

      __with_awaitable_senders() = default;

      __coroutine_handle<> __continuation_{};
    };
  }  // namespace __detail

  template <class _Promise>
  struct with_awaitable_senders : __detail::__with_awaitable_senders
  {
    template <class _Value>
    [[nodiscard]]
    constexpr auto await_transform(_Value&& __val)
      noexcept(__nothrow_callable<as_awaitable_t, _Value, _Promise&>)  //
      -> __call_result_t<as_awaitable_t, _Value, _Promise&>
    {
      static_assert(__std::derived_from<_Promise, with_awaitable_senders>);
      return as_awaitable(static_cast<_Value&&>(__val), static_cast<_Promise&>(*this));
    }

   private:
    friend _Promise;
    with_awaitable_senders() = default;
  };
#endif
}  // namespace STDEXEC
