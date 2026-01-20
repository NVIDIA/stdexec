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

#include "../coroutine.hpp" // IWYU pragma: keep for __std::coroutine_handle
#include "__as_awaitable.hpp"
#include "__concepts.hpp"

#include <exception>

namespace STDEXEC {
#if !STDEXEC_NO_STD_COROUTINES()
  template <class _Promise = void>
  class __coroutine_handle;

  template <>
  class __coroutine_handle<void> : __std::coroutine_handle<> {
   public:
    constexpr __coroutine_handle() = default;

    template <class _Promise>
    constexpr __coroutine_handle(__std::coroutine_handle<_Promise> __coro) noexcept
      : __std::coroutine_handle<>(__coro) {
      if constexpr (requires(_Promise& __promise) { __promise.unhandled_stopped(); }) {
        __stopped_callback_ = [](void* __address) noexcept -> __std::coroutine_handle<> {
          // This causes the rest of the coroutine (the part after the co_await
          // of the sender) to be skipped and invokes the calling coroutine's
          // stopped handler.
          return __std::coroutine_handle<_Promise>::from_address(__address)
            .promise()
            .unhandled_stopped();
        };
      }
      // If _Promise doesn't implement unhandled_stopped(), then if a "stopped" unwind
      // reaches this point, it's considered an unhandled exception and terminate()
      // is called.
    }

    [[nodiscard]]
    constexpr auto handle() const noexcept -> __std::coroutine_handle<> {
      return *this;
    }

    [[nodiscard]]
    constexpr auto unhandled_stopped() const noexcept -> __std::coroutine_handle<> {
      return __stopped_callback_(address());
    }

   private:
    using __stopped_callback_t = __std::coroutine_handle<> (*)(void*) noexcept;

    __stopped_callback_t __stopped_callback_ = [](void*) noexcept -> __std::coroutine_handle<> {
      std::terminate();
    };
  };

  template <class _Promise>
  class __coroutine_handle : public __coroutine_handle<> {
   public:
    constexpr __coroutine_handle() = default;

    constexpr __coroutine_handle(__std::coroutine_handle<_Promise> __coro) noexcept
      : __coroutine_handle<>{__coro} {
    }

    [[nodiscard]]
    static constexpr auto from_promise(_Promise& __promise) noexcept -> __coroutine_handle {
      return __coroutine_handle(__std::coroutine_handle<_Promise>::from_promise(__promise));
    }

    [[nodiscard]]
    constexpr auto promise() const noexcept -> _Promise& {
      return __std::coroutine_handle<_Promise>::from_address(address()).promise();
    }

    [[nodiscard]]
    constexpr auto handle() const noexcept -> __std::coroutine_handle<_Promise> {
      return __std::coroutine_handle<_Promise>::from_address(address());
    }

    [[nodiscard]]
    constexpr operator __std::coroutine_handle<_Promise>() const noexcept {
      return handle();
    }
  };

  struct __with_awaitable_senders_base {
    template <class _OtherPromise>
    constexpr void set_continuation(__std::coroutine_handle<_OtherPromise> __hcoro) noexcept {
      static_assert(!__same_as<_OtherPromise, void>);
      __continuation_ = __hcoro;
    }

    constexpr void set_continuation(__coroutine_handle<> __continuation) noexcept {
      __continuation_ = __continuation;
    }

    [[nodiscard]]
    constexpr auto continuation() const noexcept -> __coroutine_handle<> {
      return __continuation_;
    }

    [[nodiscard]]
    constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<> {
      return __continuation_.unhandled_stopped();
    }

   private:
    __coroutine_handle<> __continuation_{};
  };

  template <class _Promise>
  struct with_awaitable_senders : __with_awaitable_senders_base {
    template <class _Value>
    [[nodiscard]]
    constexpr auto
      await_transform(_Value&& __val) -> __call_result_t<as_awaitable_t, _Value, _Promise&> {
      static_assert(__std::derived_from<_Promise, with_awaitable_senders>);
      return as_awaitable(static_cast<_Value&&>(__val), static_cast<_Promise&>(*this));
    }
  };
#endif
} // namespace STDEXEC
