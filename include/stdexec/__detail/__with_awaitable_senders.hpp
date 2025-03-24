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

#include "__as_awaitable.hpp"
#include "__concepts.hpp"

#include <exception>

namespace stdexec {
#if !STDEXEC_STD_NO_COROUTINES()
  namespace __was {
    template <class _Promise = void>
    class __continuation_handle;

    template <>
    class __continuation_handle<void> {
     public:
      __continuation_handle() = default;

      template <class _Promise>
      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __coro_(__coro) {
        if constexpr (requires(_Promise& __promise) { __promise.unhandled_stopped(); }) {
          __stopped_callback_ = [](void* __address) noexcept -> __coro::coroutine_handle<> {
            // This causes the rest of the coroutine (the part after the co_await
            // of the sender) to be skipped and invokes the calling coroutine's
            // stopped handler.
            return __coro::coroutine_handle<_Promise>::from_address(__address)
              .promise()
              .unhandled_stopped();
          };
        }
        // If _Promise doesn't implement unhandled_stopped(), then if a "stopped" unwind
        // reaches this point, it's considered an unhandled exception and terminate()
        // is called.
      }

      [[nodiscard]]
      auto handle() const noexcept -> __coro::coroutine_handle<> {
        return __coro_;
      }

      [[nodiscard]]
      auto unhandled_stopped() const noexcept -> __coro::coroutine_handle<> {
        return __stopped_callback_(__coro_.address());
      }

     private:
      using __stopped_callback_t = __coro::coroutine_handle<> (*)(void*) noexcept;

      __coro::coroutine_handle<> __coro_{};
      __stopped_callback_t __stopped_callback_ = [](void*) noexcept -> __coro::coroutine_handle<> {
        std::terminate();
      };
    };

    template <class _Promise>
    class __continuation_handle {
     public:
      __continuation_handle() = default;

      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __continuation_{__coro} {
      }

      auto handle() const noexcept -> __coro::coroutine_handle<_Promise> {
        return __coro::coroutine_handle<_Promise>::from_address(__continuation_.handle().address());
      }

      [[nodiscard]]
      auto unhandled_stopped() const noexcept -> __coro::coroutine_handle<> {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    struct __with_awaitable_senders_base {
      template <class _OtherPromise>
      void set_continuation(__coro::coroutine_handle<_OtherPromise> __hcoro) noexcept {
        static_assert(!__same_as<_OtherPromise, void>);
        __continuation_ = __hcoro;
      }

      void set_continuation(__continuation_handle<> __continuation) noexcept {
        __continuation_ = __continuation;
      }

      [[nodiscard]]
      auto continuation() const noexcept -> __continuation_handle<> {
        return __continuation_;
      }

      auto unhandled_stopped() noexcept -> __coro::coroutine_handle<> {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    template <class _Promise>
    struct with_awaitable_senders : __with_awaitable_senders_base {
      template <class _Value>
      auto await_transform(_Value&& __val) -> __call_result_t<as_awaitable_t, _Value, _Promise&> {
        static_assert(derived_from<_Promise, with_awaitable_senders>);
        return as_awaitable(static_cast<_Value&&>(__val), static_cast<_Promise&>(*this));
      }
    };
  } // namespace __was

  using __was::with_awaitable_senders;
  using __was::__continuation_handle;
#endif
} // namespace stdexec
