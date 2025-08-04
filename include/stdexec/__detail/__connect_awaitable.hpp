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

#include "__awaitable.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"

#include <exception>
#include <utility>

namespace stdexec {
#if !STDEXEC_STD_NO_COROUTINES()
  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  namespace __connect_awaitable_ {
    struct __promise_base {
      auto initial_suspend() noexcept -> __coro::suspend_always {
        return {};
      }

      [[noreturn]]
      auto final_suspend() noexcept -> __coro::suspend_always {
        std::terminate();
      }

      [[noreturn]]
      void unhandled_exception() noexcept {
        std::terminate();
      }

      [[noreturn]]
      void return_void() noexcept {
        std::terminate();
      }
    };

    struct __operation_base {
      __coro::coroutine_handle<> __coro_;

      explicit __operation_base(__coro::coroutine_handle<> __hcoro) noexcept
        : __coro_(__hcoro) {
      }

      __operation_base(__operation_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {})) {
      }

      ~__operation_base() {
        if (__coro_) {
#  if STDEXEC_MSVC()
          // MSVCBUG https://developercommunity.visualstudio.com/t/Double-destroy-of-a-local-in-coroutine-d/10456428

          // Reassign __coro_ before calling destroy to make the mutation
          // observable and to hopefully ensure that the compiler does not eliminate it.
          auto __coro = __coro_;
          __coro_ = {};
          __coro.destroy();
#  else
          __coro_.destroy();
#  endif
        }
      }

      void start() & noexcept {
        __coro_.resume();
      }
    };

    template <class _ReceiverId>
    struct __promise;

    template <class _ReceiverId>
    struct __operation {
      struct __t : __operation_base {
        using promise_type = stdexec::__t<__promise<_ReceiverId>>;
        using __operation_base::__operation_base;
      };
    };

    template <class _ReceiverId>
    struct __promise {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t
        : __promise_base
        , __env::__with_await_transform<__t> {
        using __id = __promise;

#  if STDEXEC_EDG()
        __t(auto&&, _Receiver&& __rcvr) noexcept
          : __rcvr_(__rcvr) {
        }
#  else
        explicit __t(auto&, _Receiver& __rcvr) noexcept
          : __rcvr_(__rcvr) {
        }
#  endif

        auto unhandled_stopped() noexcept -> __coro::coroutine_handle<> {
          stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr_));
          // Returning noop_coroutine here causes the __connect_awaitable
          // coroutine to never resume past the point where it co_await's
          // the awaitable.
          return __coro::noop_coroutine();
        }

        auto get_return_object() noexcept -> stdexec::__t<__operation<_ReceiverId>> {
          return stdexec::__t<__operation<_ReceiverId>>{
            __coro::coroutine_handle<__t>::from_promise(*this)};
        }

        // Pass through the get_env receiver query
        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__rcvr_);
        }

        _Receiver& __rcvr_;
      };
    };

    template <receiver _Receiver>
    using __promise_t = __t<__promise<__id<_Receiver>>>;

    template <receiver _Receiver>
    using __operation_t = __t<__operation<__id<_Receiver>>>;

    struct __connect_awaitable_t {
     private:
      template <class _Fun, class... _Ts>
      static auto __co_call(_Fun __fun, _Ts&&... __as) noexcept {
        auto __fn = [&, __fun]() noexcept {
          __fun(static_cast<_Ts&&>(__as)...);
        };

        struct __awaiter {
          decltype(__fn) __fn_;

          static constexpr auto await_ready() noexcept -> bool {
            return false;
          }

          void await_suspend(__coro::coroutine_handle<>) noexcept {
            __fn_();
          }

          [[noreturn]]
          void await_resume() noexcept {
            std::terminate();
          }
        };

        return __awaiter{__fn};
      }

      template <class _Awaitable, class _Receiver>
#  if STDEXEC_GCC() && (STDEXEC_GCC_VERSION >= 12'00)
      __attribute__((__used__))
#  endif
      static auto __co_impl(_Awaitable __awaitable, _Receiver __rcvr) -> __operation_t<_Receiver> {
        using __result_t = __await_result_t<_Awaitable, __promise_t<_Receiver>>;
        std::exception_ptr __eptr;
        STDEXEC_TRY {
          if constexpr (same_as<__result_t, void>)
            co_await (
              co_await static_cast<_Awaitable&&>(__awaitable),
              __co_call(set_value, static_cast<_Receiver&&>(__rcvr)));
          else
            co_await __co_call(
              set_value,
              static_cast<_Receiver&&>(__rcvr),
              co_await static_cast<_Awaitable&&>(__awaitable));
        }
        STDEXEC_CATCH_ALL {
          __eptr = std::current_exception();
        }
        co_await __co_call(
          set_error, static_cast<_Receiver&&>(__rcvr), static_cast<std::exception_ptr&&>(__eptr));
      }

      template <receiver _Receiver, class _Awaitable>
      using __completions_t =
        completion_signatures<
          __minvoke< // set_value_t() or set_value_t(T)
            __mremove<void, __qf<set_value_t>>,
            __await_result_t<_Awaitable, __promise_t<_Receiver>>>,
          set_error_t(std::exception_ptr),
          set_stopped_t()>;

     public:
      template <class _Receiver, __awaitable<__promise_t<_Receiver>> _Awaitable>
        requires receiver_of<_Receiver, __completions_t<_Receiver, _Awaitable>>
      auto
        operator()(_Awaitable&& __awaitable, _Receiver __rcvr) const -> __operation_t<_Receiver> {
        return __co_impl(static_cast<_Awaitable&&>(__awaitable), static_cast<_Receiver&&>(__rcvr));
      }
    };
  } // namespace __connect_awaitable_

  using __connect_awaitable_::__connect_awaitable_t;
#else
  struct __connect_awaitable_t { };
#endif
  inline constexpr __connect_awaitable_t __connect_awaitable{};
} // namespace stdexec
