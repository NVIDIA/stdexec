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
#include "__concepts.hpp"
#include "__config.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"

#include <exception>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wsubobject-linkage")

namespace STDEXEC
{
#if !STDEXEC_NO_STDCPP_COROUTINES()
  /////////////////////////////////////////////////////////////////////////////
  // __connect_await
  namespace __connect_await
  {
    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise) {
      static_cast<_Tp&&>(__t).as_awaitable(__promise);
    };

    // A partial duplicate of with_awaitable_senders to avoid circular type dependencies
    template <class _Promise>
    struct __with_await_transform
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto await_transform(_Ty&& __value) noexcept -> _Ty&&
      {
        return static_cast<_Ty&&>(__value);
      }

      template <class _Ty>
        requires __has_as_awaitable_member<_Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(noexcept(__declval<_Ty>().as_awaitable(__declval<_Promise&>())))
          -> decltype(__declval<_Ty>().as_awaitable(__declval<_Promise&>()))
      {
        return static_cast<_Ty&&>(__value).as_awaitable(static_cast<_Promise&>(*this));
      }

      template <class _Ty>
        requires __has_as_awaitable_member<_Ty, _Promise&>
              || __tag_invocable<as_awaitable_t, _Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(__nothrow_tag_invocable<as_awaitable_t, _Ty, _Promise&>)
          -> __tag_invoke_result_t<as_awaitable_t, _Ty, _Promise&>
      {
        return __tag_invoke(as_awaitable,
                            static_cast<_Ty&&>(__value),
                            static_cast<_Promise&>(*this));
      }
    };

    struct __promise_base
    {
      constexpr auto initial_suspend() noexcept -> __std::suspend_always
      {
        return {};
      }

      [[noreturn]]
      auto final_suspend() noexcept -> __std::suspend_always
      {
        std::terminate();
      }

      [[noreturn]]
      void unhandled_exception() noexcept
      {
        std::terminate();
      }

      [[noreturn]]
      void return_void() noexcept
      {
        std::terminate();
      }
    };

    struct __operation_base
    {
      __std::coroutine_handle<> __coro_;

      constexpr explicit __operation_base(__std::coroutine_handle<> __hcoro) noexcept
        : __coro_(__hcoro)
      {}

      constexpr __operation_base(__operation_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {}))
      {}

      constexpr ~__operation_base()
      {
        if (__coro_)
        {
#  if STDEXEC_MSVC()
          // MSVCBUG https://developercommunity.visualstudio.com/t/Double-destroy-of-a-local-in-coroutine-d/10456428

          // Reassign __coro_ before calling destroy to make the mutation
          // observable and to hopefully ensure that the compiler does not eliminate it.
          auto __coro = __coro_;
          __coro_     = {};
          __coro.destroy();
#  else
          __coro_.destroy();
#  endif
        }
      }

      void start() & noexcept
      {
        __coro_.resume();
      }
    };

    template <class _Receiver>
    struct __promise;

    template <class _Receiver>
    struct __operation : __operation_base
    {
      using promise_type = __promise<_Receiver>;
      using __operation_base::__operation_base;
    };

    template <class _Receiver>
    struct __promise
      : __promise_base
      , __with_await_transform<__promise<_Receiver>>
    {
#  if STDEXEC_EDG()
      constexpr __promise(auto&&, _Receiver&& __rcvr) noexcept
        : __rcvr_(__rcvr)
      {}
#  else
      constexpr explicit __promise(auto&, _Receiver& __rcvr) noexcept
        : __rcvr_(__rcvr)
      {}
#  endif

      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
        // Returning noop_coroutine here causes the __connect_awaitable
        // coroutine to never resume past the point where it co_await's
        // the awaitable.
        return __std::noop_coroutine();
      }

      constexpr auto get_return_object() noexcept -> __operation<_Receiver>
      {
        return __operation<_Receiver>{__std::coroutine_handle<__promise>::from_promise(*this)};
      }

      // Pass through the get_env receiver query
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__rcvr_);
      }

      _Receiver& __rcvr_;
    };
  }  // namespace __connect_await

  struct __connect_awaitable_t
  {
   private:
    template <class _Fun, class... _Ts>
    static constexpr auto __co_call(_Fun __fun, _Ts&&... __as) noexcept
    {
      auto __fn = [&, __fun]() noexcept
      {
        __fun(static_cast<_Ts&&>(__as)...);
      };

      struct __awaiter
      {
        decltype(__fn) __fn_;

        static constexpr auto await_ready() noexcept -> bool
        {
          return false;
        }

        constexpr void await_suspend(__std::coroutine_handle<>) noexcept
        {
          __fn_();
        }

        [[noreturn]]
        void await_resume() noexcept
        {
          std::terminate();
        }
      };

      return __awaiter{__fn};
    }

    template <class _Awaitable, class _Receiver>
#  if STDEXEC_GCC() && (STDEXEC_GCC_VERSION >= 12'00)
    __attribute__((__used__))
#  endif
    static auto
    __co_impl(_Awaitable __awaitable, _Receiver __rcvr) -> __connect_await::__operation<_Receiver>
    {
      using __result_t = __await_result_t<_Awaitable, __connect_await::__promise<_Receiver>>;
      std::exception_ptr __eptr;
      STDEXEC_TRY
      {
        if constexpr (__std::same_as<__result_t, void>)
          co_await (co_await static_cast<_Awaitable&&>(__awaitable),
                    __co_call(set_value, static_cast<_Receiver&&>(__rcvr)));
        else
          co_await __co_call(set_value,
                             static_cast<_Receiver&&>(__rcvr),
                             co_await static_cast<_Awaitable&&>(__awaitable));
      }
      STDEXEC_CATCH_ALL
      {
        __eptr = std::current_exception();
      }
      co_await __co_call(set_error,
                         static_cast<_Receiver&&>(__rcvr),
                         static_cast<std::exception_ptr&&>(__eptr));
    }

    template <receiver _Receiver, class _Awaitable>
    using __completions_t =
      completion_signatures<__minvoke<  // set_value_t() or set_value_t(T)
                              __mremove<void, __qf<set_value_t>>,
                              __await_result_t<_Awaitable, __connect_await::__promise<_Receiver>>>,
                            set_error_t(std::exception_ptr),
                            set_stopped_t()>;

   public:
    template <class _Receiver, __awaitable<__connect_await::__promise<_Receiver>> _Awaitable>
      requires receiver_of<_Receiver, __completions_t<_Receiver, _Awaitable>>
    auto operator()(_Awaitable&& __awaitable, _Receiver __rcvr) const
      -> __connect_await::__operation<_Receiver>
    {
      return __co_impl(static_cast<_Awaitable&&>(__awaitable), static_cast<_Receiver&&>(__rcvr));
    }
  };

#else
  namespace __connect_await
  {
    template <class>
    struct __promise
    {};

    template <class>
    struct __with_await_transform
    {};
  }  // namespace __connect_await

  struct __connect_awaitable_t
  {};
#endif
  inline constexpr __connect_awaitable_t __connect_awaitable{};
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
