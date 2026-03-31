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

#include "../coroutine.hpp"
#include "__awaitable.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__env.hpp"
#include "__optional.hpp"
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
    static constexpr std::size_t __storage_size  = 5 * sizeof(void*);
    static constexpr std::size_t __storage_align = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    // clang-format off
    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise)
    {
      static_cast<_Tp&&>(__t).as_awaitable(__promise);
    };
    // clang-format on

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

      template <__has_as_awaitable_member<_Promise&> _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto await_transform(_Ty&& __value)                                //
        noexcept(noexcept(__declval<_Ty>().as_awaitable(__declval<_Promise&>())))  //
        -> decltype(auto)
      {
        return static_cast<_Ty&&>(__value).as_awaitable(static_cast<_Promise&>(*this));
      }

     private:
      friend _Promise;
      __with_await_transform() = default;
    };

    struct __final_awaiter
    {
      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      template <class _Promise>
      static constexpr void await_suspend(__std::coroutine_handle<_Promise> __h) noexcept
      {
        STDEXEC_TRY
        {
          __h.promise().__opstate_.__on_resume();
        }
        STDEXEC_CATCH_ALL
        {
          __std::unreachable();
        }
      }

      [[noreturn]]
      static void await_resume() noexcept
      {
        __std::unreachable();
      }
    };

    template <class _Awaitable, class _Receiver>
    struct __opstate;

    template <class _Awaitable, class _Receiver>
    struct __promise : __with_await_transform<__promise<_Awaitable, _Receiver>>
    {
      using __opstate_t = __opstate<_Awaitable, _Receiver>;

      constexpr explicit(!STDEXEC_EDG()) __promise(__opstate_t& __opstate) noexcept
        : __opstate_(__opstate)
      {}

      ~__promise()
      {
        // never invoked
        __std::unreachable();
      }

      static constexpr auto
      operator new([[maybe_unused]] std::size_t __bytes, __opstate_t& __opstate) noexcept -> void*
      {
        // the first implementation of storing the coroutine frame inline in __opstate using the
        // technique in this file is due to Lewis Baker <lewissbaker@gmail.com>, and was first
        // shared at https://godbolt.org/z/zGG9fsPrz
        STDEXEC_ASSERT(__bytes == __storage_size);
        return __opstate.__storage_;
      }

      static constexpr void operator delete(void*, std::size_t) noexcept
      {
        // never invoked
        __std::unreachable();
      }

      constexpr auto get_return_object() noexcept -> __std::coroutine_handle<__promise>
      {
        STDEXEC_TRY
        {
          return __std::coroutine_handle<__promise>::from_promise(*this);
        }
        STDEXEC_CATCH_ALL
        {
          __std::unreachable();
        }
      }

      [[noreturn]]
      static auto
      get_return_object_on_allocation_failure() noexcept -> __std::coroutine_handle<__promise>
      {
        __std::unreachable();
      }

      static constexpr auto initial_suspend() noexcept -> __std::suspend_always
      {
        return {};
      }

      [[noreturn]]
      static void unhandled_exception() noexcept
      {
        __std::unreachable();
      }

      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
      {
        __opstate_.__on_stopped();
        // Returning noop_coroutine here causes the __connect_awaitable
        // coroutine to never resume past its initial_suspend point
        return __std::noop_coroutine();
      }

      static constexpr auto final_suspend() noexcept -> __final_awaiter
      {
        return __final_awaiter{};
      }

      static constexpr void return_void() noexcept
      {
        // no-op
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__opstate_.__rcvr_);
      }

      __opstate<_Awaitable, _Receiver>& __opstate_;
    };
  }  // namespace __connect_await
}

template <class _Awaitable, class _Receiver>
struct std::coroutine_traits<
  STDEXEC::__std::coroutine_handle<STDEXEC::__connect_await::__promise<_Awaitable, _Receiver>>,
  STDEXEC::__connect_await::__opstate<_Awaitable, _Receiver>&>
{
  using promise_type = STDEXEC::__connect_await::__promise<_Awaitable, _Receiver>;
};

namespace STDEXEC
{
  namespace __connect_await
  {
    template <class _Awaitable, class _Receiver>
    struct __opstate
    {
      constexpr explicit __opstate(_Awaitable&& __awaitable, _Receiver&& __rcvr)
        noexcept(__is_nothrow)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __coro_(__co_impl(*this))
        , __awaitable1_(static_cast<_Awaitable&&>(__awaitable))
        , __awaitable2_(
            __get_awaitable(static_cast<_Awaitable&&>(__awaitable1_), __coro_.promise()))
        , __awaiter_(__get_awaiter(static_cast<__awaitable_t&&>(__awaitable2_)))
      {}

      void start() & noexcept
      {
        STDEXEC_TRY
        {
          if (!__awaiter_.await_ready())
          {
            using __suspend_result_t = decltype(__awaiter_.await_suspend(__coro_));

            // suspended
            if constexpr (std::is_void_v<__suspend_result_t>)
            {
              // void-returning await_suspend means "always suspend"
              __awaiter_.await_suspend(__coro_);
              return;
            }
            else if constexpr (std::same_as<bool, __suspend_result_t>)
            {
              if (__awaiter_.await_suspend(__coro_))
              {
                // returning true from a bool-returning await_suspend means suspend
                return;
              }
              else
              {
                // returning false means immediately resume
              }
            }
            else
            {
              static_assert(__std::convertible_to<__suspend_result_t, __std::coroutine_handle<>>);
              auto __resume_target = __awaiter_.await_suspend(__coro_);
              STDEXEC_TRY
              {
                __resume_target.resume();
              }
              STDEXEC_CATCH_ALL
              {
                STDEXEC_ASSERT(false
                               && "about to deliberately commit UB in response to a misbehaving "
                                  "awaitable");
                __std::unreachable();
              }
              return;
            }
          }

          // immediate resumption
          __on_resume();
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!noexcept(__awaiter_.await_ready())
                        || !noexcept(__awaiter_.await_suspend(__coro_)))
          {
            STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
          }
        }
      }

     private:
      using __promise_t   = __promise<_Awaitable, _Receiver>;
      using __awaitable_t = __result_of<__get_awaitable, _Awaitable, __promise_t&>;
      using __awaiter_t   = __awaiter_of_t<__awaitable_t>;

      friend __promise_t;
      friend __final_awaiter;

      static constexpr bool __is_nothrow = __nothrow_move_constructible<_Awaitable>
                                        && __noexcept_of<__get_awaitable, _Awaitable, __promise_t&>
                                        && __noexcept_of<__get_awaiter, __awaitable_t>;

      static auto __co_impl(__opstate&) noexcept -> __std::coroutine_handle<__promise_t>
      {
        co_return;
      }

      constexpr void __on_resume() noexcept
      {
        STDEXEC_TRY
        {
          if constexpr (std::is_void_v<decltype(__awaiter_.await_resume())>)
          {
            __awaiter_.await_resume();
            STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
          }
          else
          {
            STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), __awaiter_.await_resume());
          }
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (!noexcept(__awaiter_.await_resume()))
          {
            STDEXEC::set_error(static_cast<_Receiver&&>(__rcvr_), std::current_exception());
          }
        }
      }

      constexpr void __on_stopped() noexcept
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
      }

      alignas(__storage_align) std::byte __storage_[__storage_size];
      _Receiver                            __rcvr_;
      __std::coroutine_handle<__promise_t> __coro_;
      _Awaitable                           __awaitable1_;
      __awaitable_t                        __awaitable2_;
      __awaiter_t                          __awaiter_;
    };
  }  // namespace __connect_await

  struct __connect_awaitable_t
  {
    template <class _Awaitable, class _Receiver>
    using __opstate_t = __connect_await::__opstate<_Awaitable, _Receiver>;

    template <class _Awaitable, class _Receiver>
    using __promise_t = __connect_await::__promise<_Awaitable, _Receiver>;

    template <class _Awaitable, class _Receiver>
      requires __awaitable<_Awaitable, __promise_t<_Awaitable, _Receiver>>
    auto operator()(_Awaitable&& __awaitable, _Receiver __rcvr) const noexcept(
      __nothrow_constructible_from<__opstate_t<_Awaitable, _Receiver>, _Awaitable, _Receiver>)
    {
      return __opstate_t<_Awaitable, _Receiver>(static_cast<_Awaitable&&>(__awaitable),
                                                static_cast<_Receiver&&>(__rcvr));
    }
  };

#else

  namespace __connect_await
  {
    template <class, class>
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
