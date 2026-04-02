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
    STDEXEC_PRAGMA_OPTIMIZE_BEGIN()

#  if STDEXEC_MSVC()
    static constexpr std::size_t __storage_size = 256;
#  else
    static constexpr std::size_t __storage_size = 8 * sizeof(void*);
#  endif
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

    struct __awaiter_base
    {
      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      [[noreturn]]
      inline void await_resume() noexcept
      {
        __std::unreachable();
      }
    };

    inline void __destroy_coro(__std::coroutine_handle<> __coro) noexcept
    {
#  if STDEXEC_MSVC()
      // MSVCBUG https://developercommunity.visualstudio.com/t/Double-destroy-of-a-local-in-coroutine-d/10456428
      // Reassign __coro before calling destroy to make the mutation
      // observable and to hopefully ensure that the compiler does not eliminate it.
      std::exchange(__coro, {}).destroy();
#  else
      __coro.destroy();
#  endif
    }

    template <class _Awaitable, class _Receiver>
    struct __opstate;

    template <class _Awaitable, class _Receiver>
    struct __promise : __with_await_transform<__promise<_Awaitable, _Receiver>>
    {
      using __opstate_t = __opstate<_Awaitable, _Receiver>;

      struct __task : __immovable
      {
        using promise_type = __promise;

        ~__task()
        {
          __connect_await::__destroy_coro(__coro_);
        }

        __std::coroutine_handle<__promise> __coro_{};
      };

      struct __final_awaiter : __awaiter_base
      {
        void await_suspend(__std::coroutine_handle<>) noexcept
        {
          using __awaitable_t = __result_of<__get_awaitable, _Awaitable, __promise&>;
          using __awaiter_t   = __awaiter_of_t<__awaitable_t>;
          using __result_t    = decltype(__declval<__awaiter_t>().await_resume());

          if (__opstate_.__eptr_)
          {
            STDEXEC::set_error(static_cast<_Receiver&&>(__opstate_.__rcvr_),
                               std::move(__opstate_.__eptr_));
          }
          else if constexpr (__same_as<__result_t, void>)
          {
            STDEXEC_ASSERT(__opstate_.__result_.has_value());
            STDEXEC::set_value(static_cast<_Receiver&&>(__opstate_.__rcvr_));
          }
          else
          {
            STDEXEC_ASSERT(__opstate_.__result_.has_value());
            STDEXEC::set_value(static_cast<_Receiver&&>(__opstate_.__rcvr_),
                               static_cast<__result_t&&>(*__opstate_.__result_));
          }
          // This coroutine is never resumed; its work is done.
        }

        __opstate<_Awaitable, _Receiver>& __opstate_;
      };

      constexpr explicit(!STDEXEC_EDG()) __promise(__opstate_t& __opstate) noexcept
        : __opstate_(__opstate)
      {}

      static constexpr auto
      operator new([[maybe_unused]] std::size_t __bytes, __opstate_t& __opstate) noexcept -> void*
      {
        STDEXEC_ASSERT(__bytes <= sizeof(__opstate.__storage_));
        return __opstate.__storage_;
      }

      static constexpr void operator delete([[maybe_unused]] void* __ptr) noexcept
      {
        // no-op
      }

      constexpr auto get_return_object() noexcept -> __task
      {
        return __task{{}, __std::coroutine_handle<__promise>::from_promise(*this)};
      }

      [[noreturn]]
      static auto get_return_object_on_allocation_failure() noexcept -> __task
      {
        __std::unreachable();
      }

      static constexpr auto initial_suspend() noexcept -> __std::suspend_always
      {
        return {};
      }

      void unhandled_exception() noexcept
      {
        __opstate_.__eptr_ = std::current_exception();
      }

      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_.__rcvr_));
        // Returning noop_coroutine here causes the __connect_awaitable
        // coroutine to never resume past the point where it co_await's
        // the awaitable.
        return __std::noop_coroutine();
      }

      constexpr auto final_suspend() noexcept -> __final_awaiter
      {
        return __final_awaiter{{}, __opstate_};
      }

      static void return_void() noexcept
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

    template <class _Awaitable, class _Receiver>
    struct __opstate
    {
      constexpr explicit __opstate(_Awaitable&& __awaitable, _Receiver&& __rcvr)
        noexcept(__is_nothrow)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __task_(__co_impl(*this))
        , __awaitable1_(static_cast<_Awaitable&&>(__awaitable))
        , __awaitable2_(
            __get_awaitable(static_cast<_Awaitable&&>(__awaitable1_), __task_.__coro_.promise()))
        , __awaiter_(__get_awaiter(static_cast<__awaitable_t&&>(__awaitable2_)))
      {}

      void start() & noexcept
      {
        __task_.__coro_.resume();
      }

     private:
      using __promise_t   = __promise<_Awaitable, _Receiver>;
      using __task_t      = __promise_t::__task;
      using __awaitable_t = __result_of<__get_awaitable, _Awaitable, __promise_t&>;
      using __awaiter_t   = __awaiter_of_t<__awaitable_t>;
      using __result_t    = decltype(__declval<__awaiter_t>().await_resume());

      friend __promise_t;

      static constexpr bool __is_nothrow = __nothrow_move_constructible<_Awaitable>
                                        && __noexcept_of<__get_awaitable, _Awaitable, __promise_t&>
                                        && __noexcept_of<__get_awaiter, __awaitable_t>;

      static constexpr std::size_t __storage_size = __connect_await::__storage_size
                                                  + sizeof(__manual_lifetime<__result_t>)
                                                  - __same_as<__result_t, void>;

      static auto __co_impl(__opstate& __op) noexcept -> __task_t
      {
        using __op_awaiter_t = decltype(__op.__awaiter_);
        if constexpr (__same_as<decltype(*__op.__result_), void>)
        {
          co_await static_cast<__op_awaiter_t&&>(__op.__awaiter_);
          __op.__result_.emplace();
        }
        else
        {
          __op.__result_.emplace(co_await static_cast<__op_awaiter_t&&>(__op.__awaiter_));
        }
      }

      alignas(__storage_align) std::byte __storage_[__storage_size];
      _Receiver              __rcvr_;
      __promise_t::__task    __task_;
      _Awaitable             __awaitable1_;
      __awaitable_t          __awaitable2_;
      __awaiter_t            __awaiter_;
      std::exception_ptr     __eptr_{};
      __optional<__result_t> __result_{};
    };

    STDEXEC_PRAGMA_OPTIMIZE_END()
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
