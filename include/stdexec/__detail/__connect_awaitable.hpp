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
#include "../functional.hpp"
#include "__awaitable.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__env.hpp"
#include "__manual_lifetime.hpp"
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

    static constexpr std::size_t __storage_size  = 256;
    static constexpr std::size_t __storage_align = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    // clang-format off
    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise) {
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
      [[deprecated("The use of tag_invoke to customize the behavior of await_transform is "
                   "deprecated. Please provide an as_awaitable member function instead.")]]
      STDEXEC_ATTRIBUTE(nodiscard, host, device) auto await_transform(_Ty&& __value)
        noexcept(__nothrow_tag_invocable<as_awaitable_t, _Ty, _Promise&>)
          -> __tag_invoke_result_t<as_awaitable_t, _Ty, _Promise&>
      {
        return __tag_invoke(as_awaitable,
                            static_cast<_Ty&&>(__value),
                            static_cast<_Promise&>(*this));
      }
    };

    struct __awaiter_base
    {
      static constexpr auto await_ready() noexcept -> bool
      {
        return false;
      }

      [[noreturn]]
      void await_resume() noexcept
      {
        __std::unreachable();
      }
    };

    // Turn a nothrow nullary-callable into an awaitable that simply invokes the callable
    // in await_suspend.
    template <class _Receiver, class... _Args>
    struct __set_value_awaiter;

    template <class _Receiver>
    struct __set_value_awaiter<_Receiver> : __awaiter_base
    {
      constexpr void await_suspend(__std::coroutine_handle<>) noexcept
      {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_));
        // This coroutine is never resumed; its work is done.
      }

      _Receiver& __rcvr_;
    };

    template <class _Receiver, class _Arg>
    struct __set_value_awaiter<_Receiver, _Arg> : __awaiter_base
    {
      constexpr void await_suspend(__std::coroutine_handle<>) noexcept
      {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<_Arg&&>(__arg_));
        // This coroutine is never resumed; its work is done.
      }

      _Receiver& __rcvr_;
      _Arg&&     __arg_;
    };

    struct __task_base
    {
      constexpr explicit __task_base(__std::coroutine_handle<> __coro) noexcept
        : __coro_(__coro)
      {}

      constexpr __task_base(__task_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {}))
      {}

      constexpr ~__task_base()
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

      __std::coroutine_handle<> __coro_;
    };

    template <class _Receiver>
    struct __promise;

    template <class _Receiver>
    struct __task : __task_base
    {
      using promise_type = __promise<_Receiver>;
      using __task_base::__task_base;
    };

    struct __promise_base
    {
      static constexpr auto initial_suspend() noexcept -> __std::suspend_always
      {
        return {};
      }

      void unhandled_exception() noexcept
      {
        __eptr_ = std::current_exception();
      }

      [[noreturn]]
      static void return_void() noexcept
      {
        __std::unreachable();
      }

      std::exception_ptr __eptr_;
    };

    template <class _Receiver>
    struct __opstate_base
    {
      _Receiver __rcvr_;
      alignas(__storage_align) std::byte __storage_[__storage_size];
    };

    template <class _Receiver>
    struct STDEXEC_ATTRIBUTE(empty_bases) __promise
      : __promise_base
      , __with_await_transform<__promise<_Receiver>>
    {
      constexpr explicit(!STDEXEC_EDG()) __promise(__opstate_base<_Receiver>& __opstate) noexcept
        : __opstate_(__opstate)
      {}

      static constexpr auto operator new([[maybe_unused]] std::size_t __bytes,
                                         __opstate_base<_Receiver>&   __opstate) noexcept -> void*
      {
        STDEXEC_ASSERT(__bytes <= __storage_size);
        return __opstate.__storage_;
      }

      static constexpr void operator delete([[maybe_unused]] void* __ptr) noexcept
      {
        // no-op
      }

      struct __set_error_awaiter : __awaiter_base
      {
        void await_suspend(__std::coroutine_handle<>) noexcept
        {
          STDEXEC::set_error(static_cast<_Receiver&&>(__promise_.__opstate_.__rcvr_),
                             std::move(__promise_.__eptr_));
          // This coroutine is never resumed; its work is done.
        }

        __promise& __promise_;
      };

      constexpr auto final_suspend() noexcept -> __set_error_awaiter
      {
        STDEXEC_ASSERT(__eptr_);
        return __set_error_awaiter{{}, *this};
      }

      constexpr auto unhandled_stopped() noexcept -> __std::coroutine_handle<>
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_.__rcvr_));
        // Returning noop_coroutine here causes the __connect_awaitable
        // coroutine to never resume past the point where it co_await's
        // the awaitable.
        return __std::noop_coroutine();
      }

      constexpr auto get_return_object() noexcept -> __task<_Receiver>
      {
        return __task<_Receiver>{__std::coroutine_handle<__promise>::from_promise(*this)};
      }

      [[noreturn]]
      static constexpr auto get_return_object_on_allocation_failure() noexcept -> __task<_Receiver>
      {
        __std::unreachable();
      }

      constexpr auto get_env() const noexcept -> env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__opstate_.__rcvr_);
      }

      __opstate_base<_Receiver>& __opstate_;
    };

    template <class _Awaitable, class _Receiver>
    struct __opstate : __opstate_base<_Receiver>
    {
      explicit __opstate(_Awaitable&& __awaitable, _Receiver&& __rcvr)
        noexcept(__nothrow_move_constructible<_Awaitable>)
        : __opstate_base<_Receiver>{static_cast<_Receiver&&>(__rcvr)}
        , __awaitable1_(static_cast<_Awaitable&&>(__awaitable))
        , __task_(__opstate::__co_impl(*this))
      {
        auto __coro = STDEXEC::__coroutine_handle_cast<__promise<_Receiver>>(__task_.__coro_);

        __awaitable2_.__construct_from(STDEXEC::__get_awaitable,
                                       static_cast<_Awaitable&&>(__awaitable1_),
                                       __coro.promise());

        __awaiter_.__construct_from(STDEXEC::__get_awaiter,
                                    static_cast<__awaitable_t&&>(__awaitable2_.__get()));
      }

      STDEXEC_IMMOVABLE(__opstate);

      ~__opstate()
      {
        __awaiter_.__destroy();
        __awaitable2_.__destroy();
      }

      void start() & noexcept
      {
        __task_.__coro_.resume();
      }

     private:
      using __awaitable_t = __awaitable_of_t<_Awaitable, __promise<_Receiver>>;
      using __awaiter_t   = __awaiter_of_t<__awaitable_t>;

      static auto __co_impl(__opstate& __op) noexcept -> __task<_Receiver>
      {
        auto&& __awaiter = __op.__awaiter_.__get();
        using __result_t = decltype(__declval<__awaiter_t>().await_resume());

        if constexpr (__same_as<__result_t, void>)
        {
          using __set_value_t = __set_value_awaiter<_Receiver>;
          co_await (co_await static_cast<__awaiter_t&&>(__awaiter),
                    __set_value_t{{}, __op.__rcvr_});
        }
        else
        {
          using __set_value_t = __set_value_awaiter<_Receiver, __result_t>;
          co_await __set_value_t{{}, __op.__rcvr_, co_await static_cast<__awaiter_t&&>(__awaiter)};
        }
      }

      _Awaitable                       __awaitable1_;
      __manual_lifetime<__awaitable_t> __awaitable2_;
      __manual_lifetime<__awaiter_t>   __awaiter_;
      __task<_Receiver>                __task_;
    };

    STDEXEC_PRAGMA_OPTIMIZE_END()
  }  // namespace __connect_await

  struct __connect_awaitable_t
  {
    template <class _Receiver, __awaitable<__connect_await::__promise<_Receiver>> _Awaitable>
    auto operator()(_Awaitable&& __awaitable, _Receiver __rcvr) const
      noexcept(__nothrow_move_constructible<_Awaitable>)
    {
      using __result_t      = __await_result_t<_Awaitable, __connect_await::__promise<_Receiver>>;
      using __completions_t = completion_signatures<__single_value_sig_t<__result_t>,
                                                    set_error_t(std::exception_ptr),
                                                    set_stopped_t()>;
      static_assert(receiver_of<_Receiver, __completions_t>);
      return __connect_await::__opstate(static_cast<_Awaitable&&>(__awaitable),
                                        static_cast<_Receiver&&>(__rcvr));
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
