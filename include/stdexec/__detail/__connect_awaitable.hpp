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
    // four pointers' worth of space when compiling with Clang or MSVC; five with GCC
    static constexpr std::size_t __num_pointers = 4 * (STDEXEC_CLANG() + STDEXEC_MSVC())
                                                + 5 * STDEXEC_GCC();
    static constexpr std::size_t __storage_size  = __num_pointers * sizeof(void*) - 1;
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

    template <class _Derived>
    struct __awaitable_wrapper
    {
      constexpr auto& __awaiter() noexcept
      {
        return static_cast<_Derived*>(this)->__awaiter_;
      }

      constexpr auto await_ready() noexcept(noexcept(__awaiter().await_ready())) -> bool
      {
        return __awaiter().await_ready();
      }

      template <class _Promise>
      constexpr auto await_suspend(__std::coroutine_handle<_Promise> __h)
        noexcept(noexcept(__awaiter().await_suspend(__h)))
      {
        return __awaiter().await_suspend(__h);
      }

      constexpr decltype(auto) await_resume() noexcept(noexcept(__awaiter().await_resume()))
      {
        return __awaiter().await_resume();
      }
    };

    template <class _Tp, class _Promise>
    concept __has_distinct_awaitable = __has_as_awaitable_member<_Tp, _Promise>;

    template <class _Awaitable>
    concept __has_distinct_awaiter = requires(_Awaitable&& __awaitable) {
      { static_cast<_Awaitable&&>(__awaitable).operator co_await() };
    } || requires(_Awaitable&& __awaitable) {
      { operator co_await(static_cast<_Awaitable&&>(__awaitable)) };
    };

    template <class _Awaitable, class _Promise>
    struct __awaitable_state : __awaitable_wrapper<__awaitable_state<_Awaitable, _Promise>>
    {
      using __awaitable_t = __result_of<__get_awaitable, _Awaitable, _Promise&>;
      using __awaiter_t   = __awaiter_of_t<__awaitable_t>;

      static constexpr bool __is_nothrow = __noexcept_of<__get_awaitable, _Awaitable, _Promise&>
                                        && __noexcept_of<__get_awaiter, __awaitable_t>;

      struct __state : __awaitable_wrapper<__state>
      {
        __state(_Awaitable&& __source, __std::coroutine_handle<_Promise> __coro)
          noexcept(__is_nothrow)
        {
          // GCC doesn't like initializing __awaitable_ or __awaiter_ in the member initializer
          // clause when the result of __get_awaitable or __get_awaiter is immovable; it *seems*
          // like direct initialization of a member with the result of a function ought to trigger
          // C++17's mandatory copy elision, and both Clang and MSVC accept that code, but using
          // a union with in-place new works around the issue.
          new (static_cast<void*>(std::addressof(__awaitable_)))
            __awaitable_t(__get_awaitable(static_cast<_Awaitable&&>(__source), __coro.promise()));
          new (static_cast<void*>(std::addressof(__awaiter_)))
            __awaiter_t(__get_awaiter(static_cast<__awaitable_t&&>(__awaitable_)));
        }

        ~__state()
        {
          // make sure to destroy in the reverse order of construction
          std::destroy_at(std::addressof(__awaiter_));
          std::destroy_at(std::addressof(__awaitable_));
        }

        union
        {
          [[no_unique_address]]
          __awaitable_t __awaitable_;
        };

        union
        {
          [[no_unique_address]]
          __awaiter_t __awaiter_;
        };
      };

      [[no_unique_address]]
      _Awaitable __source_awaitable_;
      union
      {
        [[no_unique_address]]
        __state __awaiter_;
      };

      template <class _A>
        requires(!std::same_as<std::remove_cvref_t<_A>, __awaitable_state>)
      explicit __awaitable_state(_A&& __awaitable)
        noexcept(__nothrow_constructible_from<_Awaitable, _A>)
        : __source_awaitable_(static_cast<_A&&>(__awaitable))
      {}

      ~__awaitable_state() {}

      constexpr void construct(__std::coroutine_handle<_Promise> __coro) noexcept(__is_nothrow)
      {
        std::construct_at(&__awaiter_, static_cast<_Awaitable&&>(__source_awaitable_), __coro);
      }

      constexpr void destroy() noexcept
      {
        std::destroy_at(&__awaiter_);
      }
    };

    template <class _Awaitable, class _Promise>
      requires __awaitable<_Awaitable, _Promise>
            && (!__has_distinct_awaitable<_Awaitable, _Promise>)
            && __has_distinct_awaiter<_Awaitable>
    struct __awaitable_state<_Awaitable, _Promise>
      : __awaitable_wrapper<__awaitable_state<_Awaitable, _Promise>>
    {
      // _Awaitable has a distinct awaiter, but no distinct as_awaitable()
      // so we don't need separate storage for it
      using __awaiter_t = __awaiter_of_t<_Awaitable&&>;

      static constexpr bool __is_nothrow = __noexcept_of<__get_awaiter, _Awaitable&&>;

      struct __state : __awaitable_wrapper<__state>
      {
        __state(_Awaitable&& __source, __std::coroutine_handle<_Promise> __coro)
          noexcept(__is_nothrow)
        {
          // GCC doesn't like initializing __awaiter_ in the member initializer clause when the
          // result of __get_awaiter is immovable; it *seems* like direct initialization of a
          // member with the result of a function ought to trigger C++17's mandatory copy elision,
          // and both Clang and MSVC accept that code, but using a union with in-place new works
          // around the issue.
          new (static_cast<void*>(std::addressof(__awaiter_)))
            __awaiter_t(__get_awaiter(static_cast<_Awaitable&&>(__source)));

          [[maybe_unused]]
          auto&& __awaitable = __get_awaitable(static_cast<_Awaitable&&>(__source),
                                               __coro.promise());

          STDEXEC_ASSERT(std::addressof(__awaitable) == std::addressof(__source));
        }

        ~__state()
        {
          std::destroy_at(std::addressof(__awaiter_));
        }

        union
        {
          [[no_unique_address]]
          __awaiter_t __awaiter_;
        };
      };

      [[no_unique_address]]
      _Awaitable __source_awaitable_;
      union
      {
        [[no_unique_address]]
        __state __awaiter_;
      };

      template <class _A>
        requires(!std::same_as<std::remove_cvref_t<_A>, __awaitable_state>)
      explicit __awaitable_state(_A&& __awaitable)
        noexcept(__nothrow_constructible_from<_Awaitable, _A>)
        : __source_awaitable_(static_cast<_A&&>(__awaitable))
      {}

      ~__awaitable_state() {}

      constexpr void construct(__std::coroutine_handle<_Promise> __coro) noexcept(__is_nothrow)
      {
        std::construct_at(&__awaiter_, static_cast<_Awaitable&&>(__source_awaitable_), __coro);
      }

      constexpr void destroy() noexcept
      {
        std::destroy_at(&__awaiter_);
      }
    };

    template <class _Awaitable, class _Promise>
      requires __awaitable<_Awaitable, _Promise>  //
            && __has_distinct_awaitable<_Awaitable, _Promise>
            && (!__has_distinct_awaiter<__result_of<__get_awaitable, _Awaitable, _Promise&>>)
    struct __awaitable_state<_Awaitable, _Promise>
      : __awaitable_wrapper<__awaitable_state<_Awaitable, _Promise>>
    {
      // _Awaitable has a distinct awaitable, but no distinct awaiter
      // so we don't need separate storage for it
      using __awaiter_t = __result_of<__get_awaitable, _Awaitable, _Promise&>;

      static constexpr bool __is_nothrow = __noexcept_of<__get_awaitable, _Awaitable, _Promise&>;

      struct __state : __awaitable_wrapper<__state>
      {
        __state(_Awaitable&& __source, __std::coroutine_handle<_Promise> __coro)
          noexcept(__is_nothrow)
        {
          // GCC doesn't like initializing __awaiter_ in the member initializer clause when the
          // result of __get_awaitable is immovable; it *seems* like direct initialization of a
          // member with the result of a function ought to trigger C++17's mandatory copy elision,
          // and both Clang and MSVC accept that code, but using a union with in-place new works
          // around the issue.
          new (static_cast<void*>(std::addressof(__awaiter_)))
            __awaiter_t(__get_awaitable(static_cast<_Awaitable&&>(__source), __coro.promise()));

          [[maybe_unused]]
          auto&& __awaiter = __get_awaiter(static_cast<__awaiter_t&&>(__awaiter_));
          STDEXEC_ASSERT(std::addressof(__awaiter) == std::addressof(__awaiter_));
        }

        ~__state()
        {
          std::destroy_at(std::addressof(__awaiter_));
        }

        union
        {
          [[no_unique_address]]
          __awaiter_t __awaiter_;
        };
      };

      [[no_unique_address]]
      _Awaitable __source_awaitable_;
      union
      {
        [[no_unique_address]]
        __state __awaiter_;
      };

      template <class _A>
        requires(!std::same_as<std::remove_cvref_t<_A>, __awaitable_state>)
      explicit __awaitable_state(_A&& __awaitable)
        noexcept(__nothrow_constructible_from<_Awaitable, _A>)
        : __source_awaitable_(static_cast<_A&&>(__awaitable))
      {}

      ~__awaitable_state() {}

      constexpr void construct(__std::coroutine_handle<_Promise> __coro) noexcept(__is_nothrow)
      {
        std::construct_at(&__awaiter_, static_cast<_Awaitable&&>(__source_awaitable_), __coro);
      }

      constexpr void destroy() noexcept
      {
        std::destroy_at(&__awaiter_);
      }
    };

    template <class _Awaitable, class _Promise>
      requires __awaitable<_Awaitable, _Promise>
            && (!__has_distinct_awaitable<_Awaitable, _Promise>)
            && (!__has_distinct_awaiter<_Awaitable>)
    struct __awaitable_state<_Awaitable, _Promise>
      : __awaitable_wrapper<__awaitable_state<_Awaitable, _Promise>>
    {
      // _Awaitable has neither a distinct awaiter, nor a distinct awaitable
      // so we don't need separate storage for either
      [[no_unique_address]]
      _Awaitable __awaiter_;

      template <class _A>
        requires(!std::same_as<std::remove_cvref_t<_A>, __awaitable_state>)
      explicit __awaitable_state(_A&& __awaitable)
        noexcept(__nothrow_constructible_from<_Awaitable, _A>)
        : __awaiter_(static_cast<_A&&>(__awaitable))
      {}

      static constexpr void construct(__std::coroutine_handle<_Promise>) noexcept
      {
        // no-op
      }

      static constexpr void destroy() noexcept
      {
        // no-op
      }
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
          __h.promise().__get_opstate().__on_resume();
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

      static constexpr std::ptrdiff_t __promise_offset = sizeof(void*) * 2;

      explicit(!STDEXEC_EDG()) __promise([[maybe_unused]]
                                         __opstate_t& __opstate) noexcept
      {
        STDEXEC_ASSERT(__promise_offset
                       == reinterpret_cast<std::byte*>(this) - __opstate.__storage_);
      }

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
        STDEXEC_ASSERT(__bytes == __storage_size + 1);
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
        __get_opstate().__on_stopped();
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
        return STDEXEC::get_env(__get_opstate().__rcvr_);
      }

      __opstate<_Awaitable, _Receiver>& __get_opstate() noexcept
      {
        return *reinterpret_cast<__opstate<_Awaitable, _Receiver>*>(
          reinterpret_cast<std::byte*>(this) - __promise_offset);
      }

      __opstate<_Awaitable, _Receiver> const & __get_opstate() const noexcept
      {
        return *reinterpret_cast<__opstate<_Awaitable, _Receiver> const *>(
          reinterpret_cast<std::byte const *>(this) - __promise_offset);
      }
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
        noexcept(__nothrow_move_constructible<_Awaitable>)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __awaiter_(static_cast<_Awaitable&&>(__awaitable))
      {}

      __opstate(__opstate&&) = delete;

      ~__opstate()
      {
        if (__started_)
        {
          __awaiter_.destroy();
        }
      }

      void start() & noexcept
      {
        auto __coro = __co_impl(*this);

        STDEXEC_TRY
        {
          __awaiter_.construct(__coro);
          __started_ = true;

          if (!__awaiter_.await_ready())
          {
            using __suspend_result_t = decltype(__awaiter_.await_suspend(__coro));

            // suspended
            if constexpr (std::is_void_v<__suspend_result_t>)
            {
              // void-returning await_suspend means "always suspend"
              __awaiter_.await_suspend(__coro);
              return;
            }
            else if constexpr (std::same_as<bool, __suspend_result_t>)
            {
              if (__awaiter_.await_suspend(__coro))
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
              auto __resume_target = __awaiter_.await_suspend(__coro);
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
          if constexpr (!noexcept(__awaiter_.construct(__coro))
                        || !noexcept(__awaiter_.await_ready())
                        || !noexcept(__awaiter_.await_suspend(__coro)))
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
      [[no_unique_address]]
      bool __started_{false};
      [[no_unique_address]]
      _Receiver __rcvr_;
      [[neo_unique_addres]]
      __awaitable_state<_Awaitable, __promise_t> __awaiter_;
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
