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

#include "__detail/__awaitable.hpp"  // IWYU pragma: export
#include "__detail/__concepts.hpp"
#include "__detail/__config.hpp"

#include <exception>

#if !STDEXEC_NO_STDCPP_COROUTINES()

namespace STDEXEC
{
  template <class _Tp, __one_of<_Tp, void> _Up>
  constexpr auto __coroutine_handle_cast(__std::coroutine_handle<_Up> __h) noexcept  //
    -> __std::coroutine_handle<_Tp>
  {
    return __std::coroutine_handle<_Tp>::from_address(__h.address());
  }

  // A coroutine handle that also supports unhandled_stopped() for propagating stop
  // signals through co_awaits of senders.
  template <class _Promise = void>
  class __coroutine_handle;

  template <>
  class __coroutine_handle<void> : __std::coroutine_handle<>
  {
   public:
    constexpr __coroutine_handle() = default;

    template <class _Promise>
    constexpr __coroutine_handle(__std::coroutine_handle<_Promise> __coro) noexcept
      : __std::coroutine_handle<>(__coro)
    {
      if constexpr (requires(_Promise& __promise) { __promise.unhandled_stopped(); })
      {
        __stopped_callback_ = [](void* __address) noexcept -> __std::coroutine_handle<>
        {
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
    constexpr auto handle() const noexcept -> __std::coroutine_handle<>
    {
      return *this;
    }

    [[nodiscard]]
    constexpr auto unhandled_stopped() const noexcept -> __std::coroutine_handle<>
    {
      return __stopped_callback_(address());
    }

   private:
    using __stopped_callback_t = __std::coroutine_handle<> (*)(void*) noexcept;

    __stopped_callback_t __stopped_callback_ = [](void*) noexcept -> __std::coroutine_handle<>
    {
      std::terminate();
    };
  };

  template <class _Promise>
  class __coroutine_handle : public __coroutine_handle<>
  {
   public:
    constexpr __coroutine_handle() = default;

    constexpr __coroutine_handle(__std::coroutine_handle<_Promise> __coro) noexcept
      : __coroutine_handle<>{__coro}
    {}

    [[nodiscard]]
    static constexpr auto from_promise(_Promise& __promise) noexcept -> __coroutine_handle
    {
      return __coroutine_handle(__std::coroutine_handle<_Promise>::from_promise(__promise));
    }

    [[nodiscard]]
    constexpr auto promise() const noexcept -> _Promise&
    {
      return __std::coroutine_handle<_Promise>::from_address(address()).promise();
    }

    [[nodiscard]]
    constexpr auto handle() const noexcept -> __std::coroutine_handle<_Promise>
    {
      return __std::coroutine_handle<_Promise>::from_address(address());
    }

    [[nodiscard]]
    constexpr operator __std::coroutine_handle<_Promise>() const noexcept
    {
      return handle();
    }
  };

#  if STDEXEC_MSVC() && STDEXEC_MSVC_VERSION <= 1939
  // MSVCBUG https://developercommunity.visualstudio.com/t/destroy-coroutine-from-final_suspend-r/10096047

  // Prior to Visual Studio 17.9 (Feb, 2024), aka MSVC 19.39, MSVC incorrectly allocates the return
  // buffer for await_suspend calls within the suspended coroutine frame. When the suspended
  // coroutine is destroyed within await_suspend, the continuation coroutine handle is not only used
  // after free, but also overwritten by the debug malloc implementation when NRVO is in play.

  // This workaround delays the destruction of the suspended coroutine by wrapping the continuation
  // in another coroutine which destroys the former and transfers execution to the original
  // continuation.

  // The wrapping coroutine is thread-local and is reused within the thread for each
  // destroy-and-continue sequence. The wrapping coroutine itself is destroyed at thread exit.

  namespace __destroy_and_continue_msvc
  {
    struct __task
    {
      struct promise_type
      {
        __task get_return_object() noexcept
        {
          return {__std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        static std::suspend_never initial_suspend() noexcept
        {
          return {};
        }

        static std::suspend_never final_suspend() noexcept
        {
          STDEXEC_ASSERT(!"Should never get here");
          return {};
        }

        static void return_void() noexcept
        {
          STDEXEC_ASSERT(!"Should never get here");
        }

        static void unhandled_exception() noexcept
        {
          STDEXEC_ASSERT(!"Should never get here");
        }
      };

      __std::coroutine_handle<> __coro_;
    };

    struct __continue_t
    {
      static constexpr bool await_ready() noexcept
      {
        return false;
      }

      __std::coroutine_handle<> await_suspend(__std::coroutine_handle<>) noexcept
      {
        return __continue_;
      }

      static void await_resume() noexcept {}

      __std::coroutine_handle<> __continue_;
    };

    struct __context
    {
      __std::coroutine_handle<> __destroy_;
      __std::coroutine_handle<> __continue_;
    };

    inline __task __co_impl(__context& __c)
    {
      while (true)
      {
        co_await __continue_t{__c.__continue_};
        __c.__destroy_.destroy();
      }
    }

    struct __context_and_coro
    {
      __context_and_coro()
      {
        __context_.__continue_ = __std::noop_coroutine();
        __coro_                = __co_impl(__context_).__coro_;
      }

      ~__context_and_coro()
      {
        __coro_.destroy();
      }

      __context                 __context_;
      __std::coroutine_handle<> __coro_;
    };

    inline __std::coroutine_handle<>
    __impl(__std::coroutine_handle<> __destroy, __std::coroutine_handle<> __continue)
    {
      static thread_local __context_and_coro __c;
      __c.__context_.__destroy_  = __destroy;
      __c.__context_.__continue_ = __continue;
      return __c.__coro_;
    }
  }  // namespace __destroy_and_continue_msvc

#    define STDEXEC_CORO_DESTROY_AND_CONTINUE(__destroy, __continue)                      \
       (::STDEXEC::__destroy_and_continue_msvc::__impl(__destroy, __continue))
#  else
#    define STDEXEC_CORO_DESTROY_AND_CONTINUE(__destroy, __continue)                      \
       (__destroy.destroy(), __continue)
#  endif
}  // namespace STDEXEC

#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
