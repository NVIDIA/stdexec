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
#include "__detail/__utility.hpp"

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

  inline void __coroutine_resume_nothrow(__std::coroutine_handle<> __h) noexcept
  {
    STDEXEC_TRY
    {
      STDEXEC_ASSERT(__h);
      __h.resume();
    }
    STDEXEC_CATCH_ALL
    {
      STDEXEC_ASSERT(!"Coroutine resume threw an exception!");
      __std::unreachable();
    }
  }

  inline void __coroutine_destroy_nothrow(__std::coroutine_handle<> __h) noexcept
  {
    STDEXEC_TRY
    {
      STDEXEC_ASSERT(__h);
      __h.destroy();
    }
    STDEXEC_CATCH_ALL
    {
      STDEXEC_ASSERT(!"Coroutine destroy threw an exception!");
      __std::unreachable();
    }
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
      constexpr bool __has_unhandled_stopped = requires { __coro.promise().unhandled_stopped(); };
      static_assert(__has_unhandled_stopped,
                    "Coroutine promises used with senders must implement unhandled_stopped()");

      if constexpr (__has_unhandled_stopped)
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
      return STDEXEC::__coroutine_handle_cast<_Promise>(*this).promise();
    }

    [[nodiscard]]
    constexpr auto handle() const noexcept -> __std::coroutine_handle<_Promise>
    {
      return STDEXEC::__coroutine_handle_cast<_Promise>(*this);
    }

    [[nodiscard]]
    constexpr operator __std::coroutine_handle<_Promise>() const noexcept
    {
      return handle();
    }
  };

  namespace __detail
  {
    struct __synthetic_coro_frame
    {
      void (*__resume_)(void*) noexcept;
      // we never invoke __destroy_ so a no-op implementation is fine; we've chosen the
      // address of a no-op function rather than nullptr in case some rogue awaitable
      // *does* invoke destroy on the synthesized handle that it receives in its
      // await_suspend function
      void (*__destroy_)(void*) noexcept = &__noop_destroy;

      static void __noop_destroy(void*) noexcept
      {
        STDEXEC_ASSERT(!"Attempt to destroy a synthetic coroutine!");
      }
    };

    static constexpr std::ptrdiff_t __coro_promise_offset = static_cast<std::ptrdiff_t>(
      sizeof(__synthetic_coro_frame));
  }  // namespace __detail

#  if STDEXEC_MSVC() && STDEXEC_MSVC_VERSION <= 1939
  // MSVCBUG https://developercommunity.visualstudio.com/t/destroy-coroutine-from-final_suspend-r/10096047

  // Prior to Visual Studio 17.9 (Feb, 2024), aka MSVC 19.39, MSVC incorrectly allocates
  // the return buffer for await_suspend calls within the suspended coroutine frame. When
  // the suspended coroutine is destroyed within await_suspend, the continuation coroutine
  // handle is not only used after free, but also overwritten by the debug malloc
  // implementation when NRVO is in play.

  // This workaround delays the destruction of the suspended coroutine by wrapping the
  // continuation in another "synthetic" coroutine the resumes the continuation and *then*
  // destroys the suspended coroutine.

  // The wrapping coroutine frame is thread-local and reused within the thread for each
  // destroy-and-continue sequence.

  struct __destroy_and_continue_frame : __detail::__synthetic_coro_frame
  {
    constexpr __destroy_and_continue_frame() noexcept
      : __detail::__synthetic_coro_frame{&__destroy_and_continue_frame::__resume}
    {}

    static void __resume(void* __address) noexcept
    {
      // Make a local copy of the promise to ensure we can safely destroy the suspended
      // coroutine after resuming the continuation.
      auto __promise = static_cast<__destroy_and_continue_frame*>(__address)->__promise_;
      STDEXEC::__coroutine_resume_nothrow(__promise.__continue_);
      STDEXEC::__coroutine_destroy_nothrow(__promise.__destroy_);
    }

    struct __promise
    {
      __std::coroutine_handle<> __destroy_{};
      __std::coroutine_handle<> __continue_{};
    } __promise_;
  };

  inline auto __coroutine_destroy_and_continue(__std::coroutine_handle<> __destroy,            //
                                               __std::coroutine_handle<> __continue) noexcept  //
    -> __std::coroutine_handle<>
  {
    static constinit thread_local __destroy_and_continue_frame __fr;
    __fr.__promise_.__destroy_  = __destroy;
    __fr.__promise_.__continue_ = __continue;
    return __std::coroutine_handle<>::from_address(&__fr);
  }

#    define STDEXEC_CORO_DESTROY_AND_CONTINUE(__destroy, __continue)                      \
       ::STDEXEC::__coroutine_destroy_and_continue(__destroy, __continue)
#  else
#    define STDEXEC_CORO_DESTROY_AND_CONTINUE(__destroy, __continue)                      \
       (__destroy.destroy(), __continue)
#  endif
}  // namespace STDEXEC

#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
