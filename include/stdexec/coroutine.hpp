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

  STDEXEC_ATTRIBUTE(always_inline)
  void __coroutine_resume_nothrow(void* __address) noexcept
  {
    STDEXEC_TRY
    {
      __builtin_coro_resume(__address);
    }
    STDEXEC_CATCH_ALL
    {
      __std::unreachable();
    }
  }

  STDEXEC_ATTRIBUTE(always_inline)
  void __coroutine_resume_nothrow(__std::coroutine_handle<> __h) noexcept
  {
    STDEXEC::__coroutine_resume_nothrow(__h.address());
  }

  STDEXEC_ATTRIBUTE(always_inline)
  void __coroutine_destroy_nothrow(void* __address) noexcept
  {
    STDEXEC_TRY
    {
      __builtin_coro_destroy(__address);
    }
    STDEXEC_CATCH_ALL
    {
      __std::unreachable();
    }
  }

  STDEXEC_ATTRIBUTE(always_inline)
  void __coroutine_destroy_nothrow(__std::coroutine_handle<> __h) noexcept
  {
    STDEXEC::__coroutine_destroy_nothrow(__h.address());
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
      using __callback_fn_t = void(void*) noexcept;

      __callback_fn_t* __resume_  = &__noop_fn;
      __callback_fn_t* __destroy_ = &__noop_fn;

      static void __noop_fn(void*) noexcept {}
    };

    static constexpr std::ptrdiff_t __coro_promise_offset = static_cast<std::ptrdiff_t>(
      sizeof(__synthetic_coro_frame));
  }  // namespace __detail

#  if STDEXEC_MSVC() && STDEXEC_MSVC_VERSION < 1950
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
    static void __resume(void* __address) noexcept
    {
      // Make a local copy of the promise to ensure we can safely destroy the suspended
      // coroutine after resuming the continuation.
      auto __promise = static_cast<__destroy_and_continue_frame*>(__address)->__promise_;
      STDEXEC::__coroutine_resume_nothrow(__promise.__continue_);
      STDEXEC_ATTRIBUTE(musttail)
      return STDEXEC::__coroutine_destroy_nothrow(__promise.__destroy_.address());
    }

    struct __promise
    {
      __std::coroutine_handle<> __destroy_{};
      __std::coroutine_handle<> __continue_{};
    } __promise_;

    static thread_local __destroy_and_continue_frame value;
  };

  inline thread_local __destroy_and_continue_frame __destroy_and_continue_frame::value{
    {&__destroy_and_continue_frame::__resume},
    {}};

  struct __symmetric_transfer_frame : __detail::__synthetic_coro_frame
  {
    static void __resume(void* __address) noexcept
    {
      // Make a local copy of the promise to ensure we can safely destroy the suspended
      // coroutine after resuming the continuation.
      auto __promise = static_cast<__symmetric_transfer_frame*>(__address)->__promise_;
      STDEXEC_ATTRIBUTE(musttail)
      return STDEXEC::__coroutine_resume_nothrow(__promise.__continue_.address());
    }

    struct __promise
    {
      __std::coroutine_handle<> __continue_{};
    } __promise_;

    static thread_local __symmetric_transfer_frame value;
  };

  inline thread_local __symmetric_transfer_frame __symmetric_transfer_frame::value{
    {&__symmetric_transfer_frame::__resume},
    {}};

  inline auto __coroutine_destroy_and_continue(__std::coroutine_handle<> __destroy,            //
                                               __std::coroutine_handle<> __continue) noexcept  //
    -> __std::coroutine_handle<>
  {
    __destroy_and_continue_frame::value.__promise_.__destroy_  = __destroy;
    __destroy_and_continue_frame::value.__promise_.__continue_ = __continue;
    return __std::coroutine_handle<>::from_address(&__destroy_and_continue_frame::value);
  }

  inline auto __coroutine_destroy_and_continue(__std::coroutine_handle<> __continue) noexcept  //
    -> __std::coroutine_handle<>
  {
    __symmetric_transfer_frame::value.__promise_.__continue_ = __continue;
    return __std::coroutine_handle<>::from_address(&__symmetric_transfer_frame::value);
  }

#  else

  STDEXEC_ATTRIBUTE(always_inline)
  auto __coroutine_destroy_and_continue(__std::coroutine_handle<> __destroy,            //
                                        __std::coroutine_handle<> __continue) noexcept  //
    -> __std::coroutine_handle<>
  {
    ::STDEXEC::__coroutine_destroy_nothrow(__destroy);
    return __continue;
  }

  STDEXEC_ATTRIBUTE(always_inline)
  auto __coroutine_destroy_and_continue(__std::coroutine_handle<> __continue) noexcept  //
    -> __std::coroutine_handle<>
  {
    return __continue;
  }

#  endif  // STDEXEC_MSVC() && STDEXEC_MSVC_VERSION < 1950
}  // namespace STDEXEC

#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
