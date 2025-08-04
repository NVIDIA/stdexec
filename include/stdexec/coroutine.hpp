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

#include "__detail/__awaitable.hpp" // IWYU pragma: export
#include "__detail/__config.hpp"

#if STDEXEC_MSVC() && STDEXEC_MSVC_VERSION <= 19'39
namespace stdexec {
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

  namespace __destroy_and_continue_msvc {
    struct __task {
      struct promise_type {
        __task get_return_object() noexcept {
          return {__coro::coroutine_handle<promise_type>::from_promise(*this)};
        }

        static std::suspend_never initial_suspend() noexcept {
          return {};
        }

        static std::suspend_never final_suspend() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
          return {};
        }

        static void return_void() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }

        static void unhandled_exception() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }
      };

      __coro::coroutine_handle<> __coro_;
    };

    struct __continue_t {
      static constexpr bool await_ready() noexcept {
        return false;
      }

      __coro::coroutine_handle<> await_suspend(__coro::coroutine_handle<>) noexcept {
        return __continue_;
      }

      static void await_resume() noexcept {
      }

      __coro::coroutine_handle<> __continue_;
    };

    struct __context {
      __coro::coroutine_handle<> __destroy_;
      __coro::coroutine_handle<> __continue_;
    };

    inline __task __co_impl(__context& __c) {
      while (true) {
        co_await __continue_t{__c.__continue_};
        __c.__destroy_.destroy();
      }
    }

    struct __context_and_coro {
      __context_and_coro() {
        __context_.__continue_ = __coro::noop_coroutine();
        __coro_ = __co_impl(__context_).__coro_;
      }

      ~__context_and_coro() {
        __coro_.destroy();
      }

      __context __context_;
      __coro::coroutine_handle<> __coro_;
    };

    inline __coro::coroutine_handle<>
      __impl(__coro::coroutine_handle<> __destroy, __coro::coroutine_handle<> __continue) {
      static thread_local __context_and_coro __c;
      __c.__context_.__destroy_ = __destroy;
      __c.__context_.__continue_ = __continue;
      return __c.__coro_;
    }
  } // namespace __destroy_and_continue_msvc
} // namespace stdexec

#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue)                                      \
    (::stdexec::__destroy_and_continue_msvc::__impl(__destroy, __continue))
#else
#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue) (__destroy.destroy(), __continue)
#endif
