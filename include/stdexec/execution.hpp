/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__detail/__as_awaitable.hpp"
#include "__detail/__basic_sender.hpp"
#include "__detail/__bulk.hpp"
#include "__detail/__completion_signatures.hpp"
#include "__detail/__connect_awaitable.hpp"
#include "__detail/__continue_on.hpp"
#include "__detail/__cpo.hpp"
#include "__detail/__debug.hpp"
#include "__detail/__domain.hpp"
#include "__detail/__ensure_started.hpp"
#include "__detail/__env.hpp"
#include "__detail/__execute.hpp"
#include "__detail/__inline_scheduler.hpp"
#include "__detail/__into_variant.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__intrusive_slist.hpp"
#include "__detail/__just.hpp"
#include "__detail/__let.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__on.hpp"
#include "__detail/__operation_states.hpp"
#include "__detail/__read_env.hpp"
#include "__detail/__receivers.hpp"
#include "__detail/__receiver_adaptor.hpp"
#include "__detail/__run_loop.hpp"
#include "__detail/__schedule_from.hpp"
#include "__detail/__schedulers.hpp"
#include "__detail/__senders.hpp"
#include "__detail/__sender_adaptor_closure.hpp"
#include "__detail/__split.hpp"
#include "__detail/__start_detached.hpp"
#include "__detail/__start_on.hpp"
#include "__detail/__stopped_as_error.hpp"
#include "__detail/__stopped_as_optional.hpp"
#include "__detail/__submit.hpp"
#include "__detail/__sync_wait.hpp"
#include "__detail/__then.hpp"
#include "__detail/__transfer_just.hpp"
#include "__detail/__transform_sender.hpp"
#include "__detail/__transform_completion_signatures.hpp"
#include "__detail/__type_traits.hpp"
#include "__detail/__upon_error.hpp"
#include "__detail/__upon_stopped.hpp"
#include "__detail/__utility.hpp"
#include "__detail/__when_all.hpp"
#include "__detail/__with_awaitable_senders.hpp"
#include "__detail/__write_env.hpp"

#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

#include <atomic>
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <memory>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>
#include <cstddef>
#include <exception>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wundefined-inline")
STDEXEC_PRAGMA_IGNORE_GNU("-Wsubobject-linkage")
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

STDEXEC_PRAGMA_IGNORE_EDG(1302)
STDEXEC_PRAGMA_IGNORE_EDG(497)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Tag = set_value_t>
  concept __completes_on =
    __decays_to<__call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct __ignore_sender {
    using sender_concept = sender_t;

    template <sender _Sender>
    constexpr __ignore_sender(_Sender&&) noexcept {
    }
  };

  template <auto _Reason = "You cannot pipe one sender into another."_mstr>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;
} // namespace stdexec

#if STDEXEC_MSVC() && _MSC_VER >= 1939
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

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(stdexec::__ignore_sender, _Sender&&) noexcept -> stdexec::__ignore_sender;

#include "__detail/__p2300.hpp"

STDEXEC_PRAGMA_POP()
