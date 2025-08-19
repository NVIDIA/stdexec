/*
 * Copyright (c) 2023 Maikel Nadolski
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

// The original idea is taken from libunifex and adapted to stdexec.

#include "../stdexec/execution.hpp"
#include "../stdexec/coroutine.hpp"
#include "task.hpp"
#include "inline_scheduler.hpp" // IWYU pragma: keep
#include "any_sender_of.hpp"

#include <exception>

namespace exec {
  namespace __on_coro_disp {
    using namespace stdexec;

    using __any_scheduler = any_receiver_ref<
      completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>
    >::any_sender<>::any_scheduler<>;

    template <class _Promise>
    concept __promise_with_disposition = __at_coro_exit::__has_continuation<_Promise>
                                      && requires(_Promise& __promise) {
                                           {
                                             __promise.disposition()
                                           } -> convertible_to<task_disposition>;
                                         };

    struct __get_disposition {
      task_disposition __disposition_;

      static constexpr auto await_ready() noexcept -> bool {
        return false;
      }

      template <class _Promise>
      auto await_suspend(__coro::coroutine_handle<_Promise> __h) noexcept -> bool {
        auto& __promise = __h.promise();
        __disposition_ = __promise.__get_disposition_callback_(__promise.__parent_.address());
        return false;
      }

      [[nodiscard]]
      auto await_resume() const noexcept -> task_disposition {
        return __disposition_;
      }
    };

    template <class... _Ts>
    class [[nodiscard]] __task {
      struct __promise;
     public:
      using promise_type = __promise;

      explicit __task(__coro::coroutine_handle<__promise> __coro) noexcept
        : __coro_(__coro) {
      }

      __task(__task&& __that) noexcept
        : __coro_(std::exchange(__that.__coro_, {})) {
      }

      [[nodiscard]]
      auto await_ready() const noexcept -> bool {
        return false;
      }

      template <__promise_with_disposition _Promise>
      auto await_suspend(__coro::coroutine_handle<_Promise> __parent) noexcept -> bool {
        __coro_.promise().__parent_ = __parent;
        __coro_.promise().__get_disposition_callback_ = [](void* __parent) noexcept {
          _Promise& __promise = __coro::coroutine_handle<_Promise>::from_address(__parent)
                                  .promise();
          return __promise.disposition();
        };
        __coro_.promise().__scheduler_ = get_scheduler(get_env(__parent.promise()));
        __coro_.promise().set_continuation(__parent.promise().continuation());
        __parent.promise().set_continuation(__coro_);
        return false;
      }

      auto await_resume() noexcept -> std::tuple<_Ts&...> {
        return std::exchange(__coro_, {}).promise().__args_;
      }

     private:
      struct __final_awaitable {
        static constexpr auto await_ready() noexcept -> bool {
          return false;
        }

        static auto await_suspend(__coro::coroutine_handle<__promise> __h) noexcept
          -> __coro::coroutine_handle<> {
          __promise& __p = __h.promise();
          auto __coro = __p.__is_unhandled_stopped_ ? __p.continuation().unhandled_stopped()
                                                    : __p.continuation().handle();
          return STDEXEC_DESTROY_AND_CONTINUE(__h, __coro);
        }

        void await_resume() const noexcept {
        }
      };

      struct __env {
        const __promise& __promise_;

        [[nodiscard]]
        auto query(get_scheduler_t) const noexcept -> __any_scheduler {
          return __promise_.__scheduler_;
        }
      };

      struct __promise : with_awaitable_senders<__promise> {
#if STDEXEC_EDG()
        template <class _Action>
        __promise(_Action&&, _Ts&&... __ts) noexcept
          : __args_{__ts...} {
        }
#else
        template <class _Action>
        explicit __promise(_Action&&, _Ts&... __ts) noexcept
          : __args_{__ts...} {
        }
#endif

        auto initial_suspend() noexcept -> __coro::suspend_always {
          return {};
        }

        auto final_suspend() noexcept -> __final_awaitable {
          return {};
        }

        void return_void() noexcept {
        }

        [[noreturn]]
        void unhandled_exception() noexcept {
          std::terminate();
        }

        auto unhandled_stopped() noexcept -> __coro::coroutine_handle<__promise> {
          __is_unhandled_stopped_ = true;
          return __coro::coroutine_handle<__promise>::from_promise(*this);
        }

        auto get_return_object() noexcept -> __task {
          return __task(__coro::coroutine_handle<__promise>::from_promise(*this));
        }

        auto await_transform(__get_disposition __awaitable) noexcept -> __get_disposition {
          return __awaitable;
        }

        template <class _Awaitable>
        auto await_transform(_Awaitable&& __awaitable) noexcept -> decltype(auto) {
          return as_awaitable(
            __at_coro_exit::__die_on_stop(static_cast<_Awaitable&&>(__awaitable)), *this);
        }

        auto get_env() const noexcept -> __env {
          return {*this};
        }

        bool __is_unhandled_stopped_{false};
        std::tuple<_Ts&...> __args_{};
        using __get_disposition_callback_t = task_disposition (*)(void*) noexcept;
        __coro::coroutine_handle<> __parent_{};
        __get_disposition_callback_t __get_disposition_callback_{nullptr};
        __any_scheduler __scheduler_{stdexec::inline_scheduler{}};
      };

      __coro::coroutine_handle<__promise> __coro_;
    };

    template <task_disposition _OnCompletion>
    class __on_disp {
     private:
      template <class _Action, class... _Ts>
      static auto __impl(_Action __action, _Ts... __ts) -> __task<_Ts...> {
#if STDEXEC_EDG()
        // This works around an EDG bug where the compiler misinterprets __get_disposition:
        // operand to this co_await expression resolves to non-class "<unnamed>"
        using __get_disposition =
          std::enable_if_t<sizeof(_Action) != 0, __on_coro_disp::__get_disposition>;
#endif
        task_disposition __d = co_await __get_disposition();
        if (__d == _OnCompletion) {
          co_await static_cast<_Action&&>(__action)(static_cast<_Ts&&>(__ts)...);
        }
      }

     public:
      template <class _Action, class... _Ts>
        requires __callable<__decay_t<_Action>, __decay_t<_Ts>...>
      auto operator()(_Action&& __action, _Ts&&... __ts) const -> __task<_Ts...> {
        return __impl(static_cast<_Action&&>(__action), static_cast<_Ts&&>(__ts)...);
      }
    };

    struct __succeeded_t : __on_disp<task_disposition::succeeded> { };

    struct __stopped_t : __on_disp<task_disposition::stopped> { };

    struct __failed_t : __on_disp<task_disposition::failed> { };
  } // namespace __on_coro_disp

  inline constexpr __on_coro_disp::__succeeded_t on_coroutine_succeeded{};
  inline constexpr __on_coro_disp::__stopped_t on_coroutine_stopped{};
  inline constexpr __on_coro_disp::__failed_t on_coroutine_failed{};
} // namespace exec
