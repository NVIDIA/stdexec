/*
 * Copyright (c) 2023 Maikel Nadolski
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

// The original idea is taken from libunifex and adapted to stdexec.

#include <exception>
#include <type_traits>

#include "../stdexec/execution.hpp"
#include "task.hpp"
#include "inline_scheduler.hpp"
#include "any_sender_of.hpp"

namespace exec {
  namespace __on_coro_disp {
    using namespace stdexec;

    using __any_scheduler =                                                      //
      any_receiver_ref<                                                          //
        completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>> //
      ::any_sender<>::any_scheduler<>;

    template <class _Promise>
    concept __promise_with_disposition =              //
      __at_coro_exit::__has_continuation<_Promise> && //
      requires(_Promise& __promise) {
        { __promise.disposition() } -> convertible_to<task_disposition>;
      };

    struct __get_disposition {
      task_disposition __disposition_;

      static constexpr bool await_ready() noexcept {
        return false;
      }

      template <class _Promise>
      bool await_suspend(__coro::coroutine_handle<_Promise> __h) noexcept {
        auto& __promise = __h.promise();
        __disposition_ = //
          __promise.__get_disposition_callback_(__promise.__parent_.address());
        return false;
      }

      task_disposition await_resume() const noexcept {
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

      bool await_ready() const noexcept {
        return false;
      }

      template <__promise_with_disposition _Promise>
      bool await_suspend(__coro::coroutine_handle<_Promise> __parent) noexcept {
        __coro_.promise().__parent_ = __parent;
        __coro_.promise().__get_disposition_callback_ = //
          [](void* __parent) noexcept {
            _Promise& __promise =
              __coro::coroutine_handle<_Promise>::from_address(__parent).promise();
            return __promise.disposition();
          };
        __coro_.promise().__scheduler_ = get_scheduler(get_env(__parent.promise()));
        __coro_.promise().set_continuation(__parent.promise().continuation());
        __parent.promise().set_continuation(__coro_);
        return false;
      }

      std::tuple<_Ts&...> await_resume() noexcept {
        return std::exchange(__coro_, {}).promise().__args_;
      }

     private:
      struct __final_awaitable {
        static constexpr bool await_ready() noexcept {
          return false;
        }

        static __coro::coroutine_handle<>
          await_suspend(__coro::coroutine_handle<__promise> __h) noexcept {
          __promise& __p = __h.promise();
          auto __coro = __p.__is_unhandled_stopped_
                        ? __p.continuation().unhandled_stopped()
                        : __p.continuation().handle();
          return STDEXEC_DESTROY_AND_CONTINUE(__h, __coro);
        }

        void await_resume() const noexcept {
        }
      };

      struct __env {
        const __promise& __promise_;

        friend __any_scheduler tag_invoke(get_scheduler_t, __env __self) noexcept {
          return __self.__promise_.__scheduler_;
        }
      };

      struct __promise : with_awaitable_senders<__promise> {
        template <class _Action>
        explicit __promise(_Action&&, _Ts&... __ts) noexcept
          : __args_{__ts...} {
        }

        __coro::suspend_always initial_suspend() noexcept {
          return {};
        }

        __final_awaitable final_suspend() noexcept {
          return {};
        }

        void return_void() noexcept {
        }

        [[noreturn]] void unhandled_exception() noexcept {
          std::terminate();
        }

        __coro::coroutine_handle<__promise> unhandled_stopped() noexcept {
          __is_unhandled_stopped_ = true;
          return __coro::coroutine_handle<__promise>::from_promise(*this);
        }

        __task get_return_object() noexcept {
          return __task(__coro::coroutine_handle<__promise>::from_promise(*this));
        }

        __get_disposition await_transform(__get_disposition __awaitable) noexcept {
          return __awaitable;
        }

        template <class _Awaitable>
        decltype(auto) await_transform(_Awaitable&& __awaitable) noexcept {
          return as_awaitable(__at_coro_exit::__die_on_stop((_Awaitable&&) __awaitable), *this);
        }

        friend __env tag_invoke(get_env_t, const __promise& __self) noexcept {
          return {__self};
        }

        bool __is_unhandled_stopped_{false};
        std::tuple<_Ts&...> __args_{};
        using __get_disposition_callback_t = task_disposition (*)(void*) noexcept;
        __coro::coroutine_handle<> __parent_{};
        __get_disposition_callback_t __get_disposition_callback_{nullptr};
        __any_scheduler __scheduler_{inline_scheduler{}};
      };

      __coro::coroutine_handle<__promise> __coro_;
    };

    template <task_disposition _OnCompletion>
    class __on_disp {
     private:
      template <class _Action, class... _Ts>
      static __task<_Ts...> __impl(_Action __action, _Ts... __ts) {
        task_disposition __d = co_await __get_disposition();
        if (__d == _OnCompletion) {
          co_await ((_Action&&) __action)((_Ts&&) __ts...);
        }
      }

     public:
      template <class _Action, class... _Ts>
        requires __callable<__decay_t<_Action>, __decay_t<_Ts>...>
      __task<_Ts...> operator()(_Action&& __action, _Ts&&... __ts) const {
        return __impl((_Action&&) __action, (_Ts&&) __ts...);
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
