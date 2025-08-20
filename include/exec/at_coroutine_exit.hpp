/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <exception>

#include "../stdexec/execution.hpp"

#include "inline_scheduler.hpp"
#include "any_sender_of.hpp"

namespace exec {
  namespace __at_coro_exit {
    using namespace stdexec;

    using __any_scheduler = any_receiver_ref<
      completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>
    >::any_sender<>::any_scheduler<>;

    struct __die_on_stop_t {
      template <class _Receiver>
      struct __receiver_id {
        struct __t {
          using receiver_concept = stdexec::receiver_t;
          using __id = __receiver_id;
          _Receiver __receiver_;

          template <class... _Args>
            requires __callable<set_value_t, _Receiver, _Args...>
          void set_value(_Args&&... __args) noexcept {
            stdexec::set_value(
              static_cast<_Receiver&&>(__receiver_), static_cast<_Args&&>(__args)...);
          }

          template <class _Error>
            requires __callable<set_error_t, _Receiver, _Error>
          void set_error(_Error&& __err) noexcept {
            stdexec::set_error(static_cast<_Receiver&&>(__receiver_), static_cast<_Error&&>(__err));
          }

          [[noreturn]]
          void set_stopped() noexcept {
            std::terminate();
          }

          auto get_env() const noexcept -> env_of_t<_Receiver> {
            return stdexec::get_env(__receiver_);
          }
        };
      };

      template <class _Rec>
      using __receiver = __t<__receiver_id<_Rec>>;

      template <class _Sender>
      struct __sender_id {
        template <class... _Env>
        using __completion_signatures = __mapply<
          __mremove<set_stopped_t(), __q<completion_signatures>>,
          __completion_signatures_of_t<_Sender, _Env...>
        >;

        struct __t {
          using __id = __sender_id;
          using sender_concept = stdexec::sender_t;

          _Sender __sender_;

          template <receiver _Receiver>
            requires sender_to<_Sender, __receiver<_Receiver>>
          auto connect(_Receiver __rcvr) && noexcept
            -> connect_result_t<_Sender, __receiver<_Receiver>> {
            return stdexec::connect(
              static_cast<_Sender&&>(__sender_),
              __receiver<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
          }

          template <class... _Env>
          auto get_completion_signatures(_Env&&...) -> __completion_signatures<_Env...> {
            return {};
          }

          auto get_env() const noexcept -> env_of_t<_Sender> {
            return stdexec::get_env(__sender_);
          }
        };
      };
      template <class _Sender>
      using __sender = __t<__sender_id<__decay_t<_Sender>>>;

      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const noexcept(__nothrow_decay_copyable<_Sender>)
        -> __sender<_Sender> {
        return __sender<_Sender>{static_cast<_Sender&&>(__sndr)};
      }

      template <class _Value>
      auto operator()(_Value&& __value) const noexcept -> _Value&& {
        return static_cast<_Value&&>(__value);
      }
    };

    inline constexpr __die_on_stop_t __die_on_stop;

    template <class _Promise>
    concept __has_continuation = requires(_Promise& __promise, __coroutine_handle<> __c) {
      { __promise.continuation() } -> convertible_to<__coroutine_handle<>>;
      { __promise.set_continuation(__c) };
    };

    template <class... _Ts>
    class [[nodiscard]] __task {
      struct __promise;
     public:
      using promise_type = __promise;

#if STDEXEC_EDG()
      __task(__coro::coroutine_handle<__promise> __coro) noexcept
        : __coro_(__coro) {
      }
#else
      explicit __task(__coro::coroutine_handle<__promise> __coro) noexcept
        : __coro_(__coro) {
      }
#endif

      __task(__task&& __that) noexcept
        : __coro_(std::exchange(__that.__coro_, {})) {
      }

      [[nodiscard]]
      auto await_ready() const noexcept -> bool {
        return false;
      }

      template <__has_continuation _Promise>
      auto await_suspend(__coro::coroutine_handle<_Promise> __parent) noexcept -> bool {
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

        template <class _Awaitable>
        auto await_transform(_Awaitable&& __awaitable) noexcept -> decltype(auto) {
          return as_awaitable(__die_on_stop(static_cast<_Awaitable&&>(__awaitable)), *this);
        }

        auto get_env() const noexcept -> __env {
          return {*this};
        }

        bool __is_unhandled_stopped_{false};
        std::tuple<_Ts&...> __args_{};
        __any_scheduler __scheduler_{stdexec::inline_scheduler{}};
      };

      __coro::coroutine_handle<__promise> __coro_;
    };

    struct __at_coro_exit_t {
     private:
      template <class _Action, class... _Ts>
      static auto __impl(_Action __action, _Ts... __ts) -> __task<_Ts...> {
        co_await static_cast<_Action&&>(__action)(static_cast<_Ts&&>(__ts)...);
      }

     public:
      template <class _Action, class... _Ts>
        requires __callable<__decay_t<_Action>, __decay_t<_Ts>...>
      auto operator()(_Action&& __action, _Ts&&... __ts) const -> __task<_Ts...> {
        return __impl(static_cast<_Action&&>(__action), static_cast<_Ts&&>(__ts)...);
      }
    };
  } // namespace __at_coro_exit

  inline constexpr __at_coro_exit::__at_coro_exit_t at_coroutine_exit{};
} // namespace exec
