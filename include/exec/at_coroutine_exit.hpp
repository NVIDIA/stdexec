/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "inline_scheduler.hpp"
#include "any_sender_of.hpp"

namespace exec {
  namespace __at_coro_exit {
    using namespace stdexec;

    using __any_scheduler =                                                      //
      any_receiver_ref<                                                          //
        completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>> //
      ::any_sender<>::any_scheduler<>;

    struct __die_on_stop_t {
      template <class _Receiver>
      struct __receiver_id {
        struct __t {
          using receiver_concept = stdexec::receiver_t;
          using __id = __receiver_id;
          _Receiver __receiver_;

          template <__one_of<set_value_t, set_error_t> _Tag, __decays_to<__t> _Self, class... _Args>
            requires __callable<_Tag, _Receiver, _Args...>
          friend void tag_invoke(_Tag, _Self&& __self, _Args&&... __args) noexcept {
            _Tag{}((_Receiver&&) __self.__receiver_, (_Args&&) __args...);
          }

          template <same_as<set_stopped_t> _Tag>
          [[noreturn]] friend void tag_invoke(_Tag, __t&&) noexcept {
            std::terminate();
          }

          friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
            return get_env(__self.__receiver_);
          }
        };
      };
      template <class _Rec>
      using __receiver = __t<__receiver_id<_Rec>>;

      template <class _Sender>
      struct __sender_id {
        template <class _Env>
        using __completion_signatures = //
          __mapply<
            __remove<set_stopped_t(), __q<completion_signatures>>,
            completion_signatures_of_t<_Sender, _Env>>;

        struct __t {
          using __id = __sender_id;
          using sender_concept = stdexec::sender_t;

          _Sender __sender_;

          template <receiver _Receiver>
            requires sender_to<_Sender, __receiver<_Receiver>>
          friend connect_result_t<_Sender, __receiver<_Receiver>>
            tag_invoke(connect_t, __t&& __self, _Receiver&& __rcvr) noexcept {
            return stdexec::connect(
              (_Sender&&) __self.__sender_, __receiver<_Receiver>{(_Receiver&&) __rcvr});
          }

          template <__decays_to<__t> _Self, class _Env>
          friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
            -> __completion_signatures<_Env> {
            return {};
          }

          friend env_of_t<_Sender> tag_invoke(get_env_t, const __t& __self) noexcept {
            return get_env(__self.__sender_);
          }
        };
      };
      template <class _Sender>
      using __sender = __t<__sender_id<__decay_t<_Sender>>>;

      template <sender _Sender>
      __sender<_Sender> operator()(_Sender&& __sndr) const
        noexcept(__nothrow_decay_copyable<_Sender>) {
        return __sender<_Sender>{(_Sender&&) __sndr};
      }

      template <class _Value>
      _Value&& operator()(_Value&& __value) const noexcept {
        return (_Value&&) __value;
      }
    };

    inline constexpr __die_on_stop_t __die_on_stop;

    template <class _Promise>
    concept __has_continuation = //
      requires(_Promise& __promise, __continuation_handle<> __c) {
        { __promise.continuation() } -> convertible_to<__continuation_handle<>>;
        { __promise.set_continuation(__c) };
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

      template <__has_continuation _Promise>
      bool await_suspend(__coro::coroutine_handle<_Promise> __parent) noexcept {
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

        template <class _Awaitable>
        decltype(auto) await_transform(_Awaitable&& __awaitable) noexcept {
          return as_awaitable(__die_on_stop((_Awaitable&&) __awaitable), *this);
        }

        friend __env tag_invoke(get_env_t, const __promise& __self) noexcept {
          return {__self};
        }

        bool __is_unhandled_stopped_{false};
        std::tuple<_Ts&...> __args_{};
        __any_scheduler __scheduler_{inline_scheduler{}};
      };

      __coro::coroutine_handle<__promise> __coro_;
    };

    struct __at_coro_exit_t {
     private:
      template <class _Action, class... _Ts>
      static __task<_Ts...> __impl(_Action __action, _Ts... __ts) {
        co_await ((_Action&&) __action)((_Ts&&) __ts...);
      }

     public:
      template <class _Action, class... _Ts>
        requires __callable<__decay_t<_Action>, __decay_t<_Ts>...>
      __task<_Ts...> operator()(_Action&& __action, _Ts&&... __ts) const {
        return __impl((_Action&&) __action, (_Ts&&) __ts...);
      }
    };
  } // namespace __at_coro_exit

  inline constexpr __at_coro_exit::__at_coro_exit_t at_coroutine_exit{};
} // namespace exec
