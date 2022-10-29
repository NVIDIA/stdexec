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

#include <any>
#include <cassert>
#include <exception>
#include <utility>
#include <variant>

#include "../stdexec/coroutine.hpp"
#include "../stdexec/execution.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "scope.hpp"

_PRAGMA_PUSH()
_PRAGMA_IGNORE("-Wundefined-inline")

namespace exec {
  namespace __task {
    template <class _Ty>
      concept __stop_token_provider =
        requires(const _Ty& t) {
          stdexec::get_stop_token(t);
        };

    template <class _Ty>
      concept __indirect_stop_token_provider =
        requires(const _Ty& t) {
          { stdexec::get_env(t) } -> __stop_token_provider;
        };

    struct __forward_stop_request {
      stdexec::in_place_stop_source& __stop_source_;
      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _ParentPromise>
      struct __default_awaiter_context;

    ////////////////////////////////////////////////////////////////////////////////
    // This is the context that is associated with basic_task's promise type
    // by default. It handles forwarding of stop requests from parent to child.
    class __default_task_context_impl {
      template <class _ParentPromise>
        friend struct __default_awaiter_context;

      stdexec::in_place_stop_token __stop_token_;

      friend auto tag_invoke(
            stdexec::get_stop_token_t, const __default_task_context_impl& __self)
          noexcept -> stdexec::in_place_stop_token {
        return __self.__stop_token_;
      }

    public:
      __default_task_context_impl() = default;

      bool stop_requested() const noexcept {
        return __stop_token_.stop_requested();
      }

      template <class _ThisPromise>
        using promise_context_t = __default_task_context_impl;

      template <class _ThisPromise, class _ParentPromise = void>
        using awaiter_context_t = __default_awaiter_context<_ParentPromise>;
    };

    template <class _Ty>
      using default_task_context = __default_task_context_impl;

    // This is the context associated with basic_task's awaiter. By default
    // it does nothing.
    template <class _ParentPromise>
      struct __default_awaiter_context {
        explicit __default_awaiter_context(
            __default_task_context_impl&, _ParentPromise&) noexcept
        {}
      };

    ////////////////////////////////////////////////////////////////////////////////
    // This is the context to be associated with basic_task's awaiter when
    // the parent coroutine's promise type is known, is a __stop_token_provider,
    // and its stop token type is neither in_place_stop_token nor unstoppable.
    template <__indirect_stop_token_provider _ParentPromise>
      struct __default_awaiter_context<_ParentPromise> {
        using __stop_token_t = stdexec::stop_token_of_t<stdexec::env_of_t<_ParentPromise>>;
        using __stop_callback_t =
          typename __stop_token_t::template callback_type<__forward_stop_request>;

        explicit __default_awaiter_context(
            __default_task_context_impl& __self, _ParentPromise& __parent) noexcept
            // Register a callback that will request stop on this basic_task's
            // stop_source when stop is requested on the parent coroutine's stop
            // token.
          : __stop_callback_{
              stdexec::get_stop_token(stdexec::get_env(__parent)),
              __forward_stop_request{__stop_source_}} {
          static_assert(std::is_nothrow_constructible_v<
              __stop_callback_t, __stop_token_t, __forward_stop_request>);
          __self.__stop_token_ = __stop_source_.get_token();
        }

        stdexec::in_place_stop_source __stop_source_{};
        __stop_callback_t __stop_callback_;
      };

    // If the parent coroutine's type has a stop token of type in_place_stop_token,
    // we don't need to register a stop callback.
    template <__indirect_stop_token_provider _ParentPromise>
        requires std::same_as<
            stdexec::in_place_stop_token,
            stdexec::stop_token_of_t<stdexec::env_of_t<_ParentPromise>>>
      struct __default_awaiter_context<_ParentPromise> {
        explicit __default_awaiter_context(
            __default_task_context_impl& __self, _ParentPromise& __parent) noexcept {
          __self.__stop_token_ =
            stdexec::get_stop_token(stdexec::get_env(__parent));
        }
      };

    // If the parent coroutine's stop token is unstoppable, there's no point
    // forwarding stop tokens or stop requests at all.
    template <__indirect_stop_token_provider _ParentPromise>
        requires stdexec::unstoppable_token<
            stdexec::stop_token_of_t<stdexec::env_of_t<_ParentPromise>>>
      struct __default_awaiter_context<_ParentPromise> {
        explicit __default_awaiter_context(
            __default_task_context_impl&, _ParentPromise&) noexcept
        {}
      };

    // Finally, if we don't know the parent coroutine's promise type, assume the
    // worst and save a type-erased stop callback.
    template<>
      struct __default_awaiter_context<void> {
        template <class _Ty>
          explicit __default_awaiter_context(
            __default_task_context_impl&, _Ty&) noexcept
          {}

        template <__indirect_stop_token_provider _ParentPromise>
          explicit __default_awaiter_context(
              __default_task_context_impl& __self, _ParentPromise& __parent) {
            // Register a callback that will request stop on this basic_task's
            // stop_source when stop is requested on the parent coroutine's stop
            // token.
            using __stop_token_t =
              stdexec::stop_token_of_t<
                stdexec::env_of_t<_ParentPromise>>;
            using __stop_callback_t =
              typename __stop_token_t::template callback_type<__forward_stop_request>;

            if constexpr (std::same_as<__stop_token_t, stdexec::in_place_stop_token>) {
              __self.__stop_token_ =
                stdexec::get_stop_token(stdexec::get_env(__parent));
            } else if(auto __token =
                        stdexec::get_stop_token(
                          stdexec::get_env(__parent));
                      __token.stop_possible()) {
              __stop_callback_.emplace<__stop_callback_t>(
                  std::move(__token), __forward_stop_request{__stop_source_});
              __self.__stop_token_ = __stop_source_.get_token();
            }
          }

        stdexec::in_place_stop_source __stop_source_{};
        std::any __stop_callback_{};
      };

    template <class _Promise, class _ParentPromise = void>
      using awaiter_context_t =
        typename stdexec::env_of_t<_Promise>::
          template awaiter_context_t<_Promise, _ParentPromise>;

    ////////////////////////////////////////////////////////////////////////////////
    // In a base class so it can be specialized when _Ty is void:
    template <class _Ty>
      struct __promise_base {
        void return_value(_Ty value) noexcept {
          __data_.template emplace<1>(std::move(value));
        }
        std::variant<std::monostate, _Ty, std::exception_ptr> __data_{};
      };

    template<>
      struct __promise_base<void> {
        struct __void {};
        void return_void() noexcept {
          __data_.template emplace<1>(__void{});
        }
        std::variant<std::monostate, __void, std::exception_ptr> __data_{};
      };

    ////////////////////////////////////////////////////////////////////////////////
    // basic_task
    template <class _Ty, class _Context = default_task_context<_Ty>>
      class basic_task {
        struct __promise;
       public:
        using promise_type = __promise;

        basic_task(basic_task&& __that) noexcept
          : __coro_(std::exchange(__that.__coro_, {}))
        {}

        ~basic_task() {
          if (__coro_)
            __coro_.destroy();
        }

       private:
        struct __final_awaitable {
          static std::false_type await_ready() noexcept {
            return {};
          }
          static __coro::coroutine_handle<>
          await_suspend(__coro::coroutine_handle<__promise> __h) noexcept {
            return __h.promise().continuation();
          }
          static void await_resume() noexcept {
          }
        };

        struct __promise
          : __promise_base<_Ty>
          , stdexec::with_awaitable_senders<__promise> {
          basic_task get_return_object() noexcept {
            return basic_task(__coro::coroutine_handle<__promise>::from_promise(*this));
          }
          __coro::suspend_always initial_suspend() noexcept {
            return {};
          }
          __final_awaitable final_suspend() noexcept {
            return {};
          }
          void unhandled_exception() noexcept {
            this->__data_.template emplace<2>(std::current_exception());
          }
          using __context_t =
            typename _Context::template promise_context_t<__promise>;
          friend __context_t tag_invoke(stdexec::get_env_t, const __promise& __self) {
            return __self.__context_;
          }
          __context_t __context_;
        };

        template <class _ParentPromise = void>
          struct __task_awaitable {
            __coro::coroutine_handle<__promise> __coro_;
            std::optional<awaiter_context_t<__promise, _ParentPromise>> __context_{};

            ~__task_awaitable() {
              if (__coro_)
                __coro_.destroy();
            }

            static std::false_type await_ready() noexcept {
              return {};
            }
            template <class ParentPromise2>
              __coro::coroutine_handle<>
              await_suspend(__coro::coroutine_handle<ParentPromise2> __parent) noexcept {
                static_assert(stdexec::__one_of<_ParentPromise, ParentPromise2, void>);
                __coro_.promise().set_continuation(__parent);
                __context_.emplace(__coro_.promise().__context_, __parent.promise());
                if constexpr (requires { __coro_.promise().stop_requested() ? 0 : 1; }) {
                  if (__coro_.promise().stop_requested())
                    return __parent.promise().unhandled_stopped();
                }
                return __coro_;
              }
            _Ty await_resume() {
              __context_.reset();
              scope_guard __on_exit{
                  [this]() noexcept { std::exchange(__coro_, {}).destroy(); }};
              if (__coro_.promise().__data_.index() == 2)
                std::rethrow_exception(std::get<2>(std::move(__coro_.promise().__data_)));
              if constexpr (!std::is_void_v<_Ty>)
                return std::get<1>(std::move(__coro_.promise().__data_));
            }
          };

        // Make this task awaitable within a particular context:
        template <class _ParentPromise>
            requires std::constructible_from<
                awaiter_context_t<__promise, _ParentPromise>, __promise&, _ParentPromise&>
          friend __task_awaitable<_ParentPromise>
          tag_invoke(stdexec::as_awaitable_t, basic_task&& __self, _ParentPromise&) noexcept {
            return __task_awaitable<_ParentPromise>{std::exchange(__self.__coro_, {})};
          }

        // Make this task generally awaitable:
        friend __task_awaitable<> operator co_await(basic_task&& __self) noexcept
            requires stdexec::__valid<awaiter_context_t, __promise> {
          return __task_awaitable<>{std::exchange(__self.__coro_, {})};
        }

        // From the list of types [_Ty], remove any types that are void, and send
        //   the resulting list to __qf<set_value_t>, which uses the list of types
        //   as arguments of a function type. In other words, set_value_t() if _Ty
        //   is void, and set_value_t(_Ty) otherwise.
        using __set_value_sig_t =
          stdexec::__minvoke<
            stdexec::__remove<void, stdexec::__qf<stdexec::set_value_t>>,
            _Ty>;

        // Specify basic_task's completion signatures
        //   This is only necessary when basic_task is not generally awaitable
        //   owing to constraints imposed by its _Context parameter.
        using __task_traits_t =
          stdexec::completion_signatures<
            __set_value_sig_t,
            stdexec::set_error_t(std::exception_ptr),
            stdexec::set_stopped_t()>;

        friend auto tag_invoke(stdexec::get_completion_signatures_t, const basic_task&, auto)
          -> __task_traits_t;

        explicit basic_task(__coro::coroutine_handle<promise_type> __coro) noexcept
          : __coro_(__coro)
        {}

        __coro::coroutine_handle<promise_type> __coro_;
      };
  } // namespace __task

  template <class _Ty>
    using default_task_context =
      __task::default_task_context<_Ty>;

  template <class _Promise, class _ParentPromise = void>
    using awaiter_context_t =
      __task::awaiter_context_t<_Promise, _ParentPromise>;

  template <class _Ty, class _Context = default_task_context<_Ty>>
    using basic_task =
      __task::basic_task<_Ty, _Context>;

  template <class _Ty>
    using task =
      basic_task<_Ty, default_task_context<_Ty>>;
} // namespace exec

_PRAGMA_POP()
