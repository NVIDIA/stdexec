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

// The original idea is taken from libunifex and adapted to stdexec.

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <exception>
#include <type_traits>

#include "stdexec/execution.hpp"
#include "exec/task.hpp"

namespace exec {
namespace __at_coroutine_exit {
struct __die_on_stop_t {
  template <class _Receiver>
    struct __receiver_id {
      struct __t {
        using __id = __receiver_id;
        _Receiver __receiver_;

        template <class... _Args>
          friend void tag_invoke(stdexec::set_value_t, __t&& __self,
                                _Args&&... __args) noexcept {
            if constexpr (stdexec::__nothrow_callable<stdexec::set_value_t, _Receiver&&, _Args&&...>) {
              stdexec::set_value((_Receiver &&) __self.__receiver_, (_Args &&) __args...);
            } else {
              try {
                stdexec::set_value((_Receiver &&) __self.__receiver_, (_Args &&) __args...);
              } catch (...) {
                stdexec::set_error((_Receiver &&) __self.__receiver_, std::current_exception());
              }
            }
          }

        template <class _Error>
          friend void tag_invoke(stdexec::set_error_t, __t&& __self,
                                _Error&& __error) noexcept {
            stdexec::set_error((_Receiver &&) __self.__receiver_, (_Error &&) __error);
          }

        [[noreturn]] friend void tag_invoke(stdexec::set_stopped_t, __t&&) noexcept {
          std::terminate();
        }

        friend auto tag_invoke(stdexec::get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__receiver_);
        }
      };
    };
  template <class _R> using __receiver = stdexec::__t<__receiver_id<_R>>;

  template <class _Sig> struct __return_type {
    using type = _Sig;
  };

  template <class _R, class... _Args> struct __return_type<_R(_Args...)> {
    using type = _R;
  };

  template <class _Sig>
  using __return_type_t = typename __return_type<_Sig>::type;

  template <class _Tag, class _With = stdexec::__> struct __replace_tag {
    template <class _Arg>
    using __f = stdexec::__if_c<std::is_same_v<__return_type_t<_Arg>, _Tag>, _With, _Arg>;
  };

  template <class _Tag>
    struct __remove_signatures_with_tag {
      template <class... _Args>
      using __f =
          stdexec::__mapply<
              stdexec::__remove<stdexec::__, stdexec::__q<stdexec::completion_signatures>>,
              stdexec::__minvoke<stdexec::__transform<__replace_tag<_Tag, stdexec::__>>, _Args...>>;
    };

  template <class _Sender>
    struct __sender_id {
      using __old_sigs = stdexec::__completion_signatures_of_t<_Sender, stdexec::env_of_t<_Sender>>;
      using __completion_signatures_t =
          stdexec::__mapply<__remove_signatures_with_tag<stdexec::set_stopped_t>, __old_sigs>;
      class __t {
      public:
        using __id = __sender_id;

        using completion_signatures = __completion_signatures_t;

        explicit __t(_Sender&& sndr)
        noexcept(stdexec::__nothrow_decay_copyable<_Sender&&>)
        : __sender_((_Sender&&) sndr) {}

      private:
        _Sender __sender_;

        template <stdexec::receiver _Receiver>
            requires stdexec::sender_to<_Sender, __receiver<_Receiver>>
          friend stdexec::connect_result_t<_Sender, __receiver<_Receiver>>
          tag_invoke(stdexec::connect_t, __t&& __self, _Receiver&& __rcvr) noexcept {
            return stdexec::connect((_Sender &&) __self.__sender_,
                                    __receiver<_Receiver>{(_Receiver &&) __rcvr});
          }

        friend stdexec::env_of_t<_Sender> tag_invoke(stdexec::get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__sender_);
        }
      };
    };
  template <class _S> using __sender = stdexec::__t<__sender_id<_S>>;

  template <stdexec::sender _Sender>
    __sender<_Sender> operator()(_Sender&& __sndr) const
    noexcept(stdexec::__nothrow_decay_copyable<_Sender&&>) {
      return __sender<_Sender>((_Sender &&) __sndr);
    }

  template <class _Value> _Value&& operator()(_Value&& __value) const noexcept {
    return (_Value &&) __value;
  }
};
inline constexpr __die_on_stop_t __die_on_stop;

template <class... _Ts>
  class [[nodiscard]] __task {
    struct __promise;
  public:
    using promise_type = __promise;

    explicit __task(__coro::coroutine_handle<__promise> __coro) noexcept
    : __coro_(__coro) {}

    __task(__task&& __that) noexcept
    : __coro_(std::exchange(__that.__coro_, {})) {}

    bool await_ready() const noexcept { return false; }

    template <typename _Promise>
        requires requires (_Promise& __promise, __coro::coroutine_handle<promise_type> __h) {
          { __promise.continuation() } -> std::convertible_to<__coro::coroutine_handle<>>;
          { __promise.set_continuation(__h) };
        }
      bool await_suspend(__coro::coroutine_handle<_Promise> __parent) noexcept {
        __coro_.promise().__coro_ = __parent.promise().continuation();
        __parent.promise().set_continuation(__coro_);
        return false;
      }

    std::tuple<_Ts&...> await_resume() noexcept {
      return std::exchange(__coro_, {}).promise().__args_;
    }

  private:
    struct __final_awaitable {
      static std::false_type await_ready() noexcept {
        return {};
      }

      static __coro::coroutine_handle<>
      await_suspend(__coro::coroutine_handle<__promise> __h) noexcept {
        auto __coro = __h.promise().__coro_;
        __h.destroy();
        return __coro;
      }

      void await_resume() const noexcept {
      }
    };

    struct __promise : stdexec::with_awaitable_senders<__promise> {
      template <typename _Action>
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
        return __coro::coroutine_handle<__promise>::from_promise(*this);
      }

      __task get_return_object() noexcept {
        return __task(__coro::coroutine_handle<__promise>::from_promise(*this));
      }

      template <class _Awaitable>
        decltype(auto) await_transform(_Awaitable&& __awaitable) noexcept {
          return stdexec::as_awaitable(__die_on_stop((_Awaitable &&) __awaitable), *this);
        }

      __coro::coroutine_handle<> __coro_{};
      std::tuple<_Ts&...> __args_{};
    };

    __coro::coroutine_handle<__promise> __coro_;
  };

struct __at_coroutine_exit_t {
private:
  template <typename _Action, typename... _Ts>
    static __task<_Ts...> at_coroutine_exit(_Action&& __action, _Ts&&... __ts) {
      co_await ((_Action&&) __action)((_Ts&&) __ts...);
    }

public:
  template <typename _Action, typename... _Ts>
      requires stdexec::__callable<_Action, _Ts...>
    __task<_Ts...> operator()(_Action&& __action, _Ts&&... __ts) const {
      return __at_coroutine_exit_t::at_coroutine_exit((_Action &&) __action,
                                                    (_Ts &&) __ts...);
    }
};
inline constexpr __at_coroutine_exit_t at_coroutine_exit{};
} // namespace __at_coroutine_exit
using __at_coroutine_exit::at_coroutine_exit;
} // namespace exec
