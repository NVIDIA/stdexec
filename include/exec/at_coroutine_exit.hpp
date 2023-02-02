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
    struct __receiver {
      _Receiver __receiver_;

      template <class... _Args>
        friend void tag_invoke(stdexec::set_value_t, __receiver&& __self,
                              _Args&&... __args) noexcept {
          try {
            stdexec::set_value((_Receiver &&) __self.__receiver_, (_Args &&) __args...);
          } catch (...) {
            stdexec::set_error((_Receiver &&) __self.__receiver_, std::current_exception());
          }
        }

      template <class _Error>
        friend void tag_invoke(stdexec::set_error_t, __receiver&& __self,
                              _Error&& __error) noexcept {
          stdexec::set_error((_Receiver &&) __self.__receiver_, (_Error &&) __error);
        }

      friend [[noreturn]] void tag_invoke(stdexec::set_stopped_t, __receiver&&) noexcept {
        std::terminate();
      }

      friend auto tag_invoke(stdexec::get_env_t, const __receiver& __self) noexcept {
        return stdexec::get_env(__self.__receiver_);
      }
    };

  template <class _Sender>
    class __sender {
    public:
      // TODO remove set_error_t completion signatures from __sender
      using completion_signatures = stdexec::completion_signatures<
            stdexec::set_value_t(),
            stdexec::set_error_t(std::exception_ptr)>;

      explicit __sender(_Sender&& sndr)
      noexcept(stdexec::__nothrow_decay_copyable<_Sender&&>)
      : __sender_((_Sender&&) sndr) {}

    private:
      _Sender __sender_;

      template <stdexec::receiver _Receiver>
          requires stdexec::sender_to<_Sender, __receiver<_Receiver>>
        friend stdexec::connect_result_t<_Sender, __receiver<_Receiver>>
        tag_invoke(stdexec::connect_t, __sender&& __self, _Receiver&& __rcvr) noexcept {
          return stdexec::connect((_Sender &&) __self.__sender_,
                                  __receiver<_Receiver>{(_Receiver &&) __rcvr});
        }

      friend auto tag_invoke(stdexec::get_env_t, const __sender& __self) noexcept {
        return stdexec::get_env(__self.__sender_);
      }
    };

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
