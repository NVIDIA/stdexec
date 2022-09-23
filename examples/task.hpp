/*
 * Copyright (c) NVIDIA
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

#include <coroutine.hpp>
#include <execution.hpp>

_PRAGMA_PUSH()
_PRAGMA_IGNORE("-Wundefined-inline")

template <template<class...> class T, class... As>
  concept well_formed =
    requires { typename T<As...>; };

template <class T>
  concept stop_token_provider =
    requires(const T& t) {
      _P2300::execution::get_stop_token(t);
    };

template <class T>
  concept indirect_stop_token_provider =
    requires(const T& t) {
      { _P2300::execution::get_env(t) } -> stop_token_provider;
    };

template <std::invocable Fn>
    requires std::is_nothrow_move_constructible_v<Fn> &&
      std::is_nothrow_invocable_v<Fn>
  struct scope_guard {
    Fn fn_;
    scope_guard(Fn fn) noexcept : fn_((Fn&&) fn) {}
    ~scope_guard() { ((Fn&&) fn_)(); }
  };

struct forward_stop_request {
  _P2300::in_place_stop_source& stop_source_;
  void operator()() noexcept {
    stop_source_.request_stop();
  }
};

////////////////////////////////////////////////////////////////////////////////
// This is the context that is associated with basic_task's promise type
// by default. It handles forwarding of stop requests from parent to child.
class default_task_context_impl {
  _P2300::in_place_stop_token stop_token_;

  // This is the context associated with basic_task's awaiter. By default
  // it does nothing.
  template <class ParentPromise>
    struct awaiter_context {
      explicit awaiter_context(
          default_task_context_impl&, ParentPromise&) noexcept
      {}
    };

  friend auto tag_invoke(
        _P2300::execution::get_stop_token_t, const default_task_context_impl& self)
      noexcept -> _P2300::in_place_stop_token {
    return self.stop_token_;
  }

public:
  default_task_context_impl() = default;

  bool stop_requested() const noexcept {
    return stop_token_.stop_requested();
  }

  template <class ThisPromise>
    using promise_context_t = default_task_context_impl;

  template <class ThisPromise, class ParentPromise = void>
    using awaiter_context_t = awaiter_context<ParentPromise>;
};

////////////////////////////////////////////////////////////////////////////////
// This is the context to be associated with basic_task's awaiter when
// the parent coroutine's promise type is known, is a stop_token_provider,
// and its stop token type is neither in_place_stop_token nor unstoppable.
template <indirect_stop_token_provider ParentPromise>
  struct default_task_context_impl::awaiter_context<ParentPromise> {
    using stop_token_t = _P2300::execution::stop_token_of_t<_P2300::execution::env_of_t<ParentPromise>>;
    using stop_callback_t =
      typename stop_token_t::template callback_type<forward_stop_request>;

    explicit awaiter_context(
        default_task_context_impl& self, ParentPromise& parent) noexcept
        // Register a callback that will request stop on this basic_task's
        // stop_source when stop is requested on the parent coroutine's stop
        // token.
      : stop_callback_{
          _P2300::execution::get_stop_token(_P2300::execution::get_env(parent)),
          forward_stop_request{stop_source_}} {
      static_assert(std::is_nothrow_constructible_v<
          stop_callback_t, stop_token_t, forward_stop_request>);
      self.stop_token_ = stop_source_.get_token();
    }

    _P2300::in_place_stop_source stop_source_{};
    stop_callback_t stop_callback_;
  };

// If the parent coroutine's type has a stop token of type in_place_stop_token,
// we don't need to register a stop callback.
template <indirect_stop_token_provider ParentPromise>
    requires std::same_as<
        _P2300::in_place_stop_token,
        _P2300::execution::stop_token_of_t<_P2300::execution::env_of_t<ParentPromise>>>
  struct default_task_context_impl::awaiter_context<ParentPromise> {
    explicit awaiter_context(
        default_task_context_impl& self, ParentPromise& parent) noexcept {
      self.stop_token_ =
        _P2300::execution::get_stop_token(_P2300::execution::get_env(parent));
    }
  };

// If the parent coroutine's stop token is unstoppable, there's no point
// forwarding stop tokens or stop requests at all.
template <indirect_stop_token_provider ParentPromise>
    requires _P2300::unstoppable_token<
        _P2300::execution::stop_token_of_t<_P2300::execution::env_of_t<ParentPromise>>>
  struct default_task_context_impl::awaiter_context<ParentPromise> {
    explicit awaiter_context(
        default_task_context_impl&, ParentPromise&) noexcept
    {}
  };

// Finally, if we don't know the parent coroutine's promise type, assume the
// worst and save a type-erased stop callback.
template<>
  struct default_task_context_impl::awaiter_context<void> {
    explicit awaiter_context(
        default_task_context_impl& self, auto&) noexcept
    {}

    template <indirect_stop_token_provider ParentPromise>
      explicit awaiter_context(
          default_task_context_impl& self, ParentPromise& parent) {
        // Register a callback that will request stop on this basic_task's
        // stop_source when stop is requested on the parent coroutine's stop
        // token.
        using stop_token_t =
          _P2300::execution::stop_token_of_t<
            _P2300::execution::env_of_t<ParentPromise>>;
        using stop_callback_t =
          typename stop_token_t::template callback_type<forward_stop_request>;

        if constexpr (std::same_as<stop_token_t, _P2300::in_place_stop_token>) {
          self.stop_token_ =
            _P2300::execution::get_stop_token(_P2300::execution::get_env(parent));
        } else if(auto token =
                    _P2300::execution::get_stop_token(
                      _P2300::execution::get_env(parent));
                  token.stop_possible()) {
          stop_callback_.emplace<stop_callback_t>(
              std::move(token), forward_stop_request{stop_source_});
          self.stop_token_ = stop_source_.get_token();
        }
      }

    _P2300::in_place_stop_source stop_source_{};
    std::any stop_callback_{};
  };

template <class ValueType>
  using default_task_context = default_task_context_impl;

template <class Promise, class ParentPromise = void>
  using awaiter_context_t =
    typename _P2300::execution::env_of_t<Promise>::
      template awaiter_context_t<Promise, ParentPromise>;

////////////////////////////////////////////////////////////////////////////////
// In a base class so it can be specialized when T is void:
template <class T>
  struct _promise_base {
    void return_value(T value) noexcept {
      data_.template emplace<1>(std::move(value));
    }
    std::variant<std::monostate, T, std::exception_ptr> data_{};
  };

template<>
  struct _promise_base<void> {
    struct _void {};
    void return_void() noexcept {
      data_.template emplace<1>(_void{});
    }
    std::variant<std::monostate, _void, std::exception_ptr> data_{};
  };

////////////////////////////////////////////////////////////////////////////////
// basic_task
template <class T, class Context = default_task_context<T>>
class basic_task {
  struct _promise;
public:
  using promise_type = _promise;

  basic_task(basic_task&& that) noexcept
    : coro_(std::exchange(that.coro_, {}))
  {}

  ~basic_task() {
    if (coro_)
      coro_.destroy();
  }

private:
  struct _final_awaitable {
    static std::false_type await_ready() noexcept {
      return {};
    }
    static __coro::coroutine_handle<>
    await_suspend(__coro::coroutine_handle<_promise> h) noexcept {
      return h.promise().continuation();
    }
    static void await_resume() noexcept {
    }
  };

  struct _promise
    : _promise_base<T>
    , _P2300::execution::with_awaitable_senders<_promise> {
    basic_task get_return_object() noexcept {
      return basic_task(__coro::coroutine_handle<_promise>::from_promise(*this));
    }
    __coro::suspend_always initial_suspend() noexcept {
      return {};
    }
    _final_awaitable final_suspend() noexcept {
      return {};
    }
    void unhandled_exception() noexcept {
      this->data_.template emplace<2>(std::current_exception());
    }
    using _context_t =
      typename Context::template promise_context_t<_promise>;
    friend _context_t tag_invoke(_P2300::execution::get_env_t, const _promise& self) {
      return self.context_;
    }
    _context_t context_;
  };

  template <class ParentPromise = void>
  struct _task_awaitable {
    __coro::coroutine_handle<_promise> coro_;
    std::optional<awaiter_context_t<_promise, ParentPromise>> context_{};

    ~_task_awaitable() {
      if (coro_)
        coro_.destroy();
    }

    static std::false_type await_ready() noexcept {
      return {};
    }
    template <class ParentPromise2>
    __coro::coroutine_handle<>
    await_suspend(__coro::coroutine_handle<ParentPromise2> parent) noexcept {
      static_assert(std::__one_of<ParentPromise, ParentPromise2, void>);
      coro_.promise().set_continuation(parent);
      context_.emplace(coro_.promise().context_, parent.promise());
      if constexpr (requires { coro_.promise().stop_requested() ? 0 : 1; }) {
        if (coro_.promise().stop_requested())
          return parent.promise().unhandled_stopped();
      }
      return coro_;
    }
    T await_resume() {
      context_.reset();
      scope_guard on_exit{
          [this]() noexcept { std::exchange(coro_, {}).destroy(); }};
      if (coro_.promise().data_.index() == 2)
        std::rethrow_exception(std::get<2>(std::move(coro_.promise().data_)));
      if constexpr (!std::is_void_v<T>)
        return std::get<1>(std::move(coro_.promise().data_));
    }
  };

  // Make this task awaitable within a particular context:
  template <class ParentPromise>
    requires std::constructible_from<
        awaiter_context_t<_promise, ParentPromise>, _promise&, ParentPromise&>
  friend _task_awaitable<ParentPromise>
  tag_invoke(_P2300::execution::as_awaitable_t, basic_task&& self, ParentPromise&) noexcept {
    return _task_awaitable<ParentPromise>{std::exchange(self.coro_, {})};
  }

  // Make this task generally awaitable:
  friend _task_awaitable<> operator co_await(basic_task&& self) noexcept
      requires well_formed<awaiter_context_t, _promise> {
    return _task_awaitable<>{std::exchange(self.coro_, {})};
  }

  // Specify basic_task's completion signatures
  //   This is only necessary when basic_task is not generally awaitable
  //   owing to constraints imposed by its Context parameter.
  template <class... Ts>
    using _task_traits_t =
      _P2300::execution::completion_signatures<
        _P2300::execution::set_value_t(Ts...),
        _P2300::execution::set_error_t(std::exception_ptr),
        _P2300::execution::set_stopped_t()>;

  friend auto tag_invoke(_P2300::execution::get_completion_signatures_t, const basic_task&, auto)
    -> std::conditional_t<std::is_void_v<T>, _task_traits_t<>, _task_traits_t<T>>;

  explicit basic_task(__coro::coroutine_handle<promise_type> __coro) noexcept
    : coro_(__coro)
  {}

  __coro::coroutine_handle<promise_type> coro_;
};

template <class T>
  using task = basic_task<T, default_task_context<T>>;

_PRAGMA_POP()
