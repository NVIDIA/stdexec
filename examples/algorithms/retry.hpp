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

// Pull in the reference implementation of P2300:
#include <stdexec/execution.hpp>

template <class From, class To>
using _copy_cvref_t = stdexec::__copy_cvref_t<From, To>;

template <class From, class To>
concept _decays_to = std::same_as<std::decay_t<From>, To>;

///////////////////////////////////////////////////////////////////////////////
// retry algorithm:

// _conv needed so we can emplace construct non-movable types into
// a std::optional.
template <std::invocable F>
  requires std::is_nothrow_move_constructible_v<F>
struct _conv {
  F f_;

  explicit _conv(F f) noexcept
    : f_(static_cast<F&&>(f)) {
  }

  operator std::invoke_result_t<F>() && {
    return static_cast<F&&>(f_)();
  }
};

template <class S, class R>
struct _op;

// pass through all customizations except set_error, which retries the operation.
template <class S, class R>
struct _retry_receiver : stdexec::receiver_adaptor<_retry_receiver<S, R>> {
  _op<S, R>* o_;

  auto base() && noexcept -> R&& {
    return static_cast<R&&>(o_->r_);
  }

  auto base() const & noexcept -> const R& {
    return o_->r_;
  }

  explicit _retry_receiver(_op<S, R>* o)
    : o_(o) {
  }

  template <class Error>
  void set_error(Error&&) && noexcept {
    o_->_retry(); // This causes the op to be retried
  }
};

// Hold the nested operation state in an optional so we can
// re-construct and re-start it if the operation fails.
template <class S, class R>
struct _op {
  S s_;
  R r_;
  std::optional<stdexec::connect_result_t<S&, _retry_receiver<S, R>>> o_;

  _op(S s, R r)
    : s_(static_cast<S&&>(s))
    , r_(static_cast<R&&>(r))
    , o_{_connect()} {
  }

  _op(_op&&) = delete;

  auto _connect() noexcept {
    return _conv{[this] { return stdexec::connect(s_, _retry_receiver<S, R>{this}); }};
  }

  void _retry() noexcept {
    STDEXEC_TRY {
      o_.emplace(_connect()); // potentially throwing
      stdexec::start(*o_);
    }
    STDEXEC_CATCH_ALL {
      stdexec::set_error(static_cast<R&&>(r_), std::current_exception());
    }
  }

  void start() & noexcept {
    stdexec::start(*o_);
  }
};

template <class S>
struct _retry_sender {
  using sender_concept = stdexec::sender_t;
  S s_;

  explicit _retry_sender(S s)
    : s_(static_cast<S&&>(s)) {
  }

  template <class>
  using _error = stdexec::completion_signatures<>;
  template <class... Ts>
  using _value = stdexec::completion_signatures<stdexec::set_value_t(Ts...)>;

  template <class Self, class... Env>
  static consteval auto get_completion_signatures() -> stdexec::transform_completion_signatures<
    stdexec::completion_signatures_of_t<S&, Env...>,
    stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
    _value,
    _error
  > {
    return {};
  }

  template <stdexec::receiver R>
  auto connect(R r) && -> _op<S, R> {
    return {static_cast<S&&>(s_), static_cast<R&&>(r)};
  }

  auto get_env() const noexcept -> stdexec::env_of_t<S> {
    return stdexec::get_env(s_);
  }
};

template <stdexec::sender S>
auto retry(S s) -> stdexec::sender auto {
  return _retry_sender{static_cast<S&&>(s)};
}
