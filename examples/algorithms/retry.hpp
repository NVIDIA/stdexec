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
    : f_((F&&) f) {
  }

  operator std::invoke_result_t<F>() && {
    return ((F&&) f_)();
  }
};

template <class S, class R>
struct _op;

// pass through all customizations except set_error, which retries the operation.
template <class S, class R>
struct _retry_receiver : stdexec::receiver_adaptor<_retry_receiver<S, R>> {
  _op<S, R>* o_;

  R&& base() && noexcept {
    return (R&&) o_->r_;
  }

  const R& base() const & noexcept {
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
  std::optional< stdexec::connect_result_t<S&, _retry_receiver<S, R>>> o_;

  _op(S s, R r)
    : s_((S&&) s)
    , r_((R&&) r)
    , o_{_connect()} {
  }

  _op(_op&&) = delete;

  auto _connect() noexcept {
    return _conv{[this] {
      return stdexec::connect(s_, _retry_receiver<S, R>{this});
    }};
  }

  void _retry() noexcept {
    try {
      o_.emplace(_connect()); // potentially throwing
      stdexec::start(*o_);
    } catch (...) {
      stdexec::set_error((R&&) r_, std::current_exception());
    }
  }

  STDEXEC_DEFINE_CUSTOM(void start)(this _op& o, stdexec::start_t) noexcept {
    stdexec::start(*o.o_);
  }
};

template <class S>
struct _retry_sender {
  using is_sender = void;
  S s_;

  explicit _retry_sender(S s)
    : s_((S&&) s) {
  }

  template <class>
  using _error = stdexec::completion_signatures<>;
  template <class... Ts>
  using _value = stdexec::completion_signatures<stdexec::set_value_t(Ts...)>;

  template <class Env>
  STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(this const _retry_sender&, stdexec::get_completion_signatures_t, Env)
    -> stdexec::make_completion_signatures<
      S&,
      Env,
      stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
      _value,
      _error>;

  template <stdexec::receiver R>
  STDEXEC_DEFINE_CUSTOM(_op<S, R> connect)(this _retry_sender&& self, stdexec::connect_t, R r) {
    return {(S&&) self.s_, (R&&) r};
  }

  STDEXEC_DEFINE_CUSTOM(auto get_env)(this const _retry_sender& self, stdexec::get_env_t) //
    noexcept(noexcept(stdexec::get_env(self.s_)))                                         //
    -> std::invoke_result_t<stdexec::get_env_t, S> {
    return stdexec::get_env(self.s_);
  }
};

template <stdexec::sender S>
stdexec::sender auto retry(S s) {
  return _retry_sender{(S&&) s};
}
