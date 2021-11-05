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

// Pull in the reference implementation of P2300:
#include <execution.hpp>

template <class From, class To>
using _copy_cvref_t = std::__member_t<From, To>;

template <class From, class To>
concept _decays_to = std::same_as<std::decay_t<From>, To>;

///////////////////////////////////////////////////////////////////////////////
// retry algorithm:

// _conv needed so we can emplace construct non-movable types into
// a std::optional.
template<std::invocable F>
  requires std::is_nothrow_move_constructible_v<F>
struct _conv {
  F f_;
  explicit _conv(F f) noexcept : f_((F&&) f) {}
  operator std::invoke_result_t<F>() && {
    return ((F&&) f_)();
  }
};

// pass through all customizations except set_error, which retries the operation.
template<class O, class R>
struct _retry_receiver {
  O* o_;
  explicit _retry_receiver(O* o) : o_(o) {}
  friend void tag_invoke(std::execution::set_error_t, _retry_receiver&& self, auto&&) noexcept {
    self.o_->_retry(); // This causes the op to be retried
  }
  // Forward all other tag_invoke-based CPOs (_copy_cvref_t from P1450):
  template <_decays_to<_retry_receiver> Self, class... As, class Tag>
  friend auto tag_invoke(Tag tag, Self&& self, As&&... as)
    noexcept(std::is_nothrow_invocable_v<Tag, _copy_cvref_t<Self, R>, As...>)
    -> std::invoke_result_t<Tag, _copy_cvref_t<Self, R>, As...> {
    return ((Tag&&) tag)((_copy_cvref_t<Self, R>&&) self.o_->r_, (As&&) as...);
  }
};

template<class S>
struct _retry_sender : std::execution::sender_base {
  S s_;
  explicit _retry_sender(S s) : s_((S&&) s) {}

  // Hold the nested operation state in an optional so we can
  // re-construct and re-start it if the operation fails.
  template<class R>
  struct _op {
    S s_;
    R r_;
    std::optional<
        std::execution::connect_result_t<S&, _retry_receiver<_op, R>>> o_;

    _op(S s, R r): s_((S&&)s), r_((R&&)r), o_{_connect()} {}
    _op(_op&&) = delete;

    auto _connect() noexcept {
      return _conv{[this] {
        return std::execution::connect(s_, _retry_receiver<_op, R>{this});
      }};
    }
    void _retry() noexcept try {
      o_.emplace(_connect()); // potentially throwing
      std::execution::start(*o_);
    } catch(...) {
      std::execution::set_error((R&&) r_, std::current_exception());
    }
    friend void tag_invoke(std::execution::start_t, _op& o) noexcept {
      std::execution::start(*o.o_);
    }
  };

  template<std::execution::receiver R>
    requires std::execution::sender_to<S&, R>
  friend _op<R> tag_invoke(std::execution::connect_t, _retry_sender&& self, R r) {
    return {(S&&) self.s_, (R&&) r};
  }
};

namespace std::execution {
  template <typed_sender S>
  struct sender_traits<_retry_sender<S>> : sender_traits<S> { };
}

template<std::execution::sender S>
std::execution::sender auto retry(S s) {
  return _retry_sender{(S&&) s};
}
