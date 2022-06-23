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

namespace stdex = std::execution;

///////////////////////////////////////////////////////////////////////////////
// then algorithm:
template<class R, class F>
class _then_receiver
    : stdex::receiver_adaptor<_then_receiver<R, F>, R> {
  friend stdex::receiver_adaptor<_then_receiver, R>;
  F f_;

  template <class... As>
  using _completions =
    stdex::completion_signatures<
      stdex::set_value_t(std::invoke_result_t<F, As...>),
      stdex::set_error_t(std::exception_ptr)>;

  // Customize set_value by invoking the callable and passing the result to the inner receiver
  template<class... As>
    requires stdex::receiver_of<R, _completions<As...>>
  void set_value(As&&... as) && noexcept try {
    stdex::set_value(std::move(*this).base(), std::invoke((F&&) f_, (As&&) as...));
  } catch(...) {
    stdex::set_error(std::move(*this).base(), std::current_exception());
  }

 public:
  _then_receiver(R r, F f)
   : stdex::receiver_adaptor<_then_receiver, R>{std::move(r)}
   , f_(std::move(f)) {}
};

struct then_t;

template<stdex::sender S, class F>
struct _then_sender {
  S s_;
  F f_;

  // Compute the completion signatures
  template <class... Args>
    using _set_value_t =
      stdex::completion_signatures<
        stdex::set_value_t(std::invoke_result_t<F, Args...>)>;

  template <class Env>
    using _completions_t =
      stdex::make_completion_signatures<S, Env,
        stdex::completion_signatures<stdex::set_error_t(std::exception_ptr)>,
        _set_value_t>;

  template<class Env>
  friend auto tag_invoke(stdex::get_completion_signatures_t, _then_sender&&, Env)
    -> _completions_t<Env>;

  friend auto tag_invoke(stdex::get_descriptor_t, const _then_sender&)
    -> stdex::sender_descriptor_t<then_t(S)>;

  // Connect:
  template<class R>
    requires stdex::receiver_of<R, _completions_t<stdex::env_of_t<R>>>
  friend auto tag_invoke(stdex::connect_t, _then_sender&& self, R r)
    -> stdex::connect_result_t<S, _then_receiver<R, F>> {
      return stdex::connect(
        (S&&) self.s_, _then_receiver<R, F>{(R&&) r, (F&&) self.f_});
  }
};

struct then_t {
  template<stdex::sender S, class F>
  stdex::sender auto operator()(S s, F f) const {
    return _then_sender<S, F>{(S&&) s, (F&&) f};
  }
};
inline constexpr then_t then {};