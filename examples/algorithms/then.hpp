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

///////////////////////////////////////////////////////////////////////////////
// then algorithm:
template<class R, class F>
class _then_receiver
    : stdexec::receiver_adaptor<_then_receiver<R, F>, R> {
  friend stdexec::receiver_adaptor<_then_receiver, R>;
  F f_;

  template <class... As>
  using _completions =
    stdexec::completion_signatures<
      stdexec::set_value_t(std::invoke_result_t<F, As...>),
      stdexec::set_error_t(std::exception_ptr)>;

  // Customize set_value by invoking the callable and passing the result to the inner receiver
  template<class... As>
    requires stdexec::receiver_of<R, _completions<As...>>
  void set_value(As&&... as) && noexcept try {
    stdexec::set_value(std::move(*this).base(), std::invoke((F&&) f_, (As&&) as...));
  } catch(...) {
    stdexec::set_error(std::move(*this).base(), std::current_exception());
  }

 public:
  _then_receiver(R r, F f)
   : stdexec::receiver_adaptor<_then_receiver, R>{std::move(r)}
   , f_(std::move(f)) {}
};

template<stdexec::sender S, class F>
struct _then_sender {
  S s_;
  F f_;

  // Compute the completion signatures
  template <class... Args>
    using _set_value_t =
      stdexec::completion_signatures<
        stdexec::set_value_t(std::invoke_result_t<F, Args...>)>;

  template <class Env>
    using _completions_t =
      stdexec::make_completion_signatures<S, Env,
        stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
        _set_value_t>;

  template<class Env>
  friend auto tag_invoke(stdexec::get_completion_signatures_t, _then_sender&&, Env)
    -> _completions_t<Env>;

  // Connect:
  template<class R>
    //requires stdexec::receiver_of<R, _completions_t<stdexec::env_of_t<R>>>
    //requires stdexec::receiver_of<R, stdexec::completion_signatures_of_t<S, stdexec::env_of_t<R>>>
  friend auto tag_invoke(stdexec::connect_t, _then_sender&& self, R r)
    -> stdexec::connect_result_t<S, _then_receiver<R, F>> {
      return stdexec::connect(
        (S&&) self.s_, _then_receiver<R, F>{(R&&) r, (F&&) self.f_});
  }
};

template<stdexec::sender S, class F>
stdexec::sender auto then(S s, F f) {
  return _then_sender<S, F>{(S&&) s, (F&&) f};
}
