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

namespace exec = std::execution;

///////////////////////////////////////////////////////////////////////////////
// then algorithm:
template<class R, class F>
class _then_receiver
    : exec::receiver_adaptor<_then_receiver<R, F>, R> {
  friend exec::receiver_adaptor<_then_receiver, R>;
  F f_;

  template <class... As>
  using _completions =
    exec::completion_signatures<
      exec::set_value_t(std::invoke_result_t<F, As...>),
      exec::set_error_t(std::exception_ptr)>;

  // Customize set_value by invoking the callable and passing the result to the inner receiver
  template<class... As>
    requires exec::receiver_of<R, _completions<As...>>
  void set_value(As&&... as) && noexcept try {
    exec::set_value(std::move(*this).base(), std::invoke((F&&) f_, (As&&) as...));
  } catch(...) {
    exec::set_error(std::move(*this).base(), std::current_exception());
  }

 public:
  _then_receiver(R r, F f)
   : exec::receiver_adaptor<_then_receiver, R>{std::move(r)}
   , f_(std::move(f)) {}
};

template<exec::sender S, class F>
struct _then_sender {
  S s_;
  F f_;

  // Connect:
  template<class R>
    requires exec::sender_to<S, _then_receiver<R, F>>
  friend auto tag_invoke(exec::connect_t, _then_sender&& self, R r)
    -> exec::connect_result_t<S, _then_receiver<R, F>> {
      return exec::connect(
        (S&&) self.s_, _then_receiver<R, F>{(R&&) r, (F&&) self.f_});
  }

  // Compute the completion_signatures
  template <class...Args>
    using _set_value = exec::set_value_t(std::invoke_result_t<F, Args...>);

  template<class Env>
  friend auto tag_invoke(exec::get_completion_signatures_t, _then_sender&&, Env)
    -> exec::make_completion_signatures<S, Env,
        exec::completion_signatures<exec::set_error_t(std::exception_ptr)>,
        _set_value>;
};

template<exec::sender S, class F>
exec::sender auto then(S s, F f) {
  return _then_sender<S, F>{(S&&) s, (F&&) f};
}
