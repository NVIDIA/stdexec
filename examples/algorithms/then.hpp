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

using namespace stdexec::tags;

///////////////////////////////////////////////////////////////////////////////
// then algorithm:
template <class R, class F>
class _then_receiver : public stdexec::receiver_adaptor<_then_receiver<R, F>, R> {
  template <class... As>
  using _completions = stdexec::completion_signatures<
    stdexec::set_value_t(std::invoke_result_t<F, As...>),
    stdexec::set_error_t(std::exception_ptr)
  >;

 public:
  _then_receiver(R r, F f)
    : stdexec::receiver_adaptor<_then_receiver, R>{std::move(r)}
    , f_(std::move(f)) {
  }

  // Customize set_value by invoking the callable and passing the result to the inner receiver
  template <class... As>
    requires stdexec::receiver_of<R, _completions<As...>>
  void set_value(As&&... as) && noexcept {
    STDEXEC_TRY {
      stdexec::set_value(
        std::move(*this).base(), std::invoke(static_cast<F&&>(f_), static_cast<As&&>(as)...));
    }
    STDEXEC_CATCH_ALL {
      stdexec::set_error(std::move(*this).base(), std::current_exception());
    }
  }

 private:
  F f_;
};

template <stdexec::sender S, class F>
struct _then_sender {
  using sender_concept = stdexec::sender_t;

  S s_;
  F f_;

  // Compute the completion signatures
  template <class... Args>
  using _set_value_t =
    stdexec::completion_signatures<stdexec::set_value_t(std::invoke_result_t<F, Args...>)>;

  template <class Env>
  using _completions_t = stdexec::transform_completion_signatures_of<
    S,
    Env,
    stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>,
    _set_value_t
  >;

  template <class Env>
  auto get_completion_signatures(Env&&) && -> _completions_t<Env> {
    return {};
  }

  // Connect:
  template <stdexec::receiver R>
    requires stdexec::sender_to<S, _then_receiver<R, F>>
  auto connect(R r) && {
    return stdexec::connect(
      static_cast<S&&>(s_), _then_receiver<R, F>{static_cast<R&&>(r), static_cast<F&&>(f_)});
  }

  auto get_env() const noexcept -> decltype(auto) {
    return stdexec::get_env(s_);
  }
};

template <stdexec::sender S, class F>
auto then(S s, F f) -> stdexec::sender auto {
  return _then_sender<S, F>{static_cast<S&&>(s), static_cast<F&&>(f)};
}
