/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../../stdexec/execution.hpp"
#include <type_traits>

#include "common.cuh"

namespace nvexec {
namespace STDEXEC_STREAM_DETAIL_NS {

template <class ReceiverId, class Fun>
struct launch_receiver_t {
  using Receiver = stdexec::__t<ReceiverId>;

  class __t : stream_receiver_base {
    operation_state_base_t<ReceiverId>& op_state_;
    Fun f_;

   public:
    using __id = launch_receiver_t;

    template <class... As>
    friend void tag_invoke(stdexec::set_value_t, __t&& self, As&&... as) noexcept
      requires std::invocable<Fun, std::decay_t<As>...>
    {
      try {
        std::invoke(std::move(self.f_), self.op_state_.get_stream(), as...);
        self.op_state_.propagate_completion_signal(stdexec::set_value, (As&&) as...);
      } catch (...) {
        self.op_state_.propagate_completion_signal(stdexec::set_error, std::current_exception());
      }
    }

    template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
    friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
      self.op_state_.propagate_completion_signal(tag, (As&&) as...);
    }

    friend typename operation_state_base_t<ReceiverId>::env_t
      tag_invoke(stdexec::get_env_t, const __t& self) {
      return self.op_state_.make_env();
    }

    explicit __t(operation_state_base_t<ReceiverId>& op_state, Fun fun)
      : op_state_(op_state)
      , f_((Fun&&) fun) {
    }
  };
};

template <class SenderId, class Fun>
struct launch_sender_t {
  using Sender = stdexec::__t<SenderId>;

  struct __t : stream_sender_base {
    using __id = launch_sender_t;

    Sender sndr_;
    Fun fun_;

    template <class Receiver>
    using receiver_t = stdexec::__t<launch_receiver_t<stdexec::__id<Receiver>, Fun>>;

    template <class... As>
    using set_value_t = stdexec::completion_signatures< stdexec::set_value_t(As...)>;

    using set_error_t = stdexec::completion_signatures< stdexec::set_error_t(std::exception_ptr)>;

    template <class Self, class Env>
    using completion_signatures = //
      stdexec::__make_completion_signatures<
        stdexec::__copy_cvref_t<Self, Sender>,
        Env,
        set_error_t,
        stdexec::__q<set_value_t>>;

    template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
      requires stdexec::receiver_of<
        Receiver,
        completion_signatures<Self, stdexec::env_of_t<Receiver>>>
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<stdexec::__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
      return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
        ((Self&&) self).sndr_,
        (Receiver&&) rcvr,
        [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
          -> receiver_t<Receiver> { return receiver_t<Receiver>(stream_provider, self.fun_); });
    }

    template <stdexec::__decays_to<__t> Self, class Env>
    friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
      -> stdexec::dependent_completion_signatures<Env>;

    template <stdexec::__decays_to<__t> Self, class Env>
    friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env>
      requires true;

    friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
      noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
        -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
      return stdexec::get_env(self.sndr_);
    }
  };
};

struct launch_t {
  template <class Sender, class Fun>
    using sender_t = stdexec::__t<launch_sender_t<
      stdexec::__id<::cuda::std::decay_t<Sender>>, Fun
    >>;

  template <stdexec::sender Sender, stdexec::__movable_value Fun>
    sender_t<Sender, Fun> operator()(Sender&& sndr, Fun&& fun) const {
      return {{}, (Sender&&) sndr, (Fun&&) fun};
    }

  template <stdexec::__movable_value Fun>
    stdexec::__binder_back<launch_t, Fun> operator()(Fun&& fun) const {
      return {{}, {}, (Fun&&) fun};
    }
};

} // namespace STDEXEC_STREAM_DETAIL_NS

inline constexpr STDEXEC_STREAM_DETAIL_NS::launch_t launch{};

} // namespace nvexec

