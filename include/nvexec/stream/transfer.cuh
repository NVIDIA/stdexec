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

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
 
namespace transfer {
  template <class SenderId, class ReceiverId>
    struct operation_state_t : operation_state_base_t<ReceiverId> {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;
      using variant_t = variant_storage_t<Sender, Env>;

      struct receiver_t {
        operation_state_t &op_state_;

        template <stdexec::__one_of<stdexec::set_value_t,
                                    stdexec::set_error_t,
                                    stdexec::set_stopped_t> Tag,
                  class... As >
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          Tag{}(std::move(self.op_state_.receiver_), (As&&)as...);
        }

        friend Env
        tag_invoke(stdexec::get_env_t, const receiver_t& self) {
          return self.operation_state_.make_env();
        }
      };

      using task_t = continuation_task_t<receiver_t, variant_t>;

      cudaError_t status_{cudaSuccess};
      context_state_t context_state_;

      queue::host_ptr<variant_t> storage_;
      task_t *task_;

      ::cuda::std::atomic_flag started_;

      using enqueue_receiver = stdexec::__t<stream_enqueue_receiver<stdexec::__x<Env>, stdexec::__x<variant_t>>>;
      using inner_op_state_t = stdexec::connect_result_t<Sender, enqueue_receiver>;
      inner_op_state_t inner_op_;

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        op.started_.test_and_set(::cuda::std::memory_order::relaxed);

        if (op.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          stdexec::set_error(std::move(op.receiver_), std::move(op.status_));
          return;
        }

        stdexec::start(op.inner_op_);
      }

      operation_state_t(Sender&& sender, Receiver &&receiver, context_state_t context_state)
        : operation_state_base_t<ReceiverId>((Receiver&&)receiver, context_state, true)
        , context_state_(context_state)
        , storage_(queue::make_host<variant_t>(this->status_, context_state.pinned_resource_))
        , task_(queue::make_host<task_t>(this->status_, context_state.pinned_resource_, receiver_t{*this}, storage_.get(), this->get_stream(), context_state.pinned_resource_).release())
        , started_(ATOMIC_FLAG_INIT)
        , inner_op_{
            stdexec::connect(
                (Sender&&)sender,
                enqueue_receiver{
                  this->make_env(), 
                  storage_.get(), 
                  task_, 
                  context_state_.hub_->producer()})} {
        if (this->status_ == cudaSuccess) {
          this->status_ = task_->status_;
        }
      }

      STDEXEC_IMMOVABLE(operation_state_t);
    };
}

template <class SenderId>
  struct transfer_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = transfer_sender_t;

      template <class Self, class Receiver>
        using op_state_th = 
          transfer::operation_state_t<
            stdexec::__x<stdexec::__member_t<Self, Sender>>, 
            stdexec::__id<Receiver>>;

      context_state_t context_state_;
      Sender sndr_;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::sender_to<stdexec::__member_t<Self, Sender>, Receiver>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> op_state_th<Self, Receiver> {
        return op_state_th<Self, Receiver>{
          (Sender&&)self.sndr_, 
          (Receiver&&)rcvr,
          self.context_state_};
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env) ->
          stdexec::make_completion_signatures<
            stdexec::__member_t<_Self, Sender>,
            _Env,
            stdexec::completion_signatures<
              stdexec::set_stopped_t(),
              stdexec::set_error_t(cudaError_t)
            >
          > requires true;

      template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
        requires stdexec::__callable<Tag, const Sender&, As...>
      friend auto tag_invoke(Tag tag, const __t& self, As&&... as)
        noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
        -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
        return ((Tag&&) tag)(self.sndr_, (As&&) as...);
      }

      __t(context_state_t context_state, Sender sndr)
        : context_state_(context_state)
        , sndr_{(Sender&&)sndr} {
      }
    };
  };
}
