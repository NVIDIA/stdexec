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
    struct operation_state_t : stream_op_state_base {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = stdexec::env_of_t<Receiver>;
      using variant_t = variant_storage_t<Sender, Env>;

      struct receiver_t {
        operation_state_t &op_state_;

        template <stdexec::__one_of<stdexec::set_value_t,
                                    stdexec::set_error_t,
                                    stdexec::set_stopped_t> Tag,
                  class... As  _NVCXX_CAPTURE_PACK(As)>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          _NVCXX_EXPAND_PACK(As, as,
            Tag{}(std::move(self.op_state_.receiver_), (As&&)as...);
          );
        }

        friend stdexec::env_of_t<stdexec::__t<ReceiverId>>
        tag_invoke(stdexec::get_env_t, const receiver_t& self) {
          return stdexec::get_env(self.operation_state_.receiver_);
        }
      };

      using task_t = continuation_task_t<receiver_t, variant_t>;

      bool owner_{false};
      cudaStream_t stream_{0};
      cudaError_t status_{cudaSuccess};

      queue::task_hub_t* hub_;
      queue::host_ptr<variant_t> storage_;
      task_t *task_;

      ::cuda::std::atomic_flag started_;

      Receiver receiver_;

      using enqueue_receiver = stream_enqueue_receiver<stdexec::__x<Env>, stdexec::__x<variant_t>>;
      using inner_op_state_t = stdexec::connect_result_t<Sender, enqueue_receiver>;
      inner_op_state_t inner_op_;

      cudaStream_t allocate() {
        if (stream_ == 0) {
          owner_ = true;
          status_ = STDEXEC_DBG_ERR(cudaStreamCreate(&stream_));
        }

        return stream_;
      }

      cudaStream_t get_stream() {
        cudaStream_t stream{};

        if constexpr (std::is_base_of_v<stream_op_state_base, inner_op_state_t>) {
          stream = inner_op_.get_stream();
        } else {
          stream = this->allocate();
        }

        return stream;
      }

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        op.started_.test_and_set(::cuda::std::memory_order::relaxed);
        op.stream_ = op.get_stream();

        if (op.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          stdexec::set_error(std::move(op.receiver_), std::move(op.status_));
          return;
        }

        stdexec::start(op.inner_op_);
      }

      operation_state_t(queue::task_hub_t* hub, Sender&& sender, Receiver &&receiver)
        : hub_(hub)
        , storage_(queue::make_host<variant_t>(this->status_))
        , task_(queue::make_host<task_t>(this->status_, receiver_t{*this}, storage_.get()).release())
        , started_(ATOMIC_FLAG_INIT)
        , receiver_((Receiver&&)receiver)
        , inner_op_{
            stdexec::connect(
                (Sender&&)sender,
                enqueue_receiver{
                  stdexec::get_env(receiver_), 
                  storage_.get(), 
                  task_, 
                  hub_->producer()})} {
        if (this->status_ == cudaSuccess) {
          this->status_ = task_->status_;
        }
      }

      ~operation_state_t() {
        if (owner_) {
          STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
          stream_ = 0;
          owner_ = false;
        }
      }

      STDEXEC_IMMOVABLE(operation_state_t);
    };
}

template <class SenderId>
  struct transfer_sender_t : stream_sender_base {
    using Sender = stdexec::__t<SenderId>;

    template <class Self, class Receiver>
      using op_state_th = 
        transfer::operation_state_t<
          stdexec::__x<stdexec::__member_t<Self, Sender>>, 
          stdexec::__x<Receiver>>;

    queue::task_hub_t* hub_;
    Sender sndr_;

    template <stdexec::__decays_to<transfer_sender_t> Self, stdexec::receiver Receiver>
      requires stdexec::sender_to<stdexec::__member_t<Self, Sender>, Receiver>
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
      -> op_state_th<Self, Receiver> {
      return op_state_th<Self, Receiver>{
        self.hub_, 
        (Sender&&)self.sndr_, 
        (Receiver&&)rcvr};
    }

    template <stdexec::__decays_to<transfer_sender_t> Self, class Env>
    friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
      -> stdexec::dependent_completion_signatures<Env>;

    template <stdexec::__decays_to<transfer_sender_t> _Self, class _Env>
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
    friend auto tag_invoke(Tag tag, const transfer_sender_t& self, As&&... as)
      noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
      -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }

    transfer_sender_t(queue::task_hub_t* hub, Sender sndr)
      : hub_(hub)
      , sndr_{(Sender&&)sndr} {
    }
  };

}
