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

namespace nvexec::STDEXEC_STREAM_DETAIL_NS::submit {

template <class SenderId, class ReceiverId>
struct op_state_t {
  using Sender = stdexec::__t<SenderId>;
  using Receiver = stdexec::__t<ReceiverId>;
  struct receiver_t : stream_receiver_base {
    op_state_t* op_state_;

    template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
      requires stdexec::__callable<Tag, Receiver, As...>
    friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as)
        noexcept(stdexec::__nothrow_callable<Tag, Receiver, As...>) {
      // Delete the state as cleanup:
      std::unique_ptr<op_state_t> g{self.op_state_};
      return tag((Receiver&&) self.op_state_->rcvr_, (As&&) as...);
    }
    // Forward all receiever queries.
    friend auto tag_invoke(stdexec::get_env_t, const receiver_t& self)
      -> stdexec::env_of_t<Receiver> {
      return stdexec::get_env((const Receiver&) self.op_state_->rcvr_);
    }
  };
  Receiver rcvr_;
  stdexec::connect_result_t<Sender, receiver_t> op_state_;

  template <stdexec::__decays_to<Receiver> CvrefReceiver>
  op_state_t(Sender&& sndr, CvrefReceiver&& rcvr)
    : rcvr_((CvrefReceiver&&) rcvr)
    , op_state_(stdexec::connect((Sender&&) sndr, receiver_t{{}, this}))
  {}
};

struct submit_t {
  template <stdexec::receiver Receiver, stdexec::sender_to<Receiver> Sender>
  void operator()(Sender&& sndr, Receiver&& rcvr) const noexcept(false) {
    stdexec::start((new op_state_t<stdexec::__x<Sender>, stdexec::__x<std::decay_t<Receiver>>>{
        (Sender&&) sndr, (Receiver&&) rcvr})->op_state_);
  }
};

}
