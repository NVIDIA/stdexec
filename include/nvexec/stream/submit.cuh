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

namespace nvexec::STDEXEC_STREAM_DETAIL_NS::_submit {

  template <class SenderId, class ReceiverId>
  struct op_state_t {
    using Sender = stdexec::__t<SenderId>;
    using Receiver = stdexec::__t<ReceiverId>;

    struct receiver_t : stream_receiver_base {
      op_state_t* op_state_;

      template <same_as<set_value_t> Tag, class... Args>
        requires __callable<Tag, Receiver, Args...>
      STDEXEC_DEFINE_CUSTOM(void set_value)(this receiver_t&& self, Tag, Args&&... as) noexcept {
        std::unique_ptr<op_state_t> g{self.op_state_};
        return Tag()((Receiver&&) self.op_state_->rcvr_, (Args&&) as...);
      }

      template <same_as<set_error_t> Tag, class Error>
        requires __callable<Tag, Receiver, Error>
      STDEXEC_DEFINE_CUSTOM(void set_error)(this receiver_t&& self, Tag, Error&& err) noexcept {
        std::unique_ptr<op_state_t> g{self.op_state_};
        return Tag()((Receiver&&) self.op_state_->rcvr_, (Error&&) err);
      }

      template <same_as<set_stopped_t> Tag>
        requires __callable<Tag, Receiver>
      STDEXEC_DEFINE_CUSTOM(void set_stopped)(this receiver_t&& self, Tag) noexcept {
        std::unique_ptr<op_state_t> g{self.op_state_};
        return Tag()((Receiver&&) self.op_state_->rcvr_);
      }

      // Forward all receiever queries.
      STDEXEC_DEFINE_CUSTOM(auto get_env)(this const receiver_t& self, get_env_t)
        -> env_of_t<Receiver> {
        return stdexec::get_env((const Receiver&) self.op_state_->rcvr_);
      }
    };

    Receiver rcvr_;
    connect_result_t<Sender, receiver_t> op_state_;

    template <__decays_to<Receiver> CvrefReceiver>
    op_state_t(Sender&& sndr, CvrefReceiver&& rcvr)
      : rcvr_((CvrefReceiver&&) rcvr)
      , op_state_(connect((Sender&&) sndr, receiver_t{{}, this})) {
    }
  };

  struct submit_t {
    template <receiver Receiver, sender_to<Receiver> Sender>
    void operator()(Sender&& sndr, Receiver&& rcvr) const noexcept(false) {
      stdexec::start((new op_state_t<stdexec::__id<Sender>, stdexec::__id<__decay_t<Receiver>>>{
               (Sender&&) sndr, (Receiver&&) rcvr})
              ->op_state_);
    }
  };

}
