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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"
#include <memory>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS::_submit {

  template <class SenderId, class ReceiverId>
  struct op_state_t {
    using Sender = stdexec::__t<SenderId>;
    using Receiver = stdexec::__t<ReceiverId>;

    struct receiver_t : stream_receiver_base {
      op_state_t* op_state_;

      template <class... As>
        requires __callable<set_value_t, Receiver, As...>
      void set_value(As&&... as) noexcept {
        // Delete the state as cleanup:
        std::unique_ptr<op_state_t> g{op_state_};
        stdexec::set_value(static_cast<Receiver&&>(op_state_->rcvr_), static_cast<As&&>(as)...);
      }

      template <class Error>
        requires __callable<set_error_t, Receiver, Error>
      void set_error(Error&& err) noexcept {
        // Delete the state as cleanup:
        std::unique_ptr<op_state_t> g{op_state_};
        stdexec::set_error(static_cast<Receiver&&>(op_state_->rcvr_), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
        requires __callable<set_stopped_t, Receiver>
      {
        // Delete the state as cleanup:
        std::unique_ptr<op_state_t> g{op_state_};
        stdexec::set_stopped(static_cast<Receiver&&>(op_state_->rcvr_));
      }

      // Forward all receiever queries.
      auto get_env() const noexcept -> env_of_t<Receiver> {
        return stdexec::get_env(op_state_->rcvr_);
      }
    };

    Receiver rcvr_;
    connect_result_t<Sender, receiver_t> op_state_;

    template <__decays_to<Receiver> CvrefReceiver>
    op_state_t(Sender&& sndr, CvrefReceiver&& rcvr)
      : rcvr_(static_cast<CvrefReceiver&&>(rcvr))
      , op_state_(connect(static_cast<Sender&&>(sndr), receiver_t{{}, this})) {
    }
  };

  struct submit_t {
    template <receiver Receiver, sender_to<Receiver> Sender>
    void operator()(Sender&& sndr, Receiver&& rcvr) const noexcept(false) {
      start((new op_state_t<stdexec::__id<Sender>, stdexec::__id<__decay_t<Receiver>>>{
               static_cast<Sender&&>(sndr), static_cast<Receiver&&>(rcvr)})
              ->op_state_);
    }
  };

} // namespace nvexec::STDEXEC_STREAM_DETAIL_NS::_submit
