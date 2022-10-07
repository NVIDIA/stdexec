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
#include "../detail/variant.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

  namespace schedule_from {
    template <class CvrefSenderId, class ReceiverId>
    struct receiver_t {
      using Sender = stdexec::__cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;

      struct __t : stream_receiver_base {
        using __id = receiver_t;
        using storage_t = variant_storage_t<Sender, Env>;

        constexpr static std::size_t memory_allocation_size = sizeof(storage_t);

        operation_state_base_t<ReceiverId>& operation_state_;

        template <
          stdexec::__one_of<stdexec::set_value_t, stdexec::set_error_t, stdexec::set_stopped_t> Tag,
          class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          storage_t* storage = static_cast<storage_t*>(self.operation_state_.temp_storage_);
          storage->template emplace<decayed_tuple<Tag, As...>>(Tag{}, (As&&) as...);

          ::nvexec::visit(
            [&](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&](auto tag, auto&... tas) noexcept {
                  self.operation_state_.template propagate_completion_signal(tag, tas...);
                },
                tpl);
            },
            *storage);
        }

        friend Env tag_invoke(stdexec::get_env_t, const __t& self) {
          return self.operation_state_.make_env();
        }
      };
    };

    template <class Sender>
    struct source_sender_t : stream_sender_base {
      template <stdexec::__decays_to<source_sender_t> Self, stdexec::receiver Receiver>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> stdexec::connect_result_t<stdexec::__copy_cvref_t<Self, Sender>, Receiver> {
        return stdexec::connect(((Self&&) self).sender_, (Receiver&&) rcvr);
      }

      friend auto tag_invoke(stdexec::get_env_t, const source_sender_t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        // TODO - this code is not exercised by any test
        return stdexec::get_env(self.sndr_);
      }

      template <stdexec::__decays_to<source_sender_t> _Self, class _Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env)
        -> stdexec::make_completion_signatures< stdexec::__copy_cvref_t<_Self, Sender>, _Env>;

      Sender sender_;
    };
  }

  template <class Scheduler, class SenderId>
  struct schedule_from_sender_t {
    using Sender = stdexec::__t<SenderId>;
    using source_sender_th = schedule_from::source_sender_t<Sender>;

    struct __env {
      context_state_t context_state_;

      template <
        stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t, stdexec::set_error_t> _Tag>
      friend Scheduler
        tag_invoke(stdexec::get_completion_scheduler_t<_Tag>, const __env& __self) noexcept {
        return {__self.context_state_};
      }
    };

    struct __t : stream_sender_base {
      using __id = schedule_from_sender_t;
      __env env_;
      source_sender_th sndr_;

      template <class Self, class Receiver>
      using receiver_t = //
        stdexec::__t<
          schedule_from::receiver_t< stdexec::__cvref_id<Self, Sender>, stdexec::__id<Receiver>>>;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::sender_to<stdexec::__copy_cvref_t<Self, source_sender_th>, Receiver>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> stream_op_state_t<
        stdexec::__copy_cvref_t<Self, source_sender_th>,
        receiver_t<Self, Receiver>,
        Receiver> {
        return stream_op_state<stdexec::__copy_cvref_t<Self, source_sender_th>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Self, Receiver> {
            return receiver_t<Self, Receiver>{{}, stream_provider};
          },
          self.env_.context_state_);
      }

      friend const __env& tag_invoke(stdexec::get_env_t, const __t& __self) noexcept {
        return __self.env_;
      }

      template <stdexec::__decays_to<__t> _Self, class _Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, _Self&&, _Env)
        -> stdexec::make_completion_signatures<
          stdexec::__copy_cvref_t<_Self, Sender>,
          _Env,
          stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>>;

      __t(context_state_t context_state, Sender sndr)
        : env_{context_state}
        , sndr_{{}, (Sender&&) sndr} {
      }
    };
  };
}
