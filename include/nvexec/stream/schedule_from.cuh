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

  namespace _sched_from {
    template <class CvrefSenderId, class ReceiverId>
    struct receiver_t {
      using Sender = __cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;

      struct __t : stream_receiver_base {
        using __id = receiver_t;
        using storage_t = variant_storage_t<Sender, Env>;

        constexpr static std::size_t memory_allocation_size = sizeof(storage_t);

        operation_state_base_t<ReceiverId>& operation_state_;

        template < __completion_tag Tag, class... As>
        friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          storage_t* storage = static_cast<storage_t*>(self.operation_state_.temp_storage_);
          storage->template emplace<decayed_tuple<Tag, As...>>(Tag(), (As&&) as...);

          visit(
            [&](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&]<class Tag2, class... Bs>(Tag2, Bs&... tas) noexcept {
                  self.operation_state_.propagate_completion_signal(Tag2(), std::move(tas)...);
                },
                tpl);
            },
            *storage);
        }

        friend Env tag_invoke(get_env_t, const __t& self) {
          return self.operation_state_.make_env();
        }
      };
    };

    template <class Sender>
    struct source_sender_t : stream_sender_base {
      template <__decays_to<source_sender_t> Self, receiver Receiver>
      friend auto tag_invoke(connect_t, Self&& self, Receiver&& rcvr)
        -> connect_result_t<__copy_cvref_t<Self, Sender>, Receiver> {
        return connect(((Self&&) self).sender_, (Receiver&&) rcvr);
      }

      friend auto tag_invoke(get_env_t, const source_sender_t& self) //
        noexcept(__nothrow_callable<get_env_t, const Sender&>)
          -> __call_result_t<get_env_t, const Sender&> {
        // TODO - this code is not exercised by any test
        return get_env(self.sndr_);
      }

      template <__decays_to<source_sender_t> _Self, class _Env>
      friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
        -> make_completion_signatures< __copy_cvref_t<_Self, Sender>, _Env>;

      Sender sender_;
    };

    template <class... _Ty>
    using value_completions_t = //
      completion_signatures<set_value_t(__decay_t<_Ty>&&...)>;

    template <class _Ty>
    using error_completions_t = //
      completion_signatures<set_error_t(__decay_t<_Ty>&&)>;
  }

  template <class Scheduler, class SenderId>
  struct schedule_from_sender_t {
    using Sender = stdexec::__t<SenderId>;
    using source_sender_th = _sched_from::source_sender_t<Sender>;

    struct __env {
      context_state_t context_state_;

      template < __one_of<set_value_t, set_stopped_t, set_error_t> _Tag>
      friend Scheduler tag_invoke(get_completion_scheduler_t<_Tag>, const __env& __self) noexcept {
        return {__self.context_state_};
      }
    };

    struct __t : stream_sender_base {
      using __id = schedule_from_sender_t;
      __env env_;
      source_sender_th sndr_;

      template <class Self, class Receiver>
      using receiver_t = //
        stdexec::__t< _sched_from::receiver_t< __cvref_id<Self, Sender>, stdexec::__id<Receiver>>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires sender_to<__copy_cvref_t<Self, source_sender_th>, Receiver>
      friend auto tag_invoke(connect_t, Self&& self, Receiver&& rcvr) -> stream_op_state_t<
        __copy_cvref_t<Self, source_sender_th>,
        receiver_t<Self, Receiver>,
        Receiver> {
        return stream_op_state<__copy_cvref_t<Self, source_sender_th>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Self, Receiver> {
            return receiver_t<Self, Receiver>{{}, stream_provider};
          },
          self.env_.context_state_);
      }

      friend const __env& tag_invoke(get_env_t, const __t& __self) noexcept {
        return __self.env_;
      }

      template <__decays_to<__t> _Self, class _Env>
      friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
        -> make_completion_signatures<
          __copy_cvref_t<_Self, Sender>,
          _Env,
          completion_signatures<set_error_t(cudaError_t)>,
          _sched_from::value_completions_t,
          _sched_from::error_completions_t>;

      __t(context_state_t context_state, Sender sndr)
        : env_{context_state}
        , sndr_{{}, (Sender&&) sndr} {
      }
    };
  };
}
