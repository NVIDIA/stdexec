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
#include <utility>

#include "../detail/cuda_atomic.cuh" // IWYU pragma: keep

#include "common.cuh"

namespace nvexec::_strm {

  namespace _continues_on {
    template <class CvrefSenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = __cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;

      struct __t : operation_state_base_t<ReceiverId> {
        using __id = operation_state_t;
        using variant_t = variant_storage_t<Sender, Env>;

        struct receiver_t {
          __t& op_state_;

          template <class... _Args>
          void set_value(_Args&&... __args) noexcept {
            stdexec::set_value(std::move(op_state_.rcvr_), static_cast<_Args&&>(__args)...);
          }

          template <class _Error>
          void set_error(_Error&& __err) noexcept {
            stdexec::set_error(std::move(op_state_.rcvr_), static_cast<_Error&&>(__err));
          }

          void set_stopped() noexcept {
            stdexec::set_stopped(std::move(op_state_.rcvr_));
          }

          [[nodiscard]]
          auto get_env() const noexcept -> Env {
            return op_state_.make_env();
          }
        };

        using task_t = continuation_task_t<receiver_t, variant_t>;

        cudaError_t status_{cudaSuccess};
        context_state_t context_state_;

        host_ptr<variant_t> storage_;
        task_t* task_;

        ::cuda::std::atomic_flag started_{};

        using enqueue_receiver =
          stdexec::__t<stream_enqueue_receiver<stdexec::__cvref_id<Env>, variant_t>>;
        using inner_op_state_t = connect_result_t<Sender, enqueue_receiver>;
        host_ptr<__decay_t<Env>> env_{};
        inner_op_state_t inner_op_;

        void start() & noexcept {
          started_.test_and_set(::cuda::std::memory_order::relaxed);

          if (status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            stdexec::set_error(std::move(this->rcvr_), std::move(status_));
            return;
          }

          stdexec::start(inner_op_);
        }

        __t(Sender&& sender, Receiver&& rcvr, context_state_t context_state)
          : operation_state_base_t<ReceiverId>(static_cast<Receiver&&>(rcvr), context_state)
          , context_state_(context_state)
          , storage_(make_host<variant_t>(this->status_, context_state.pinned_resource_))
          , task_(
              make_host<task_t>(
                this->status_,
                context_state.pinned_resource_,
                receiver_t{*this},
                storage_.get(),
                this->get_stream(),
                context_state.pinned_resource_)
                .release())
          , env_(make_host(this->status_, context_state_.pinned_resource_, this->make_env()))
          , inner_op_{connect(
              static_cast<Sender&&>(sender),
              enqueue_receiver{
                env_.get(),
                storage_.get(),
                task_,
                context_state_.hub_->producer()})} {
          if (this->status_ == cudaSuccess) {
            this->status_ = task_->status_;
          }
        }

        STDEXEC_IMMOVABLE(__t);
      };
    };
  } // namespace _continues_on

  template <class Scheduler, class SenderId>
  struct continues_on_sender_t {
    using Sender = stdexec::__t<SenderId>;
    using LateDomain = __detail::__early_domain_of_t<Sender, __none_such>;

    struct __t : stream_sender_base {
      using __id = continues_on_sender_t;

      template <class Self, class Receiver>
      using op_state_th = stdexec::__t<
        _continues_on::operation_state_t<__cvref_id<Self, Sender>, stdexec::__id<Receiver>>
      >;

      Scheduler sched_;
      context_state_t context_state_;
      Sender sndr_;

      template <class... Ts>
      using _set_value_t = completion_signatures<set_value_t(stdexec::__decay_t<Ts>...)>;

      template <class Ty>
      using _set_error_t = completion_signatures<set_error_t(stdexec::__decay_t<Ty>)>;

      template <class Self, class... Env>
      using _completion_signatures_t = transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        completion_signatures<set_stopped_t(), set_error_t(cudaError_t)>,
        _set_value_t,
        _set_error_t
      >;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
      static auto connect(Self&& self, Receiver rcvr) -> op_state_th<Self, Receiver> {
        return op_state_th<Self, Receiver>{
          static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr), self.context_state_};
      }

      template <__decays_to<__t> Self, class... Env>
      static auto
        get_completion_signatures(Self&&, Env&&...) -> _completion_signatures_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> __sched_attrs<Scheduler, LateDomain> {
        return {sched_, {}};
      }

      __t(Scheduler sched, context_state_t context_state, Sender sndr)
        : sched_(sched)
        , context_state_(context_state)
        , sndr_{static_cast<Sender&&>(sndr)} {
      }
    };
  };

  template <>
  struct transform_sender_for<stdexec::continues_on_t> {
    template <class Sender>
    using _current_scheduler_t =
      __result_of<get_completion_scheduler<set_value_t>, env_of_t<Sender>>;

    template <class Sched, class Sender>
      requires gpu_stream_scheduler<_current_scheduler_t<Sender>>
    auto operator()(__ignore, Sched sched, Sender&& sndr) const {
      using _sender_t = __t<continues_on_sender_t<Sched, __id<__decay_t<Sender>>>>;
      auto stream_sched = get_completion_scheduler<set_value_t>(get_env(sndr));
      return schedule_from(
        static_cast<Sched&&>(sched),
        _sender_t{sched, stream_sched.context_state_, static_cast<Sender&&>(sndr)});
    }
  };

} // namespace nvexec::_strm

namespace stdexec::__detail {
  template <class Scheduler, class SenderId>
  inline constexpr __mconst<
    nvexec::_strm::continues_on_sender_t<Scheduler, __name_of<__t<SenderId>>>
  >
    __name_of_v<nvexec::_strm::continues_on_sender_t<Scheduler, SenderId>>{};
} // namespace stdexec::__detail
