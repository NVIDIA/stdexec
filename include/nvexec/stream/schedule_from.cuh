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

namespace nvexec {
  struct CANNOT_DISPATCH_THE_SCHEDULE_FROM_ALGORITHM_TO_THE_CUDA_STREAM_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_CUDA_STREAM_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_CUDA_STREAM_SCHEDULER_BEFORE_THE_SCHEDULE_FROM_ALGORITHM;

  namespace _strm {

    namespace _schfr {
      template <class CvrefSenderId, class ReceiverId>
      struct operation_state_t {
        using Sender = __cvref_t<CvrefSenderId>;
        using Receiver = STDEXEC::__t<ReceiverId>;
        using Env = operation_state_base_t<ReceiverId>::env_t;

        struct __t : operation_state_base_t<ReceiverId> {
          struct receiver_t;
          using __id = operation_state_t;
          using variant_t = variant_storage_t<Sender, Env>;
          using task_t = continuation_task_t<receiver_t, variant_t>;
          using enqueue_receiver =
            STDEXEC::__t<stream_enqueue_receiver<STDEXEC::__cvref_id<Env>, variant_t>>;
          using inner_op_state_t = connect_result_t<Sender, enqueue_receiver>;

          struct receiver_t {
            using receiver_concept = STDEXEC::receiver_t;

            template <class... _Args>
            void set_value(_Args&&... __args) noexcept {
              STDEXEC::set_value(std::move(op_state_.rcvr_), static_cast<_Args&&>(__args)...);
            }

            template <class _Error>
            void set_error(_Error&& __err) noexcept {
              STDEXEC::set_error(std::move(op_state_.rcvr_), static_cast<_Error&&>(__err));
            }

            void set_stopped() noexcept {
              STDEXEC::set_stopped(std::move(op_state_.rcvr_));
            }

            [[nodiscard]]
            auto get_env() const noexcept -> Env {
              return op_state_.make_env();
            }

            __t& op_state_;
          };

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

          void start() & noexcept {
            started_.test_and_set(::cuda::std::memory_order::relaxed);

            if (status_ != cudaSuccess) {
              // Couldn't allocate memory for operation state, complete with error
              STDEXEC::set_error(std::move(this->rcvr_), std::move(status_));
              return;
            }

            STDEXEC::start(inner_op_);
          }

          cudaError_t status_{cudaSuccess};
          context_state_t context_state_;
          host_ptr<variant_t> storage_;
          task_t* task_;
          ::cuda::std::atomic_flag started_{};
          host_ptr<__decay_t<Env>> env_{};
          inner_op_state_t inner_op_;
        };
      };
    } // namespace _schfr

    template <class SenderId>
    struct schedule_from_sender_t {
      using Sender = STDEXEC::__t<SenderId>;

      struct __t : stream_sender_base {
        using __id = schedule_from_sender_t;

        template <class Self, class Receiver>
        using op_state_th =
          STDEXEC::__t<_schfr::operation_state_t<__cvref_id<Self, Sender>, STDEXEC::__id<Receiver>>>;

        template <class... Ts>
        using _set_value_t = completion_signatures<set_value_t(STDEXEC::__decay_t<Ts>...)>;

        template <class Ty>
        using _set_error_t = completion_signatures<set_error_t(STDEXEC::__decay_t<Ty>)>;

        template <class Self, class... Env>
        using _completion_signatures_t = transform_completion_signatures<
          __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
          completion_signatures<set_stopped_t(), set_error_t(cudaError_t)>,
          _set_value_t,
          _set_error_t
        >;

        __t(context_state_t context_state, Sender sndr)
          : context_state_(context_state)
          , sndr_{static_cast<Sender&&>(sndr)} {
        }

        template <__decays_to<__t> Self, receiver Receiver>
          requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
        STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
          -> op_state_th<Self, Receiver> {
          return op_state_th<Self, Receiver>{
            static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr), self.context_state_};
        }
        STDEXEC_EXPLICIT_THIS_END(connect)

        template <__decays_to<__t> Self, class... Env>
        static consteval auto get_completion_signatures() //
          -> _completion_signatures_t<Self, Env...> {
          return {};
        }

        auto get_env() const noexcept -> STDEXEC::__fwd_env_t<STDEXEC::env_of_t<Sender>> {
          return STDEXEC::__fwd_env(STDEXEC::get_env(sndr_));
        }

        context_state_t context_state_;
        Sender sndr_;
      };
    };

    template <class Env>
    struct transform_sender_for<STDEXEC::schedule_from_t, Env> {
      template <class Sender>
      using _current_scheduler_t =
        __result_of<get_completion_scheduler<set_value_t>, env_of_t<Sender>, const Env&>;

      template <class Sender>
      auto operator()(__ignore, __ignore, Sender&& sndr) const {
        if constexpr (stream_completing_sender<Sender, Env>) {
          using _sender_t = __t<schedule_from_sender_t<__id<__decay_t<Sender>>>>;
          auto stream_sched = get_completion_scheduler<set_value_t>(get_env(sndr), env_);
          return _sender_t{stream_sched.context_state_, static_cast<Sender&&>(sndr)};
        } else {
          return STDEXEC::__not_a_sender<
            STDEXEC::_WHAT_<>(
              CANNOT_DISPATCH_THE_SCHEDULE_FROM_ALGORITHM_TO_THE_CUDA_STREAM_SCHEDULER),
            STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_CUDA_STREAM_SCHEDULER_IN_THE_ENVIRONMENT),
            STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, STDEXEC::schedule_from_t),
            // STDEXEC::_TO_FIX_THIS_ERROR_(
            //   ADD_A_CONTINUES_ON_TRANSITION_TO_THE_CUDA_STREAM_SCHEDULER_BEFORE_THE_SCHEDULE_FROM_ALGORITHM),
            STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
            STDEXEC::_WITH_ENVIRONMENT_(Env)
          >{};
        }
      }

      const Env& env_;
    };

  } // namespace _strm
} // namespace nvexec

namespace STDEXEC::__detail {
  template <class SenderId>
  inline constexpr __mconst<nvexec::_strm::schedule_from_sender_t<__demangle_t<__t<SenderId>>>>
    __demangle_v<nvexec::_strm::schedule_from_sender_t<SenderId>>{};
} // namespace STDEXEC::__detail
