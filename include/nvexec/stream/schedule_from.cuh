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
      template <class Sender, class Receiver>
      struct opstate : _strm::opstate_base<Receiver> {
        using env_t = _strm::opstate_base<Receiver>::env_t;
        struct receiver;
        using variant_t = variant_storage_t<Sender, env_t>;
        using task_t = continuation_task<receiver, variant_t>;
        using enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
        using inner_opstate_t = connect_result_t<Sender, enqueue_receiver_t>;

        struct receiver {
          using receiver_concept = STDEXEC::receiver_t;

          template <class... Args>
          void set_value(Args&&... args) noexcept {
            STDEXEC::set_value(std::move(opstate_.rcvr_), static_cast<Args&&>(args)...);
          }

          template <class Error>
          void set_error(Error&& __err) noexcept {
            STDEXEC::set_error(std::move(opstate_.rcvr_), static_cast<Error&&>(__err));
          }

          void set_stopped() noexcept {
            STDEXEC::set_stopped(std::move(opstate_.rcvr_));
          }

          [[nodiscard]]
          auto get_env() const noexcept -> env_t {
            return opstate_.make_env();
          }

          opstate& opstate_;
        };

        opstate(Sender&& sender, Receiver&& rcvr, context ctx)
          : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), ctx)
          , ctx_(ctx)
          , storage_(host_allocate<variant_t>(this->status_, ctx.pinned_resource_))
          , task_(
              host_allocate<task_t>(
                this->status_,
                ctx.pinned_resource_,
                receiver{*this},
                storage_.get(),
                this->get_stream(),
                ctx.pinned_resource_)
                .release())
          , env_(host_allocate(this->status_, ctx_.pinned_resource_, this->make_env()))
          , inner_op_{connect(
              static_cast<Sender&&>(sender),
              enqueue_receiver_t{env_.get(), storage_.get(), task_, ctx_.hub_->producer()})} {
          if (this->status_ == cudaSuccess) {
            this->status_ = task_->status_;
          }
        }

        STDEXEC_IMMOVABLE(opstate);

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
        context ctx_;
        host_ptr_t<variant_t> storage_;
        task_t* task_;
        ::cuda::std::atomic_flag started_{};
        host_ptr_t<__decay_t<env_t>> env_{};
        inner_opstate_t inner_op_;
      };
    } // namespace _schfr

    template <class Sender>
    struct schedule_from_sender : stream_sender_base {
      template <class Self, class Receiver>
      using opstate_t = _schfr::opstate<__copy_cvref_t<Self, Sender>, Receiver>;

      template <class... Ts>
      using _set_value_t = completion_signatures<set_value_t(__decay_t<Ts>...)>;

      template <class Ty>
      using _set_error_t = completion_signatures<set_error_t(__decay_t<Ty>)>;

      template <class Self, class... Env>
      using _completions_t = transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        completion_signatures<set_stopped_t(), set_error_t(cudaError_t)>,
        _set_value_t,
        _set_error_t
      >;

      schedule_from_sender(context ctx, Sender sndr)
        : ctx_(ctx)
        , sndr_{static_cast<Sender&&>(sndr)} {
      }

      template <__decays_to<schedule_from_sender> Self, STDEXEC::receiver Receiver>
        requires receiver_of<Receiver, _completions_t<Self, env_of_t<Receiver>>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        -> opstate_t<Self, Receiver> {
        return opstate_t<Self, Receiver>{
          static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr), self.ctx_};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<schedule_from_sender> Self, class... Env>
      static consteval auto get_completion_signatures() //
        -> _completions_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> STDEXEC::__fwd_env_t<STDEXEC::env_of_t<Sender>> {
        return STDEXEC::__fwd_env(STDEXEC::get_env(sndr_));
      }

      context ctx_;
      Sender sndr_;
    };

    template <class Env>
    struct transform_sender_for<STDEXEC::schedule_from_t, Env> {
      template <class Sender>
      using _current_scheduler_t =
        __result_of<get_completion_scheduler<set_value_t>, env_of_t<Sender>, const Env&>;

      template <class Sender>
      auto operator()(__ignore, __ignore, Sender&& sndr) const {
        if constexpr (stream_completing_sender<Sender, Env>) {
          using _sender_t = schedule_from_sender<__decay_t<Sender>>;
          auto stream_sched = get_completion_scheduler<set_value_t>(get_env(sndr), env_);
          return _sender_t{stream_sched.ctx_, static_cast<Sender&&>(sndr)};
        } else {
          return STDEXEC::__not_a_sender<
            STDEXEC::_WHAT_(
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
  template <class Sender>
  extern __declfn_t<nvexec::_strm::schedule_from_sender<__demangle_t<Sender>>>
    __demangle_v<nvexec::_strm::schedule_from_sender<Sender>>;
} // namespace STDEXEC::__detail
