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
#include "../detail/queue.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

namespace when_all {

enum state_t { started, error, stopped };

struct on_stop_requested {
  stdexec::in_place_stop_source& stop_source_;
  void operator()() noexcept {
    stop_source_.request_stop();
  }
};

template <class Env>
  using env_t =
    exec::make_env_t<Env, exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>>;

template <class...>
  using swallow_values = stdexec::completion_signatures<>;

template <class Env, class... Senders>
  struct traits {
    using __t = stdexec::dependent_completion_signatures<Env>;
  };

template <class TupleT, class... As>
__launch_bounds__(1)
__global__ void copy_kernel(TupleT* tpl, As&&... as) {
  *tpl = decayed_tuple<As...>((As&&)as...);
}

template <class Env, class... Senders>
    requires ((stdexec::__v<stdexec::__count_of<stdexec::set_value_t, Senders, Env>> <= 1) &&...)
  struct traits<Env, Senders...> {
    using non_values =
      stdexec::__concat_completion_signatures_t<
        stdexec::completion_signatures<
          stdexec::set_error_t(cudaError_t),
          stdexec::set_stopped_t()>,
        stdexec::make_completion_signatures<
          Senders,
          Env,
          stdexec::completion_signatures<>,
          swallow_values>...>;
    using values =
      stdexec::__minvoke<
        stdexec::__concat<stdexec::__qf<stdexec::set_value_t>>,
        stdexec::__value_types_of_t<
          Senders,
          Env,
          stdexec::__q<stdexec::__types>,
          stdexec::__single_or<stdexec::__types<>>>...>;
    using __t =
      stdexec::__if_c<
        (stdexec::__sends<stdexec::set_value_t, Senders, Env> &&...),
        stdexec::__minvoke<
          stdexec::__push_back<stdexec::__q<stdexec::completion_signatures>>, non_values, values>,
        non_values>;
  };
}

template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  struct when_all_sender_t : stream_sender_base {
    template <class... Sndrs>
      explicit when_all_sender_t(queue::task_hub_t* hub, Sndrs&&... __sndrs)
        : hub_(hub)
        , sndrs_((Sndrs&&) __sndrs...)
      {}

   private:
    const queue::task_hub_t* hub_{};

    template <class CvrefEnv>
      using completion_sigs =
        stdexec::__t<when_all::traits<
          when_all::env_t<std::remove_cvref_t<CvrefEnv>>,
          stdexec::__member_t<CvrefEnv, stdexec::__t<SenderIds>>...>>;

    template <class Traits>
      using sends_values =
        stdexec::__bool<stdexec::__v<typename Traits::template
          __gather_sigs<stdexec::set_value_t, stdexec::__mconst<int>, stdexec::__mcount>> != 0>;

    template <class CvrefReceiverId>
      struct operation_t;

    template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t : stdexec::receiver_adaptor<receiver_t<CvrefReceiverId, Index>>
                        , stream_receiver_base {
        using WhenAll = stdexec::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = stdexec::__t<std::decay_t<CvrefReceiverId>>;
        using SenderId = nvexec::detail::nth_type<Index, SenderIds...>;
        using Traits =
          completion_sigs<
            stdexec::__member_t<CvrefReceiverId, stdexec::env_of_t<Receiver>>>;

        Receiver&& base() && noexcept {
          return (Receiver&&) op_state_->recvr_;
        }

        const Receiver& base() const & noexcept {
          return op_state_->recvr_;
        }

        template <class Error>
          void set_error(Error&& err, when_all::state_t expected) noexcept {
            // TODO: _What memory orderings are actually needed here?
            if (op_state_->state_.compare_exchange_strong(expected, when_all::error)) {
              op_state_->stop_source_.request_stop();
              // We won the race, free to write the error into the operation
              // state without worry.
              op_state_->errors_.template emplace<std::decay_t<Error>>((Error&&) err);
            }
            op_state_->arrive();
          }

        template <class... Values>
          void set_value(Values&&... vals) && noexcept {
            if constexpr (sends_values<Traits>::value) {
              // We only need to bother recording the completion values
              // if we're not already in the "error" or "stopped" state.
              if (op_state_->state_ == when_all::started) {
                cudaStream_t stream = std::get<Index>(op_state_->child_states_).stream_;
                if constexpr (sizeof...(Values)) {
                  when_all::copy_kernel<<<1, 1, 0, stream>>>(&get<Index>(*op_state_->values_), (Values&&)vals...);
                }

                if constexpr (stream_receiver<Receiver>) {
                  if (op_state_->status_ == cudaSuccess) {
                    op_state_->status_ = 
                      STDEXEC_DBG_ERR(cudaEventRecord(op_state_->events_[Index], stream));
                  }
                }
              }
            }
            op_state_->arrive();
          }

        template <class Error>
            requires std::tag_invocable<stdexec::set_error_t, Receiver, Error>
          void set_error(Error&& err) && noexcept {
            set_error((Error&&) err, when_all::started);
          }

        void set_stopped() && noexcept {
          when_all::state_t expected = when_all::started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (op_state_->state_.compare_exchange_strong(expected, when_all::stopped)) {
            op_state_->stop_source_.request_stop();
          }
          op_state_->arrive();
        }

        auto get_env() const
          -> exec::make_env_t<stdexec::env_of_t<Receiver>, exec::with_t<stdexec::get_stop_token_t, stdexec::in_place_stop_token>> {
          return exec::make_env(
            stdexec::get_env(base()),
            stdexec::__with(stdexec::get_stop_token, op_state_->stop_source_.get_token()));
        }

        operation_t<CvrefReceiverId>* op_state_;
      };

    template <class CvrefReceiverId>
      struct operation_t : stream_op_state_base {
        using WhenAll = stdexec::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = stdexec::__t<std::decay_t<CvrefReceiverId>>;
        using Env = stdexec::env_of_t<Receiver>;
        using CvrefEnv = stdexec::__member_t<CvrefReceiverId, Env>;
        using Traits = completion_sigs<CvrefEnv>;

        cudaStream_t stream_{0};
        cudaError_t status_{cudaSuccess};

        cudaStream_t get_stream() {
          if (!stream_) {
            status_ = STDEXEC_DBG_ERR(cudaStreamCreate(&stream_));
          }
          return stream_;
        }

        template <class Sender, std::size_t Index>
          using child_op_state =
            exit_operation_state_t<
              Sender&&, 
              receiver_t<CvrefReceiverId, Index>>;

        using Indices = std::index_sequence_for<SenderIds...>;

        template <size_t... Is>
          static auto connect_children_(std::index_sequence<Is...>)
            -> std::tuple<child_op_state<stdexec::__t<SenderIds>, Is>...>;

        using child_op_states_tuple_t =
            decltype(operation_t::connect_children_(Indices{}));

        void arrive() noexcept {
          if (0 == --count_) {
            complete();
          }
        }

        template <class OpT>
        static void sync(OpT& op) noexcept {
          if constexpr (std::is_base_of_v<stream_op_state_base, OpT>) {
            if (op.stream_) {
              if (op.status_ == cudaSuccess) {
                op.status_ = STDEXEC_DBG_ERR(cudaStreamSynchronize(op.stream_));
              }
            }
          }
        }

        void complete() noexcept {
          // Stop callback is no longer needed. Destroy it.
          on_stop_.reset();

          // Synchronize streams
          if (status_ == cudaSuccess) {
            if constexpr (stream_receiver<Receiver>) {
              for (int i = 0; i < sizeof...(SenderIds); i++) {
                if (status_ == cudaSuccess) {
                  status_ = STDEXEC_DBG_ERR(cudaStreamWaitEvent(stream_, events_[i]));
                }
              }
            } else {
              std::apply([this](auto&... ops) { (sync(ops), ...); }, child_states_);
            }
          }

          if (status_ == cudaSuccess) {
            // All child operations have completed and arrived at the barrier.
            switch(state_.load(std::memory_order_relaxed)) {
            case when_all::started:
              if constexpr (sends_values<Traits>::value) {
                // All child operations completed successfully:
                ::cuda::std::apply(
                  [this](auto&... opt_vals) -> void {
                    std::apply(
                      [this](auto&... all_vals) -> void {
                        stdexec::set_value((Receiver&&) recvr_, all_vals...);
                      },
                      std::tuple_cat(
                        ::cuda::std::apply(
                          [](auto&... vals) { return std::tie(vals...); },
                          opt_vals
                        )...
                      )
                    );
                  },
                  *values_
                );
              }
              break;
            case when_all::error:
              std::visit([this](auto& err) noexcept {
                  stdexec::set_error((Receiver&&) recvr_, std::move(err));
              }, errors_);
              break;
            case when_all::stopped:
              stdexec::set_stopped((Receiver&&) recvr_);
              break;
            default:
              ;
            }
          } else {
            stdexec::set_error((Receiver&&)recvr_, std::move(status_));
          }
        }

        template <size_t... Is>
          operation_t(WhenAll&& when_all, Receiver rcvr, std::index_sequence<Is...>)
            : recvr_((Receiver&&) rcvr) 
            , child_states_{
                stdexec::__conv{[&when_all, this]() {
                  operation_t* parent_op = this;
                  queue::task_hub_t* hub = 
                    const_cast<queue::task_hub_t*>(
                      stdexec::get_completion_scheduler<stdexec::set_value_t>(
                        std::get<Is>(when_all.sndrs_)
                      ).hub_);
                  return exit_op_state<decltype(std::get<Is>(((WhenAll&&)when_all).sndrs_)),
                                       receiver_t<CvrefReceiverId, Is>>(
                           hub,
                           std::get<Is>(((WhenAll&&) when_all).sndrs_), 
                           receiver_t<CvrefReceiverId, Is>{{}, {}, parent_op});
                }}...
              } {
            status_ = STDEXEC_DBG_ERR(cudaMallocManaged(&values_, sizeof(child_values_tuple_t)));
          }
        operation_t(WhenAll&& when_all, Receiver rcvr)
          : operation_t((WhenAll&&) when_all, (Receiver&&) rcvr, Indices{})
        {
          for (int i = 0; i < sizeof...(SenderIds); i++) {
            if (status_ == cudaSuccess) {
              status_ = STDEXEC_DBG_ERR(cudaEventCreate(&events_[i]));
            }
          }
        }

        ~operation_t() {
          STDEXEC_DBG_ERR(cudaFree(values_));

          if (stream_) {
            STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
          }

          for (int i = 0; i < sizeof...(SenderIds); i++) {
            STDEXEC_DBG_ERR(cudaEventDestroy(events_[i]));
          }
        }

        STDEXEC_IMMOVABLE(operation_t);

        friend void tag_invoke(stdexec::start_t, operation_t& self) noexcept {
          (void)self.get_stream();

          // register stop callback:
          self.on_stop_.emplace(
              stdexec::get_stop_token(stdexec::get_env(self.recvr_)),
              when_all::on_stop_requested{self.stop_source_});
          if (self.stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            stdexec::set_stopped((Receiver&&) self.recvr_);
          } else {
            std::apply([](auto&&... __child_ops) noexcept -> void {
              (stdexec::start(__child_ops), ...);
            }, self.child_states_);
          }
        }

        // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
        using child_values_tuple_t =
          stdexec::__if<
            sends_values<Traits>,
            stdexec::__minvoke<
              stdexec::__q<::cuda::std::tuple>,
              stdexec::__value_types_of_t<
                stdexec::__t<SenderIds>,
                when_all::env_t<Env>,
                stdexec::__q<decayed_tuple>,
                stdexec::__single_or<void>>...>,
            stdexec::__>;

        Receiver recvr_;
        child_op_states_tuple_t child_states_;
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        std::array<cudaEvent_t, sizeof...(SenderIds)> events_;
        // Could be non-atomic here and atomic_ref everywhere except __completion_fn
        std::atomic<when_all::state_t> state_{when_all::started};
        stdexec::error_types_of_t<when_all_sender_t, when_all::env_t<Env>, stdexec::__variant> errors_{};
        child_values_tuple_t* values_{};
        stdexec::in_place_stop_source stop_source_{};
        std::optional<typename stdexec::stop_token_of_t<stdexec::env_of_t<Receiver>&>::template
            callback_type<when_all::on_stop_requested>> on_stop_{};
      };

    template <stdexec::__decays_to<when_all_sender_t> Self, stdexec::receiver Receiver>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> operation_t<stdexec::__member_t<Self, stdexec::__x<std::decay_t<Receiver>>>> {
        return {(Self&&) self, (Receiver&&) rcvr};
      }

    template <stdexec::__decays_to<when_all_sender_t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_sigs<stdexec::__member_t<Self, Env>>;

    template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t> _Tag>
        requires WithCompletionScheduler
      friend Scheduler tag_invoke(stdexec::get_completion_scheduler_t<_Tag>, const when_all_sender_t& __self) noexcept {
        return Scheduler(__self.hub_);
      }

    std::tuple<stdexec::__t<SenderIds>...> sndrs_;
  };
}

