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

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"
#include "queue.cuh"
#include "schedulers/detail/throw_on_cuda_error.cuh"

namespace example::cuda::stream {

namespace when_all {

enum state_t { started, error, stopped };

struct on_stop_requested {
  std::in_place_stop_source& stop_source_;
  void operator()() noexcept {
    stop_source_.request_stop();
  }
};

template <class Env>
  using env_t =
    _P2300::execution::make_env_t<Env, _P2300::execution::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>>;

template <class...>
  using swallow_values = std::execution::completion_signatures<>;

template <class Env, class... Senders>
  struct traits {
    using __t = std::execution::dependent_completion_signatures<Env>;
  };

template <class TupleT, class... As>
__launch_bounds__(1)
__global__ void copy_kernel(TupleT* tpl, As&&... as) {
  *tpl = decayed_tuple<As...>((As&&)as...);
}

template <class Env, class... Senders>
    requires ((_P2300::__v<_P2300::execution::__count_of<std::execution::set_value_t, Senders, Env>> <= 1) &&...)
  struct traits<Env, Senders...> {
    using non_values =
      _P2300::execution::__concat_completion_signatures_t<
        std::execution::completion_signatures<
          std::execution::set_error_t(cudaError_t),
          std::execution::set_stopped_t()>,
        std::execution::make_completion_signatures<
          Senders,
          Env,
          std::execution::completion_signatures<>,
          swallow_values>...>;
    using values =
      _P2300::__minvoke<
        _P2300::__concat<_P2300::__qf<std::execution::set_value_t>>,
        _P2300::execution::__value_types_of_t<
          Senders,
          Env,
          _P2300::__q<_P2300::__types>,
          _P2300::__single_or<_P2300::__types<>>>...>;
    using __t =
      _P2300::__if_c<
        (_P2300::execution::__sends<std::execution::set_value_t, Senders, Env> &&...),
        _P2300::__minvoke2<
          _P2300::__push_back<_P2300::__q<std::execution::completion_signatures>>, non_values, values>,
        non_values>;
  };
}

template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  struct when_all_sender_t : sender_base_t {
    template <class... Sndrs>
      explicit when_all_sender_t(detail::queue::task_hub_t* hub, Sndrs&&... __sndrs)
        : hub_(hub)
        , sndrs_((Sndrs&&) __sndrs...)
      {}

   private:
    const detail::queue::task_hub_t* hub_{};

    template <class CvrefEnv>
      using completion_sigs =
        _P2300::__t<when_all::traits<
          when_all::env_t<std::remove_cvref_t<CvrefEnv>>,
          _P2300::__member_t<CvrefEnv, _P2300::__t<SenderIds>>...>>;

    template <class Traits>
      using sends_values =
        _P2300::__bool<_P2300::__v<typename Traits::template
          __gather_sigs<std::execution::set_value_t, _P2300::__mconst<int>, _P2300::__mcount>> != 0>;

    template <class CvrefReceiverId>
      struct operation_t;

    template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t : std::execution::receiver_adaptor<receiver_t<CvrefReceiverId, Index>>
                        , receiver_base_t {
        using WhenAll = _P2300::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = _P2300::__t<std::decay_t<CvrefReceiverId>>;
        using SenderId = example::cuda::detail::nth_type<Index, SenderIds...>;
        using Traits =
          completion_sigs<
            _P2300::__member_t<CvrefReceiverId, std::execution::env_of_t<Receiver>>>;

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
            requires std::tag_invocable<std::execution::set_error_t, Receiver, Error>
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
          -> _P2300::execution::make_env_t<std::execution::env_of_t<Receiver>, _P2300::execution::with_t<std::execution::get_stop_token_t, std::in_place_stop_token>> {
          return _P2300::execution::make_env(
            std::execution::get_env(base()),
            _P2300::execution::with(std::execution::get_stop_token, op_state_->stop_source_.get_token()));
        }

        operation_t<CvrefReceiverId>* op_state_;
      };

    template <class CvrefReceiverId>
      struct operation_t : detail::op_state_base_t {
        using WhenAll = _P2300::__member_t<CvrefReceiverId, when_all_sender_t>;
        using Receiver = _P2300::__t<std::decay_t<CvrefReceiverId>>;
        using Env = std::execution::env_of_t<Receiver>;
        using CvrefEnv = _P2300::__member_t<CvrefReceiverId, Env>;
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
            std::execution::connect_result_t<
              _P2300::__member_t<WhenAll, Sender>,
              receiver_t<CvrefReceiverId, Index>>;

        using Indices = std::index_sequence_for<SenderIds...>;

        template <size_t... Is>
          static auto connect_children_(std::index_sequence<Is...>)
            -> std::tuple<child_op_state<_P2300::__t<SenderIds>, Is>...>;

        using child_op_states_tuple_t =
            decltype((connect_children_)(Indices{}));

        void arrive() noexcept {
          if (0 == --count_) {
            complete();
          }
        }

        template <class OpT>
        static void sync(OpT& op) noexcept {
          if constexpr (std::is_base_of_v<detail::op_state_base_t, OpT>) {
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
                apply(
                  [this](auto&... opt_vals) -> void {
                    std::apply(
                      [this](auto&... all_vals) -> void {
                        std::execution::set_value((Receiver&&) recvr_, all_vals...);
                      },
                      std::tuple_cat(
                        apply(
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
                  std::execution::set_error((Receiver&&) recvr_, std::move(err));
              }, errors_);
              break;
            case when_all::stopped:
              std::execution::set_stopped((Receiver&&) recvr_);
              break;
            default:
              ;
            }
          } else {
            std::execution::set_error((Receiver&&)recvr_, std::move(status_));
          }
        }

        template <size_t... Is>
          operation_t(WhenAll&& when_all, Receiver rcvr, std::index_sequence<Is...>)
            : child_states_{
              _P2300::execution::__conv{[&when_all, this]() {
                  return std::execution::connect(
                      std::get<Is>(((WhenAll&&) when_all).sndrs_),
                      receiver_t<CvrefReceiverId, Is>{{}, {}, this});
                }}...
              }
            , recvr_((Receiver&&) rcvr) {
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

        _P2300_IMMOVABLE(operation_t);

        friend void tag_invoke(std::execution::start_t, operation_t& self) noexcept {
          (void)self.get_stream();

          // register stop callback:
          self.on_stop_.emplace(
              std::execution::get_stop_token(std::execution::get_env(self.recvr_)),
              when_all::on_stop_requested{self.stop_source_});
          if (self.stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            std::execution::set_stopped((Receiver&&) self.recvr_);
          } else {
            std::apply([](auto&&... __child_ops) noexcept -> void {
              (std::execution::start(__child_ops), ...);
            }, self.child_states_);
          }
        }

        // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
        using child_values_tuple_t =
          _P2300::__if<
            sends_values<Traits>,
            _P2300::__minvoke<
              _P2300::__q<tuple_t>,
              _P2300::execution::__value_types_of_t<
                _P2300::__t<SenderIds>,
                when_all::env_t<Env>,
                _P2300::__q<decayed_tuple>,
                _P2300::__single_or<void>>...>,
            _P2300::__>;

        child_op_states_tuple_t child_states_;
        Receiver recvr_;
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        std::array<cudaEvent_t, sizeof...(SenderIds)> events_;
        // Could be non-atomic here and atomic_ref everywhere except __completion_fn
        std::atomic<when_all::state_t> state_{when_all::started};
        std::execution::error_types_of_t<when_all_sender_t, when_all::env_t<Env>, _P2300::execution::__variant> errors_{};
        child_values_tuple_t* values_{};
        std::in_place_stop_source stop_source_{};
        std::optional<typename std::execution::stop_token_of_t<std::execution::env_of_t<Receiver>&>::template
            callback_type<when_all::on_stop_requested>> on_stop_{};
      };

    template <_P2300::__decays_to<when_all_sender_t> Self, std::execution::receiver Receiver>
      friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
        -> operation_t<_P2300::__member_t<Self, _P2300::__x<std::decay_t<Receiver>>>> {
        return {(Self&&) self, (Receiver&&) rcvr};
      }

    template <_P2300::__decays_to<when_all_sender_t> Self, class Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
        -> completion_sigs<_P2300::__member_t<Self, Env>>;

    template <_P2300::__one_of<std::execution::set_value_t, std::execution::set_stopped_t> _Tag>
      requires WithCompletionScheduler
    friend Scheduler tag_invoke(std::execution::get_completion_scheduler_t<_Tag>, const when_all_sender_t& __self) noexcept {
      return Scheduler(__self.hub_);
    }

    std::tuple<_P2300::__t<SenderIds>...> sndrs_;
  };
}

