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

#include "../../exec/env.hpp"
#include "../../stdexec/execution.hpp"
#include <type_traits>

#include "common.cuh"
#include "../detail/queue.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

  namespace _when_all {

    enum state_t {
      started,
      error,
      stopped
    };

    struct on_stop_requested {
      in_place_stop_source& stop_source_;

      void operator()() noexcept {
        stop_source_.request_stop();
      }
    };

    template <class Env>
    using env_t = exec::make_env_t<Env, exec::with_t<get_stop_token_t, in_place_stop_token>>;

    template <class...>
    using swallow_values = completion_signatures<>;

    template <class Sender, class Env>
    using too_many_completions = __mbool<(1 < __v<__count_of<set_value_t, Sender, Env>>)>;

    template <class Env, class... Senders>
    struct completions {
      using InvalidArg = //
        __minvoke< __mfind_if<__mbind_back_q<too_many_completions, Env>, __q<__mfront>>, Senders...>;

      using __t = stdexec::__when_all::__too_many_value_completions_error<InvalidArg, Env>;
    };

    template <class... As, class TupleT>
    __launch_bounds__(1) __global__ void copy_kernel(TupleT* tpl, As... as) {
      static_assert(trivially_copyable<As...>);
      *tpl = decayed_tuple<As...>(static_cast<As&&>(as)...);
    }

    template <class Env, class... Senders>
      requires(!__v<too_many_completions<Senders, Env>> && ...)
    struct completions<Env, Senders...> {
      using non_values = //
        __concat_completion_signatures_t<
          completion_signatures< set_error_t(cudaError_t), set_stopped_t()>,
          __try_make_completion_signatures<
            Senders,
            Env,
            completion_signatures<>,
            __q<swallow_values>>...>;
      using values = //
        __minvoke<
          __mconcat<__qf<set_value_t>>,
          __value_types_of_t< Senders, Env, __q<__types>, __msingle_or<__types<>>>...>;
      using __t = //
        __if_c<
          (__sends<set_value_t, Senders, Env> && ...),
          __minvoke< __push_back<__q<completion_signatures>>, non_values, values>,
          non_values>;
    };
  }

  template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  struct when_all_sender_t {
    struct __t : stream_sender_base {
     private:
      struct env {
        context_state_t context_state_;

        template <__one_of<set_value_t, set_stopped_t> _Tag>
          requires WithCompletionScheduler
        friend Scheduler tag_invoke(get_completion_scheduler_t<_Tag>, const env& self) noexcept {
          return Scheduler(self.context_state_);
        }
      };
     public:
      using __id = when_all_sender_t;

      template <class... Sndrs>
      explicit __t(context_state_t context_state, Sndrs&&... __sndrs)
        : env_{context_state}
        , sndrs_((Sndrs&&) __sndrs...) {
      }

     private:
      env env_;

      template <class Env, class Cvref>
      using completion_sigs = //
        stdexec::__t<_when_all::completions<
          _when_all::env_t<Env>,
          __copy_cvref_t<Cvref, stdexec::__t<SenderIds>>...>>;

      template <class Completions>
      using sends_values = //
        __mbool< __v< __gather_signal< set_value_t, Completions, __mconst<int>, __msize>> != 0>;

      template <class CvrefReceiverId>
      struct operation_t;

      template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t {
        using WhenAll = __copy_cvref_t<CvrefReceiverId, stdexec::__t<when_all_sender_t>>;
        using Receiver = stdexec::__t<__decay_t<CvrefReceiverId>>;
        using Env = //
          make_terminal_stream_env_t< exec::make_env_t<
            env_of_t<Receiver>,
            exec::with_t< get_stop_token_t, in_place_stop_token>>>;

        struct __t
          : receiver_adaptor<__t>
          , stream_receiver_base {
          using __id = receiver_t;
          using SenderId = nvexec::detail::nth_type<Index, SenderIds...>;
          using Completions = completion_sigs<env_of_t<Receiver>, CvrefReceiverId>;

          Receiver&& base() && noexcept {
            return (Receiver&&) op_state_->recvr_;
          }

          const Receiver& base() const & noexcept {
            return op_state_->recvr_;
          }

          template <class Error>
          void set_error(Error&& err, _when_all::state_t expected) noexcept {
            // TODO: _What memory orderings are actually needed here?
            if (op_state_->state_.compare_exchange_strong(expected, _when_all::error)) {
              op_state_->stop_source_.request_stop();
              // We won the race, free to write the error into the operation
              // state without worry.
              op_state_->errors_.template emplace<__decay_t<Error>>((Error&&) err);
            }
            op_state_->arrive();
          }

          template <class... Values>
          void set_value(Values&&... vals) && noexcept {
            if constexpr (sends_values<Completions>::value) {
              // We only need to bother recording the completion values
              // if we're not already in the "error" or "stopped" state.
              if (op_state_->state_ == _when_all::started) {
                cudaStream_t stream = std::get<Index>(op_state_->child_states_).get_stream();
                if constexpr (sizeof...(Values)) {
                  _when_all::copy_kernel<Values&&...>
                    <<<1, 1, 0, stream>>>(&get<Index>(*op_state_->values_), (Values&&) vals...);
                }

                if constexpr (stream_receiver<Receiver>) {
                  if (op_state_->status_ == cudaSuccess) {
                    op_state_->status_ = STDEXEC_DBG_ERR(
                      cudaEventRecord(op_state_->events_[Index], stream));
                  }
                }
              }
            }
            op_state_->arrive();
          }

          template <class Error>
            requires tag_invocable<set_error_t, Receiver, Error>
          void set_error(Error&& err) && noexcept {
            set_error((Error&&) err, _when_all::started);
          }

          void set_stopped() && noexcept {
            _when_all::state_t expected = _when_all::started;
            // Transition to the "stopped" state if and only if we're in the
            // "started" state. (If this fails, it's because we're in an
            // error state, which trumps cancellation.)
            if (op_state_->state_.compare_exchange_strong(expected, _when_all::stopped)) {
              op_state_->stop_source_.request_stop();
            }
            op_state_->arrive();
          }

          Env get_env() const noexcept {
            auto env = make_terminal_stream_env(
              exec::make_env(
                stdexec::get_env(base()),
                __mkprop(op_state_->stop_source_.get_token(), get_stop_token)),
              &const_cast<stream_provider_t&>(op_state_->stream_providers_[Index]));

            return env;
          }

          operation_t<CvrefReceiverId>* op_state_;
        };
      };

      template <class CvrefReceiverId>
      struct operation_t : stream_op_state_base {
        using WhenAll = __copy_cvref_t<CvrefReceiverId, stdexec::__t<when_all_sender_t>>;
        using Receiver = stdexec::__t<__decay_t<CvrefReceiverId>>;
        using Env = env_of_t<Receiver>;
        using Completions = completion_sigs<Env, CvrefReceiverId>;

        cudaError_t status_{cudaSuccess};

        template <class Sender, std::size_t Index>
        using child_op_state =
          exit_operation_state_t< Sender&&, stdexec::__t<receiver_t<CvrefReceiverId, Index>>>;

        using Indices = std::index_sequence_for<SenderIds...>;

        template <size_t... Is>
        static auto connect_children_(std::index_sequence<Is...>)
          -> std::tuple<child_op_state<stdexec::__t<SenderIds>, Is>...>;

        using child_op_states_tuple_t = decltype(operation_t::connect_children_(Indices{}));

        void arrive() noexcept {
          if (0 == --count_) {
            complete();
          }
        }

        template <class OpT>
        static void sync(OpT& op) noexcept {
          if constexpr (STDEXEC_IS_BASE_OF(stream_op_state_base, OpT)) {
            if (op.stream_provider_.status_ == cudaSuccess) {
              op.stream_provider_.status_ = STDEXEC_DBG_ERR(cudaStreamSynchronize(op.get_stream()));
            }
          }
        }

        void complete() noexcept {
          // Stop callback is no longer needed. Destroy it.
          on_stop_.reset();

          // Synchronize streams
          if (status_ == cudaSuccess) {
            if constexpr (stream_receiver<Receiver>) {
              auto env = get_env(recvr_);
              stream_provider_t* stream_provider = get_stream_provider(env);
              cudaStream_t stream = stream_provider->own_stream_.value();

              for (int i = 0; i < sizeof...(SenderIds); i++) {
                if (status_ == cudaSuccess) {
                  status_ = STDEXEC_DBG_ERR(cudaStreamWaitEvent(stream, events_[i], 0));
                }
              }
            } else {
              std::apply([](auto&... ops) { (sync(ops), ...); }, child_states_);
            }
          }

          if (status_ == cudaSuccess) {
            // All child operations have completed and arrived at the barrier.
            switch (state_.load(std::memory_order_relaxed)) {
            case _when_all::started:
              if constexpr (sends_values<Completions>::value) {
                // All child operations completed successfully:
                ::cuda::std::apply(
                  [this](auto&... opt_vals) -> void {
                    std::apply(
                      [this](auto&... all_vals) -> void {
                        stdexec::set_value((Receiver&&) recvr_, std::move(all_vals)...);
                      },
                      std::tuple_cat(::cuda::std::apply(
                        [](auto&... vals) { return std::tie(vals...); }, opt_vals)...));
                  },
                  *values_);
              }
              break;
            case _when_all::error:
              std::visit(
                [this](auto& err) noexcept {
                  stdexec::set_error((Receiver&&) recvr_, std::move(err));
                },
                errors_);
              break;
            case _when_all::stopped:
              stdexec::set_stopped((Receiver&&) recvr_);
              break;
            default:;
            }
          } else {
            stdexec::set_error((Receiver&&) recvr_, std::move(status_));
          }
        }

        template <size_t Index>
        context_state_t get_context_state(WhenAll& when_all) {
          auto sch = get_completion_scheduler<set_value_t>(
            get_env(std::get<Index>(when_all.sndrs_)));
          return sch.context_state_;
        }

        template <size_t... Is>
        operation_t(WhenAll&& when_all, Receiver rcvr, std::index_sequence<Is...>)
          : recvr_((Receiver&&) rcvr)
          , stream_providers_{get_context_state<Is>(when_all)...}
          , child_states_{__conv{[&when_all, this]() {
            operation_t* parent_op = this;
            context_state_t context_state = get_context_state<Is>(when_all);

            return exit_op_state<
              decltype(std::get<Is>(((WhenAll&&) when_all).sndrs_)),
              stdexec::__t<receiver_t<CvrefReceiverId, Is>>>(
              std::get<Is>(((WhenAll&&) when_all).sndrs_),
              stdexec::__t<receiver_t<CvrefReceiverId, Is>>{{}, {}, parent_op},
              context_state);
          }}...} {
          status_ = STDEXEC_DBG_ERR(cudaMallocManaged(&values_, sizeof(child_values_tuple_t)));
        }

        operation_t(WhenAll&& when_all, Receiver rcvr)
          : operation_t((WhenAll&&) when_all, (Receiver&&) rcvr, Indices{}) {
          for (int i = 0; i < sizeof...(SenderIds); i++) {
            if (status_ == cudaSuccess) {
              status_ = STDEXEC_DBG_ERR(cudaEventCreate(&events_[i]));
            }
          }
        }

        ~operation_t() {
          STDEXEC_DBG_ERR(cudaFree(values_));

          for (int i = 0; i < sizeof...(SenderIds); i++) {
            STDEXEC_DBG_ERR(cudaEventDestroy(events_[i]));
          }
        }

        STDEXEC_IMMOVABLE(operation_t);

        friend void tag_invoke(start_t, operation_t& self) noexcept {
          // register stop callback:
          self.on_stop_.emplace(
            get_stop_token(get_env(self.recvr_)), _when_all::on_stop_requested{self.stop_source_});
          if (self.stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            stdexec::set_stopped((Receiver&&) self.recvr_);
          } else {
            if constexpr (sizeof...(SenderIds) == 0) {
              self.complete();
            } else {
              std::apply(
                [](auto&&... __child_ops) noexcept -> void { (start(__child_ops), ...); },
                self.child_states_);
            }
          }
        }

        // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
        using child_values_tuple_t = //
          __if<
            sends_values<Completions>,
            __minvoke<
              __q<::cuda::std::tuple>,
              __value_types_of_t<
                stdexec::__t<SenderIds>,
                _when_all::env_t<Env>,
                __q<decayed_tuple>,
                __msingle_or<void>>...>,
            __>;

        Receiver recvr_;
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        std::array<stream_provider_t, sizeof...(SenderIds)> stream_providers_;
        std::array<cudaEvent_t, sizeof...(SenderIds)> events_;
        child_op_states_tuple_t child_states_;
        // Could be non-atomic here and atomic_ref everywhere except __completion_fn
        std::atomic<_when_all::state_t> state_{_when_all::started};

        error_types_of_t<stdexec::__t<when_all_sender_t>, _when_all::env_t<Env>, __variant>
          errors_{};
        child_values_tuple_t* values_{};
        in_place_stop_source stop_source_{};
        std::optional<typename stop_token_of_t< env_of_t<Receiver>&>::template callback_type<
          _when_all::on_stop_requested>>
          on_stop_{};
      };

      template <__decays_to<__t> Self, receiver Receiver>
      friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr)
        -> operation_t<__copy_cvref_t<Self, stdexec::__id<__decay_t<Receiver>>>> {
        return {(Self&&) self, (Receiver&&) rcvr};
      }

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> completion_sigs<Env, Self> {
        return {};
      }

      friend const env& tag_invoke(get_env_t, const __t& __self) noexcept {
        return __self.env_;
      }

      std::tuple<stdexec::__t<SenderIds>...> sndrs_;
    };
  };
}

namespace stdexec::__detail {
  template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::
      when_all_sender_t<WithCompletionScheduler, Scheduler, __name_of<__t<SenderIds>>...>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::
                  when_all_sender_t<WithCompletionScheduler, Scheduler, SenderIds...>>{};
}
