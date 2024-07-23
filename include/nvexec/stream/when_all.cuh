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
      inplace_stop_source& stop_source_;

      void operator()() noexcept {
        stop_source_.request_stop();
      }
    };

    template <class Env>
    using env_t = exec::make_env_t<Env, stdexec::prop<get_stop_token_t, inplace_stop_token>>;

    template <class Sender, class... Env>
    concept valid_child_sender =
      sender_in<Sender, Env...> &&
      requires {
        requires (__v<__count_of<set_value_t, Sender, Env...>> <= 1);
      };

    template <class Sender, class... Env>
    concept too_many_completions_sender =
      sender_in<Sender, Env...> &&
      requires {
        requires (__v<__count_of<set_value_t, Sender, Env...>> > 1);
      };

    template <class Env, class... Senders>
    struct completions {};

    template <class... Env, class... Senders>
      requires(too_many_completions_sender<Senders, Env...> || ...)
    struct completions<__types<Env...>, Senders...> {
      static constexpr std::size_t position_of() noexcept {
        constexpr bool which[] = {too_many_completions_sender<Senders, Env...>...};
        return __pos_of(which, which + sizeof...(Senders));
      }

      using InvalidArg = __m_at_c<position_of(), Senders...>;
      using __t = stdexec::__when_all::__too_many_value_completions_error<InvalidArg, Env...>;
    };

    template <class... As, class TupleT>
    __launch_bounds__(1) __global__ void copy_kernel(TupleT* tpl, As... as) {
      static_assert(trivially_copyable<As...>);
      *tpl = __decayed_tuple<As...>{{static_cast<As&&>(as)}...};
    }

    template <class... Env, class... Senders>
      requires(valid_child_sender<Senders, Env...> && ...)
    struct completions<__types<Env...>, Senders...> {
      using non_values = //
        __meval<
          __concat_completion_signatures,
          completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
          transform_completion_signatures<
            __completion_signatures_of_t<Senders, Env...>,
            completion_signatures<>,
            __mconst<completion_signatures<>>::__f>...>;
      using values = //
        __minvoke<
          __mconcat<__qf<set_value_t>>,
          __value_types_t<
            __completion_signatures_of_t<Senders, Env...>, __q<__types>, __msingle_or<__types<>>>...>;
      using __t = //
        __if_c<
          (__sends<set_value_t, Senders, Env...> && ...),
          __minvoke<__mpush_back<__q<completion_signatures>>, non_values, values>,
          non_values>;
    };
  } // namespace _when_all

  template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  struct when_all_sender_t {
    struct __t : stream_sender_base {
     private:
      struct env {
        context_state_t context_state_;

        template <__one_of<set_value_t, set_stopped_t> _Tag>
          requires WithCompletionScheduler
        Scheduler query(get_completion_scheduler_t<_Tag>) const noexcept {
          return Scheduler(context_state_);
        }
      };
     public:
      using __id = when_all_sender_t;

      template <class... Sndrs>
      explicit __t(context_state_t context_state, Sndrs&&... __sndrs)
        : env_{context_state}
        , sndrs_{{static_cast<Sndrs&&>(__sndrs)}...} {
      }

     private:
      env env_;

      template <class Cvref, class... Env>
      using completion_sigs = //
        stdexec::__t<_when_all::completions<
          __types<_when_all::env_t<Env>...>,
          __copy_cvref_t<Cvref, stdexec::__t<SenderIds>>...>>;

      template <class Completions>
      using sends_values = //
        __gather_completion_signatures<
          Completions,
          set_value_t,
          __mconst<__mbool<true>>::__f,
          __mconst<__mbool<false>>::__f,
          __mor_t>;

      template <class CvrefReceiverId>
      struct operation_t;

      template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t {
        using WhenAll = __copy_cvref_t<CvrefReceiverId, stdexec::__t<when_all_sender_t>>;
        using Receiver = stdexec::__t<__decay_t<CvrefReceiverId>>;
        using Env = //
          make_terminal_stream_env_t<
            exec::make_env_t<env_of_t<Receiver>, stdexec::prop<get_stop_token_t, inplace_stop_token>>>;

        struct __t : stream_receiver_base {
          using receiver_concept = stdexec::receiver_t;
          using __id = receiver_t;
          using SenderId = nvexec::detail::nth_type<Index, SenderIds...>;
          using Completions = completion_sigs<env_of_t<Receiver>, CvrefReceiverId>;

          template <class Error>
          void _set_error_impl(Error&& err, _when_all::state_t expected) noexcept {
            // TODO: What memory orderings are actually needed here?
            if (op_state_->state_.compare_exchange_strong(expected, _when_all::error)) {
              op_state_->stop_source_.request_stop();
              // We won the race, free to write the error into the operation
              // state without worry.
              op_state_->errors_.template emplace<__decay_t<Error>>(static_cast<Error&&>(err));
            }
            op_state_->arrive();
          }

          template <class... Values>
          void set_value(Values&&... vals) && noexcept {
            if constexpr (__v<sends_values<Completions>>) {
              // We only need to bother recording the completion values
              // if we're not already in the "error" or "stopped" state.
              if (op_state_->state_ == _when_all::started) {
                cudaStream_t stream = __tup::__get<Index>(op_state_->child_states_).get_stream();
                if constexpr (sizeof...(Values)) {
                  _when_all::copy_kernel<Values&&...><<<1, 1, 0, stream>>>(
                    &__tup::__get<Index>(*op_state_->values_), static_cast<Values&&>(vals)...);
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
            requires tag_invocable<set_error_t, Receiver, Error>
          void set_error(Error&& err) && noexcept {
            _set_error_impl(static_cast<Error&&>(err), _when_all::started);
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
                stdexec::get_env(op_state_->rcvr_),
                stdexec::prop{get_stop_token, op_state_->stop_source_.get_token()}),
              &const_cast<stream_provider_t&>(op_state_->stream_providers_[Index]));

            return env;
          }

          operation_t<CvrefReceiverId>* op_state_;
        };
      };

      template <class CvrefReceiverId>
      struct operation_t : stream_op_state_base {
        using WhenAll = __copy_cvref_t<CvrefReceiverId, stdexec::__t<when_all_sender_t>>;
        using Receiver = stdexec::__t<__decay_t<CvrefReceiverId>>; // NOT __cvref_t
        using Env = env_of_t<Receiver>;
        using Completions = completion_sigs<Env, CvrefReceiverId>;

        cudaError_t status_{cudaSuccess};

        template <class SenderId, std::size_t Index>
        using child_op_state_t = exit_operation_state_t<
          __copy_cvref_t<WhenAll, stdexec::__t<SenderId>>,
          stdexec::__t<receiver_t<CvrefReceiverId, Index>>>;

        using Indices = __indices_for<SenderIds...>;

        template <size_t... Is>
        static auto connect_children_(operation_t* parent_op, WhenAll&& when_all, __indices<Is...>)
          -> __tuple_for<child_op_state_t<SenderIds, Is>...> {

          using __child_ops_t = __tuple_for<child_op_state_t<SenderIds, Is>...>;
          return when_all.sndrs_.apply(
            [parent_op]<class... Children>(Children&&... children) -> __child_ops_t {
              return __child_ops_t{{STDEXEC_STREAM_DETAIL_NS::exit_op_state(
                static_cast<Children&&>(children),
                stdexec::__t<receiver_t<CvrefReceiverId, Is>>{{}, parent_op},
                stdexec::get_completion_scheduler<set_value_t>(stdexec::get_env(children))
                  .context_state_)}...};
            },
            static_cast<WhenAll&&>(when_all).sndrs_);
        }

        using child_op_states_tuple_t =
          decltype(operation_t::connect_children_({}, __declval<WhenAll>(), Indices{}));

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
              auto env = stdexec::get_env(rcvr_);
              stream_provider_t* stream_provider = get_stream_provider(env);
              cudaStream_t stream = stream_provider->own_stream_.value();

              for (int i = 0; i < sizeof...(SenderIds); i++) {
                if (status_ == cudaSuccess) {
                  status_ = STDEXEC_DBG_ERR(cudaStreamWaitEvent(stream, events_[i], 0));
                }
              }
            } else {
              child_states_.apply([](auto&... ops) { (sync(ops), ...); }, child_states_);
            }
          }

          if (status_ == cudaSuccess) {
            // All child operations have completed and arrived at the barrier.
            switch (state_.load(std::memory_order_relaxed)) {
            case _when_all::started:
              if constexpr (__v<sends_values<Completions>>) {
                // All child operations completed successfully:
                values_->apply(
                  [this](auto&... opt_vals) -> void {
                    __tup::__cat_apply(
                      [this](auto&... all_vals) -> void {
                        stdexec::set_value(static_cast<Receiver&&>(rcvr_), std::move(all_vals)...);
                      },
                      opt_vals...);
                  },
                  *values_);
              }
              break;
            case _when_all::error:
              errors_.visit(
                [this](auto& err) noexcept {
                  stdexec::set_error(static_cast<Receiver&&>(rcvr_), std::move(err));
                },
                errors_);
              break;
            case _when_all::stopped:
              stdexec::set_stopped(static_cast<Receiver&&>(rcvr_));
              break;
            default:;
            }
          } else {
            stdexec::set_error(static_cast<Receiver&&>(rcvr_), std::move(status_));
          }
        }

        using stream_providers_t = std::array<stream_provider_t, sizeof...(SenderIds)>;

        static stream_providers_t get_stream_providers(WhenAll& when_all) {
          return when_all.sndrs_.apply(
            [](auto&... sndrs) -> stream_providers_t {
              return stream_providers_t{
                stdexec::get_completion_scheduler<set_value_t>(stdexec::get_env(sndrs))
                  .context_state_...};
            },
            when_all.sndrs_);
        }

        operation_t(WhenAll&& when_all, Receiver rcvr)
          : rcvr_(static_cast<Receiver&&>(rcvr))
          , stream_providers_{operation_t::get_stream_providers(when_all)}
          , child_states_{
              operation_t::connect_children_(this, static_cast<WhenAll&&>(when_all), Indices{})} {
          status_ = STDEXEC_DBG_ERR(cudaMallocManaged(&values_, sizeof(child_values_tuple_t)));
          for (std::size_t i = 0; i < sizeof...(SenderIds); ++i) {
            if (status_ == cudaSuccess) {
              status_ = STDEXEC_DBG_ERR(cudaEventCreate(&events_[i], cudaEventDisableTiming));
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

        void start() & noexcept {
          // register stop callback:
          auto tok = stdexec::get_stop_token(stdexec::get_env(rcvr_));
          on_stop_.emplace(std::move(tok), _when_all::on_stop_requested{stop_source_});
          if (stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            stdexec::set_stopped(static_cast<Receiver&&>(rcvr_));
          } else {
            if constexpr (sizeof...(SenderIds) == 0) {
              complete();
            } else {
              child_states_.apply(
                [](auto&... __child_ops) noexcept -> void { (stdexec::start(__child_ops), ...); },
                child_states_);
            }
          }
        }

        // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
        using child_values_tuple_t = //
          __if<
            sends_values<Completions>,
            __minvoke<
              __q<__tuple_for>,
              __value_types_of_t<
                stdexec::__t<SenderIds>,
                _when_all::env_t<Env>,
                __q<__decayed_tuple>,
                __msingle_or<void>>...>,
            __>;

        using errors_variant_t = //
          error_types_of_t<
            stdexec::__t<when_all_sender_t>,
            _when_all::env_t<Env>,
            __uniqued_variant_for>;

        Receiver rcvr_;
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        std::array<stream_provider_t, sizeof...(SenderIds)> stream_providers_;
        std::array<cudaEvent_t, sizeof...(SenderIds)> events_;
        child_op_states_tuple_t child_states_;
        // Could be non-atomic here and atomic_ref everywhere except __completion_fn
        std::atomic<_when_all::state_t> state_{_when_all::started};

        errors_variant_t errors_{};
        child_values_tuple_t* values_{};
        inplace_stop_source stop_source_{};

        using stop_callback_t =
          stop_callback_for_t<stop_token_of_t<env_of_t<Receiver>&>, _when_all::on_stop_requested>;
        std::optional<stop_callback_t> on_stop_{};
      };

     public:
      template <__decays_to<__t> Self, receiver Receiver>
      static auto connect(Self&& self, Receiver rcvr)
        -> operation_t<__copy_cvref_t<Self, stdexec::__id<Receiver>>> {
        return {static_cast<Self&&>(self), static_cast<Receiver&&>(rcvr)};
      }

      template <__decays_to<__t> Self, class... Env>
      static auto get_completion_signatures(Self&&, Env&&...) -> completion_sigs<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> const env& {
        return env_;
      }

     private:
      __tuple_for<stdexec::__t<SenderIds>...> sndrs_;
    };
  };
} // namespace nvexec::STDEXEC_STREAM_DETAIL_NS

namespace stdexec::__detail {
  template <bool WithCompletionScheduler, class Scheduler, class... SenderIds>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::
      when_all_sender_t<WithCompletionScheduler, Scheduler, __name_of<__t<SenderIds>>...>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::
                  when_all_sender_t<WithCompletionScheduler, Scheduler, SenderIds...>>{};
} // namespace stdexec::__detail
