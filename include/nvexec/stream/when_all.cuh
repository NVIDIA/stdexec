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
#include "../../exec/env.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "common.cuh"
#include "../detail/event.cuh"
#include "../detail/throw_on_cuda_error.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nvexec::_strm {

  namespace _when_all {

    enum state_t : std::uint32_t {
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
    concept valid_child_sender = sender_in<Sender, Env...> && requires {
      requires(__v<__count_of<set_value_t, Sender, Env...>> <= 1);
    };

    template <class Sender, class... Env>
    concept too_many_completions_sender = sender_in<Sender, Env...> && requires {
      requires(__v<__count_of<set_value_t, Sender, Env...>> > 1);
    };

    template <class Env, class... Senders>
    struct completions { };

    template <class... Env, class... Senders>
      requires(too_many_completions_sender<Senders, Env...> || ...)
    struct completions<__types<Env...>, Senders...> {
      static constexpr auto position_of() noexcept -> std::size_t {
        constexpr bool which[] = {too_many_completions_sender<Senders, Env...>...};
        return __pos_of(which, which + sizeof...(Senders));
      }

      using InvalidArg = __m_at_c<position_of(), Senders...>;
      using __t = stdexec::__when_all::__too_many_value_completions_error<InvalidArg, Env...>;
    };

    template <class... As, class TupleT>
    __launch_bounds__(1) __global__ void copy_kernel(TupleT* tpl, As... as) {
      static_assert(trivially_copyable<As...>);
      *tpl = __decayed_tuple<As...>{static_cast<As&&>(as)...};
    }

    template <class... Env, class... Senders>
      requires(valid_child_sender<Senders, Env...> && ...)
    struct completions<__types<Env...>, Senders...> {
      using non_values = __meval<
        __concat_completion_signatures,
        completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
        transform_completion_signatures<
          __completion_signatures_of_t<Senders, Env...>,
          completion_signatures<>,
          __mconst<completion_signatures<>>::__f
        >...
      >;
      using values = __minvoke<
        __mconcat<__qf<set_value_t>>,
        __value_types_t<
          __completion_signatures_of_t<Senders, Env...>,
          __q<__types>,
          __msingle_or<__types<>>
        >...
      >;
      using __t = __if_c<
        (__sends<set_value_t, Senders, Env...> && ...),
        __minvoke<__mpush_back<__q<completion_signatures>>, non_values, values>,
        non_values
      >;
    };

    inline constexpr auto _sync_op = []<class OpT>(OpT& op) noexcept {
      if constexpr (STDEXEC_IS_BASE_OF(stream_op_state_base, OpT)) {
        if (op.stream_provider_.status_ == cudaSuccess) {
          op.stream_provider_
            .status_ = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(op.get_stream()));
        }
      }
    };
  } // namespace _when_all

  template <class WhenAllTag, class Scheduler, class... SenderIds>
  struct when_all_sender_t {
    struct type;
    using __t = type;

    struct type : stream_sender_base {
     private:
      struct env {
        context_state_t context_state_;
        using sched_domain_t = __query_result_or_t<get_domain_t, Scheduler, default_domain>;

        auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> Scheduler
          requires stdexec::__same_as<WhenAllTag, transfer_when_all_t>
        {
          return Scheduler(context_state_);
        }

        constexpr auto query(get_domain_t) const noexcept {
          if constexpr (stdexec::__same_as<WhenAllTag, transfer_when_all_t>) {
            return sched_domain_t{};
          } else {
            static_assert(
              sizeof...(SenderIds) == 0
              || stdexec::__same_as<stream_domain, __common_domain_t<stdexec::__t<SenderIds>...>>);
            return stream_domain{};
          }
        }

        constexpr auto query(get_domain_override_t) const noexcept
          requires stdexec::__same_as<WhenAllTag, transfer_when_all_t>
        {
          static_assert(
            sizeof...(SenderIds) == 0
            || stdexec::__same_as<stream_domain, __common_domain_t<stdexec::__t<SenderIds>...>>);
          return stream_domain{};
        }
      };
     public:
      using __id = when_all_sender_t;

      template <class... Sndrs>
      explicit type(context_state_t context_state, Sndrs&&... __sndrs)
        : env_{context_state}
        , sndrs_{static_cast<Sndrs&&>(__sndrs)...} {
      }

     private:
      env env_;

      template <class Cvref, class... Env>
      using completion_sigs = stdexec::__t<_when_all::completions<
        __types<_when_all::env_t<Env>...>,
        __copy_cvref_t<Cvref, stdexec::__t<SenderIds>>...
      >>;

      template <class Completions>
      using sends_values = __gather_completion_signatures<
        Completions,
        set_value_t,
        __mconst<__mbool<true>>::__f,
        __mconst<__mbool<false>>::__f,
        __mor_t
      >;

      template <class CvrefReceiverId>
      struct operation_t;

      template <class CvrefReceiverId, std::size_t Index>
      struct receiver_t {
        using WhenAll = __copy_cvref_t<CvrefReceiverId, stdexec::__t<when_all_sender_t>>;
        using Receiver = stdexec::__t<__decay_t<CvrefReceiverId>>;
        using SenderId = __m_at_c<Index, SenderIds...>;
        using Completions = completion_sigs<env_of_t<Receiver>, CvrefReceiverId>;
        using Env = make_terminal_stream_env_t<
          stdexec::env<stdexec::prop<get_stop_token_t, inplace_stop_token>, env_of_t<Receiver>>
        >;

        struct type;
        using __t = type;

        struct type : stream_receiver_base {
          using receiver_concept = stdexec::receiver_t;
          using __id = receiver_t;

          template <class... Values>
          STDEXEC_ATTRIBUTE(always_inline)
          void set_value(Values&&... vals) && noexcept {
            op_state_->template _set_value<Index>(static_cast<Values&&>(vals)...);
          }

          template <class Error>
          STDEXEC_ATTRIBUTE(always_inline)
          void set_error(Error&& err) && noexcept {
            op_state_->_set_error(static_cast<Error&&>(err));
          }

          STDEXEC_ATTRIBUTE(always_inline) void set_stopped() && noexcept {
            op_state_->_set_stopped();
          }

          [[nodiscard]]
          auto get_env() const noexcept -> Env {
            return make_terminal_stream_env(
              stdexec::env{
                stdexec::prop{get_stop_token, op_state_->stop_source_.get_token()},
                stdexec::get_env(op_state_->rcvr_)
            },
              &const_cast<stream_provider_t&>(op_state_->stream_providers_[Index]));
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

        template <class SenderId, std::size_t Index>
        using child_op_state_t = exit_operation_state_t<
          __copy_cvref_t<WhenAll, stdexec::__t<SenderId>>,
          stdexec::__t<receiver_t<CvrefReceiverId, Index>>
        >;

        using Indices = __indices_for<SenderIds...>;

        template <size_t... Is>
        static auto connect_children_(operation_t* parent_op, WhenAll&& when_all, __indices<Is...>)
          -> __tuple_for<child_op_state_t<SenderIds, Is>...> {

          using __child_ops_t = __tuple_for<child_op_state_t<SenderIds, Is>...>;
          return when_all.sndrs_.apply(
            [parent_op]<class... Children>(Children&&... children) -> __child_ops_t {
              return __child_ops_t{_strm::exit_op_state(
                static_cast<Children&&>(children),
                stdexec::__t<receiver_t<CvrefReceiverId, Is>>{{}, parent_op},
                stdexec::get_completion_scheduler<set_value_t>(stdexec::get_env(children))
                  .context_state_)...};
            },
            static_cast<WhenAll&&>(when_all).sndrs_);
        }

        using child_op_states_tuple_t =
          decltype(operation_t::connect_children_({}, __declval<WhenAll>(), Indices{}));

        void arrive() noexcept {
          if (1 == count_.fetch_sub(1)) {
            complete();
          }
        }

        void complete() noexcept {
          // Stop callback is no longer needed. Destroy it.
          on_stop_.reset();

          // See if any child operations completed with an error status:
          for (auto status: statuses_) {
            if (status != cudaSuccess) {
              status_ = status;
              break;
            }
          }

          // Synchronize streams
          if (status_ == cudaSuccess) {
            if constexpr (stream_receiver<Receiver>) {
              auto env = stdexec::get_env(rcvr_);
              stream_provider_t* stream_provider = get_stream_provider(env);
              cudaStream_t stream = *stream_provider->own_stream_;

              for (int i = 0; i < sizeof...(SenderIds); ++i) {
                if (status_ == cudaSuccess) {
                  status_ = events_[i].try_wait(stream);
                }
              }
            } else {
              // Synchronize the streams of all the child operations
              child_states_.for_each(_when_all::_sync_op, child_states_);
            }
          }

          if (status_ == cudaSuccess) {
            // All child operations have completed and arrived at the barrier.
            switch (state_.load(std::memory_order_relaxed)) {
            case _when_all::started:
              if constexpr (__v<sends_values<Completions>>) {
                // All child operations completed successfully:
                values_->apply(
                  [this]<class... Tuples>(Tuples&&... value_tupls) noexcept -> void {
                    __tup::__cat_apply(
                      __mk_completion_fn(stdexec::set_value, rcvr_),
                      static_cast<Tuples&&>(value_tupls)...);
                  },
                  static_cast<child_values_tuple_t&&>(*values_));
              }
              break;
            case _when_all::error:
              errors_.visit(
                __mk_completion_fn(stdexec::set_error, rcvr_),
                static_cast<errors_variant_t&&>(errors_));
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

        static auto get_stream_providers(WhenAll& when_all) -> stream_providers_t {
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
          status_ = STDEXEC_LOG_CUDA_API(cudaMallocManaged(&values_, sizeof(child_values_tuple_t)));
        }

        ~operation_t() {
          STDEXEC_ASSERT_CUDA_API(cudaFree(values_));
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
            child_states_.for_each(stdexec::start, child_states_);
            if constexpr (sizeof...(SenderIds) == 0) {
              complete();
            }
          }
        }

        template <class Error>
        void _set_error_impl(Error&& err) noexcept {
          // Transition to the "error" state and switch on the prior state.
          // TODO: What memory orderings are actually needed here?
          switch (state_.exchange(_when_all::error)) {
          case _when_all::started:
            // We must request stop. When the previous state is "error" or "stopped", then stop
            // has already been requested.
            stop_source_.request_stop();
            [[fallthrough]];
          case _when_all::stopped:
            // We are the first child to complete with an error, so we must save the error.
            // (Any subsequent errors are ignored.)
            static_assert(__nothrow_constructible_from<__decay_t<Error>, Error>);
            errors_.template emplace<__decay_t<Error>>(static_cast<Error&&>(err));
            break;
          case _when_all::error:; // We're already in the "error" state. Ignore the error.
          }
        }

        template <std::size_t Index, class... Args>
        void _set_value(Args&&... args) noexcept {
          if constexpr (__v<sends_values<Completions>>) {
            // We only need to bother recording the completion values
            // if we're not already in the "error" or "stopped" state.
            if (state_.load() == _when_all::started) {
              cudaStream_t stream = child_states_.template __get<Index>(child_states_).get_stream();
              if constexpr (sizeof...(Args)) {
                _when_all::copy_kernel<Args&&...><<<1, 1, 0, stream>>>(
                  &(values_->template __get<Index>(*values_)), static_cast<Args&&>(args)...);
                statuses_[Index] = cudaGetLastError();
              }

              if constexpr (stream_receiver<Receiver>) {
                if (statuses_[Index] == cudaSuccess) {
                  statuses_[Index] = events_[Index].try_record(stream);
                }
              }
            }
          }
          arrive();
        }

        template <class Error>
        void _set_error(Error&& err) noexcept {
          _set_error_impl(static_cast<Error&&>(err));
          arrive();
        }

        void _set_stopped() noexcept {
          auto expected = _when_all::started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (state_.compare_exchange_strong(expected, _when_all::stopped)) {
            stop_source_.request_stop();
          }
          arrive();
        }

        // tuple<tuple<Vs1...>, tuple<Vs2...>, ...>
        using child_values_tuple_t = __if<
          sends_values<Completions>,
          __minvoke<
            __qq<__tuple_for>,
            __value_types_of_t<
              stdexec::__t<SenderIds>,
              _when_all::env_t<Env>,
              __qq<__decayed_tuple>,
              __msingle_or<void>
            >...
          >,
          __
        >;

        using errors_variant_t = error_types_of_t<
          stdexec::__t<when_all_sender_t>,
          _when_all::env_t<Env>,
          __uniqued_variant_for
        >;

        Receiver rcvr_;
        cudaError_t status_{cudaSuccess};
        std::atomic<std::size_t> count_{sizeof...(SenderIds)};
        std::array<stream_provider_t, sizeof...(SenderIds)> stream_providers_;
        std::array<detail::cuda_event, sizeof...(SenderIds)> events_{};
        std::array<cudaError_t, sizeof...(SenderIds)> statuses_{}; // all initialized to cudaSuccess
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
      template <__decays_to<type> Self, receiver Receiver>
      static auto connect(Self&& self, Receiver rcvr)
        -> operation_t<__copy_cvref_t<Self, stdexec::__id<Receiver>>> {
        return {static_cast<Self&&>(self), static_cast<Receiver&&>(rcvr)};
      }

      template <__decays_to<type> Self, class... Env>
      static auto get_completion_signatures(Self&&, Env&&...) -> completion_sigs<Self, Env...> {
        return {};
      }

      [[nodiscard]]
      auto get_env() const noexcept -> const env& {
        return env_;
      }

     private:
      __tuple_for<stdexec::__t<SenderIds>...> sndrs_;
    };
  };

  template <>
  struct transform_sender_for<stdexec::when_all_t> {
    template <stream_completing_sender... Senders>
    auto operator()(__ignore, __ignore, Senders&&... sndrs) const {
      using __sender_t =
        __t<when_all_sender_t<stdexec::when_all_t, stream_scheduler, __id<__decay_t<Senders>>...>>;
      return __sender_t{
        context_state_t{nullptr, nullptr, nullptr, nullptr},
        static_cast<Senders&&>(sndrs)...
      };
    }
  };

  template <>
  struct transform_sender_for<stdexec::transfer_when_all_t> {
    template <gpu_stream_scheduler Scheduler, stream_completing_sender... Senders>
    auto operator()(__ignore, Scheduler sched, Senders&&... sndrs) const {
      using __sender_t = __t<when_all_sender_t<
        stdexec::transfer_when_all_t,
        stream_scheduler,
        __id<__decay_t<Senders>>...
      >>;
      return __sender_t{sched.context_state_, static_cast<Senders&&>(sndrs)...};
    }
  };
} // namespace nvexec::_strm

namespace stdexec::__detail {
  template <class WhenAllTag, class Scheduler, class... SenderIds>
  inline constexpr __mconst<
    nvexec::_strm::when_all_sender_t<WhenAllTag, Scheduler, __name_of<__t<SenderIds>>...>
  >
    __name_of_v<nvexec::_strm::when_all_sender_t<WhenAllTag, Scheduler, SenderIds...>>{};
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
