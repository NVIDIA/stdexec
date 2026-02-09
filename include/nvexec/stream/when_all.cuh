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

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "../detail/event.cuh"
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nvexec::_strm {
  namespace _when_all {
    enum disposition : std::uint32_t {
      started,
      error,
      stopped
    };

    template <class Env>
    using env_t = STDEXEC::env<STDEXEC::prop<get_stop_token_t, inplace_stop_token>, Env>;

    template <class Sender, class... Env>
    concept valid_child_sender = sender_in<Sender, Env...> && requires {
      requires(STDEXEC::__count_of<set_value_t, Sender, Env...>::value <= 1);
    };

    template <class Sender, class... Env>
    concept too_many_completions_sender = sender_in<Sender, Env...> && requires {
      requires(STDEXEC::__count_of<set_value_t, Sender, Env...>::value > 1);
    };

    template <class... Args, class TupleT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(TupleT* tpl, Args... args) {
      static_assert(trivially_copyable<Args...>);
      *tpl = __decayed_tuple<Args...>{static_cast<Args&&>(args)...};
    }

    template <class Env, class... Senders>
    struct completions { };

    template <class... Env, class... Senders>
      requires(too_many_completions_sender<Senders, Env...> || ...)
    struct completions<__mlist<Env...>, Senders...> {
      static constexpr auto position_of() noexcept -> std::size_t {
        constexpr bool which[] = {too_many_completions_sender<Senders, Env...>...};
        return __pos_of(which, which + sizeof...(Senders));
      }

      using invalid_arg_t = __m_at_c<position_of(), Senders...>;
      using __t = STDEXEC::__when_all::__too_many_value_completions_error_t<invalid_arg_t, Env...>;
    };

    template <class... Env, class... Senders>
      requires(valid_child_sender<Senders, Env...> && ...)
    struct completions<__mlist<Env...>, Senders...> {
      using non_values_t = __minvoke_q<
        __concat_completion_signatures_t,
        completion_signatures<set_error_t(cudaError_t), set_stopped_t()>,
        transform_completion_signatures<
          __completion_signatures_of_t<Senders, Env...>,
          completion_signatures<>,
          __mconst<completion_signatures<>>::__f
        >...
      >;

      using values_t = __minvoke<
        __mconcat<__qf<set_value_t>>,
        __value_types_t<
          __completion_signatures_of_t<Senders, Env...>,
          __q<__mlist>,
          __msingle_or<__mlist<>>
        >...
      >;

      using __t = __if_c<
        (__sends<set_value_t, Senders, Env...> && ...),
        __minvoke<__mpush_back<__q<completion_signatures>>, non_values_t, values_t>,
        non_values_t
      >;
    };

    inline constexpr auto _sync_op = []<class OpT>(OpT& op) noexcept {
      if constexpr (STDEXEC_IS_BASE_OF(stream_opstate_base, OpT)) {
        if (op.stream_provider_.status_ == cudaSuccess) {
          op.stream_provider_
            .status_ = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(op.get_stream()));
        }
      }
    };
  } // namespace _when_all

  template <class WhenAllTag, class Scheduler, class... Senders>
  struct when_all_sender : stream_sender_base {
   private:
    struct attrs;

    template <class CvReceiver>
    struct opstate;

    template <class Cv, class... Env>
    using _completions_t = STDEXEC::__t<
      _when_all::completions<__mlist<_when_all::env_t<Env>...>, __copy_cvref_t<Cv, Senders>...>
    >;

   public:
    template <class... Sndrs>
    explicit when_all_sender(context ctx, Sndrs&&... sndrs)
      : attrs_{ctx}
      , sndrs_{static_cast<Sndrs&&>(sndrs)...} {
    }

    template <__decays_to<when_all_sender> Self, receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> opstate<__copy_cvref_t<Self, Receiver>> {
      return {static_cast<Self&&>(self), static_cast<Receiver&&>(rcvr)};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <__decays_to<when_all_sender> Self, class... Env>
    static consteval auto get_completion_signatures() -> _completions_t<Self, Env...> {
      return {};
    }

    [[nodiscard]]
    auto get_env() const noexcept -> const attrs& {
      return attrs_;
    }

   private:
    template <class Completions>
    using sends_values_t = __gather_completion_signatures_t<
      Completions,
      set_value_t,
      __mconst<__mbool<true>>::__f,
      __mconst<__mbool<false>>::__f,
      __mor_t
    >;

    struct attrs {
      template <class... Env>
      using sched_domain_t = __call_result_or_t<
        get_completion_domain_t<set_value_t>,
        indeterminate_domain<>,
        Scheduler,
        Env...
      >;

      auto query(get_completion_scheduler_t<set_value_t>, __ignore = {}) const noexcept -> Scheduler
        requires STDEXEC::__same_as<WhenAllTag, transfer_when_all_t>
      {
        return Scheduler(ctx_);
      }

      template <class... Env>
      constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept {
        if constexpr (STDEXEC::__same_as<WhenAllTag, transfer_when_all_t>) {
          return sched_domain_t<Env...>{};
        } else {
          static_assert(
            sizeof...(Senders) == 0
            || STDEXEC::__same_as<
              stream_domain,
              __common_domain_t<STDEXEC::__completion_domain_of_t<set_value_t, Senders, Env...>...>
            >);
          return stream_domain{};
        }
      }

      context ctx_;
    };

    template <class CvReceiver, std::size_t Index>
    struct receiver : stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;
      using when_all_sender_t = __copy_cvref_t<CvReceiver, when_all_sender>;
      using receiver_t = __decay_t<CvReceiver>;
      using sender_t = __m_at_c<Index, Senders...>;
      using completions_t = when_all_sender::_completions_t<env_of_t<receiver_t>, CvReceiver>;
      using env_t = make_terminal_stream_env_t<
        STDEXEC::env<STDEXEC::prop<get_stop_token_t, inplace_stop_token>, env_of_t<receiver_t>>
      >;

      template <class... Values>
      STDEXEC_ATTRIBUTE(always_inline)
      void set_value(Values&&... vals) && noexcept {
        opstate_->template _set_value<Index>(static_cast<Values&&>(vals)...);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(always_inline)
      void set_error(Error&& err) && noexcept {
        opstate_->_set_error(static_cast<Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(always_inline) void set_stopped() && noexcept {
        opstate_->_set_stopped();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return make_terminal_stream_env(
          STDEXEC::env{
            STDEXEC::prop{get_stop_token, opstate_->stop_source_.get_token()},
            STDEXEC::get_env(opstate_->rcvr_)
        },
          &const_cast<stream_provider&>(opstate_->stream_providers_[Index]));
      }

      opstate<CvReceiver>* opstate_;
    };

    template <class CvReceiver>
    struct opstate : stream_opstate_base {
     private:
      using _when_all_sender_t = __copy_cvref_t<CvReceiver, when_all_sender>;
      using _receiver_t = __decay_t<CvReceiver>;

     public:
      opstate(_when_all_sender_t&& sndr, _receiver_t rcvr)
        : rcvr_(static_cast<_receiver_t&&>(rcvr))
        , stream_providers_{opstate::_get_stream_providers(sndr, rcvr_)}
        , child_states_{opstate::_connect_children(
            this,
            static_cast<_when_all_sender_t&&>(sndr),
            _indices_t{})} {
        status_ = STDEXEC_LOG_CUDA_API(cudaMallocManaged(&values_, sizeof(_child_values_t)));
      }

      ~opstate() {
        STDEXEC_ASSERT_CUDA_API(cudaFree(values_));
      }

      STDEXEC_IMMOVABLE(opstate);

      void start() & noexcept {
        // register stop callback:
        auto tok = STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_));
        on_stop_.emplace(std::move(tok), __forward_stop_request{stop_source_});
        if (stop_source_.stop_requested()) {
          // Stop has already been requested. Don't bother starting
          // the child operations.
          STDEXEC::set_stopped(static_cast<_receiver_t&&>(rcvr_));
        } else {
          STDEXEC::__apply(STDEXEC::__for_each{STDEXEC::start}, child_states_);
          if constexpr (sizeof...(Senders) == 0) {
            _complete();
          }
        }
      }

     private:
      template <class, std::size_t>
      friend struct receiver;

      using _env_t = env_of_t<_receiver_t>;
      using _completions_t = when_all_sender::_completions_t<_env_t, CvReceiver>;
      using _indices_t = __indices_for<Senders...>;
      using _stream_providers_t = std::array<stream_provider, sizeof...(Senders)>;
      using _stop_callback_t =
        stop_callback_for_t<stop_token_of_t<env_of_t<_receiver_t>>, __forward_stop_request>;

      template <class Sender, std::size_t Index>
      using _child_opstate_t =
        exit_opstate_t<__copy_cvref_t<_when_all_sender_t, Sender>, receiver<CvReceiver, Index>>;

      // tuple<tuple<Vs1...>, tuple<Vs2...>, ...>
      using _child_values_t = __if<
        sends_values_t<_completions_t>,
        __minvoke<
          __qq<__tuple>,
          __value_types_of_t<
            Senders,
            _when_all::env_t<_env_t>,
            __qq<__decayed_tuple>,
            __msingle_or<void>
          >...
        >,
        __
      >;

      using _errors_t =
        error_types_of_t<when_all_sender, _when_all::env_t<_env_t>, __uniqued_variant>;

      template <size_t... Is>
      static auto
        _connect_children(opstate* parent_op, _when_all_sender_t&& when_all, __indices<Is...>)
          -> __tuple<_child_opstate_t<Senders, Is>...> {
        using _child_opstates_t = __tuple<_child_opstate_t<Senders, Is>...>;
        return STDEXEC::__apply(
          [parent_op]<class... Children>(Children&&... children) -> _child_opstates_t {
            return _child_opstates_t{_strm::exit_opstate(
              static_cast<Children&&>(children),
              receiver<CvReceiver, Is>{{}, parent_op},
              STDEXEC::get_completion_scheduler<set_value_t>(
                STDEXEC::get_env(children), STDEXEC::get_env(parent_op->rcvr_))
                .ctx_)...};
          },
          static_cast<_when_all_sender_t&&>(when_all).sndrs_);
      }

      using _child_opstates_t =
        decltype(opstate::_connect_children({}, __declval<_when_all_sender_t>(), _indices_t{}));

      void _arrive() noexcept {
        if (1 == count_.fetch_sub(1)) {
          _complete();
        }
      }

      void _complete() noexcept {
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
          if constexpr (stream_receiver<_receiver_t>) {
            auto env = STDEXEC::get_env(rcvr_);
            stream_provider* stream_provider = get_stream_provider(env);
            cudaStream_t stream = *stream_provider->own_stream_;

            for (int i = 0; i < sizeof...(Senders); ++i) {
              if (status_ == cudaSuccess) {
                status_ = events_[i].try_wait(stream);
              }
            }
          } else {
            // Synchronize the streams of all the child operations
            STDEXEC::__apply(STDEXEC::__for_each{_when_all::_sync_op}, child_states_);
          }
        }

        if (status_ == cudaSuccess) {
          // All child operations have completed and arrived at the barrier.
          switch (state_.load(std::memory_order_relaxed)) {
          case _when_all::started:
            if constexpr (sends_values_t<_completions_t>::value) {
              // All child operations completed successfully:
              STDEXEC::__apply(
                [this]<class... Tuples>(Tuples&&... value_tupls) noexcept -> void {
                  STDEXEC::__cat_apply(
                    __mk_completion_fn(STDEXEC::set_value, rcvr_),
                    static_cast<Tuples&&>(value_tupls)...);
                },
                static_cast<_child_values_t&&>(*values_));
            }
            break;
          case _when_all::error:
            STDEXEC::__visit(
              __mk_completion_fn(STDEXEC::set_error, rcvr_), static_cast<_errors_t&&>(errors_));
            break;
          case _when_all::stopped:
            STDEXEC::set_stopped(static_cast<_receiver_t&&>(rcvr_));
            break;
          default:;
          }
        } else {
          STDEXEC::set_error(static_cast<_receiver_t&&>(rcvr_), std::move(status_));
        }
      }

      static auto _get_stream_providers(_when_all_sender_t& when_all, _receiver_t& rcvr)
        -> _stream_providers_t {
        return STDEXEC::__apply(
          [](auto& rcvr, auto&... sndrs) -> _stream_providers_t {
            return _stream_providers_t{STDEXEC::get_completion_scheduler<set_value_t>(
                                         STDEXEC::get_env(sndrs), STDEXEC::get_env(rcvr))
                                         .ctx_...};
          },
          when_all.sndrs_,
          rcvr);
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
        if constexpr (sends_values_t<_completions_t>::value) {
          // We only need to bother recording the completion values
          // if we're not already in the "error" or "stopped" state.
          if (state_.load() == _when_all::started) {
            cudaStream_t stream = STDEXEC::__get<Index>(child_states_).get_stream();
            if constexpr (sizeof...(Args)) {
              _when_all::copy_kernel<Args&&...><<<1, 1, 0, stream>>>(
                &STDEXEC::__get<Index>(*values_), static_cast<Args&&>(args)...);
              statuses_[Index] = cudaGetLastError();
            }

            if constexpr (stream_receiver<_receiver_t>) {
              if (statuses_[Index] == cudaSuccess) {
                statuses_[Index] = events_[Index].try_record(stream);
              }
            }
          }
        }
        _arrive();
      }

      template <class Error>
      void _set_error(Error&& err) noexcept {
        _set_error_impl(static_cast<Error&&>(err));
        _arrive();
      }

      void _set_stopped() noexcept {
        auto expected = _when_all::started;
        // Transition to the "stopped" state if and only if we're in the
        // "started" state. (If this fails, it's because we're in an
        // error state, which trumps cancellation.)
        if (state_.compare_exchange_strong(expected, _when_all::stopped)) {
          stop_source_.request_stop();
        }
        _arrive();
      }

      _receiver_t rcvr_;
      cudaError_t status_{cudaSuccess};
      std::atomic<std::size_t> count_{sizeof...(Senders)};
      std::array<stream_provider, sizeof...(Senders)> stream_providers_;
      std::array<detail::cuda_event, sizeof...(Senders)> events_{};
      std::array<cudaError_t, sizeof...(Senders)> statuses_{}; // all initialized to cudaSuccess
      _child_opstates_t child_states_;
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      std::atomic<_when_all::disposition> state_{_when_all::started};

      _errors_t errors_{__no_init};
      _child_values_t* values_{};
      inplace_stop_source stop_source_{};

      std::optional<_stop_callback_t> on_stop_{};
    };

    attrs attrs_;
    STDEXEC::__tuple<Senders...> sndrs_;
  };

  template <class Env>
  struct transform_sender_for<STDEXEC::when_all_t, Env> {
    template <stream_completing_sender<Env>... CvSenders>
    constexpr auto operator()(__ignore, __ignore, CvSenders&&... sndrs) const {
      using sender_t =
        when_all_sender<STDEXEC::when_all_t, stream_scheduler, __decay_t<CvSenders>...>;
      return sender_t{
        context{nullptr, nullptr, nullptr, nullptr},
        static_cast<CvSenders&&>(sndrs)...
      };
    }

    const Env& env_;
  };

  template <class Env>
  struct transform_sender_for<STDEXEC::transfer_when_all_t, Env> {
    template <gpu_stream_scheduler<Env> Scheduler, stream_completing_sender<Env>... CvSenders>
    auto operator()(__ignore, Scheduler sched, CvSenders&&... sndrs) const {
      using sender_t =
        when_all_sender<STDEXEC::transfer_when_all_t, stream_scheduler, __decay_t<CvSenders>...>;
      return sender_t{sched.ctx_, static_cast<CvSenders&&>(sndrs)...};
    }

    const Env& env_;
  };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class WhenAllTag, class Scheduler, class... Senders>
  extern __declfn_t<nvexec::_strm::when_all_sender<WhenAllTag, Scheduler, __demangle_t<Senders>...>>
    __demangle_v<nvexec::_strm::when_all_sender<WhenAllTag, Scheduler, Senders...>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
