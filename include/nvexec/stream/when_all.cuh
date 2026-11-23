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

#include "../../exec/stoppable_base.hpp"
#include "../../stdexec/execution.hpp"

#include "../detail/event.cuh"
#include "../detail/throw_on_cuda_error.cuh"
#include "common.cuh"

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nv::execution::_strm
{
  // when_all as a facade algorithm. The stream_domain customizes when_all(sndrs...) by
  // re-tagging it with this tag (see transform_sender_for<when_all_t> below).
  struct when_all_t : stream_tag_base
  {};

  namespace _when_all
  {
    using STDEXEC::__when_all::_THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_;
    using STDEXEC::__when_all::_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_;

    enum disposition : std::uint32_t
    {
      started,
      error,
      stopped
    };

    template <class... Args, class TupleT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void copy_kernel(TupleT* tpl, Args... args)
    {
      static_assert(trivially_copyable<Args...>);
      *tpl = __decayed_tuple<Args...>{static_cast<Args&&>(args)...};
    }

    // The environment of a child: the stop token of when_all's stop source, and the
    // child's own stream
    template <class Env>
    using local_env_t = STDEXEC::env<STDEXEC::prop<get_stop_token_t, inplace_stop_token>, Env>;

    template <class Env>
    using child_env_t = terminal_stream_env_t<local_env_t<Env>>;

    // A child's stream, as connect_stream_child needs it
    struct stream_view
    {
      [[nodiscard]]
      auto get_stream() const -> cudaStream_t
      {
        return provider_->own_stream_.value();
      }

      [[nodiscard]]
      auto get_stream_context() const -> context
      {
        return provider_->context_;
      }

      context          ctx_;
      stream_provider* provider_;
    };

    template <class Sig>
    struct value_of
    {
      using __t = __mlist<>;
    };

    template <class... Vs>
    struct value_of<set_value_t(Vs...)>
    {
      using __t = __mlist<__decayed_tuple<Vs...>>;
    };

    template <class Sig>
    struct error_of
    {
      using __t = __mlist<>;
    };

    template <class Error>
    struct error_of<set_error_t(Error)>
    {
      using __t = __mlist<__decay_t<Error>>;
    };

    template <class Sigs>
    struct child_results;

    template <class... Sigs>
    struct child_results<completion_signatures<Sigs...>>
    {
      using values_t = __mcall<__mconcat<>, typename value_of<Sigs>::__t...>;
      using errors_t = __mcall<__mconcat<>, typename error_of<Sigs>::__t...>;
    };

    struct algo;

    // The types of when_all's state, computed from the children's completions
    template <class Sender, class Receiver, class Indices>
    struct traits;

    template <class Sender, class Receiver, std::size_t... Is>
    struct traits<Sender, Receiver, __indices<Is...>>
    {
      template <std::size_t I>
      using sigs_t =
        STDEXEC::__facade::__completions_of_t<algo, STDEXEC::__facade::__child<I>, Sender, Receiver>;

      static constexpr bool too_many_values = ((sigs_t<Is>::__count(set_value) > 1) || ...);
      static constexpr bool all_send_values = ((sigs_t<Is>::__count(set_value) == 1) && ...);
      static constexpr bool sends_stopped   = ((sigs_t<Is>::__count(set_stopped) != 0) || ...);

      // A tuple of the children's value tuples, if each child can complete with values
      using values_t =
        __if_c<all_send_values,
               __mcall<__mconcat<__qq<__tuple>>, typename child_results<sigs_t<Is>>::values_t...>,
               __ignore>;

      using errors_t = __mcall<__mconcat<__qq<__uniqued_variant>>,
                               typename child_results<sigs_t<Is>>::errors_t...>;
    };

    template <class Sender, class Receiver>
    using traits_t =
      traits<Sender, Receiver, __make_indices<STDEXEC::__facade::__nbr_children_v<Sender>>>;

    template <class Sender, class Receiver>
    struct state
      : ::exec::stoppable_base_for_t<stop_token_of_t<env_of_t<Receiver>>, state<Sender, Receiver>>
    {
      using receiver_t                 = Receiver;
      using traits_t                   = _when_all::traits_t<Sender, Receiver>;
      using values_t                   = traits_t::values_t;
      using errors_t                   = traits_t::errors_t;
      static constexpr std::size_t nbr = STDEXEC::__facade::__nbr_children_v<Sender>;

      template <class... Contexts>
      explicit state(Receiver rcvr, Contexts... ctxs)
        : rcvr_(static_cast<Receiver&&>(rcvr))
        , stream_providers_{ctxs...}
      {
        if constexpr (!__same_as<values_t, __ignore>)
        {
          status_ = STDEXEC_LOG_CUDA_API(cudaMallocManaged(&values_, sizeof(values_t)));
        }
      }

      ~state()
      {
        if (values_ != nullptr)
        {
          STDEXEC_ASSERT_CUDA_API(cudaFree(values_));
        }
      }

      STDEXEC_IMMOVABLE(state);

      // Called when the receiver's stop token is stopped
      void on_stop_requested() noexcept
      {
        stop_source_.request_stop();
      }

      void register_stop_callback() noexcept
      {
        this->register_stoppable(STDEXEC::get_stop_token(STDEXEC::get_env(rcvr_)));
      }

      void deregister_stop_callback() noexcept
      {
        this->deregister_stoppable();
      }

      Receiver                            rcvr_;
      cudaError_t                         status_{cudaSuccess};
      std::atomic<std::size_t>            count_{nbr};
      std::array<stream_provider, nbr>    stream_providers_;
      std::array<detail::cuda_event, nbr> events_{};
      std::array<cudaError_t, nbr>        statuses_{};  // all initialized to cudaSuccess
      std::atomic<disposition>            disp_{started};
      errors_t                            errors_{__no_init};
      __if_c<__same_as<values_t, __ignore>, void, values_t>* values_{};
      inplace_stop_source                                    stop_source_{};
    };

    // when_all(sndrs...): the children run concurrently, each on its own stream, and the last
    // one to complete completes the operation, once the streams have finished their work:
    // with all their values if each child succeeded, and otherwise with an error or stopped.
    // An error or stopped requests stop on a stop source whose token the children see.
    struct algo : stream_algorithm
    {
      using stream_algorithm::__env_for;
      using stream_algorithm::__get_env;

      template <std::size_t I, class Sender, class Env>
      static auto __env_for(STDEXEC::__facade::__child<I>, Sender const &, Env&& env) noexcept
        -> child_env_t<Env>
      {
        return _strm::make_terminal_stream_env(
          local_env_t<Env>{
            STDEXEC::prop{get_stop_token, inplace_stop_token{}},
            static_cast<Env&&>(env)
        },
          static_cast<stream_provider*>(nullptr));
      }

      template <std::size_t I, class State>
      static auto __get_env(STDEXEC::__facade::__child<I>, State const & state) noexcept
        -> child_env_t<env_of_t<typename State::receiver_t>>
      {
        using env_t = env_of_t<typename State::receiver_t>;
        return _strm::make_terminal_stream_env(
          local_env_t<env_t>{
            STDEXEC::prop{get_stop_token, state.stop_source_.get_token()},
            STDEXEC::get_env(state.rcvr_)
        },
          &const_cast<stream_provider&>(state.stream_providers_[I]));
      }

      // The children run on their own streams.
      template <std::size_t I, class State, class Sender, class Receiver>
      static auto
      __connect(STDEXEC::__facade::__child<I>, State& state, Sender&& sndr, Receiver rcvr)
        -> stream_child_op_t<Sender, Receiver>
      {
        stream_provider& provider = state.stream_providers_[I];
        stream_view      view{provider.context_, &provider};
        return _strm::connect_stream_child(view,
                                           static_cast<Sender&&>(sndr),
                                           static_cast<Receiver&&>(rcvr));
      }

      // A child that completes successfully in more than one way is an error: use
      // when_all_with_variant instead. (Not __if_c, which propagates an error argument.)
      template <class Sender, class Receiver>
      using _state_t = std::conditional_t<
        traits_t<Sender, Receiver>::too_many_values,
        __mexception<_WHAT_(_INVALID_ARGUMENT_),
                     _WHERE_(_IN_ALGORITHM_, STDEXEC::when_all_t),
                     _WHY_(_THE_GIVEN_SENDER_CAN_COMPLETE_SUCCESSFULLY_IN_MORE_THAN_ONE_WAY_),
                     _TO_FIX_THIS_ERROR_(_USE_WHEN_ALL_WITH_VARIANT_INSTEAD_)>,
        state<Sender, Receiver>>;

      template <class Sender, class Receiver>
      static auto __get_state(Sender&& sndr, Receiver rcvr) -> _state_t<Sender, Receiver>
      {
        if constexpr (traits_t<Sender, Receiver>::too_many_values)
          return {};
        else
          return algo::_make_state(sndr,
                                   rcvr,
                                   __make_indices<STDEXEC::__facade::__nbr_children_v<Sender>>());
      }

      template <class State, class Children>
      static auto __start(State& state, Children& children) noexcept
      {
        state.register_stop_callback();
        // If stop has already been requested, don't bother starting the children.
        return STDEXEC::__facade::__choose(
          state.stop_source_.stop_requested(),
          [&]() noexcept { return STDEXEC::__facade::__send(state.rcvr_, set_stopped_t()); },
          [&]() noexcept { return children.__start_all(); });
      }

      // A child completed with values. Copy them, on the child's stream, unless the operation
      // has failed.
      template <std::size_t I, class State, class... Args>
      static auto
      __complete(STDEXEC::__facade::__child<I>, State& state, set_value_t, Args&&... args) noexcept
      {
        if constexpr (!__same_as<typename State::values_t, __ignore>)
        {
          if (state.disp_.load() == started && state.values_ != nullptr)
          {
            cudaStream_t stream = state.stream_providers_[I].own_stream_.value();
            if constexpr (sizeof...(Args) != 0)
            {
              copy_kernel<Args&&...><<<1, 1, 0, stream>>>(&STDEXEC::__get<I>(*state.values_),
                                                          static_cast<Args&&>(args)...);
              state.statuses_[I] = cudaGetLastError();
            }

            if constexpr (stream_receiver<typename State::receiver_t>)
            {
              if (state.statuses_[I] == cudaSuccess)
              {
                state.statuses_[I] = state.events_[I].try_record(stream);
              }
            }
          }
        }
        return algo::_arrive(state);
      }

      // A child failed. The operation cannot complete with values now.
      template <std::size_t I, class State, class Error>
      static auto
      __complete(STDEXEC::__facade::__child<I>, State& state, set_error_t, Error&& err) noexcept
      {
        // Transition to the "error" state and switch on the prior state.
        switch (state.disp_.exchange(error))
        {
        case started:
          // We must request stop. When the previous state is "error" or "stopped", then stop
          // has already been requested.
          state.stop_source_.request_stop();
          [[fallthrough]];
        case stopped:
          // We are the first child to complete with an error, so we must save the error.
          // (Any subsequent errors are ignored.)
          static_assert(__nothrow_constructible_from<__decay_t<Error>, Error>);
          state.errors_.template emplace<__decay_t<Error>>(static_cast<Error&&>(err));
          break;
        case error:;  // We're already in the "error" state. Ignore the error.
        }
        return algo::_arrive(state);
      }

      template <std::size_t I, class State>
      static auto __complete(STDEXEC::__facade::__child<I>, State& state, set_stopped_t) noexcept
      {
        auto expected = started;
        // Transition to the "stopped" state if and only if we're in the "started" state. (If
        // this fails, it's because we're in an error state, which trumps cancellation.)
        if (state.disp_.compare_exchange_strong(expected, stopped))
        {
          state.stop_source_.request_stop();
        }
        return algo::_arrive(state);
      }

     private:
      template <class Sender, class Receiver, std::size_t... Is>
      static auto
      _make_state(Sender& sndr, Receiver& rcvr, __indices<Is...>) -> state<Sender, Receiver>
      {
        return state<Sender, Receiver>(static_cast<Receiver&&>(rcvr),
                                       algo::_child_context<Is>(sndr, rcvr)...);
      }

      // The context of the scheduler on which the child completes, whose stream it uses
      template <std::size_t I, class Sender, class Receiver>
      static auto _child_context(Sender const & sndr, Receiver const & rcvr) -> context
      {
        return STDEXEC::get_completion_scheduler<set_value_t>(
                 STDEXEC::__facade::__nested_attrs_of<algo>(STDEXEC::__facade::__child<I>(),
                                                            sndr,
                                                            STDEXEC::get_env(rcvr)),
                 algo::__env_for(STDEXEC::__facade::__child<I>(), sndr, STDEXEC::get_env(rcvr)))
          .ctx_;
      }

      // Is this the last child to complete? If so, it completes the operation.
      template <class State>
      static auto _arrive(State& state) noexcept
      {
        return STDEXEC::__facade::__when(1 == state.count_.fetch_sub(1),
                                         [&]() noexcept { return algo::_complete(state); });
      }

      template <class State>
      static auto _complete(State& state) noexcept
      {
        using receiver_t = State::receiver_t;

        // Stop callback is no longer needed. Deregister it.
        state.deregister_stop_callback();

        // See if any child operations completed with an error status:
        for (auto status: state.statuses_)
        {
          if (status != cudaSuccess)
          {
            state.status_ = status;
            break;
          }
        }

        // Wait for the children's streams
        if (state.status_ == cudaSuccess)
        {
          if constexpr (stream_receiver<receiver_t>
                        && __callable<get_stream_provider_t, env_of_t<receiver_t> const &>)
          {
            auto             env      = STDEXEC::get_env(state.rcvr_);
            stream_provider* provider = get_stream_provider(env);
            cudaStream_t     stream   = *provider->own_stream_;

            for (auto& event: state.events_)
            {
              if (state.status_ == cudaSuccess)
              {
                state.status_ = event.try_wait(stream);
              }
            }
          }
          else
          {
            for (auto& provider: state.stream_providers_)
            {
              if (state.status_ == cudaSuccess)
              {
                state.status_ = STDEXEC_LOG_CUDA_API(
                  cudaStreamSynchronize(provider.own_stream_.value()));
              }
            }
          }
        }

        return STDEXEC::__facade::__choose(
          state.status_ == cudaSuccess,
          [&]() noexcept
          {
            auto const disp = state.disp_.load(std::memory_order_relaxed);
            return STDEXEC::__facade::__choose(
              disp == started,
              [&]() noexcept { return algo::_send_values(state); },
              [&]() noexcept
              {
                return STDEXEC::__facade::__choose(
                  disp == error,
                  [&]() noexcept { return algo::_send_error(state); },
                  [&]() noexcept { return algo::_send_stopped(state); });
              });
          },
          [&]() noexcept
          {
            return STDEXEC::__facade::__send(state.rcvr_,
                                             set_error_t(),
                                             cudaError_t(state.status_));
          });
      }

      template <class State>
      static auto _send_values(State& state) noexcept
      {
        if constexpr (__same_as<typename State::values_t, __ignore>)
        {
          // Unreachable: some child cannot complete with a value, so the operation cannot.
          return STDEXEC::__facade::__effects<>();
        }
        else
        {
          return STDEXEC::__apply(
            [&]<class... Tuples>(Tuples&... tupls) noexcept
            {
              return STDEXEC::__cat_apply(
                [&]<class... Vs>(Vs&&... vs) noexcept
                {
                  return STDEXEC::__facade::__send(state.rcvr_,
                                                   set_value_t(),
                                                   static_cast<Vs&&>(vs)...);
                },
                static_cast<Tuples&&>(tupls)...);
            },
            *state.values_);
        }
      }

      template <class State>
      static auto _send_error(State& state) noexcept
      {
        return STDEXEC::__facade::__visit(
          state.errors_,
          [&]<class Error>(Error& err) noexcept
          {
            return STDEXEC::__facade::__send(state.rcvr_, set_error_t(), static_cast<Error&&>(err));
          });
      }

      template <class State>
      static auto _send_stopped(State& state) noexcept
      {
        return STDEXEC::__facade::__send(state.rcvr_, set_stopped_t());
      }
    };
  }  // namespace _when_all

  template <>
  struct transform_sender_for<STDEXEC::when_all_t>
  {
    template <class Env, class Data, class... CvSenders>
    constexpr auto operator()(Env const &, __ignore, Data&& data, CvSenders&&... sndrs) const
    {
      if constexpr ((stream_completing_sender<CvSenders, Env> && ...))
      {
        return STDEXEC::__facade::__sender{_strm::when_all_t(),
                                           static_cast<Data&&>(data),
                                           static_cast<CvSenders&&>(sndrs)...};
      }
      else
      {
        // Find the first sender that does not have a stream completion scheduler:
        STDEXEC_CONSTEXPR_LOCAL bool _map[]       = {!stream_completing_sender<CvSenders, Env>...};
        STDEXEC_CONSTEXPR_LOCAL std::size_t index = __pos_of(_map, _map + sizeof...(CvSenders));
        using _invalid_sender_t                   = __m_at_c<index, CvSenders...>;
        return _strm::_no_stream_scheduler_in_env<STDEXEC::when_all_t, _invalid_sender_t, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::when_all_t>
  : nv::execution::_strm::_when_all::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
