/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "../../stdexec/__detail/__optional.hpp"
#include "../../stdexec/execution.hpp"

#include "../../exec/repeat_n.hpp"

#include "common.cuh"

#include <cstddef>
#include <exception>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // exec::repeat_n as a facade algorithm. The stream_domain customizes it by re-tagging it
  // with this tag (see transform_sender_for<exec::repeat_n_t> below).
  struct repeat_n_t : stream_tag_base
  {};

  namespace _repeat_n
  {
    using namespace STDEXEC;

    // The keys of the nested operations: the schedule operation on the stream that begins
    // each iteration (and ends the loop, when no iterations are left), and the body: a copy
    // of the child, which each iteration connects.
    struct sched_op
    {};

    struct body
    {};

    struct algo;

    template <class Sender>
    using child_t = __decay_t<__facade::__nth_child_of_t<Sender, 0>>;

    // The stream scheduler on which the child completes with a value. The operation runs
    // on its stream.
    template <class Sender, class Env>
    auto _sched_of(Sender const & sndr, Env const & env) noexcept
    {
      return get_completion_scheduler<set_value_t>(
        __facade::__nested_attrs_of<algo>(__facade::__child<0>(), sndr, env),
        _strm::make_stream_env(env, static_cast<stream_provider*>(nullptr)));
    }

    template <class Sender, class Env>
    using sched_t = decltype(_repeat_n::_sched_of(__declval<__decay_t<Sender> const &>(),
                                                  __declval<Env const &>()));

    template <class Sender, class Receiver>
    struct state : _strm::opstate_base<Receiver>
    {
      using child_t      = _repeat_n::child_t<Sender>;
      using sched_t      = _repeat_n::sched_t<Sender, env_of_t<Receiver>>;
      using sched_rcvr_t = __facade::__receiver_t<algo, sched_op, state, Sender, Receiver>;
      using body_rcvr_t  = __facade::__receiver_t<algo, body, state, Sender, Receiver>;
      using sched_op_t   = connect_result_t<schedule_result_t<sched_t const &>, sched_rcvr_t>;
      using body_op_t    = stream_child_op_t<child_t, body_rcvr_t>;
      using env2_t =
        decltype(STDEXEC::__mk_sch_env(__declval<sched_t>(),
                                       __declval<__decay_t<env_of_t<Receiver>> const &>()));

      static constexpr bool nothrow_connect_body =
        __nothrow_constructible_from<child_t, child_t&>
        && noexcept(_strm::connect_stream_child(__declval<state&>(),
                                                __declval<child_t>(),
                                                __declval<body_rcvr_t>()));

      explicit state(Sender&& sndr, Receiver rcvr, sched_t sched)
        : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), sched.ctx_)
        , count_(__facade::__get_data(sndr))
        , child_(__facade::__get_child<0>(static_cast<Sender&&>(sndr)))
        , sched_(static_cast<sched_t&&>(sched))
        , env2_(STDEXEC::__mk_sch_env(sched_, STDEXEC::get_env(this->rcvr_)))
      {}

      STDEXEC_IMMOVABLE(state);

      // Destroys the operations, before the operation completes
      void cleanup() noexcept
      {
        body_op_.reset();
        sched_op_.reset();
      }

      std::size_t            count_;
      child_t                child_;
      sched_t                sched_;
      env2_t const           env2_;
      __optional<sched_op_t> sched_op_{};
      __optional<body_op_t>  body_op_{};
    };

    // repeat_n(sndr, count), for a child that completes on a stream scheduler: each
    // iteration schedules on the stream, and then starts a copy of the child, which must
    // complete with no values. The schedule operation ends the loop when no iterations are
    // left, so the operation's value completion always happens there, on the stream (as
    // does a CUDA error). The other completions of the child are the operation's.
    struct algo : stream_algorithm
    {
      static constexpr bool __connects_children = false;

      // Without an environment, the stream scheduler is not known.
      template <class Sender>
      static constexpr bool __depends_on_env = true;

      using __dispatch_source = __facade::__via<sched_op, set_value_t, __facade::__from_start>;

      // The operations share the stream. (The walk checks the child in the same
      // environment.) The body also sees the stream scheduler as its start scheduler.
      template <class Key, class Sender, class Env>
      static auto __env_for(Key, Sender const &, Env&& env) noexcept
      {
        return _strm::make_stream_env(static_cast<Env&&>(env),
                                      static_cast<stream_provider*>(nullptr));
      }

      template <class Key, class State>
      static auto __get_env(Key, State const & state) noexcept
      {
        return state.make_env();
      }

      template <class Sender, class Env>
      static auto __env_for(body, Sender const & sndr, Env&& env) noexcept
      {
        return __env::__join(STDEXEC::__mk_sch_env(_repeat_n::_sched_of(sndr, env), env),
                             _strm::make_stream_env(static_cast<Env&&>(env),
                                                    static_cast<stream_provider*>(nullptr)));
      }

      template <class State>
      static auto __get_env(body, State const & state) noexcept
      {
        using env2_t = __decay_t<decltype(state.env2_)>;
        return __env::__join(env2_t(state.env2_), state.make_env());
      }

      using stream_algorithm::__nested_sender;

      // The schedule operation's sender. (Without an environment, only its type is known.)
      template <class Sender, class... Env>
      static auto __nested_sender(sched_op, Sender const & sndr, Env const &... env) noexcept
      {
        if constexpr (sizeof...(Env) == 1)
          return STDEXEC::schedule(_repeat_n::_sched_of(sndr, env...));
        else
          return __mtype<schedule_result_t<__mfront<stream_scheduler, Sender>>>();
      }

      template <class Sender, class Env>
      static auto __nested_attrs(sched_op, Sender const & sndr, Env const & env) noexcept
      {
        using sched_t = decltype(_repeat_n::_sched_of(sndr, env));
        return __facade::__schedule_attrs<sched_t>{_repeat_n::_sched_of(sndr, env)};
      }

      // The body is a copy of the child, as an rvalue. (Its attributes are the child's.)
      template <class Sender, class... Env>
      static auto
      __nested_sender(body, Sender const & sndr, Env const &...) noexcept -> child_t<Sender>&&
      {
        using child_t = _repeat_n::child_t<Sender>;
        return static_cast<child_t&&>(const_cast<child_t&>(__facade::__get_child<0>(sndr)));
      }

      template <class Sender, class Receiver>
      static auto __get_state(Sender&& sndr, Receiver rcvr)
      {
        using child_t = _repeat_n::child_t<Sender>;
        if constexpr (!__std::copy_constructible<child_t>)
        {
          return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                              _WHERE_(_IN_ALGORITHM_, exec::repeat_n_t),
                              _WHY_(exec::__repeat::_THE_INPUT_SENDER_MUST_BE_COPYABLE_),
                              _WITH_SENDER_<child_t>>();
        }
        else
        {
          auto sched = _repeat_n::_sched_of(sndr, STDEXEC::get_env(rcvr));
          return state<Sender, Receiver>(static_cast<Sender&&>(sndr),
                                         static_cast<Receiver&&>(rcvr),
                                         std::move(sched));
        }
      }

      // Begins an iteration: the schedule operation
      template <class State>
      static auto _begin_iteration(State& state) noexcept
      {
        using sched_rcvr_t = State::sched_rcvr_t;
        auto& op           = state.sched_op_.__emplace_from(STDEXEC::connect,
                                                  STDEXEC::schedule(state.sched_),
                                                  sched_rcvr_t{&state});
        return __facade::__start_nested<sched_op>(op);
      }

      template <class State, class Children>
      static auto __start(State& state, Children&) noexcept
      {
        cudaError_t status = state.stream_provider_.status_;
        return __facade::__choose(
          status == cudaSuccess,
          [&]() noexcept { return algo::_begin_iteration(state); },
          [&]() noexcept { return _strm::_report_error(state, status); });
      }

      // On the stream: the loop is done if no iterations are left. Otherwise the iteration
      // connects and starts its body.
      template <class State>
      static auto __complete(sched_op, State& state, set_value_t) noexcept
      {
        auto start_body = [&]() noexcept
        {
          using body_rcvr_t      = State::body_rcvr_t;
          auto connect_and_start = [&]()
          {
            using child_t = State::child_t;
            auto& op      = state.body_op_.__emplace_from(
              [&]
              {
                return _strm::connect_stream_child(state,
                                                   child_t(state.child_),
                                                   body_rcvr_t{&state});
              });
            return __facade::__start_nested<body>(op);
          };

          if constexpr (State::nothrow_connect_body)
          {
            return connect_and_start();
          }
          else
          {
            using result_t =
              __facade::__union_t<decltype(connect_and_start()),
                                  __facade::__effects<set_error_t(std::exception_ptr)>>;
            STDEXEC_TRY
            {
              return result_t(connect_and_start());
            }
            STDEXEC_CATCH_ALL
            {
              state.cleanup();
              return result_t(_strm::_propagate(state, set_error_t(), std::current_exception()));
            }
          }
        };

        return __facade::__choose(
          state.count_ == 0,
          [&]() noexcept
          {
            state.cleanup();
            return _strm::_propagate(state, set_value_t());
          },
          start_body);
      }

      // The body completed with no values: one fewer iteration is left.
      template <class State, class... Args>
      static auto __complete(body, State& state, set_value_t, Args&&...) noexcept
      {
        if constexpr (sizeof...(Args) != 0)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_),
                         _WHERE_(_IN_ALGORITHM_, exec::repeat_n_t),
                         _WHY_(exec::_THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_),
                         _WITH_SENDER_<typename State::child_t>>());
        }
        else
        {
          --state.count_;
          return algo::_begin_iteration(state);
        }
      }

      // The other completions of the body and the schedule operation are the operation's.
      // An error is copied before the operation that sent it is destroyed, because it may
      // be in it.
      template <class Key, class State, class Error>
        requires __one_of<Key, sched_op, body>
      static auto __complete(Key, State& state, set_error_t, Error&& err) noexcept
      {
        auto send_copy = [&]()
        {
          __decay_t<Error> copy(static_cast<Error&&>(err));
          state.cleanup();
          return _strm::_propagate(state, set_error_t(), static_cast<__decay_t<Error>&&>(copy));
        };

        if constexpr (__nothrow_decay_copyable<Error>)
        {
          return send_copy();
        }
        else
        {
          using result_t =
            __facade::__union_t<decltype(send_copy()),
                                __facade::__effects<set_error_t(std::exception_ptr)>>;
          STDEXEC_TRY
          {
            return result_t(send_copy());
          }
          STDEXEC_CATCH_ALL
          {
            state.cleanup();
            return result_t(_strm::_propagate(state, set_error_t(), std::current_exception()));
          }
        }
      }

      template <class Key, class State>
        requires __one_of<Key, sched_op, body>
      static auto __complete(Key, State& state, set_stopped_t) noexcept
      {
        state.cleanup();
        return _strm::_propagate(state, set_stopped_t());
      }

      // CUDA errors (see _report_error)
      template <class State, class Tag, class... Args>
      static auto __complete(stream_error, State& state, Tag tag, Args&&... args) noexcept
      {
        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }
    };
  }  // namespace _repeat_n

  template <>
  struct transform_sender_for<exec::repeat_n_t>
  {
    template <class Env, class Sender>
    auto operator()(Env const &, STDEXEC::__ignore, std::size_t count, Sender&& sndr) const
    {
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::repeat_n_t(), count, static_cast<Sender&&>(sndr)};
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<exec::repeat_n_t, Sender, Env>();
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::repeat_n_t>
  : nv::execution::_strm::_repeat_n::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
