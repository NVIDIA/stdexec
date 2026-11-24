/*
 * Copyright (c) 2026 NVIDIA Corporation
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
#include "../../stdexec/functional.hpp"

#include "let_xxx.cuh"

#include "common.cuh"

namespace nv::execution::_strm
{
  // starts_on as a facade algorithm. The stream_domain customizes starts_on(sch, sndr) for
  // a stream scheduler by re-tagging it with this tag (see transform_sender_for<starts_on_t>
  // below).
  struct starts_on_t : stream_tag_base
  {};

  namespace _starts_on
  {
    // The key of the schedule operation
    struct sched_op
    {};

    struct algo;

    template <class Sender, class Receiver>
    struct state : _strm::opstate_base<Receiver>
    {
      using scheduler_t = __decay_t<STDEXEC::__facade::__data_of_t<Sender>>;
      using child_t     = STDEXEC::__facade::__nth_child_of_t<Sender, 0>;
      using env2_t =
        decltype(STDEXEC::__mk_sch_env(__declval<scheduler_t>(),
                                       __declval<__decay_t<env_of_t<Receiver>> const &>()));
      using sched_sndr_t = schedule_result_t<scheduler_t const &>;
      using sched_rcvr_t = STDEXEC::__facade::__receiver_t<algo, sched_op, state, Sender, Receiver>;
      using child_rcvr_t = STDEXEC::__facade::
        __receiver_t<algo, STDEXEC::__facade::__child<0>, state, Sender, Receiver>;

      explicit state(Sender&& sndr, Receiver rcvr)
        : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr),
                                        STDEXEC::__facade::__get_data(sndr).ctx_)
        , env2_(STDEXEC::__mk_sch_env(STDEXEC::__facade::__get_data(sndr),
                                      STDEXEC::get_env(this->rcvr_)))
        , sched_op_(STDEXEC::connect(STDEXEC::schedule(static_cast<scheduler_t const &>(
                                       STDEXEC::__facade::__get_data(sndr))),
                                     sched_rcvr_t{this}))
        , child_op_(_strm::connect_stream_child(*this,
                                                STDEXEC::__facade::__get_child<0>(
                                                  static_cast<Sender&&>(sndr)),
                                                child_rcvr_t{this}))
      {}

      STDEXEC_IMMOVABLE(state);

      env2_t const                                 env2_;
      connect_result_t<sched_sndr_t, sched_rcvr_t> sched_op_;
      stream_child_op_t<child_t, child_rcvr_t>     child_op_;
    };

    // starts_on(sch, sndr): schedules on the stream scheduler sch, and then starts sndr,
    // which sees sch as its start scheduler, and the stream in its environment. The
    // completions of sndr, and the error and stopped completions of the schedule
    // operation, are the operation's.
    struct algo : stream_algorithm
    {
      using stream_algorithm::__env_for;
      using stream_algorithm::__get_env;

      static constexpr bool __connects_children = false;

      using __sched_value_t =
        STDEXEC::__facade::__via<sched_op, set_value_t, STDEXEC::__facade::__from_start>;
      using __dispatch_source = __sched_value_t;

      template <class Tag>
      using __empty_channel_source = __if_c<__same_as<Tag, set_value_t>, __sched_value_t, void>;

      // The schedule operation and the child share the stream. The child also sees the
      // scheduler as its start scheduler.
      template <class Sender, class Env>
      static auto __env_for(sched_op, Sender const &, Env&& env) noexcept
      {
        return _strm::make_stream_env(static_cast<Env&&>(env),
                                      static_cast<stream_provider*>(nullptr));
      }

      template <class State>
      static auto __get_env(sched_op, State const & state) noexcept
      {
        return state.make_env();
      }

      template <class Sender, class Env>
      static auto __env_for(STDEXEC::__facade::__child<0>, Sender const & sndr, Env&& env) noexcept
      {
        return STDEXEC::__env::__join(
          STDEXEC::__mk_sch_env(STDEXEC::__facade::__get_data(sndr), env),
          _strm::make_stream_env(static_cast<Env&&>(env), static_cast<stream_provider*>(nullptr)));
      }

      template <class State>
      static auto __get_env(STDEXEC::__facade::__child<0>, State const & state) noexcept
      {
        using env2_t = __decay_t<decltype(state.env2_)>;
        return STDEXEC::__env::__join(env2_t(state.env2_), state.make_env());
      }

      using stream_algorithm::__nested_sender;

      template <class Sender, class... Env>
      static auto __nested_sender(sched_op, Sender const & sndr, Env const &...)
      {
        return STDEXEC::schedule(STDEXEC::__facade::__get_data(sndr));
      }

      template <class Sender, class... Env>
      static auto __nested_attrs(sched_op, Sender const & sndr, Env const &...) noexcept
      {
        return STDEXEC::__facade::__schedule_attrs<decltype(STDEXEC::__facade::__get_data(sndr))>{
          STDEXEC::__facade::__get_data(sndr)};
      }

      template <class Sender, class Receiver>
      static auto __get_state(Sender&& sndr, Receiver rcvr) -> state<Sender, Receiver>
      {
        return state<Sender, Receiver>(static_cast<Sender&&>(sndr), static_cast<Receiver&&>(rcvr));
      }

      template <class State, class Children>
      static auto __start(State& state, Children&) noexcept
      {
        cudaError_t status = state.stream_provider_.status_;
        return STDEXEC::__facade::__choose(
          status == cudaSuccess,
          [&]() noexcept { return STDEXEC::__facade::__start_nested<sched_op>(state.sched_op_); },
          [&]() noexcept { return _strm::_report_error(state, status); });
      }

      // On the scheduler: start the child.
      template <class State>
      static auto __complete(sched_op, State& state, set_value_t) noexcept
      {
        return STDEXEC::__facade::__start_nested<STDEXEC::__facade::__child<0>>(state.child_op_);
      }

      // The child completed, or the schedule operation failed or was stopped.
      template <class Key, class State, class Tag, class... Args>
      static auto __complete(Key, State& state, Tag tag, Args&&... args) noexcept
      {
        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }
    };
  }  // namespace _starts_on

  template <>
  struct transform_sender_for<STDEXEC::starts_on_t>
  {
    template <class Env, STDEXEC::scheduler Scheduler, STDEXEC::sender Sender>
    auto operator()(Env const &, STDEXEC::starts_on_t, Scheduler&& sched, Sender&& sndr) const
    {
      if constexpr (gpu_stream_scheduler<__decay_t<Scheduler>, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::starts_on_t(),
                                           static_cast<Scheduler&&>(sched),
                                           static_cast<Sender&&>(sndr)};
      }
      else
      {
        return STDEXEC::let_value(STDEXEC::schedule(sched),
                                  STDEXEC::__always(static_cast<Sender&&>(sndr)));
      }
    }
  };
}  // namespace nv::execution::_strm

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::starts_on_t>
  : nv::execution::_strm::_starts_on::algo
{};
