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

#include "../../stdexec/__detail/__variant.hpp"
#include "../../stdexec/execution.hpp"

#include "common.cuh"

#include <cuda/std/utility>

#include <cstddef>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution::_strm
{
  // let_value, let_error and let_stopped as facade algorithms. The stream_domain customizes
  // them by re-tagging them with these tags (see _transform_let_sender below).
  template <class LetTag>
  struct let_t : stream_tag_base
  {};

  namespace _let
  {
    using namespace STDEXEC;

    template <class... Args, class Fun, class ResultSenderT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _let_kernel(Fun fn, ResultSenderT* result, Args... args)
    {
      static_assert(trivially_copyable<Fun, Args...>);
      new (result) ResultSenderT(::cuda::std::move(fn)(static_cast<Args&&>(args)...));
    }

    // The key of the operation of the sender that the function returns, given the child's
    // results (decayed)
    template <class... Ts>
    struct result_op
    {};

    // The sender that the function returns
    template <class Fun, class... Ts>
    using result_sender_t = __remove_rvalue_reference_t<__call_result_t<Fun, Ts&...>>;

    // A placeholder in the state's variant of operations
    struct no_op
    {};

    template <class LetTag, class SetTag>
    struct algo;

    template <class LetTag, class SetTag, class Sender, class Receiver>
    struct state : _strm::opstate_base<Receiver>
    {
      using algo_t  = algo<LetTag, SetTag>;
      using fn_t    = __decay_t<__facade::__data_of_t<Sender>>;
      using child_t = __facade::__nth_child_of_t<Sender, 0>;
      using child_rcvr_t =
        __facade::__receiver_t<algo_t, __facade::__child<0>, state, Sender, Receiver>;
      using env2_t = decltype(__mk_secondary_env_t<SetTag>()(
        __declval<__decay_t<child_t> const &>(),
        __declval<__decay_t<env_of_t<Receiver>> const &>()));

      template <class... Ts>
      using result_rcvr_t =
        __facade::__receiver_t<algo_t, result_op<Ts...>, state, Sender, Receiver>;

      // The child's SetTag results, decayed
      using args_lists_t = STDEXEC::__let::__args_lists<
        SetTag,
        __facade::__completions_of_t<algo_t, __facade::__child<0>, Sender, Receiver>>::__t;

      template <class List>
      struct op_of
      {
        using __t = no_op;
      };

      template <class... Ts>
        requires sender_to<result_sender_t<fn_t, Ts...>, result_rcvr_t<Ts...>>
      struct op_of<__mlist<Ts...>>
      {
        using __t = stream_child_op_t<result_sender_t<fn_t, Ts...>, result_rcvr_t<Ts...>>;
      };

      template <class... Lists>
      struct variant_of
      {
        using __t = __variant<no_op, typename op_of<Lists>::__t...>;
      };

      using result_ops_t = __mapply_q<variant_of, args_lists_t>::__t;

      explicit state(Sender&& sndr, Receiver rcvr, context ctx)
        : _strm::opstate_base<Receiver>(static_cast<Receiver&&>(rcvr), ctx)
        , fn_(__facade::__get_data(static_cast<Sender&&>(sndr)))
        , env2_(__mk_secondary_env_t<SetTag>()(__facade::__get_child<0>(sndr),
                                               STDEXEC::get_env(this->rcvr_)))
        , child_op_(
            _strm::connect_stream_child(*this,
                                        __facade::__get_child<0>(static_cast<Sender&&>(sndr)),
                                        child_rcvr_t{this}))
      {}

      ~state()
      {
        if (this->temp_storage_)
        {
          this->ctx_.managed_resource_->deallocate(this->temp_storage_, temp_size_);
        }
      }

      STDEXEC_IMMOVABLE(state);

      fn_t         fn_;
      env2_t const env2_;
      std::size_t  temp_size_{0};
      // The child's operation is kept alive until the operation's: a stream operation must
      // not be destroyed while it completes.
      stream_child_op_t<child_t, child_rcvr_t> child_op_;
      result_ops_t                             result_ops_{__no_init};
    };

    // let_xxx(sndr, fn): when the child completes with SetTag, fn is called with its
    // results in a kernel on the stream, and the sender it returns is connected and started.
    // Its completions, and the child's other completions, are the operation's.
    template <class LetTag, class SetTag>
    struct algo : stream_algorithm
    {
      using stream_algorithm::__env_for;
      using stream_algorithm::__get_env;

      template <class Sender>
      static constexpr bool __depends_on_env = true;

      static constexpr bool __connects_children = false;

      // Customizations are found where the child completes with SetTag, which is where the
      // function is called.
      using __dispatch_source =
        __facade::__via<__facade::__child<0>, SetTag, __facade::__from_start>;

      // The child shares the stream.
      template <class Sender, class Env>
      static auto __env_for(__facade::__child<0>, Sender const &, Env&& env) noexcept
      {
        return _strm::make_stream_env(static_cast<Env&&>(env),
                                      static_cast<stream_provider*>(nullptr));
      }

      template <class State>
      static auto __get_env(__facade::__child<0>, State const & state) noexcept
      {
        return state.make_env();
      }

      // The sender that the function returns sees where the child completed, and the
      // stream.
      template <class... Ts, class Sender, class Env>
      static auto __env_for(result_op<Ts...>, Sender const & sndr, Env&& env) noexcept
      {
        return __env::__join(__mk_secondary_env_t<SetTag>()(__facade::__get_child<0>(sndr), env),
                             _strm::make_stream_env(static_cast<Env&&>(env),
                                                    static_cast<stream_provider*>(nullptr)));
      }

      template <class... Ts, class State>
      static auto __get_env(result_op<Ts...>, State const & state) noexcept
      {
        using env2_t = __decay_t<decltype(state.env2_)>;
        return __env::__join(env2_t(state.env2_), state.make_env());
      }

      // The sender that the function returns is created at run time, so it is known only by
      // its type.
      template <class... Ts, class Sender, class... Env>
      static auto __nested_sender(result_op<Ts...>, Sender const &, Env const &...) noexcept
        -> __mtype<result_sender_t<__decay_t<__facade::__data_of_t<Sender>>, Ts...>>
      {
        return {};
      }

      using stream_algorithm::__nested_sender;

      // The operation runs on the stream of the scheduler on which the child completes.
      template <class Sender, class Receiver>
      static auto
      __get_state(Sender&& sndr, Receiver rcvr) -> state<LetTag, SetTag, Sender, Receiver>
      {
        auto sched = get_completion_scheduler<set_value_t>(
          __facade::__nested_attrs_of<algo>(__facade::__child<0>(), sndr, STDEXEC::get_env(rcvr)),
          algo::__env_for(__facade::__child<0>(), sndr, STDEXEC::get_env(rcvr)));
        context ctx = sched.ctx_;
        return state<LetTag, SetTag, Sender, Receiver>(static_cast<Sender&&>(sndr),
                                                       static_cast<Receiver&&>(rcvr),
                                                       ctx);
      }

      template <class State, class Children>
      static auto __start(State& state, Children&) noexcept
      {
        cudaError_t status = state.stream_provider_.status_;
        return __facade::__choose(
          status == cudaSuccess,
          [&]() noexcept
          { return __facade::__start_nested<__facade::__child<0>>(state.child_op_); },
          [&]() noexcept { return _strm::_report_error(state, status); });
      }

      // The child completed with SetTag: call the function in a kernel, and connect and start
      // the sender it returns.
      template <class State, class... Args>
      static auto __complete(__facade::__child<0>, State& state, SetTag, Args&&... args) noexcept
      {
        using fn_t = State::fn_t;
        if constexpr (!__invocable<fn_t, __decay_t<Args>&...>)
        {
          return __facade::__type_error(__callable_error_t<LetTag, fn_t, __decay_t<Args>&...>());
        }
        else if constexpr (!sender<result_sender_t<fn_t, __decay_t<Args>...>>)
        {
          return __facade::__type_error(__mexception<_WHAT_(_FUNCTION_DOES_NOT_RETURN_A_SENDER_),
                                                     _WHERE_(_IN_ALGORITHM_, LetTag),
                                                     _WITH_FUNCTION_(fn_t),
                                                     _WITH_ARGUMENTS_(Args...)>());
        }
        else
        {
          using sndr2_t = result_sender_t<fn_t, __decay_t<Args>...>;
          using key_t   = result_op<__decay_t<Args>...>;
          using rcvr2_t = State::template result_rcvr_t<__decay_t<Args>...>;

          if constexpr (!sender_to<sndr2_t, rcvr2_t>)
          {
            // If the sender does not type-check in its environment, the walk reports why.
            if constexpr (!sender_in<sndr2_t, env_of_t<rcvr2_t>>)
              return __facade::__effects<__facade::__starts<key_t>>();
            else
              return __facade::__type_error(
                __mexception<_WHAT_(_FUNCTION_RETURNS_A_SENDER_THAT_CANNOT_BE_CONNECTED_),
                             _WHERE_(_IN_ALGORITHM_, LetTag),
                             _WITH_FUNCTION_(fn_t),
                             _WITH_SENDER_<sndr2_t>>());
          }
          else
          {
            using result_t =
              __facade::__union_t<__facade::__effects<__facade::__starts<key_t>>,
                                  __facade::__effects<__facade::__starts<stream_error>>>;
            cudaStream_t stream = state.get_stream();
            void*        memory = nullptr;
            STDEXEC_TRY
            {
              memory = state.ctx_.managed_resource_->allocate(sizeof(sndr2_t));
            }
            STDEXEC_CATCH_ALL {}
            if (memory == nullptr)
            {
              return result_t(_strm::_report_error(state, cudaErrorMemoryAllocation));
            }
            state.temp_storage_ = memory;
            state.temp_size_    = sizeof(sndr2_t);
            auto* sndr2         = static_cast<sndr2_t*>(memory);

            _let_kernel<Args&&...>
              <<<1, 1, 0, stream>>>(std::move(state.fn_), sndr2, static_cast<Args&&>(args)...);

            // The sender is read on the host to connect it.
            cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
            if (status != cudaSuccess)
            {
              return result_t(_strm::_report_error(state, status));
            }

            state.defer_temp_storage_destruction(sndr2);
            auto& op = state.result_ops_.__emplace_from(_strm::connect_stream_child,
                                                        state,
                                                        static_cast<sndr2_t&&>(*sndr2),
                                                        rcvr2_t{&state});
            return result_t(__facade::__start_nested<key_t>(op));
          }
        }
      }

      // The child's other completions, and those of the sender that the function returned
      template <class Key, class State, class Tag, class... Args>
      static auto __complete(Key, State& state, Tag tag, Args&&... args) noexcept
      {
        return _strm::_propagate(state, tag, static_cast<Args&&>(args)...);
      }
    };
  }  // namespace _let

  template <class LetTag>
  struct _transform_let_sender
  {
    template <class Env, class Fun, class Sender>
    auto operator()(Env const &, __ignore, Fun fn, Sender&& sndr) const
    {
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        return STDEXEC::__facade::__sender{_strm::let_t<LetTag>(),
                                           static_cast<Fun&&>(fn),
                                           static_cast<Sender&&>(sndr)};
      }
      else
      {
        return _strm::_no_stream_scheduler_in_env<LetTag, Sender, Env>();
      }
    }
  };

  template <>
  struct transform_sender_for<STDEXEC::let_value_t> : _transform_let_sender<STDEXEC::let_value_t>
  {};

  template <>
  struct transform_sender_for<STDEXEC::let_error_t> : _transform_let_sender<STDEXEC::let_error_t>
  {};

  template <>
  struct transform_sender_for<STDEXEC::let_stopped_t>
    : _transform_let_sender<STDEXEC::let_stopped_t>
  {};
}  // namespace nv::execution::_strm

template <class LetTag>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::let_t<LetTag>>
  : nv::execution::_strm::_let::algo<LetTag, STDEXEC::__t<LetTag>>
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
