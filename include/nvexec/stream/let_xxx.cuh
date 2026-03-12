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

    template <class Sender, class Receiver, class Fun, class SetTag>
    struct _opstate;

    template <class Fun>
    struct _mk_result_sender_fn
    {
      template <class... Args>
      using __f = __remove_rvalue_reference_t<__call_result_t<Fun, __decay_t<Args>&...>>;
    };

    template <class Sender, class PropagateReceiver, class Fun, class SetTag>
      requires sender_in<Sender, env_of_t<PropagateReceiver>>
    struct _max_sender_size
    {
      using _env_t              = env_of_t<PropagateReceiver>;
      using _result_sender_size = __mcompose<__msizeof, _mk_result_sender_fn<Fun>>;
      using _value_t =
        __gather_completions_of_t<SetTag, Sender, _env_t, _result_sender_size, maxsize>;

      static constexpr std::size_t value = _value_t::value;
    };

    // The environment of the receiver used to connect the secondary (result) sender must
    // correctly report the scheduler and domain on which the sender's operation will be
    // started.
    inline constexpr auto _mk_sch_env =
      []<class CvSender, class Receiver, class SetTag>(CvSender&& sndr, Receiver&& rcvr, SetTag)
    {
      using cv_fn = __copy_cvref_fn<CvSender>;
      return __mk_secondary_env_t<SetTag>()(cv_fn{}, sndr, STDEXEC::get_env(rcvr));
    };

    template <class CvSender, class Receiver, class SetTag>
    using _sch_env_t = __result_of<_mk_sch_env, CvSender, Receiver, SetTag>;

    inline constexpr auto _mk_env2 =
      []<class SchEnv, class Receiver>([[maybe_unused]]
                                       SchEnv const &                        sch_env,
                                       _strm::opstate_base<Receiver> const & opstate)
    {
      //return opstate.make_env();
      return __env::__join(sch_env, opstate.make_env());
    };

    template <class CvSender, class Receiver, class SetTag>
    using _env2_t = __result_of<_mk_env2,
                                _sch_env_t<CvSender, Receiver, SetTag> const &,
                                _strm::opstate_base<Receiver> const &>;

    template <class CvSender, class Receiver, class Fun, class SetTag>
    using _propagate_receiver_t = propagate_receiver<_opstate<CvSender, Receiver, Fun, SetTag>,
                                                     _env2_t<CvSender, Receiver, SetTag>>;

    template <class Sender, class Receiver, class Fun, class SetTag>
    using _mk_opstate_fn = __mcompose<
      __mbind_back_q<connect_result_t, _propagate_receiver_t<Sender, Receiver, Fun, SetTag>>,
      _mk_result_sender_fn<Fun>>;

    template <class SetTag, class Sig>
    struct _tfx_signal_fn
    {
      template <class, class...>
      using __f = completion_signatures<Sig>;
    };

    template <class SetTag, class... Args>
    struct _tfx_signal_fn<SetTag, SetTag(Args...)>
    {
      template <class Fun, class... StreamEnv>
      using __f = __transform_completion_signatures_t<
        __completion_signatures_of_t<__minvoke<_mk_result_sender_fn<Fun>, Args...>, StreamEnv...>,
        completion_signatures<set_error_t(cudaError_t)>>;
    };

    template <class Sig, class Fun, class SetTag, class... StreamEnv>
    using _tfx_signal_t = __minvoke<_tfx_signal_fn<SetTag, Sig>, Fun, StreamEnv...>;

    template <class Sender, class Receiver, class Fun, class SetTag, class... Tuples>
    struct _receiver : public stream_receiver_base
    {
      using _env_t           = _strm::opstate_base<Receiver>::env_t;
      using _result_tuples_t = __mlist<Tuples...>;

      static constexpr std::size_t memory_allocation_size() noexcept
      {
        using _propagate_receiver_t = _let::_propagate_receiver_t<Sender, Receiver, Fun, SetTag>;
        return _max_sender_size<Sender, _propagate_receiver_t, Fun, SetTag>::value;
      }

      template <__same_as<SetTag> Tag, class... Args>
      void _complete(Tag, Args&&... args) noexcept
      {
        using result_sender_t = __minvoke<_mk_result_sender_fn<Fun>, Args...>;

        cudaStream_t stream        = opstate_->get_stream();
        auto*        result_sender = static_cast<result_sender_t*>(opstate_->temp_storage_);
        _let_kernel<Args&&...><<<1, 1, 0, stream>>>(std::move(opstate_->fun_),
                                                    result_sender,
                                                    static_cast<Args&&>(args)...);

        cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
        if (status == cudaSuccess)
        {
          opstate_->defer_temp_storage_destruction(result_sender);
          auto& op = opstate_->_connect_result_sender(std::move(*result_sender));
          STDEXEC::start(op);
        }
        else
        {
          opstate_->propagate_completion_signal(STDEXEC::set_error, cudaError_t(status));
        }
      }

      template <class SetTag2, class... Args>
      void _complete(SetTag2, Args&&... args) noexcept
      {
        opstate_->propagate_completion_signal(SetTag2(), static_cast<Args&&>(args)...);
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept
      {
        _complete(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __err) noexcept
      {
        _complete(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept
      {
        _complete(set_stopped_t());
      }

      auto get_env() const noexcept -> _env_t
      {
        return static_cast<_strm::opstate_base<Receiver>&>(*opstate_).make_env();
      }

      _opstate<Sender, Receiver, Fun, SetTag>* opstate_;
    };

    template <class Sender, class Receiver, class Fun, class SetTag>
    using _receiver_t = __gather_completions_of_t<
      SetTag,
      Sender,
      stream_env_t<env_of_t<Receiver>>,
      __q<__decayed_std_tuple>,
      __munique<__mbind_front_q<_receiver, Sender, Receiver, Fun, SetTag>>>;

    template <class Sender, class Receiver, class Fun, class SetTag>
    using _opstate_base_t =
      _strm::opstate<Sender, _receiver_t<Sender, Receiver, Fun, SetTag>, Receiver>;

    template <class CvSender, class Receiver, class Fun, class SetTag>
    struct _opstate : _opstate_base_t<CvSender, Receiver, Fun, SetTag>
    {
      using _env2_t                = _sch_env_t<CvSender, Receiver, SetTag>;
      using _receiver_t            = _let::_receiver_t<CvSender, Receiver, Fun, SetTag>;
      using _result_tuples_t       = _receiver_t::_result_tuples_t;
      using _mk_opstate_fn_t       = _mk_opstate_fn<CvSender, Receiver, Fun, SetTag>;
      using _mk_opstate_variant_fn = __mtransform<__muncurry<_mk_opstate_fn_t>, __qq<__variant>>;
      using _opstate_variant_t     = __mapply<_mk_opstate_variant_fn, _result_tuples_t>;
      using _propagate_receiver_t  = _let::_propagate_receiver_t<CvSender, Receiver, Fun, SetTag>;

      explicit _opstate(CvSender&& sndr, Receiver rcvr, Fun fun)
        : _opstate(static_cast<CvSender&&>(sndr),
                   static_cast<Receiver&&>(rcvr),
                   static_cast<Fun&&>(fun),
                   _mk_sch_env(sndr, rcvr, SetTag{}))
      {}

      explicit _opstate(CvSender&& sndr, Receiver&& rcvr, Fun fun, _env2_t env2)
        : _opstate_base_t<CvSender, Receiver, Fun, SetTag>(
            static_cast<CvSender&&>(sndr),
            static_cast<Receiver&&>(rcvr),
            [this](__ignore) noexcept { return _receiver_t{{}, this}; },
            get_scheduler(env2).ctx_)
        , fun_(static_cast<Fun&&>(fun))
        , env2_(env2)
      {}

      STDEXEC_IMMOVABLE(_opstate);

      [[nodiscard]]
      auto make_env() const noexcept -> _let::_env2_t<CvSender, Receiver, SetTag>
      {
        return _let::_mk_env2(env2_, *this);
      }

      template <class ResultSender>
      auto _connect_result_sender(ResultSender&& sndr)
        -> connect_result_t<ResultSender, _propagate_receiver_t>&
      {
        return opstate3_.__emplace_from(STDEXEC::connect,
                                        static_cast<ResultSender&&>(sndr),
                                        _propagate_receiver_t(*this));
      }

      Fun                fun_;
      _env2_t            env2_;
      _opstate_variant_t opstate3_{__no_init};
    };
  }  // namespace _let

  template <class Sender, class Fun, class SetTag>
  struct let_sender : public stream_sender_base
  {
   private:
    template <class Self, class Receiver>
    using _opstate_t = _let::_opstate<__copy_cvref_t<Self, Sender>, Receiver, Fun, SetTag>;

    template <class Self, class Receiver>
    using _receiver_t = _let::_receiver_t<__copy_cvref_t<Self, Sender>, Receiver, Fun, SetTag>;

    template <class CvSender, class... StreamEnv>
    using _completions_t =
      __mapply<__mtransform<__mbind_back_q<_let::_tfx_signal_t, Fun, SetTag, StreamEnv...>,
                            __mtry_q<__concat_completion_signatures_t>>,
               __completion_signatures_of_t<CvSender, StreamEnv...>>;

   public:
    explicit let_sender(Sender sndr, Fun fun, SetTag)
      noexcept(__nothrow_move_constructible<Sender, Fun>)
      : sndr_(static_cast<Sender&&>(sndr))
      , fun_(static_cast<Fun&&>(fun))
    {}

    [[nodiscard]]
    auto get_env() const noexcept -> stream_sender_attrs<Sender>
    {
      return {&sndr_};
    }

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return _completions_t<__copy_cvref_t<Self, Sender>, stream_env_t<Env>...>{};
    }

    template <class Self, receiver Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> _opstate_t<Self, Receiver>
    {
      return _opstate_t<Self, Receiver>{static_cast<Self&&>(self).sndr_,
                                        static_cast<Receiver&&>(rcvr),
                                        static_cast<Self&&>(self).fun_};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

   private:
    Sender sndr_;
    Fun    fun_;
  };

  template <class Sender, class Fun, class SetTag>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    let_sender(Sender, Fun, SetTag) -> let_sender<Sender, Fun, SetTag>;

  template <class SetTag>
  struct _transform_let_sender
  {
    template <class Env, class Fun, class Sender>
    auto operator()(Env const &, __ignore, Fun fn, Sender&& sndr) const
    {
      if constexpr (stream_completing_sender<Sender, Env>)
      {
        return let_sender{static_cast<Sender&&>(sndr), static_cast<Fun&&>(fn), SetTag{}};
      }
      else
      {
        using _let_t = decltype(STDEXEC::__let::__let_from_set<SetTag>);
        return _strm::_no_stream_scheduler_in_env<_let_t, Sender, Env>();
      }
    }
  };

  template <>
  struct transform_sender_for<STDEXEC::let_value_t> : _transform_let_sender<set_value_t>
  {};

  template <>
  struct transform_sender_for<STDEXEC::let_error_t> : _transform_let_sender<set_error_t>
  {};

  template <>
  struct transform_sender_for<STDEXEC::let_stopped_t> : _transform_let_sender<set_stopped_t>
  {};
}  // namespace nv::execution::_strm

namespace nvexec = nv::execution;

namespace STDEXEC::__detail
{
  template <class Sender, class Fun, class SetTag>
  extern __declfn_t<nvexec::_strm::let_sender<__demangle_t<Sender>, Fun, SetTag>>
    __demangle_v<nvexec::_strm::let_sender<Sender, Fun, SetTag>>;
}  // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
