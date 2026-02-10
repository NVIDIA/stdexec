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

#include "common.cuh"

#include <cuda/std/utility>

#include <cstddef>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace let_xxx {
    using namespace STDEXEC;

    template <class... Args, class Fun, class ResultSenderT>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void _let_xxx_kernel(Fun fn, ResultSenderT* result, Args... args) {
      static_assert(trivially_copyable<Fun, Args...>);
      new (result) ResultSenderT(::cuda::std::move(fn)(static_cast<Args&&>(args)...));
    }

    template <class Tp>
    using __decay_ref_t = STDEXEC::__decay_t<Tp>&;

    template <class Fun>
    using _mk_result_sender_t =
      __mtransform<__q<__decay_ref_t>, __mbind_front_q<__call_result_t, Fun>>;

    template <class Sender, class PropagateReceiver, class Fun, class SetTag>
      requires sender_in<Sender, env_of_t<PropagateReceiver>>
    struct __max_sender_size {
      struct _result_sender_size {
        template <class... Args>
        using __f = __msize_t<sizeof(__minvoke<_mk_result_sender_t<Fun>, Args...>)>;
      };

      static constexpr std::size_t value = __gather_completions_of_t<
        SetTag,
        Sender,
        env_of_t<PropagateReceiver>,
        _result_sender_size,
        maxsize
      >::value;
    };

    template <class Receiver, class Fun>
    using opstate_for_t = __mcompose<
      __mbind_back_q<connect_result_t, propagate_receiver<Receiver>>,
      _mk_result_sender_t<Fun>
    >;

    template <class Set, class Sig>
    struct __tfx_signal_fn {
      template <class, class...>
      using __f = completion_signatures<Sig>;
    };

    template <class Set, class... Args>
    struct __tfx_signal_fn<Set, Set(Args...)> {
      template <class Fun, class... StreamEnv>
      using __f = transform_completion_signatures<
        __completion_signatures_of_t<__minvoke<_mk_result_sender_t<Fun>, Args...>, StreamEnv...>,
        completion_signatures<set_error_t(cudaError_t)>
      >;
    };

    template <class Sig, class Fun, class Set, class... StreamEnv>
    using __tfx_signal_t = __minvoke<__tfx_signal_fn<Set, Sig>, Fun, StreamEnv...>;

    template <class Sender, class Receiver, class Fun, class Let>
    struct opstate;

    template <class Sender, class Receiver, class Fun, class Set, class... Tuples>
    struct receiver : public stream_receiver_base {
      using env_t = _strm::opstate_base<Receiver>::env_t;
      static constexpr std::size_t memory_allocation_size() noexcept {
        return __max_sender_size<Sender, propagate_receiver<Receiver>, Fun, Set>::value;
      }

      template <__same_as<Set> Tag, class... Args>
      void _complete(Tag, Args&&... args) noexcept {
        using result_sender_t = __minvoke<_mk_result_sender_t<Fun>, Args...>;
        using opstate_t = __minvoke<opstate_for_t<Receiver, Fun>, Args...>;

        cudaStream_t stream = opstate_->get_stream();
        auto* result_sender = static_cast<result_sender_t*>(opstate_->temp_storage_);
        _let_xxx_kernel<Args&&...><<<1, 1, 0, stream>>>(
          std::move(opstate_->fun_), result_sender, static_cast<Args&&>(args)...);

        if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
            status == cudaSuccess) {
          opstate_->defer_temp_storage_destruction(result_sender);
          auto& op = opstate_->opstate3_.template emplace<opstate_t>(__emplace_from{[&] {
            return connect(
              std::move(*result_sender),
              propagate_receiver<Receiver>{
                {}, static_cast<_strm::opstate_base<Receiver>&>(*opstate_)});
          }});
          STDEXEC::start(op);
        } else {
          opstate_->propagate_completion_signal(STDEXEC::set_error, std::move(status));
        }
      }

      template <class Tag, class... Args>
      void _complete(Tag, Args&&... args) noexcept {
        static_assert(__nothrow_callable<Tag, Receiver, Args...>);
        opstate_->propagate_completion_signal(Tag(), static_cast<Args&&>(args)...);
      }

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        _complete(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __err) noexcept {
        _complete(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept {
        _complete(set_stopped_t());
      }

      auto get_env() const noexcept -> env_t {
        return opstate_->make_env();
      }

      using opstate_variant_t = __minvoke<
        __mtransform<__muncurry<opstate_for_t<Receiver, Fun>>, __qq<__nullable_std_variant>>,
        Tuples...
      >;

      opstate<Sender, Receiver, Fun, Set>* opstate_;
    };

    template <class Sender, class Receiver, class Fun, class Set>
    using receiver_t = __gather_completions_of_t<
      Set,
      Sender,
      stream_env_t<env_of_t<Receiver>>,
      __q<__decayed_std_tuple>,
      __munique<__mbind_front_q<receiver, Sender, Receiver, Fun, Set>>
    >;

    template <class Sender, class Receiver, class Fun, class Set>
    using opstate_base_t =
      _strm::opstate<Sender, receiver_t<Sender, Receiver, Fun, Set>, Receiver>;

    template <class Sender, class Receiver, class Fun, class Set>
    struct opstate : opstate_base_t<Sender, Receiver, Fun, Set> {
      using receiver_t = receiver_t<Sender, Receiver, Fun, Set>;
      using opstate_variant_t = receiver_t::opstate_variant_t;

      opstate(Sender&& sndr, Receiver rcvr, Fun fun)
        : opstate_base_t<Sender, Receiver, Fun, Set>(
            static_cast<Sender&&>(sndr),
            static_cast<Receiver&&>(rcvr),
            [this](_strm::opstate_base<Receiver>&) -> receiver_t { return receiver_t{{}, this}; },
            get_completion_scheduler<set_value_t>(get_env(sndr), get_env(rcvr)).ctx_)
        , fun_(static_cast<Fun&&>(fun)) {
      }

      STDEXEC_IMMOVABLE(opstate);

      Fun fun_;
      opstate_variant_t opstate3_;
    };
  } // namespace let_xxx

  template <class Sender, class Fun, class Set>
  struct let_sender : public stream_sender_base {
    template <class Self, class Receiver>
    using opstate_t = let_xxx::opstate<__copy_cvref_t<Self, Sender>, Receiver, Fun, Set>;

    template <class Self, class Receiver>
    using _receiver_t = let_xxx::receiver_t<__copy_cvref_t<Self, Sender>, Receiver, Fun, Set>;

    template <class CvSender, class... StreamEnv>
    using _completions_t = __mapply<
      __mtransform<
        __mbind_back_q<let_xxx::__tfx_signal_t, Fun, Set, StreamEnv...>,
        __mtry_q<__concat_completion_signatures_t>
      >,
      __completion_signatures_of_t<CvSender, StreamEnv...>
    >;

    template <__decays_to<let_sender> Self, receiver Receiver>
      requires receiver_of<
        Receiver,
        _completions_t<__copy_cvref_t<Self, Sender>, stream_env_t<env_of_t<Receiver>>>
      >
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
      -> opstate_t<Self, Receiver> {
      return opstate_t<Self, Receiver>{
        static_cast<Self&&>(self).sndr_,
        static_cast<Receiver&&>(rcvr),
        static_cast<Self&&>(self).fun_};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    auto get_env() const noexcept -> stream_sender_attrs<Sender> {
      return {&sndr_};
    }

    template <__decays_to<let_sender> Self, class... Env>
    static consteval auto get_completion_signatures()
      -> _completions_t<__copy_cvref_t<Self, Sender>, stream_env_t<Env>...> {
      return {};
    }

    Sender sndr_;
    Fun fun_;
  };

  template <class Set, class Env>
  struct _transform_let_xxx_sender {
    template <class Fun, stream_completing_sender<Env> Sender>
    auto operator()(__ignore, Fun fn, Sender&& sndr) const {
      using __sender_t = let_sender<__decay_t<Sender>, Fun, Set>;
      return __sender_t{{}, static_cast<Sender&&>(sndr), static_cast<Fun&&>(fn)};
    }

    const Env& env_;
  };

  template <class Env>
  struct transform_sender_for<STDEXEC::let_value_t, Env>
    : _transform_let_xxx_sender<set_value_t, Env> { };

  template <class Env>
  struct transform_sender_for<STDEXEC::let_error_t, Env>
    : _transform_let_xxx_sender<set_error_t, Env> { };

  template <class Env>
  struct transform_sender_for<STDEXEC::let_stopped_t, Env>
    : _transform_let_xxx_sender<set_stopped_t, Env> { };
} // namespace nvexec::_strm

namespace STDEXEC::__detail {
  template <class Sender, class Fun, class Set>
  extern __declfn_t<nvexec::_strm::let_sender<__demangle_t<Sender>, Fun, Set>>
    __demangle_v<nvexec::_strm::let_sender<Sender, Fun, Set>>;
} // namespace STDEXEC::__detail

STDEXEC_PRAGMA_POP()
