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
#include <algorithm>
#include <cstddef>
#include <utility>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {
  namespace let_xxx {
    template <class... As, class Fun, class ResultSenderT>
    __launch_bounds__(1) __global__
      void kernel_with_result(Fun fn, ResultSenderT* result, As... as) {
      static_assert(trivially_copyable<Fun, As...>);
      new (result) ResultSenderT(::cuda::std::move(fn)(static_cast<As&&>(as)...));
    }

    template <class _Tp>
    using __decay_ref = __decay_t<_Tp>&;

    template <class _Fun>
    using __result_sender_fn = //
      __mtransform<__q<__decay_ref>, __mbind_front_q<__call_result_t, _Fun>>;

    template <class... Sizes>
    struct max_in_pack {
      static constexpr std::size_t value = std::max({std::size_t{}, __v<Sizes>...});
    };

    template <class _Sender, class _PropagateReceiver, class _Fun, class _SetTag>
      requires sender_in<_Sender, env_of_t<_PropagateReceiver>>
    struct __max_sender_size {
      template <class... _As>
      struct __sender_size_for_ {
        using __t = __msize_t<sizeof(__minvoke<__result_sender_fn<_Fun>, _As...>)>;
      };
      template <class... _As>
      using __sender_size_for_t = stdexec::__t<__sender_size_for_<_As...>>;

      static constexpr std::size_t value = //
        __v<__gather_completions_of<
          _SetTag,
          _Sender,
          env_of_t<_PropagateReceiver>,
          __q<__sender_size_for_t>,
          __q<max_in_pack>>>;
    };

    template <class _Receiver, class _Fun>
    using __op_state_for = //
      __mcompose<
        __mbind_back_q<connect_result_t, stdexec::__t<propagate_receiver_t<stdexec::__id<_Receiver>>>>,
        __result_sender_fn<_Fun>>;

    template <class _Set, class _Sig>
    struct __tfx_signal_ {
      template <class, class...>
      using __f = completion_signatures<_Sig>;
    };

    template <class _Set, class... _Args>
    struct __tfx_signal_<_Set, _Set(_Args...)> {
      template <class _Fun, class... _StreamEnv>
      using __f = //
        transform_completion_signatures<
          __completion_signatures_of_t<__minvoke<__result_sender_fn<_Fun>, _Args...>, _StreamEnv...>,
          completion_signatures<set_error_t(cudaError_t)>>;
    };

    template <class _Sig, class _Fun, class _Set, class... _StreamEnv>
    using __tfx_signal_t = __minvoke<__tfx_signal_<_Set, _Sig>, _Fun, _StreamEnv...>;

    template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
    struct __operation;

    template <class _SenderId, class _ReceiverId, class _Fun, class _Let, class... _Tuples>
    struct __receiver_ {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _PropagateReceiver = stdexec::__t<propagate_receiver_t<_ReceiverId>>;
      using _Env = typename operation_state_base_t<_ReceiverId>::env_t;

      struct __t : public stream_receiver_base {
        using __id = __receiver_;

        static constexpr std::size_t memory_allocation_size =
          __v<__max_sender_size<_Sender, _PropagateReceiver, _Fun, _Let>>;

        template <__one_of<_Let> _Tag, class... _As>
          requires __minvocable<__result_sender_fn<_Fun>, _As...>
                && sender_to<__minvoke<__result_sender_fn<_Fun>, _As...>, _PropagateReceiver>
        void __complete(_Tag, _As&&... __as) noexcept {
          using result_sender_t = __minvoke<__result_sender_fn<_Fun>, _As...>;
          using op_state_t = __minvoke<__op_state_for<_Receiver, _Fun>, _As...>;

          cudaStream_t stream = __op_state_->get_stream();
          auto* result_sender = static_cast<result_sender_t*>(__op_state_->temp_storage_);
          kernel_with_result<_As&&...><<<1, 1, 0, stream>>>(
            std::move(__op_state_->__fun_), result_sender, static_cast<_As&&>(__as)...);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
              status == cudaSuccess) {
            __op_state_->defer_temp_storage_destruction(result_sender);
            auto& __op = __op_state_->__op_state3_.template emplace<op_state_t>(__emplace_from{[&] {
              return connect(
                std::move(*result_sender),
                stdexec::__t<propagate_receiver_t<_ReceiverId>>{
                  {}, static_cast<operation_state_base_t<_ReceiverId>&>(*__op_state_)});
            }});
            stdexec::start(__op);
          } else {
            __op_state_->propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <class _Tag, class... _As>
          requires __none_of<_Tag, _Let> && __callable<_Tag, _Receiver, _As...>
        void __complete(_Tag, _As&&... __as) noexcept {
          static_assert(__nothrow_callable<_Tag, _Receiver, _As...>);
          __op_state_->propagate_completion_signal(_Tag(), static_cast<_As&&>(__as)...);
        }

        template <class... _Args>
        void set_value(_Args&&... __args) noexcept {
          __complete(set_value_t(), static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          __complete(set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          __complete(set_stopped_t());
        }

        auto get_env() const noexcept -> _Env {
          return __op_state_->make_env();
        }

        using __op_state_variant_t = //
          __minvoke<
            __mtransform<__muncurry<__op_state_for<_Receiver, _Fun>>, __qq<__nullable_std_variant>>,
            _Tuples...>;

        __operation<_SenderId, _ReceiverId, _Fun, _Let>* __op_state_;
      };
    };

    template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
    using __receiver = //
      stdexec::__t<__gather_completions_of<
        _Let,
        stdexec::__t<_SenderId>,
        stream_env<env_of_t<stdexec::__t<_ReceiverId>>>,
        __q<__decayed_std_tuple>,
        __munique<__mbind_front_q<__receiver_, _SenderId, _ReceiverId, _Fun, _Let>>>>;

    template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
    using __operation_base = //
      operation_state_t<
        _SenderId,
        stdexec::__id<__receiver<_SenderId, _ReceiverId, _Fun, _Let>>,
        _ReceiverId>;

    template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
    struct __operation : __operation_base<_SenderId, _ReceiverId, _Fun, _Let> {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = __receiver<_SenderId, _ReceiverId, _Fun, _Let>;
      using __op_state_variant_t = typename __receiver_t::__op_state_variant_t;

      template <class _Receiver2>
      __operation(_Sender&& __sndr, _Receiver2&& __rcvr, _Fun __fun)
        : __operation_base<_SenderId, _ReceiverId, _Fun, _Let>(
            static_cast<_Sender&&>(__sndr),
            static_cast<_Receiver2&&>(__rcvr),
            [this](operation_state_base_t<stdexec::__id<_Receiver2>>&) -> __receiver_t {
              return __receiver_t{{}, this};
            },
            get_completion_scheduler<set_value_t>(get_env(__sndr)).context_state_)
        , __fun_(static_cast<_Fun&&>(__fun)) {
      }

      STDEXEC_IMMOVABLE(__operation);

      _Fun __fun_;

      __op_state_variant_t __op_state3_;
    };
  } // namespace let_xxx

  template <class _SenderId, class _Fun, class _Set>
  struct let_sender_t {
    using _Sender = stdexec::__t<_SenderId>;

    struct __t : stream_sender_base {
      using __id = let_sender_t;

      template <class _Self, class _Receiver>
      using __operation_t = //
        let_xxx::__operation<
          stdexec::__id<__copy_cvref_t<_Self, _Sender>>,
          stdexec::__id<__decay_t<_Receiver>>,
          _Fun,
          _Set>;
      template <class _Self, class _Receiver>
      using __receiver_t = //
        stdexec::__t<let_xxx::__receiver<
          stdexec::__id<__copy_cvref_t<_Self, _Sender>>,
          stdexec::__id<__decay_t<_Receiver>>,
          _Fun,
          _Set>>;

      template <class _Sender, class... _Env>
      using __completions = //
        __mapply<
          __mtransform<
            __mbind_back_q<let_xxx::__tfx_signal_t, _Fun, _Set, _Env...>,
            __mtry_q<__concat_completion_signatures>>,
          __completion_signatures_of_t<_Sender, _Env...>>;

      template <__decays_to<__t> _Self, receiver _Receiver>
        requires receiver_of<                          //
                   _Receiver,                          //
                   __completions<                      //
                     __copy_cvref_t<_Self, _Sender>,   //
                     stream_env<env_of_t<_Receiver>>>> //
      static auto connect(_Self&& __self, _Receiver __rcvr) -> __operation_t<_Self, _Receiver> {
        return __operation_t<_Self, _Receiver>{
          static_cast<_Self&&>(__self).__sndr_,
          static_cast<_Receiver&&>(__rcvr),
          static_cast<_Self&&>(__self).__fun_};
      }

      auto get_env() const noexcept -> env_of_t<const _Sender&> {
        return stdexec::get_env(__sndr_);
      }

      template <__decays_to<__t> _Self, class... _Env>
      static auto get_completion_signatures(_Self&&, _Env&&...)
        -> __completions<__copy_cvref_t<_Self, _Sender>, stream_env<_Env>...> {
        return {};
      }

      _Sender __sndr_;
      _Fun __fun_;
    };
  };

  template <class Set>
  struct _transform_let_xxx_sender {
    template <class Fun, stream_completing_sender Sender>
    auto operator()(__ignore, Fun fn, Sender&& sndr) const {
      using __sender_t = __t<let_sender_t<__id<__decay_t<Sender>>, Fun, Set>>;
      return __sender_t{{}, static_cast<Sender&&>(sndr), static_cast<Fun&&>(fn)};
    }
  };

  template <>
  struct transform_sender_for<stdexec::let_value_t> : _transform_let_xxx_sender<set_value_t> { };

  template <>
  struct transform_sender_for<stdexec::let_error_t> : _transform_let_xxx_sender<set_error_t> { };

  template <>
  struct transform_sender_for<stdexec::let_stopped_t>
    : _transform_let_xxx_sender<set_stopped_t> { };

} // namespace nvexec::_strm

namespace stdexec::__detail {
  template <class SenderId, class Fun, class Set>
  inline constexpr __mconst<nvexec::_strm::let_sender_t<__name_of<__t<SenderId>>, Fun, Set>>
    __name_of_v<nvexec::_strm::let_sender_t<SenderId, Fun, Set>>{};
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
