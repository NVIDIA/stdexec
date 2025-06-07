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

    template <class Tp>
    using __decay_ref = __decay_t<Tp>&;

    template <class Fun>
    using __result_sender_fn =
      __mtransform<__q<__decay_ref>, __mbind_front_q<__call_result_t, Fun>>;

    template <class... Sizes>
    struct max_in_pack {
      static constexpr std::size_t value = std::max({std::size_t{}, __v<Sizes>...});
    };

    template <class Sender, class PropagateReceiver, class Fun, class SetTag>
      requires sender_in<Sender, env_of_t<PropagateReceiver>>
    struct __max_sender_size {
      template <class... As>
      struct __sender_size_for_ {
        using __t = __msize_t<sizeof(__minvoke<__result_sender_fn<Fun>, As...>)>;
      };
      template <class... As>
      using __sender_size_for_t = stdexec::__t<__sender_size_for_<As...>>;

      static constexpr std::size_t value = __v<__gather_completions_of<
        SetTag,
        Sender,
        env_of_t<PropagateReceiver>,
        __q<__sender_size_for_t>,
        __q<max_in_pack>
      >>;
    };

    template <class Receiver, class Fun>
    using op_state_for = __mcompose<
      __mbind_back_q<connect_result_t, stdexec::__t<propagate_receiver_t<stdexec::__id<Receiver>>>>,
      __result_sender_fn<Fun>
    >;

    template <class Set, class Sig>
    struct __tfx_signal_ {
      template <class, class...>
      using __f = completion_signatures<Sig>;
    };

    template <class Set, class... Args>
    struct __tfx_signal_<Set, Set(Args...)> {
      template <class Fun, class... StreamEnv>
      using __f = transform_completion_signatures<
        __completion_signatures_of_t<__minvoke<__result_sender_fn<Fun>, Args...>, StreamEnv...>,
        completion_signatures<set_error_t(cudaError_t)>
      >;
    };

    template <class Sig, class Fun, class Set, class... StreamEnv>
    using __tfx_signal_t = __minvoke<__tfx_signal_<Set, Sig>, Fun, StreamEnv...>;

    template <class SenderId, class ReceiverId, class Fun, class Let>
    struct operation;

    template <class SenderId, class ReceiverId, class Fun, class Let, class... Tuples>
    struct _receiver_ {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using PropagateReceiver = stdexec::__t<propagate_receiver_t<ReceiverId>>;
      using Env = typename operation_state_base_t<ReceiverId>::env_t;

      struct __t : public stream_receiver_base {
        using __id = _receiver_;

        static constexpr std::size_t memory_allocation_size =
          __v<__max_sender_size<Sender, PropagateReceiver, Fun, Let>>;

        template <__one_of<Let> Tag, class... As>
          requires __minvocable<__result_sender_fn<Fun>, As...>
                && sender_to<__minvoke<__result_sender_fn<Fun>, As...>, PropagateReceiver>
        void _complete(Tag, As&&... __as) noexcept {
          using result_sender_t = __minvoke<__result_sender_fn<Fun>, As...>;
          using op_state_t = __minvoke<op_state_for<Receiver, Fun>, As...>;

          cudaStream_t stream = op_state_->get_stream();
          auto* result_sender = static_cast<result_sender_t*>(op_state_->temp_storage_);
          kernel_with_result<As&&...><<<1, 1, 0, stream>>>(
            std::move(op_state_->fun_), result_sender, static_cast<As&&>(__as)...);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
              status == cudaSuccess) {
            op_state_->defer_temp_storage_destruction(result_sender);
            auto& op = op_state_->op_state3_.template emplace<op_state_t>(__emplace_from{[&] {
              return connect(
                std::move(*result_sender),
                stdexec::__t<propagate_receiver_t<ReceiverId>>{
                  {}, static_cast<operation_state_base_t<ReceiverId>&>(*op_state_)});
            }});
            stdexec::start(op);
          } else {
            op_state_->propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <class Tag, class... As>
          requires __none_of<Tag, Let> && __callable<Tag, Receiver, As...>
        void _complete(Tag, As&&... __as) noexcept {
          static_assert(__nothrow_callable<Tag, Receiver, As...>);
          op_state_->propagate_completion_signal(Tag(), static_cast<As&&>(__as)...);
        }

        template <class... Args>
        void set_value(Args&&... __args) noexcept {
          _complete(set_value_t(), static_cast<Args&&>(__args)...);
        }

        template <class Error>
        void set_error(Error&& __err) noexcept {
          _complete(set_error_t(), static_cast<Error&&>(__err));
        }

        void set_stopped() noexcept {
          _complete(set_stopped_t());
        }

        auto get_env() const noexcept -> Env {
          return op_state_->make_env();
        }

        using op_state_variant_t = __minvoke<
          __mtransform<__muncurry<op_state_for<Receiver, Fun>>, __qq<__nullable_std_variant>>,
          Tuples...
        >;

        operation<SenderId, ReceiverId, Fun, Let>* op_state_;
      };
    };

    template <class SenderId, class ReceiverId, class Fun, class Let>
    using __receiver = stdexec::__t<__gather_completions_of<
      Let,
      stdexec::__t<SenderId>,
      stream_env<env_of_t<stdexec::__t<ReceiverId>>>,
      __q<__decayed_std_tuple>,
      __munique<__mbind_front_q<_receiver_, SenderId, ReceiverId, Fun, Let>>
    >>;

    template <class SenderId, class ReceiverId, class Fun, class Let>
    using operation_base = operation_state_t<
      SenderId,
      stdexec::__id<__receiver<SenderId, ReceiverId, Fun, Let>>,
      ReceiverId
    >;

    template <class SenderId, class ReceiverId, class Fun, class Let>
    struct operation : operation_base<SenderId, ReceiverId, Fun, Let> {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using _receiver_t = __receiver<SenderId, ReceiverId, Fun, Let>;
      using op_state_variant_t = typename _receiver_t::op_state_variant_t;

      template <class Receiver2>
      operation(Sender&& sndr, Receiver2&& rcvr, Fun fun)
        : operation_base<SenderId, ReceiverId, Fun, Let>(
            static_cast<Sender&&>(sndr),
            static_cast<Receiver2&&>(rcvr),
            [this](operation_state_base_t<stdexec::__id<Receiver2>>&) -> _receiver_t {
              return _receiver_t{{}, this};
            },
            get_completion_scheduler<set_value_t>(get_env(sndr)).context_state_)
        , fun_(static_cast<Fun&&>(fun)) {
      }

      STDEXEC_IMMOVABLE(operation);

      Fun fun_;

      op_state_variant_t op_state3_;
    };
  } // namespace let_xxx

  template <class SenderId, class Fun, class Set>
  struct let_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = let_sender_t;

      template <class Self, class Receiver>
      using operation_t = let_xxx::operation<
        stdexec::__id<__copy_cvref_t<Self, Sender>>,
        stdexec::__id<__decay_t<Receiver>>,
        Fun,
        Set
      >;
      template <class Self, class Receiver>
      using _receiver_t = stdexec::__t<let_xxx::__receiver<
        stdexec::__id<__copy_cvref_t<Self, Sender>>,
        stdexec::__id<__decay_t<Receiver>>,
        Fun,
        Set
      >>;

      template <class Sender, class... Env>
      using __completions = __mapply<
        __mtransform<
          __mbind_back_q<let_xxx::__tfx_signal_t, Fun, Set, Env...>,
          __mtry_q<__concat_completion_signatures>
        >,
        __completion_signatures_of_t<Sender, Env...>
      >;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<
          Receiver,
          __completions<__copy_cvref_t<Self, Sender>, stream_env<env_of_t<Receiver>>>
        >
      static auto connect(Self&& self, Receiver rcvr) -> operation_t<Self, Receiver> {
        return operation_t<Self, Receiver>{
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr),
          static_cast<Self&&>(self).fun_};
      }

      auto get_env() const noexcept -> stream_sender_attrs<Sender> {
        return {&sndr_};
      }

      template <__decays_to<__t> Self, class... Env>
      static auto get_completion_signatures(Self&&, Env&&...)
        -> __completions<__copy_cvref_t<Self, Sender>, stream_env<Env>...> {
        return {};
      }

      Sender sndr_;
      Fun fun_;
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
