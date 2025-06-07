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
#include <cstddef>
#include <exception>
#include <type_traits>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {

  namespace _upon_stopped {
    template <class Fun>
    __launch_bounds__(1) __global__ void kernel(Fun fn) {
      static_assert(trivially_copyable<Fun>);
      ::cuda::std::move(fn)();
    }

    template <class Fun, class ResultT>
    __launch_bounds__(1) __global__ void kernel_with_result(Fun fn, ResultT* result) {
      static_assert(trivially_copyable<Fun>);
      new (result) ResultT(::cuda::std::move(fn)());
    }

    template <class T>
    inline constexpr std::size_t size_of_ = sizeof(T);

    template <>
    inline constexpr std::size_t size_of_<void> = 0;

    template <class ReceiverId, class Fun>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using result_t = std::invoke_result_t<Fun>;
        using env_t = typename operation_state_base_t<ReceiverId>::env_t;

        Fun f_;
        operation_state_base_t<ReceiverId>& op_state_;

       public:
        using __id = receiver_t;

        static constexpr std::size_t memory_allocation_size = size_of_<result_t>;

        template <class... _As>
        void set_value(_As&&... __as) noexcept {
          op_state_.propagate_completion_signal(set_value_t(), static_cast<_As&&>(__as)...);
        }

        template <class _Error>
        void set_error(_Error __err) noexcept {
          op_state_.propagate_completion_signal(set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
          cudaStream_t stream = op_state_.get_stream();

          if constexpr (does_not_return_a_value) {
            kernel<<<1, 1, 0, stream>>>(std::move(f_));
            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status == cudaSuccess) {
              op_state_.propagate_completion_signal(stdexec::set_value);
            } else {
              op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          } else {
            using decayed_result_t = __decay_t<result_t>;
            auto* d_result = static_cast<decayed_result_t*>(op_state_.temp_storage_);
            kernel_with_result<<<1, 1, 0, stream>>>(std::move(f_), d_result);
            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status == cudaSuccess) {
              op_state_.defer_temp_storage_destruction(d_result);
              op_state_.propagate_completion_signal(stdexec::set_value, *d_result);
            } else {
              op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }
        }

        [[nodiscard]]
        auto get_env() const noexcept -> env_t {
          return op_state_.make_env();
        }

        explicit __t(Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : f_(static_cast<Fun&&>(fun))
          , op_state_(op_state) {
        }
      };
    };
  } // namespace _upon_stopped

  template <class SenderId, class Fun>
  struct upon_stopped_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = upon_stopped_sender_t;
      Sender sndr_;
      Fun fun_;

      using _set_error_t = completion_signatures<set_error_t(std::exception_ptr)>;

      template <class Receiver>
      using receiver_t = stdexec::__t<_upon_stopped::receiver_t<stdexec::__id<Receiver>, Fun>>;

      template <class Self, class... Env>
      using completion_signatures = transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        __with_error_invoke_t<
          __callable_error<"In nvexec::upon_stopped(Sender, Function)..."_mstr>,
          set_stopped_t,
          Fun,
          __copy_cvref_t<Self, Sender>,
          Env...
        >,
        __sigs::__default_set_value,
        __sigs::__default_set_error,
        __set_value_invoke_t<Fun>
      >;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, completion_signatures<Self, env_of_t<Receiver>>>
      static auto connect(Self&& self, Receiver rcvr)
        -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<__copy_cvref_t<Self, Sender>>(
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr),
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> { return receiver_t<Receiver>(self.fun_, stream_provider); });
      }

      template <__decays_to<__t> Self, class... Env>
      static auto
        get_completion_signatures(Self&&, Env&&...) -> completion_signatures<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> stream_sender_attrs<Sender> {
        return {&sndr_};
      }
    };
  };

  template <>
  struct transform_sender_for<stdexec::upon_stopped_t> {
    template <class Fn, stream_completing_sender Sender>
    auto operator()(__ignore, Fn fun, Sender&& sndr) const {
      using _sender_t = __t<upon_stopped_sender_t<__id<__decay_t<Sender>>, Fn>>;
      return _sender_t{{}, static_cast<Sender&&>(sndr), static_cast<Fn&&>(fun)};
    }
  };
} // namespace nvexec::_strm

namespace stdexec::__detail {
  template <class SenderId, class Fun>
  inline constexpr __mconst<nvexec::_strm::upon_stopped_sender_t<__name_of<__t<SenderId>>, Fun>>
    __name_of_v<nvexec::_strm::upon_stopped_sender_t<SenderId, Fun>>{};
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
