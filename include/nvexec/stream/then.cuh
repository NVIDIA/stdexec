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
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec::_strm {

  namespace _then {
    template <class... As, class Fun>
    __launch_bounds__(1) __global__ void kernel(Fun fn, As... as) {
      static_assert(trivially_copyable<Fun, As...>);
      ::cuda::std::move(fn)(static_cast<As&&>(as)...);
    }

    template <class... As, class Fun, class ResultT>
    __launch_bounds__(1) __global__ void kernel_with_result(Fun fn, ResultT* result, As... as) {
      static_assert(trivially_copyable<Fun, As...>);
      new (result) ResultT(::cuda::std::move(fn)(static_cast<As&&>(as)...));
    }

    template <std::size_t MemoryAllocationSize, class ReceiverId, class Fun>
    struct receiver_t {
      using Receiver = stdexec::__t<ReceiverId>;

      class __t : public stream_receiver_base {
        Fun f_;
        operation_state_base_t<ReceiverId>& op_state_;

       public:
        using __id = receiver_t;
        static constexpr std::size_t memory_allocation_size = MemoryAllocationSize;

        template <class... As>
          requires std::invocable<Fun, __decay_t<As>...>
        void set_value(As&&... as) noexcept {
          using result_t = std::invoke_result_t<Fun, __decay_t<As>...>;
          constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
          operation_state_base_t<ReceiverId>& op_state = op_state_;
          cudaStream_t stream = op_state.get_stream();

          if constexpr (does_not_return_a_value) {
            kernel<As&&...><<<1, 1, 0, stream>>>(std::move(f_), static_cast<As&&>(as)...);

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status == cudaSuccess) {
              op_state.propagate_completion_signal(stdexec::set_value);
            } else {
              op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          } else {
            using decayed_result_t = __decay_t<result_t>;
            auto* d_result = static_cast<decayed_result_t*>(op_state.temp_storage_);
            kernel_with_result<As&&...>
              <<<1, 1, 0, stream>>>(std::move(f_), d_result, static_cast<As&&>(as)...);
            op_state.defer_temp_storage_destruction(d_result);

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status == cudaSuccess) {
              op_state.propagate_completion_signal(stdexec::set_value, std::move(*d_result));
            } else {
              op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
          op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept {
          op_state_.propagate_completion_signal(set_stopped_t());
        }

        auto get_env() const noexcept -> typename operation_state_base_t<ReceiverId>::env_t {
          return op_state_.make_env();
        }

        explicit __t(Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : f_(static_cast<Fun&&>(fun))
          , op_state_(op_state) {
        }
      };
    };

  } // namespace _then

  template <class SenderId, class Fun>
  struct then_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = then_sender_t;
      Sender sndr_;
      Fun fun_;

      template <class T, int = 0>
      struct size_of_ {
        using __t = __msize_t<sizeof(T)>;
      };

      template <int W>
      struct size_of_<void, W> {
        using __t = __msize_t<0>;
      };

      template <class... As>
      struct result_size_for {
        using __t = typename size_of_<__call_result_t<Fun, As...>>::__t;
      };

      template <class... Sizes>
      struct max_in_pack {
        static constexpr std::size_t value = std::max({std::size_t{}, __v<Sizes>...});
      };

      template <class Receiver>
        requires sender_in<Sender, env_of_t<Receiver>>
      struct max_result_size {
        template <class... _As>
        using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

        static constexpr std::size_t value = __v<__gather_completions_of<
          set_value_t,
          Sender,
          env_of_t<Receiver>,
          __q<result_size_for_t>,
          __q<max_in_pack>
        >>;
      };

      template <class Receiver>
      using receiver_t = stdexec::__t<
        _then::receiver_t<max_result_size<Receiver>::value, stdexec::__id<Receiver>, Fun>
      >;

      template <class _Error>
      using _set_error_t = completion_signatures<set_error_t(_Error)>;

      template <class Self, class... Env>
      using __error_completions_t = __meval<
        __concat_completion_signatures,
        __with_error_invoke_t<
          __callable_error<"In nvexec::then(Sender, Function)..."_mstr>,
          set_value_t,
          Fun,
          __copy_cvref_t<Self, Sender>,
          Env...
        >,
        completion_signatures<set_error_t(cudaError_t)>
      >;

      template <class... As>
      using _set_value_t = __set_value_invoke_t<Fun, As...>;

      template <class Self, class... Env>
      using _completion_signatures_t = transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        __error_completions_t<Self, Env...>,
        _set_value_t,
        _set_error_t
      >;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
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
        get_completion_signatures(Self&&, Env&&...) -> _completion_signatures_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> stream_sender_attrs<Sender> {
        return {&sndr_};
      }
    };
  };

  template <>
  struct transform_sender_for<stdexec::then_t> {
    template <class Fn, stream_completing_sender Sender>
    auto operator()(__ignore, Fn fun, Sender&& sndr) const {
      using _sender_t = __t<then_sender_t<__id<__decay_t<Sender>>, Fn>>;
      return _sender_t{{}, static_cast<Sender&&>(sndr), static_cast<Fn&&>(fun)};
    }
  };
} // namespace nvexec::_strm

namespace stdexec::__detail {
  template <class SenderId, class Fun>
  inline constexpr __mconst<nvexec::_strm::then_sender_t<__name_of<__t<SenderId>>, Fun>>
    __name_of_v<nvexec::_strm::then_sender_t<SenderId, Fun>>{};
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
