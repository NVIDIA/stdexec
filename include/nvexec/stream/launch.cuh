/*
 * Copyright (c) 2022-2024 NVIDIA Corporation
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
#include <concepts>
#include <cstddef>
#include <exception>

#include <cuda/std/utility>

#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec {
  namespace _strm {
    struct launch_params {
      std::size_t grid_size = 1;
      std::size_t block_size = 1;
      std::size_t shared_memory = 0;
    };

    namespace _launch {
      template <class... As, class Fun>
      __global__ void kernel(Fun fn, cudaStream_t stream, As... as) {
        static_assert(trivially_copyable<Fun, As...>);
        ::cuda::std::move(fn)(stream, as...);
      }

      template <class ReceiverId, class Fun>
      struct receiver_t {
        using Receiver = stdexec::__t<ReceiverId>;

        class __t : public stream_receiver_base {
          operation_state_base_t<ReceiverId>& op_state_;
          Fun fun_;
          launch_params params_;

         public:
          using __id = receiver_t;

          template <class... As>
            requires std::invocable<Fun, cudaStream_t, As&...>
          void set_value(As&&... as) noexcept {
            cudaStream_t stream = op_state_.get_stream();
            launch_params p = params_;
            kernel<As&...><<<p.grid_size, p.block_size, p.shared_memory, stream>>>(
              std::move(fun_), stream, as...);

            if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
                status == cudaSuccess) {
              op_state_.propagate_completion_signal(stdexec::set_value, static_cast<As&&>(as)...);
            } else {
              op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }

          template <class Error>
          void set_error(Error&& err) noexcept {
            op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
          }

          void set_stopped() noexcept {
            op_state_.propagate_completion_signal(set_stopped_t());
          }

          [[nodiscard]]
          auto get_env() const noexcept -> typename operation_state_base_t<ReceiverId>::env_t {
            return op_state_.make_env();
          }

          explicit __t(operation_state_base_t<ReceiverId>& op_state, Fun fun, launch_params params)
            : op_state_(op_state)
            , fun_(static_cast<Fun&&>(fun))
            , params_(params) {
          }
        };
      };

      template <class Fun, class... As>
      using launch_error_t =
        __minvoke<__callable_error<"In nvexec::launch()..."_mstr>, Fun, cudaStream_t, As&...>;

      template <class Fun, class... As>
      using _set_value_t = __minvoke<
        __if_c<
          __callable<Fun, cudaStream_t, As&...>,
          __mcompose<__q<completion_signatures>, __qf<set_value_t>>,
          __mbind_front_q<launch_error_t, Fun>
        >,
        As...
      >;

      template <class Fun, class CvrefSender, class... Env>
      using completions_t = transform_completion_signatures<
        __completion_signatures_of_t<CvrefSender, Env...>,
        completion_signatures<set_error_t(std::exception_ptr)>,
        __mbind_front_q<_set_value_t, Fun>::template __f
      >;
    } // namespace _launch

    template <class SenderId, class Fun>
    struct launch_sender_t {
      using Sender = stdexec::__t<SenderId>;

      struct __t : stream_sender_base {
        using __id = launch_sender_t;

        Sender sndr_;
        Fun fun_;
        launch_params params_;

        template <class Self, class... Env>
        using completions_t = _launch::completions_t<Fun, __copy_cvref_t<Self, Sender>, Env...>;

        template <class Receiver>
        using receiver_t = stdexec::__t<_launch::receiver_t<stdexec::__id<Receiver>, Fun>>;

        template <__decays_to<__t> Self, receiver Receiver>
          requires receiver_of<Receiver, completions_t<Self, env_of_t<Receiver>>>
        static auto connect(Self&& self, Receiver rcvr)
          -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
          return stream_op_state<__copy_cvref_t<Self, Sender>>(
            static_cast<Self&&>(self).sndr_,
            static_cast<Receiver&&>(rcvr),
            [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
              -> receiver_t<Receiver> {
              return receiver_t<Receiver>(stream_provider, self.fun_, self.params_);
            });
        }

        template <__decays_to<__t> Self, class... Env>
        static auto get_completion_signatures(Self&&, Env&&...) -> completions_t<Self, Env...> {
          return {};
        }

        auto get_env() const noexcept -> stream_sender_attrs<Sender> {
          return {&sndr_};
        }
      };
    };

    struct launch_t {
      template <class Sender, class Fun>
      using sender_t = stdexec::__t<launch_sender_t<stdexec::__id<__decay_t<Sender>>, Fun>>;

      template <sender Sender, __movable_value Fun>
      auto operator()(Sender&& sndr, Fun&& fun) const -> sender_t<Sender, Fun> {
        return {{}, static_cast<Sender&&>(sndr), static_cast<Fun&&>(fun), {}};
      }

      template <sender Sender, __movable_value Fun>
      auto
        operator()(Sender&& sndr, launch_params params, Fun&& fun) const -> sender_t<Sender, Fun> {
        return {{}, static_cast<Sender&&>(sndr), static_cast<Fun&&>(fun), params};
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(Fun&& fun) const -> __binder_back<launch_t, Fun> {
        return {{static_cast<Fun&&>(fun)}};
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(launch_params params, Fun&& fun) const
        -> __binder_back<launch_t, launch_params, Fun> {
        return {
          {params, static_cast<Fun&&>(fun)},
          {},
          {}
        };
      }
    };

  } // namespace _strm

  inline constexpr _strm::launch_t launch{};

} // namespace nvexec

namespace stdexec::__detail {
  template <class SenderId, class Fun>
  inline constexpr __mconst<nvexec::_strm::launch_sender_t<__name_of<__t<SenderId>>, Fun>>
    __name_of_v<nvexec::_strm::launch_sender_t<SenderId, Fun>>{};
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
