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
#pragma once

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"

namespace example::cuda::stream {

namespace bulk {
  template <int BlockThreads, std::integral Shape, class Fun, class... As>
    __launch_bounds__(BlockThreads)
    __global__ void kernel(Shape shape, Fun fn, As&&... as) {
      const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

      if (tid < static_cast<int>(shape)) {
        fn(tid, as...);
      }
    }

  template <class ReceiverId, std::integral Shape, class Fun>
    class receiver_t : public receiver_base_t {
      using Receiver = _P2300::__t<ReceiverId>;

      Shape shape_;
      Fun f_;

      operation_state_base_t<ReceiverId>& op_state_;

    public:
      template <class... As _NVCXX_CAPTURE_PACK(As)>
        friend void tag_invoke(std::execution::set_value_t, receiver_t&& self, As&&... as)
          noexcept requires _P2300::__callable<Fun, Shape, As...> {
          _NVCXX_EXPAND_PACK(As, as,
            constexpr int block_threads = 256;
            const int grid_blocks = (static_cast<int>(self.shape_) + block_threads - 1) / block_threads;
            kernel<block_threads, Shape, Fun, As...>
                    <<<grid_blocks, block_threads, 0, self.op_state_.stream_>>>(
                      self.shape_, self.f_, (As&&)as...);

            self.op_state_.propagate_completion_signal(std::execution::set_value, (As&&)as...);
          );
        }

      template <_P2300::__one_of<std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        }

      friend std::execution::env_of_t<Receiver> tag_invoke(std::execution::get_env_t, const receiver_t& self) {
        return std::execution::get_env(self.op_state_.receiver_);
      }

      explicit receiver_t(Shape shape, Fun fun, operation_state_base_t<ReceiverId>& op_state)
        : shape_(shape)
        , f_((Fun&&) fun)
        , op_state_(op_state)
      {}
    };
}

template <class SenderId, std::integral Shape, class FunId>
  struct bulk_sender_t : gpu_sender_base_t {
    using Sender = _P2300::__t<SenderId>;
    using Fun = _P2300::__t<FunId>;

    Sender sndr_;
    Shape shape_;
    Fun fun_;

    using set_error_t =
      std::execution::completion_signatures<
        std::execution::set_error_t(cudaError_t)>;

    template <class Receiver>
      using receiver_t = bulk::receiver_t<_P2300::__x<Receiver>, Shape, Fun>;

    template <class... Tys>
    using set_value_t =
      std::execution::completion_signatures<
        std::execution::set_value_t(Tys...)>;

    template <class Self, class Env>
      using completion_signatures =
        std::execution::__make_completion_signatures<
          _P2300::__member_t<Self, Sender>,
          Env,
          set_error_t,
          _P2300::__q<set_value_t>>;

    template <_P2300::__decays_to<bulk_sender_t> Self, std::execution::receiver Receiver>
      requires std::execution::receiver_of<Receiver, completion_signatures<Self, std::execution::env_of_t<Receiver>>>
    friend auto tag_invoke(std::execution::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<_P2300::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<_P2300::__member_t<Self, Sender>>(
            ((Self&&)self).sndr_,
            (Receiver&&)rcvr,
            [&](operation_state_base_t<_P2300::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
              return receiver_t<Receiver>(self.shape_, self.fun_, stream_provider);
            });
      }

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> std::execution::dependent_completion_signatures<Env>;

    template <_P2300::__decays_to<bulk_sender_t> Self, class Env>
    friend auto tag_invoke(std::execution::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <std::execution::tag_category<std::execution::forwarding_sender_query> Tag, class... As>
      requires _P2300::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const bulk_sender_t& self, As&&... as)
      noexcept(_P2300::__nothrow_callable<Tag, const Sender&, As...>)
      -> _P2300::__call_result_if_t<std::execution::tag_category<Tag, std::execution::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}

