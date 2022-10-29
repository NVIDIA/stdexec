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

#include "../../stdexec/execution.hpp"
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

namespace upon_error {

template <class Fun, class... As>
  __launch_bounds__(1)
  __global__ void kernel(Fun fn, As... as) {
    fn(as...);
  }

template <class Fun, class ResultT, class... As>
  __launch_bounds__(1)
  __global__ void kernel_with_result(Fun fn, ResultT* result, As... as) {
    new (result) ResultT(fn(as...));
  }

template <std::size_t MemoryAllocationSize, class ReceiverId, class Fun>
  class receiver_t : public stream_receiver_base {

    Fun f_;
    operation_state_base_t<ReceiverId> &op_state_;

  public:
    constexpr static std::size_t memory_allocation_size = MemoryAllocationSize;

    template <class Error>
      friend void tag_invoke(stdexec::set_error_t, receiver_t&& self, Error&& error) 
          noexcept requires std::invocable<Fun, Error> {
        using result_t = std::invoke_result_t<Fun, std::decay_t<Error>>;
        constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
        cudaStream_t stream = self.op_state_.stream_;

        if constexpr (does_not_return_a_value) {
          kernel<Fun, Error><<<1, 1, 0, stream>>>(self.f_, (Error&&)error);
          if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        } else {
          using decayed_result_t = std::decay_t<result_t>;
          decayed_result_t *d_result = reinterpret_cast<decayed_result_t*>(self.op_state_.temp_storage_);
          kernel_with_result<Fun, Error><<<1, 1, 0, stream>>>(self.f_, d_result, error);
          if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value, *d_result);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }
      }

    template <stdexec::__one_of<stdexec::set_value_t,
                                stdexec::set_stopped_t> Tag, 
              class... As _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.op_state_.propagate_completion_signal(tag, (As&&)as...);
        );
      }

    friend stdexec::env_of_t<stdexec::__t<ReceiverId>>
    tag_invoke(stdexec::get_env_t, const receiver_t& self) {
      return stdexec::get_env(self.op_state_.receiver_);
    }

    explicit receiver_t(Fun fun, operation_state_base_t<ReceiverId> &op_state)
      : f_((Fun&&) fun)
      , op_state_(op_state)
    {}
  };
}

template <class SenderId, class FunId>
  struct upon_error_sender_t : stream_sender_base {
    using Sender = stdexec::__t<SenderId>;
    using Fun = stdexec::__t<FunId>;

    Sender sndr_;
    Fun fun_;

    template <class T, int = 0>
      struct size_of_ {
        using __t = stdexec::__index<sizeof(T)>;
      };

    template <int W>
      struct size_of_<void, W> {
        using __t = stdexec::__index<0>;
      };
    
    template <class... As>
      struct result_size_for {
        using __t = typename size_of_<stdexec::__call_result_t<Fun, As...>>::__t;
      };

    template <class... Sizes>
      struct max_in_pack {
        static constexpr std::size_t value = std::max({std::size_t{}, stdexec::__v<Sizes>...});
      };

    template <class Receiver>
        requires stdexec::sender<Sender, stdexec::env_of_t<Receiver>>
      struct max_result_size {
        template <class... _As>
          using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

        static constexpr std::size_t value =
          stdexec::__v<
            stdexec::__gather_sigs_t<
              stdexec::set_error_t, 
              Sender,  
              stdexec::env_of_t<Receiver>, 
              stdexec::__q<result_size_for_t>, 
              stdexec::__q<max_in_pack>>>;
      };

    template <class Receiver>
      using receiver_t = 
        upon_error::receiver_t<
          max_result_size<Receiver>::value, 
          stdexec::__x<Receiver>, 
          Fun>;

    template <class Self, class Env>
      using completion_signatures =
        stdexec::__make_completion_signatures<
          stdexec::__member_t<Self, Sender>,
          Env,
          stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
          stdexec::__q<stdexec::__compl_sigs::__default_set_value>,
          stdexec::__mbind_front_q<stdexec::__set_value_invoke_t, Fun>>;

    template <stdexec::__decays_to<upon_error_sender_t> Self, stdexec::receiver Receiver>
      requires stdexec::receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
      -> stream_op_state_t<stdexec::__member_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<stdexec::__member_t<Self, Sender>>(
            ((Self&&)self).sndr_,
            (Receiver&&)rcvr,
            [&](operation_state_base_t<stdexec::__x<Receiver>>& stream_provider) -> receiver_t<Receiver> {
              return receiver_t<Receiver>(self.fun_, stream_provider);
            });
    }

    template <stdexec::__decays_to<upon_error_sender_t> Self, class Env>
    friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
      -> stdexec::dependent_completion_signatures<Env>;

    template <stdexec::__decays_to<upon_error_sender_t> Self, class Env>
    friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
      -> completion_signatures<Self, Env> requires true;

    template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
      requires stdexec::__callable<Tag, const Sender&, As...>
    friend auto tag_invoke(Tag tag, const upon_error_sender_t& self, As&&... as)
      noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
      -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
      return ((Tag&&) tag)(self.sndr_, (As&&) as...);
    }
  };

}
