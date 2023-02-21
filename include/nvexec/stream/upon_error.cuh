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
    __launch_bounds__(1) __global__ void kernel(Fun fn, As... as) {
      fn(as...);
    }

    template <class Fun, class ResultT, class... As>
    __launch_bounds__(1) __global__ void kernel_with_result(Fun fn, ResultT* result, As... as) {
      new (result) ResultT(fn(as...));
    }

    template <std::size_t MemoryAllocationSize, class ReceiverId, class Fun>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using env_t = typename operation_state_base_t<ReceiverId>::env_t;

        Fun f_;
        operation_state_base_t<ReceiverId>& op_state_;

    public:
        using __id = receiver_t;

        constexpr static std::size_t memory_allocation_size = MemoryAllocationSize;

        template <class Error>
        friend void tag_invoke(stdexec::set_error_t, __t&& self, Error&& error) noexcept
          requires std::invocable<Fun, Error>
        {
          using result_t = std::invoke_result_t<Fun, std::decay_t<Error>>;
          constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
          cudaStream_t stream = self.op_state_.get_stream();

          if constexpr (does_not_return_a_value) {
            kernel<Fun, Error><<<1, 1, 0, stream>>>(self.f_, (Error&&) error);
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(stdexec::set_value);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          } else {
            using decayed_result_t = std::decay_t<result_t>;
            decayed_result_t* d_result = static_cast<decayed_result_t*>(
              self.op_state_.temp_storage_);
            kernel_with_result<Fun, Error><<<1, 1, 0, stream>>>(self.f_, d_result, error);
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(stdexec::set_value, *d_result);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }
        }

        template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend env_t tag_invoke(stdexec::get_env_t, const __t& self) {
          return self.op_state_.make_env();
        }

        explicit __t(Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : f_((Fun&&) fun)
          , op_state_(op_state) {
        }
      };
    };
  }

  template <class SenderId, class Fun>
  struct upon_error_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = upon_error_sender_t;
      Sender sndr_;
      Fun fun_;

      template <class T, int = 0>
      struct size_of_ {
        using __t = stdexec::__msize_t<sizeof(T)>;
      };

      template <int W>
      struct size_of_<void, W> {
        using __t = stdexec::__msize_t<0>;
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
        requires stdexec::sender_in<Sender, stdexec::env_of_t<Receiver>>
      struct max_result_size {
        template <class... _As>
        using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

        static constexpr std::size_t value = //
          stdexec::__v< stdexec::__gather_completions_for<
            stdexec::set_error_t,
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__q<result_size_for_t>,
            stdexec::__q<max_in_pack>>>;
      };

      template <class Receiver>
      using receiver_t = //
        stdexec::__t<
          upon_error::receiver_t< max_result_size<Receiver>::value, stdexec::__id<Receiver>, Fun>>;

      template <class Self, class Env>
      using completion_signatures = //
        stdexec::__make_completion_signatures<
          stdexec::__copy_cvref_t<Self, Sender>,
          Env,
          stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
          stdexec::__q<stdexec::__compl_sigs::__default_set_value>,
          stdexec::__mbind_front_q<stdexec::__set_value_invoke_t, Fun>>;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<
          Receiver,
          completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> stream_op_state_t<stdexec::__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> { return receiver_t<Receiver>(self.fun_, stream_provider); });
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env>
        requires true;

      friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }
    };
  };
}
