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

  namespace _upon_error {
    template <class... As, class Fun>
    __launch_bounds__(1) __global__ void kernel(Fun fn, As... as) {
      static_assert(trivially_copyable<Fun, As...>);
      ::cuda::std::move(fn)(static_cast<As&&>(as)...);
    }

    template <class... As, class Fun, class ResultVariant>
    __launch_bounds__(1) __global__
      void kernel_with_result(Fun fn, ResultVariant* result, As... as) {
      static_assert(trivially_copyable<Fun, As...>);
      using result_t = __decay_t<__call_result_t<Fun, As...>>;
      result->template emplace<result_t>(::cuda::std::move(fn)(static_cast<As&&>(as)...));
    }

    template <class ReceiverId, class Fun, class TempStorage>
    struct receiver_t {
      using Receiver = stdexec::__t<ReceiverId>;
      using Env = make_stream_env_t<env_of_t<Receiver>>;

      class __t : stream_receiver_base {
        Fun f_;
        using op_state_t = operation_state_base_t<ReceiverId, TempStorage>;
        op_state_t& op_state_;

       public:
        using __id = receiver_t;

        using temporary_storage_type = TempStorage;

        template <same_as<set_error_t> _Tag, class Error>
        friend void tag_invoke(_Tag, __t&& self, Error&& error) noexcept
          requires std::invocable<Fun, Error>
        {
          using result_t = __decay_t<__call_result_t<Fun, Error>>;
          constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
          cudaStream_t stream = self.op_state_.get_stream();

          if constexpr (does_not_return_a_value) {
            kernel<Error><<<1, 1, 0, stream>>>(std::move(self.f_), (Error&&) error);
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(stdexec::set_value);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          } else {
            temporary_storage_type& d_result = self.op_state_.temp_storage_->emplace();
            kernel_with_result<Error>
              <<<1, 1, 0, stream>>>(std::move(self.f_), &d_result, (Error&&) error);
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(
                stdexec::set_value, std::move(d_result.template get<result_t>()));
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }
        }

        template <__one_of<set_value_t, set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(Tag(), (As&&) as...);
        }

        friend Env tag_invoke(get_env_t, const __t& self) noexcept {
          return self.op_state_.make_env();
        }

        explicit __t(Fun fun, op_state_t& op_state)
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

      // A nullable variant of all the possible callable result types:
      template <class Receiver>
        requires sender_in<Sender, env_of_t<Receiver>>
      using temp_storage_t = //
        __gather_completions_for<
          set_value_t,
          Sender,
          env_of_t<Receiver>,
          __mbind_front_q<__call_result_t, Fun>,
          __transform<__q<__decay_t>, __remove<void, __q<unique_nullable_variant_t>>>>;

      template <class Receiver>
      using receiver_t = //
        stdexec::__t<
          _upon_error::receiver_t<stdexec::__id<Receiver>, Fun, temp_storage_t<Receiver>>>;

      template <class Self, class Env>
      using completion_signatures = //
        __try_make_completion_signatures<
          __copy_cvref_t<Self, Sender>,
          Env,
          completion_signatures<set_error_t(cudaError_t)>,
          __q<__compl_sigs::__default_set_value>,
          __mbind_front_q<__set_value_invoke_t, Fun>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of< Receiver, completion_signatures<Self, env_of_t<Receiver>>>
      friend auto tag_invoke(connect_t, Self&& self, Receiver&& rcvr)
        -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        using inner_receiver_t = receiver_t<Receiver>;
        return stream_op_state<__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&]<class StreamProvider>(StreamProvider& stream_provider) -> inner_receiver_t { //
            return inner_receiver_t(self.fun_, stream_provider);
          });
      }

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> dependent_completion_signatures<Env>;

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> completion_signatures<Self, Env>
        requires true;

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
        return get_env(self.sndr_);
      }
    };
  };
}
