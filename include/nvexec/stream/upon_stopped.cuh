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

        constexpr static std::size_t memory_allocation_size = size_of_<result_t>;

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& self) noexcept {
          constexpr bool does_not_return_a_value = std::is_same_v<void, result_t>;
          cudaStream_t stream = self.op_state_.get_stream();

          if constexpr (does_not_return_a_value) {
            kernel<<<1, 1, 0, stream>>>(std::move(self.f_));
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.propagate_completion_signal(stdexec::set_value);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          } else {
            using decayed_result_t = __decay_t<result_t>;
            decayed_result_t* d_result = static_cast<decayed_result_t*>(
              self.op_state_.temp_storage_);
            kernel_with_result<<<1, 1, 0, stream>>>(std::move(self.f_), d_result);
            if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError());
                status == cudaSuccess) {
              self.op_state_.defer_temp_storage_destruction(d_result);
              self.op_state_.propagate_completion_signal(stdexec::set_value, *d_result);
            } else {
              self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            }
          }
        }

        template <__one_of<set_value_t, set_error_t> Tag, class... As>
        friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(Tag(), (As&&) as...);
        }

        friend env_t tag_invoke(get_env_t, const __t& self) noexcept {
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
  struct upon_stopped_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = upon_stopped_sender_t;
      Sender sndr_;
      Fun fun_;

      using _set_error_t = completion_signatures< set_error_t(std::exception_ptr)>;

      template <class Receiver>
      using receiver_t = stdexec::__t< _upon_stopped::receiver_t<stdexec::__id<Receiver>, Fun>>;

      template <class Self, class Env>
      using completion_signatures = //
        __meval<
          __try_make_completion_signatures,
          __copy_cvref_t<Self, Sender>,
          Env,
          __with_error_invoke_t<
            set_stopped_t,
            Fun,
            __copy_cvref_t<Self, Sender>,
            Env,
            __callable_error<"In nvexec::upon_stopped(Sender, Function)..."__csz>>,
          __q<__compl_sigs::__default_set_value>,
          __q<__compl_sigs::__default_set_error>,
          __set_value_invoke_t<Fun>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of< Receiver, completion_signatures<Self, env_of_t<Receiver>>>
      friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr)
        -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> { return receiver_t<Receiver>(self.fun_, stream_provider); });
      }

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> completion_signatures<Self, Env> {
        return {};
      }

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
        return get_env(self.sndr_);
      }
    };
  };
}

namespace stdexec::__detail {
  template <class SenderId, class Fun>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::upon_stopped_sender_t<__name_of<__t<SenderId>>, Fun>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::upon_stopped_sender_t<SenderId, Fun>>{};
}
