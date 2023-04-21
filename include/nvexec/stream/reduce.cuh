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
#include <ranges>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "algorithm_base.cuh"
#include "common.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS {
    namespace reduce_ {
      template <class SenderId, class ReceiverId, class Fun>
      struct receiver_t : public ::nvexec::STDEXEC_STREAM_DETAIL_NS::__algo_base::receiver_t<SenderId, ReceiverId, Fun, receiver_t<SenderId, ReceiverId, Fun>> {
        using base = ::nvexec::STDEXEC_STREAM_DETAIL_NS::__algo_base::receiver_t<SenderId, ReceiverId, Fun, receiver_t<SenderId, ReceiverId, Fun>>;

        template<class Range>
        using result_t = typename ::nvexec::STDEXEC_STREAM_DETAIL_NS::__algo_base::binary_invoke_result_t<Range, Fun>;

        template <class Range>
        static void set_value_impl(base::__t&& self, Range&& range) noexcept {
          cudaStream_t stream = self.op_state_.get_stream();

          using value_t = result_t<Range>;
          value_t* d_out = static_cast<value_t*>(self.op_state_.temp_storage_);

          void* d_temp_storage{};
          std::size_t temp_storage_size{};

          auto first = begin(range);
          auto last = end(range);

          std::size_t num_items = std::distance(first, last);

          cudaError_t status;

          do {
            if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                  d_temp_storage,
                  temp_storage_size,
                  first,
                  d_out,
                  num_items,
                  self.payload_,
                  value_t{},
                  stream));
                status != cudaSuccess) {
              break;
            }

            if (status = STDEXEC_DBG_ERR(
                  cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
                status != cudaSuccess) {
              break;
            }

            if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                  d_temp_storage,
                  temp_storage_size,
                  first,
                  d_out,
                  num_items,
                  self.payload_,
                  value_t{},
                  stream));
                status != cudaSuccess) {
              break;
            }

            status = STDEXEC_DBG_ERR(cudaFreeAsync(d_temp_storage, stream));
          } while (false);

          if (status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }
      };

      template <class SenderId, class Fun>
      struct sender_t {
        using Sender = stdexec::__t<SenderId>;

        template<class Receiver>
        using receiver_t = stdexec::__t<reduce_::receiver_t< SenderId, stdexec::__id<Receiver>, Fun>>;

        template<class Range>
        using result_t = ::cuda::std::decay_t< ::cuda::std::invoke_result_t<
                            Fun,
                            ::std::ranges::range_value_t<Range>,
                            ::std::ranges::range_value_t<Range>>>;

        using PAYLOAD = Fun;
        struct __t : stream_sender_base {
          using __id = sender_t;

          Sender sndr_;
          PAYLOAD payload_;

          template <class... Range>
            requires(sizeof...(Range) == 1)
          using set_value_t = stdexec::completion_signatures<stdexec::set_value_t(
            std::add_lvalue_reference_t<result_t<Range>>...)>;

          template <class Self, class Env>
          using completion_signatures = //
            stdexec::make_completion_signatures<
              stdexec::__copy_cvref_t<Self, Sender>,
              Env,
              stdexec::completion_signatures<stdexec::set_error_t(cudaError_t)>,
              set_value_t >;

          template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
            requires stdexec::
              receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
            friend auto
            tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> stream_op_state_t<
              stdexec::__copy_cvref_t<Self, Sender>,
              receiver_t<Receiver>,
              Receiver> {
            return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
              ((Self&&) self).sndr_,
              (Receiver&&) rcvr,
              [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
                -> receiver_t<Receiver> { return receiver_t<Receiver>(self.payload_, stream_provider); });
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


    struct reduce_t {
      template <class Sender, class PAYLOAD>
      using __sender =
        stdexec::__t<reduce_::sender_t<stdexec::__id<stdexec::__decay_t<Sender>>, PAYLOAD>>;

      template <stdexec::sender Sender, stdexec::__movable_value PAYLOAD>
      __sender<Sender, PAYLOAD> operator()(Sender&& __sndr, PAYLOAD __fun) const {
        return __sender<Sender, PAYLOAD>{{}, (Sender&&) __sndr, (PAYLOAD&&) __fun};
      }

      template <class PAYLOAD = cub::Sum>
      stdexec::__binder_back<reduce_t, PAYLOAD> operator()(PAYLOAD __fun = {}) const {
        return {{}, {}, {(PAYLOAD&&) __fun}};
      }
    };
  }

  inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};
}
