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
      template <class SenderId, class ReceiverId, class InitT, class Fun>
      struct receiver_t
        : public __algo_range_init_fun::receiver_t<
            SenderId,
            ReceiverId,
            InitT,
            Fun,
            receiver_t<SenderId, ReceiverId, InitT, Fun>> {
        using base = __algo_range_init_fun::
          receiver_t<SenderId, ReceiverId, InitT, Fun, receiver_t<SenderId, ReceiverId, InitT, Fun>>;

        template <class Range>
        using result_t = typename __algo_range_init_fun::binary_invoke_result_t<Range, InitT, Fun>;

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

          if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                d_temp_storage,
                temp_storage_size,
                first,
                d_out,
                num_items,
                self.fun_,
                self.init_,
                stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          if (status = STDEXEC_DBG_ERR( //
                cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          if (status = STDEXEC_DBG_ERR(cub::DeviceReduce::Reduce(
                d_temp_storage,
                temp_storage_size,
                first,
                d_out,
                num_items,
                self.fun_,
                self.init_,
                stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          status = STDEXEC_DBG_ERR(cudaFreeAsync(d_temp_storage, stream));
          if (status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }
      };

      template <class SenderId, class InitT, class Fun>
      struct sender_t
        : public __algo_range_init_fun::
            sender_t<SenderId, InitT, Fun, sender_t<SenderId, InitT, Fun>> {
        template <class Receiver>
        using receiver_t =
          stdexec::__t<reduce_::receiver_t< SenderId, stdexec::__id<Receiver>, InitT, Fun>>;

        template <class Range>
        using _set_value_t = completion_signatures<set_value_t(
          ::std::add_lvalue_reference_t<
            typename __algo_range_init_fun::binary_invoke_result_t<Range, InitT, Fun>>)>;
      };
    }

    struct reduce_t {
      template <class Sender, class InitT, class Fun>
      using __sender =
        stdexec::__t<reduce_::sender_t<stdexec::__id<__decay_t<Sender>>, InitT, Fun>>;

      template < sender Sender, __movable_value InitT, __movable_value Fun = cub::Sum>
      __sender<Sender, InitT, Fun> operator()(Sender&& sndr, InitT init, Fun fun) const {
        return __sender<Sender, InitT, Fun>{{}, (Sender&&) sndr, (InitT&&) init, (Fun&&) fun};
      }

      template <class InitT, class Fun = cub::Sum>
      __binder_back<reduce_t, InitT, Fun> operator()(InitT init, Fun fun = {}) const {
        return {
          {},
          {},
          {(InitT&&) init, (Fun&&) fun}
        };
      }
    };
  }

  inline constexpr STDEXEC_STREAM_DETAIL_NS::reduce_t reduce{};
}
