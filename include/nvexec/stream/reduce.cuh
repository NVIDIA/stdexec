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
#include <type_traits>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "algorithm_base.cuh"
#include "../detail/throw_on_cuda_error.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nvexec {
  namespace _strm {
    namespace reduce_ {
      template <class SenderId, class ReceiverId, class InitT, class Fun>
      struct receiver_t
        : public __algo_range_init_fun::receiver_t<
            SenderId,
            ReceiverId,
            InitT,
            Fun,
            receiver_t<SenderId, ReceiverId, InitT, Fun>
          > {
        using base = __algo_range_init_fun::receiver_t<
          SenderId,
          ReceiverId,
          InitT,
          Fun,
          receiver_t<SenderId, ReceiverId, InitT, Fun>
        >;

        template <class Range>
        using result_t = typename __algo_range_init_fun::binary_invoke_result_t<Range, InitT, Fun>;

        template <class Range>
        static void set_value_impl(typename base::__t&& self, Range&& range) noexcept {
          cudaError_t status{cudaSuccess};
          cudaStream_t stream = self.op_state_.get_stream();

          // `range` is produced asynchronously, so we need to wait for it to be ready
          if (status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream)); status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          using value_t = result_t<Range>;
          auto* d_out = static_cast<value_t*>(self.op_state_.temp_storage_);

          void* d_temp_storage{};
          std::size_t temp_storage_size{};

          auto first = begin(range);
          auto last = end(range);

          std::size_t num_items = std::distance(first, last);

          if (status = STDEXEC_LOG_CUDA_API(
                cub::DeviceReduce::Reduce(
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

          if (status = STDEXEC_LOG_CUDA_API(
                cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
              status != cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
            return;
          }

          if (status = STDEXEC_LOG_CUDA_API(
                cub::DeviceReduce::Reduce(
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

          status = STDEXEC_LOG_CUDA_API(cudaFreeAsync(d_temp_storage, stream));
          self.op_state_.defer_temp_storage_destruction(d_out);

          if (status == cudaSuccess) {
            self.op_state_.propagate_completion_signal(stdexec::set_value, *d_out);
          } else {
            self.op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }
      };

      template <class SenderId, class InitT, class Fun>
      struct sender_t
        : public __algo_range_init_fun::sender_t<
            SenderId,
            InitT,
            Fun,
            sender_t<SenderId, InitT, Fun>
          > {
        template <class Receiver>
        using receiver_t =
          stdexec::__t<reduce_::receiver_t<SenderId, stdexec::__id<Receiver>, InitT, Fun>>;

        template <class Range>
        using _set_value_t = completion_signatures<set_value_t(
          ::std::add_lvalue_reference_t<
            typename __algo_range_init_fun::binary_invoke_result_t<Range, InitT, Fun>
          >)>;
      };
    } // namespace reduce_

    struct reduce_t {
      template <class Sender, class InitT, class Fun>
      using __sender =
        stdexec::__t<reduce_::sender_t<stdexec::__id<__decay_t<Sender>>, InitT, Fun>>;

      template <sender Sender, __movable_value InitT, __movable_value Fun = cuda::std::plus<>>
      auto operator()(Sender&& sndr, InitT init, Fun fun) const -> __sender<Sender, InitT, Fun> {
        return __sender<Sender, InitT, Fun>{
          {}, static_cast<Sender&&>(sndr), static_cast<InitT&&>(init), static_cast<Fun&&>(fun)};
      }

      template <class InitT, class Fun = cuda::std::plus<>>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(InitT init, Fun fun = {}) const -> __binder_back<reduce_t, InitT, Fun> {
        return {
          {static_cast<InitT&&>(init), static_cast<Fun&&>(fun)},
          {},
          {}
        };
      }
    };
  } // namespace _strm

  inline constexpr _strm::reduce_t reduce{};
} // namespace nvexec

namespace stdexec::__detail {
  template <class SenderId, class Init, class Fun>
  extern __mconst<nvexec::_strm::reduce_::sender_t<__name_of<__t<SenderId>>, Init, Fun>>
    __name_of_v<nvexec::_strm::reduce_::sender_t<SenderId, Init, Fun>>;
} // namespace stdexec::__detail

STDEXEC_PRAGMA_POP()
