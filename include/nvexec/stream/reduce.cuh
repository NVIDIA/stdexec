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

#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "../detail/throw_on_cuda_error.cuh"
#include "adaptor.cuh"
#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace nv::execution
{
  namespace _strm
  {
    struct reduce_t;

    namespace reduce_
    {
      template <class Range, class InitT, class Fun>
      using binary_invoke_result_t = ::cuda::std::decay_t<
        ::cuda::std::invoke_result_t<Fun, std::ranges::range_reference_t<Range>, InitT>>;

      // The data of a reduce sender
      template <class InitT, class Fun>
      struct data
      {
        STDEXEC_ATTRIBUTE(no_unique_address) InitT init_;
        STDEXEC_ATTRIBUTE(no_unique_address) Fun fun_;
      };

      struct _REDUCE_REQUIRES_A_SENDER_OF_ONE_RANGE_;

      // reduce(sndr, init, fn): when the child completes with a range, the range is reduced
      // with fn and init on the stream, and the result is the operation's value (an lvalue in
      // managed memory). The child's other completions are passed on.
      struct algo : stream_adaptor<algo>
      {
        template <class State, class... Ranges>
        static auto __complete(STDEXEC::__facade::__child<0>,
                               State& state,
                               set_value_t,
                               Ranges&&... ranges) noexcept
        {
          if constexpr (sizeof...(Ranges) != 1)
          {
            return STDEXEC::__facade::__type_error(
              __mexception<_WHAT_(_REDUCE_REQUIRES_A_SENDER_OF_ONE_RANGE_),
                           _WHERE_(_IN_ALGORITHM_, reduce_t),
                           _WITH_ARGUMENTS_(Ranges...)>());
          }
          else
          {
            return algo::_reduce(state, ranges...);
          }
        }

        using stream_adaptor<algo>::__complete;

       private:
        template <class State, class Range>
        static auto _reduce(State& state, Range& range) noexcept
        {
          using init_t   = decltype(state.data_.init_);
          using fun_t    = decltype(state.data_.fun_);
          using value_t  = binary_invoke_result_t<Range, init_t, fun_t>;
          using result_t = STDEXEC::__facade::__union_t<
            STDEXEC::__facade::__effects<set_value_t(value_t&)>,
            STDEXEC::__facade::__effects<STDEXEC::__facade::__starts<stream_error>>>;

          cudaStream_t stream = state.get_stream();

          // `range` is produced asynchronously, so we need to wait for it to be ready
          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cudaStreamSynchronize(stream));
              status != cudaSuccess)
          {
            return result_t(_strm::_report_error(state, status));
          }

          auto* d_out = static_cast<value_t*>(
            _strm::_allocate_temp_storage(state, sizeof(value_t)));
          if (d_out == nullptr)
          {
            return result_t(_strm::_report_error(state, cudaErrorMemoryAllocation));
          }

          void*       d_temp_storage{};
          std::size_t temp_storage_size{};

          auto first = std::ranges::begin(range);
          auto last  = std::ranges::end(range);

          std::size_t num_items = std::distance(first, last);

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cub::DeviceReduce::Reduce(d_temp_storage,
                                                                                  temp_storage_size,
                                                                                  first,
                                                                                  d_out,
                                                                                  num_items,
                                                                                  state.data_.fun_,
                                                                                  state.data_.init_,
                                                                                  stream));
              status != cudaSuccess)
          {
            return result_t(_strm::_report_error(state, status));
          }

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(
                cudaMallocAsync(&d_temp_storage, temp_storage_size, stream));
              status != cudaSuccess)
          {
            return result_t(_strm::_report_error(state, status));
          }

          if (cudaError_t status = STDEXEC_LOG_CUDA_API(cub::DeviceReduce::Reduce(d_temp_storage,
                                                                                  temp_storage_size,
                                                                                  first,
                                                                                  d_out,
                                                                                  num_items,
                                                                                  state.data_.fun_,
                                                                                  state.data_.init_,
                                                                                  stream));
              status != cudaSuccess)
          {
            return result_t(_strm::_report_error(state, status));
          }

          cudaError_t status = STDEXEC_LOG_CUDA_API(cudaFreeAsync(d_temp_storage, stream));
          state.defer_temp_storage_destruction(d_out);

          if (status != cudaSuccess)
          {
            return result_t(_strm::_report_error(state, status));
          }
          return result_t(_strm::_propagate(state, set_value_t(), *d_out));
        }
      };
    }  // namespace reduce_

    struct reduce_t : stream_tag_base
    {
      template <sender CvSender, __movable_value InitT, __movable_value Fun = cuda::std::plus<>>
      auto operator()(CvSender&& sndr, InitT init, Fun fun) const
      {
        return STDEXEC::__facade::__sender{
          reduce_t(),
          reduce_::data<InitT, Fun>{static_cast<InitT&&>(init), static_cast<Fun&&>(fun)},
          static_cast<CvSender&&>(sndr)
        };
      }

      template <class InitT, class Fun = cuda::std::plus<>>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(InitT init, Fun fun = {}) const noexcept(__nothrow_decay_copyable<InitT, Fun>)
      {
        return STDEXEC::__closure(*this, static_cast<InitT&&>(init), static_cast<Fun&&>(fun));
      }
    };
  }  // namespace _strm

  inline constexpr _strm::reduce_t reduce{};
}  // namespace nv::execution

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::reduce_t>
  : nv::execution::_strm::reduce_::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
