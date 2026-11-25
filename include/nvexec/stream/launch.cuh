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

#include <cuda/std/utility>

#include "adaptor.cuh"
#include "common.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nv::execution
{
  namespace _strm
  {
    struct launch_t;

    struct launch_params
    {
      std::size_t grid_size     = 1;
      std::size_t block_size    = 1;
      std::size_t shared_memory = 0;
    };

    namespace _launch
    {
      template <class... Args, class Fun>
      __global__ void _launch_kernel(Fun fn, cudaStream_t stream, Args... args)
      {
        static_assert(trivially_copyable<Fun, Args...>);
        ::cuda::std::move(fn)(stream, args...);
      }

      // The data of a launch sender
      template <class Fun>
      struct data
      {
        Fun           fun_;
        launch_params params_;
      };

      // launch(sndr, params, fn): when the child completes with values, a kernel with the
      // given launch parameters calls fn with the stream and the values, and then the values
      // are sent. The child's other completions are passed on.
      struct algo : stream_adaptor<algo>
      {
        template <class State, class... Args>
        static auto __complete(STDEXEC::__facade::__child<0>,
                               State& state,
                               set_value_t,
                               Args&&... args) noexcept
        {
          using fun_t = decltype(state.data_.fun_);
          if constexpr (!std::invocable<fun_t, cudaStream_t, Args&...>)
          {
            return STDEXEC::__facade::__type_error(
              __callable_error_t<launch_t, fun_t, cudaStream_t, Args&...>());
          }
          else
          {
            cudaStream_t  stream = state.get_stream();
            launch_params p      = state.data_.params_;
            _launch_kernel<Args&...>
              <<<p.grid_size, p.block_size, p.shared_memory, stream>>>(std::move(state.data_.fun_),
                                                                       stream,
                                                                       args...);
            cudaError_t status = STDEXEC_LOG_CUDA_API(cudaPeekAtLastError());
            return STDEXEC::__facade::__choose(
              status == cudaSuccess,
              [&]() noexcept
              { return _strm::_propagate(state, set_value_t(), static_cast<Args&&>(args)...); },
              [&]() noexcept { return _strm::_report_error(state, status); });
          }
        }

        using stream_adaptor<algo>::__complete;
      };
    }  // namespace _launch

    struct launch_t : stream_tag_base
    {
      template <sender CvSender, __movable_value Fun>
      constexpr auto operator()(CvSender&& sndr, Fun&& fun) const
      {
        return STDEXEC::__facade::__sender{
          launch_t(),
          _launch::data<__decay_t<Fun>>{static_cast<Fun&&>(fun), {}},
          static_cast<CvSender&&>(sndr)
        };
      }

      template <sender CvSender, __movable_value Fun>
      constexpr auto operator()(CvSender&& sndr, launch_params params, Fun&& fun) const
      {
        return STDEXEC::__facade::__sender{
          launch_t(),
          _launch::data<__decay_t<Fun>>{static_cast<Fun&&>(fun), params},
          static_cast<CvSender&&>(sndr)
        };
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(Fun&& fun) const
      {
        return STDEXEC::__closure(*this, static_cast<Fun&&>(fun));
      }

      template <__movable_value Fun>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(launch_params params, Fun&& fun) const
      {
        return STDEXEC::__closure(*this, params, static_cast<Fun&&>(fun));
      }
    };
  }  // namespace _strm

  inline constexpr _strm::launch_t launch{};

}  // namespace nv::execution

template <>
struct STDEXEC::__facade::__algorithm<nv::execution::_strm::launch_t>
  : nv::execution::_strm::_launch::algo
{};

namespace nvexec = nv::execution;

STDEXEC_PRAGMA_POP()
