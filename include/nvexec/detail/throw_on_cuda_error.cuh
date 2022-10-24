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

#include "config.cuh"

#include <stdexcept>
#include <cstdio>

namespace nvexec {
  namespace detail {
    inline cudaError_t debug_cuda_error(
        cudaError_t error,
        [[maybe_unused]] char const* file_name,
        [[maybe_unused]] int line) {
      // Clear the global CUDA error state which may have been set by the last
      // call. Otherwise, errors may "leak" to unrelated calls.
      cudaGetLastError();

#if defined(STDEXEC_STDERR)
      if (error != cudaSuccess) {
        std::printf("CUDA error %s [%s:%d]: %s\n", 
                    cudaGetErrorName(error),
                    file_name, line, 
                    cudaGetErrorString(error));
      }
#endif

      return error;
    }
  }

  #define STDEXEC_DBG_ERR(E)                                         \
    ::nvexec::detail::debug_cuda_error(E, __FILE__, __LINE__) \
    /**/
}
