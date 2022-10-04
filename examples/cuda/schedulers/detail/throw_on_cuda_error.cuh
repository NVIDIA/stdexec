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

#include <execution.hpp>

#include <stdexcept>

namespace example::cuda {

  inline void throw_on_cuda_error(cudaError_t error, char const* file_name, int line)
  {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    cudaGetLastError();

    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA Error: ")
                             + file_name
                             + ":"
                             + std::to_string(line)
                             + ": "
                             + cudaGetErrorName(error)
                             + ": "
                             + cudaGetErrorString(error));
    }
  }

  #define THROW_ON_CUDA_ERROR(...)                     \
    ::example::cuda::throw_on_cuda_error(__VA_ARGS__, __FILE__, __LINE__); \
    /**/

}
