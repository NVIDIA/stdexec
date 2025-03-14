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

#include "config.cuh"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace nvexec::detail {
  class cuda_error : public ::std::runtime_error {
   private:
    struct __msg_storage {
      char __buffer[256]; // NOLINT
    };

    static auto
      __format_cuda_error(const int __status, const char* __msg, char* __msg_buffer) noexcept
      -> char* {
      ::snprintf(__msg_buffer, 256, "cudaError %d: %s", __status, __msg);
      return __msg_buffer;
    }

   public:
    cuda_error(const int __status, const char* __msg, __msg_storage __msg_buffer = {0}) noexcept
      : ::std::runtime_error(__format_cuda_error(__status, __msg, __msg_buffer.__buffer)) {
    }
  };

  inline auto debug_cuda_error(
    cudaError_t error,
    [[maybe_unused]] char const * file_name,
    [[maybe_unused]] int line) -> cudaError_t {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    cudaGetLastError();

#if defined(STDEXEC_STDERR)
      if (error != cudaSuccess) {
        std::printf(
          "CUDA error %s [%s:%d]: %s\n",
          cudaGetErrorName(error),
          file_name,
          line,
          cudaGetErrorString(error));
      }
#endif

      return error;
  }
} // namespace nvexec::detail

#define STDEXEC_DBG_ERR(E) ::nvexec::detail::debug_cuda_error(E, __FILE__, __LINE__) /**/
