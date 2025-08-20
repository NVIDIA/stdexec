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

#include "config.cuh"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace nvexec::detail {
  struct _msg_storage {
    char buffer[256];
  };

  inline auto _format_cuda_error(
    cudaError_t status,
    const char* file_name,
    int line,
    _msg_storage& storage) noexcept -> const char* {
    std::snprintf(
      storage.buffer,
      sizeof(storage.buffer),
      "CUDA ERROR: %s:%d: %s: %s",
      file_name,
      line,
      ::cudaGetErrorName(status),
      ::cudaGetErrorString(status));
    return storage.buffer;
  }

  class cuda_error : public ::std::runtime_error {
   public:
    cuda_error(cudaError_t status, const char* file_name, int line, _msg_storage storage = {})
      : ::std::runtime_error(detail::_format_cuda_error(status, file_name, line, storage)) {
#if defined(STDEXEC_STDERR)
      std::printf("%s\n", storage.buffer);
#endif
    }
  };

  [[noreturn]]
  inline void throw_on_cuda_error(cudaError_t status, char const * file_name, int line) {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    ::cudaGetLastError();
    throw ::nvexec::detail::cuda_error(status, file_name, line);
  }

  [[nodiscard]]
  inline auto log_on_cuda_error(cudaError_t status, [[maybe_unused]] char const *const file_name, [[maybe_unused]] const int line) noexcept
    -> ::cudaError_t {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    ::cudaGetLastError();
#if defined(STDEXEC_STDERR)
    if (status != ::cudaSuccess) {
      _msg_storage storage{};
      std::printf("%s\n", detail::_format_cuda_error(status, file_name, line, storage));
    }
#endif
    return status;
  }

  // clang-format off
#define STDEXEC_TRY_CUDA_API(...)                                                                  \
  if (const ::cudaError_t status = (__VA_ARGS__); status == ::cudaSuccess) {                       \
  } else {                                                                                         \
    ::nvexec::detail::throw_on_cuda_error(status, __FILE__, __LINE__);                             \
  }

#define STDEXEC_ASSERT_CUDA_API(...)                                                               \
  if (const ::cudaError_t status = (__VA_ARGS__); status == ::cudaSuccess) {                       \
  } else {                                                                                         \
    [[maybe_unused]] auto _ign = ::nvexec::detail::log_on_cuda_error(status, __FILE__, __LINE__);  \
    STDEXEC_ASSERT(!"CUDA ERROR: " __FILE__                                                        \
                   STDEXEC_STRINGIZE(:__LINE__: STDEXEC_STRINGIZE(__VA_ARGS__))); /*NOLINT*/       \
  }

#define STDEXEC_LOG_CUDA_API(...) \
  ::nvexec::detail::log_on_cuda_error((__VA_ARGS__), __FILE__, __LINE__)
  // clang-format on
} // namespace nvexec::detail
