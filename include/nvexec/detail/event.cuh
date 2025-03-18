/*
 * Copyright (c) 2025 NVIDIA Corporation
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
#include "cuda_fwd.cuh"
#include "throw_on_cuda_error.cuh"

#include <utility>

namespace nvexec::detail {
  struct cuda_event {
    cuda_event() {
      if (auto status =
            STDEXEC_DBG_ERR(::cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
          status != cudaSuccess) {
        throw cuda_error(status, "cudaEventCreate");
      }
    }

    cuda_event(cuda_event&& other) noexcept
      : event_(std::exchange(other.event_, nullptr)) {
    }

    ~cuda_event() {
      if (event_ != nullptr) {
        STDEXEC_DBG_ERR(::cudaEventDestroy(event_));
      }
    }

    auto operator=(cuda_event&& other) noexcept -> cuda_event& {
      event_ = std::exchange(other.event_, nullptr);
      return *this;
    }

    auto try_record(cudaStream_t stream) noexcept -> cudaError_t {
      return STDEXEC_DBG_ERR(::cudaEventRecord(event_, stream));
    }

    auto try_wait(cudaStream_t stream) noexcept -> cudaError_t {
      return STDEXEC_DBG_ERR(::cudaStreamWaitEvent(stream, event_, 0));
    }

   private:
    cudaEvent_t event_{};
  };
} // namespace nvexec::detail
