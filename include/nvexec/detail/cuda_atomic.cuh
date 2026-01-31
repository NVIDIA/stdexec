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

#include "cuda_fwd.cuh"

#include <cuda/atomic> // IWYU pragma: export

#if STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORKAROUND()
_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __detail {

  template <class Tag>
  inline __device__ void __atomic_thread_fence_cuda(memory_order order, Tag) {
    return __atomic_thread_fence_cuda((int) order, Tag());
  }

  inline __device__ void __atomic_signal_fence_cuda(memory_order order) {
    return __atomic_signal_fence_cuda((int) order);
  }

  template <class Type, class Tag>
  inline __device__ Type
    __atomic_load_n_cuda(const volatile Type *ptr, memory_order order, Tag) {
    return __atomic_load_n_cuda(ptr, (int) order, Tag());
  }

  template <class Type, class Tag>
  inline __device__ void
    __atomic_store_n_cuda(volatile Type *ptr, Type val, memory_order order, Tag) {
    return __atomic_store_n_cuda(ptr, val, (int) order, Tag());
  }

  template <class Type, class Tag>
  inline __device__ Type
    __atomic_exchange_n_cuda(volatile Type *ptr, Type val, memory_order order, Tag) {
    return __atomic_exchange_n_cuda(ptr, val, (int) order, Tag());
  }

  template <class Type, class Tag>
  inline __device__ bool __atomic_compare_exchange_cuda(
    volatile Type *ptr,
    Type *expected,
    const Type *desired,
    bool weak,
    memory_order __success_memorder,
    memory_order __failure_memorder,
    Tag) {
    return __atomic_compare_exchange_cuda(
      ptr,
      expected,
      desired,
      weak,
      (int) __success_memorder,
      (int) __failure_memorder,
      Tag());
  }

} // namespace __detail

_LIBCUDACXX_END_NAMESPACE_STD
#endif // STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORKAROUND()
