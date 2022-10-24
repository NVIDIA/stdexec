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

#ifdef _CUDA_ATOMIC
#error This file must be included before <cuda/atomic> from libcudacxx
#endif

#include "config.cuh"
#include <cuda/std/detail/__config>

#if _LIBCUDACXX_STD_VER > 17 && _LIBCUDACXX_CUDA_API_VERSION < 1009000
#define STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORAROUND() 1
#else
#define STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORAROUND() 0
#endif

#if STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORAROUND()
_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class memory_order : unsigned int;

namespace __detail {

  template <class _Tag>
    __device__ void __atomic_thread_fence_cuda(memory_order __memorder, _Tag);

  __device__ void __atomic_signal_fence_cuda(memory_order __order);

  template<class _Type, class _Tag>
    __device__ _Type __atomic_load_n_cuda(const volatile _Type *__ptr, memory_order __memorder, _Tag);

  template<class _Type, class _Tag>
    __device__ void __atomic_store_n_cuda(volatile _Type *__ptr, _Type, memory_order __memorder, _Tag);

  template<class _Type, class _Tag>
    __device__ _Type __atomic_exchange_n_cuda(volatile _Type *__ptr, _Type __val, memory_order __memorder, _Tag);

  template<class _Type, class _Tag>
    __device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool __weak, memory_order __success_memorder, memory_order __failure_memorder, _Tag);

} // namespace __detail

_LIBCUDACXX_END_NAMESPACE_STD
#endif // STDEXEC_LIBCUDACXX_NEEDS_ATOMIC_WORAROUND()
