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
#pragma once

#include "__config.hpp"

#if __has_include(<cuda/std/atomic>)
#  include <cuda/std/atomic>
#  define STDEXEC_HAS_CUDA_STD_ATOMIC() 1
#else
#  include <atomic>
#  define STDEXEC_HAS_CUDA_STD_ATOMIC() 0
#endif

namespace stdexec::__std {
#if __has_include(<cuda/std/atomic>)

using cuda::std::atomic;
using cuda::std::atomic_ref;
using cuda::std::atomic_flag;
using cuda::std::atomic_ptrdiff_t;
using cuda::std::memory_order;
using cuda::std::memory_order_relaxed;
using cuda::std::memory_order_acquire;
using cuda::std::memory_order_release;
using cuda::std::memory_order_acq_rel;
using cuda::std::memory_order_seq_cst;
using cuda::std::atomic_thread_fence;
using cuda::std::atomic_signal_fence;

#else

using std::atomic;
using std::atomic_ref;
using std::atomic_flag;
using std::atomic_ptrdiff_t;
using std::memory_order;
using std::memory_order_relaxed;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_acq_rel;
using std::memory_order_seq_cst;
using std::atomic_thread_fence;
using std::atomic_signal_fence;

#endif
} // namespace stdexec::__std
