/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <atomic>

#if __cpp_lib_atomic_ref >= 2018'06L
template <class T>
using __atomic_ref = std::atomic_ref<T>;
#else
#  include <concepts>

// clang-12 does not know about std::atomic_ref yet
// Here we implement only what we need
template <std::integral _Ty>
class __atomic_ref {
  _Ty* __ptr_;

  static constexpr int __map_memory_order(std::memory_order __order) {
    constexpr int __map[] = {
      __ATOMIC_RELAXED,
      __ATOMIC_CONSUME,
      __ATOMIC_ACQUIRE,
      __ATOMIC_RELEASE,
      __ATOMIC_ACQ_REL,
      __ATOMIC_SEQ_CST,
    };
    return __map[static_cast<int>(__order)];
  }

 public:
  __atomic_ref(_Ty& __ref) noexcept
    : __ptr_(&__ref) {
  }

  __atomic_ref(const __atomic_ref&) = delete;
  __atomic_ref& operator=(const __atomic_ref&) = delete;

  __atomic_ref(__atomic_ref&&) = delete;
  __atomic_ref& operator=(__atomic_ref&&) = delete;

  _Ty load(std::memory_order __order = std::memory_order_seq_cst) const noexcept {
    return __atomic_load_n(__ptr_, __map_memory_order(__order));
  }

  void store(_Ty __desired, std::memory_order __order = std::memory_order_seq_cst) noexcept {
    __atomic_store_n(__ptr_, __desired, __map_memory_order(__order));
  }
};
#endif
