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

#include "__concepts.hpp" // IWYU pragma: keep for __std::integral
#include "__config.hpp"

#if __has_include(<cuda/std/atomic>)
#  include <cuda/std/atomic>
#  define STDEXEC_HAS_CUDA_STD_ATOMIC() 1
#else
#  include <atomic>
#  define STDEXEC_HAS_CUDA_STD_ATOMIC() 0
#endif

#include <memory>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wdeprecated-declarations")
STDEXEC_PRAGMA_IGNORE_EDG(deprecated_entity)
STDEXEC_PRAGMA_IGNORE_MSVC(4996) // 'foo': was declared deprecated

namespace STDEXEC::__std {
#if STDEXEC_HAS_CUDA_STD_ATOMIC()

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

#else // ^^^ STDEXEC_HAS_CUDA_STD_ATOMIC() / vvv !STDEXEC_HAS_CUDA_STD_ATOMIC()

  using std::atomic;
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

#  if __cpp_lib_atomic_ref >= 2018'06L && !defined(STDEXEC_RELACY)
  using std::atomic_ref;
#  else
  inline constexpr int __atomic_flag_map[] = {
    __ATOMIC_RELAXED,
    __ATOMIC_CONSUME,
    __ATOMIC_ACQUIRE,
    __ATOMIC_RELEASE,
    __ATOMIC_ACQ_REL,
    __ATOMIC_SEQ_CST,
  };

  // clang-12 does not know about std::atomic_ref yet
  // Here we implement only what we need
  template <integral _Ty>
  class atomic_ref {
    _Ty* __ptr_;

    static constexpr int __map_memory_order(memory_order __order) {
      return __atomic_flag_map[static_cast<int>(__order)];
    }

   public:
    atomic_ref(_Ty& __ref) noexcept
      : __ptr_(&__ref) {
    }

    atomic_ref(const atomic_ref&) = delete;
    atomic_ref& operator=(const atomic_ref&) = delete;

    atomic_ref(atomic_ref&&) = delete;
    atomic_ref& operator=(atomic_ref&&) = delete;

    _Ty load(memory_order __order = memory_order_seq_cst) const noexcept {
      return __atomic_load_n(__ptr_, __map_memory_order(__order));
    }

    void store(_Ty __desired, memory_order __order = memory_order_seq_cst) noexcept {
      __atomic_store_n(__ptr_, __desired, __map_memory_order(__order));
    }
  };
#  endif // ^^^ __cpp_lib_atomic_ref < 2018'06L

#endif // ^^^ !STDEXEC_HAS_CUDA_STD_ATOMIC()

  constexpr memory_order __memory_order_load(memory_order __order) noexcept {
    return __order == memory_order_acq_rel ? memory_order_acquire
         : __order == memory_order_release ? memory_order_relaxed
                                           : __order;
  }

#if __cpp_lib_atomic_shared_ptr >= 2017'11L && !STDEXEC_HAS_CUDA_STD_ATOMIC()
  template <class _Ty>
  using __atomic_shared_ptr = std::atomic<std::shared_ptr<_Ty>>;
#else
  template <typename _Ty>
  struct __atomic_shared_ptr {
    using value_type = std::shared_ptr<_Ty>;

    __atomic_shared_ptr() = default;

    __atomic_shared_ptr(std::nullptr_t) noexcept {
    }

    __atomic_shared_ptr(std::shared_ptr<_Ty> __ptr) noexcept
      : __ptr_{std::move(__ptr)} {
    }

    __atomic_shared_ptr(__atomic_shared_ptr&&) = delete;
    __atomic_shared_ptr(__atomic_shared_ptr const &) = delete;

    void operator=(std::nullptr_t) noexcept {
      store(nullptr);
    }

    void operator=(std::shared_ptr<_Ty> __ptr) noexcept {
      store(std::move(__ptr));
    }

    void operator=(__atomic_shared_ptr&&) = delete;
    void operator=(__atomic_shared_ptr const &) = delete;

    /* implicit */ operator std::shared_ptr<_Ty>() const noexcept {
      return load();
    }

    [[nodiscard]]
    bool is_lock_free() const noexcept {
      return std::atomic_is_lock_free(&__ptr_);
    }

    std::shared_ptr<_Ty> load(memory_order __order = memory_order_seq_cst) const noexcept {
      return std::atomic_load_explicit(&__ptr_, static_cast<std::memory_order>(__order));
    }

    void store(std::shared_ptr<_Ty> __ptr, memory_order __order = memory_order_seq_cst) noexcept {
      std::atomic_store_explicit(&__ptr_, std::move(__ptr), static_cast<std::memory_order>(__order));
    }

    std::shared_ptr<_Ty>
      exchange(std::shared_ptr<_Ty> __ptr, memory_order __order = memory_order_seq_cst) noexcept {
      return std::atomic_exchange_explicit(
        &__ptr_, std::move(__ptr), static_cast<std::memory_order>(__order));
    }

    bool compare_exchange_weak(
      std::shared_ptr<_Ty>& __expected,
      std::shared_ptr<_Ty> __desired,
      memory_order __success,
      memory_order __failure) noexcept {
      return std::atomic_compare_exchange_weak_explicit(
        &__ptr_,
        &__expected,
        std::move(__desired),
        static_cast<std::memory_order>(__success),
        static_cast<std::memory_order>(__failure));
    }

    bool compare_exchange_weak(
      std::shared_ptr<_Ty>& __expected,
      std::shared_ptr<_Ty> __desired,
      memory_order __order = memory_order_seq_cst) noexcept {
      return std::atomic_compare_exchange_weak_explicit(
        &__ptr_,
        &__expected,
        std::move(__desired),
        static_cast<std::memory_order>(__order),
        static_cast<std::memory_order>(__std::__memory_order_load(__order)));
    }

    bool compare_exchange_strong(
      std::shared_ptr<_Ty>& __expected,
      std::shared_ptr<_Ty> __desired,
      memory_order __upon_success,
      memory_order __upon_failure) noexcept {
      return std::atomic_compare_exchange_strong_explicit(
        &__ptr_,
        &__expected,
        std::move(__desired),
        static_cast<std::memory_order>(__upon_success),
        static_cast<std::memory_order>(__upon_failure));
    }

    bool compare_exchange_strong(
      std::shared_ptr<_Ty>& __expected,
      std::shared_ptr<_Ty> __desired,
      memory_order __order = memory_order_seq_cst) noexcept {
      return std::atomic_compare_exchange_strong_explicit(
        &__ptr_,
        &__expected,
        std::move(__desired),
        static_cast<std::memory_order>(__order),
        static_cast<std::memory_order>(__std::__memory_order_load(__order)));
    }

   private:
    std::shared_ptr<_Ty> __ptr_;
  };

  template <typename _Ty>
  __atomic_shared_ptr(std::shared_ptr<_Ty>) -> __atomic_shared_ptr<_Ty>;

#endif // ^^^ __cpp_lib_atomic_shared_ptr < 2017'11L

} // namespace STDEXEC::__std

STDEXEC_PRAGMA_POP()
