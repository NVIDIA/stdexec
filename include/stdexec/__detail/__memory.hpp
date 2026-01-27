/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__execution_fwd.hpp"
#include "__scope.hpp"

// include these after __execution_fwd.hpp
#include <memory> // IWYU pragma: export

namespace STDEXEC {
  namespace __detail {
    template <class _Alloc>
    struct __alloc_deleter {
      using __pointer = std::allocator_traits<_Alloc>::pointer;

      void operator()(__pointer __ptr) const {
        _Alloc __alloc(__alloc_);
        std::allocator_traits<_Alloc>::destroy(__alloc, std::addressof(*__ptr));
        std::allocator_traits<_Alloc>::deallocate(__alloc, __ptr, 1);
      }

      _Alloc __alloc_;
    };
  } // namespace __detail

  template <class _Ty, class _Alloc, class... _Args>
  [[nodiscard]]
  constexpr auto __allocate_unique(const _Alloc& __alloc, _Args&&... __args)
    -> std::unique_ptr<_Ty, __detail::__alloc_deleter<_Alloc>> {
    using __value_t = std::allocator_traits<_Alloc>::value_type;
    using __deleter_t = __detail::__alloc_deleter<_Alloc>;
    static_assert(__same_as<__value_t const, _Ty const>, "Allocator has the wrong value_type");

    _Alloc __alloc2(__alloc);
    auto __ptr = std::allocator_traits<_Alloc>::allocate(__alloc2, 1);
    __scope_guard __guard{
      &std::allocator_traits<_Alloc>::deallocate, std::ref(__alloc2), __ptr, 1ul};
    std::allocator_traits<_Alloc>::construct(
      __alloc2, std::addressof(*__ptr), static_cast<_Args&&>(__args)...);
    __guard.__dismiss();
    return std::unique_ptr<_Ty, __deleter_t>(__ptr, __deleter_t{__alloc2});
  }

  template <class _Ty, class _Alloc>
  [[nodiscard]]
  constexpr auto __rebind_allocator(const _Alloc& __alloc) noexcept {
    using __rebound_alloc_t = std::allocator_traits<_Alloc>::template rebind_alloc<_Ty>;
    static_assert(noexcept(__rebound_alloc_t(__alloc)));
    return __rebound_alloc_t(__alloc);
  }
} // namespace STDEXEC
