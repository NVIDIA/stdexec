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

#include "__any.hpp"
#include "__concepts.hpp"
#include "__memory.hpp"
#include "__typeinfo.hpp"

namespace STDEXEC {
  namespace __detail {
    template <class _Base>
    struct __byte_allocator;

    template <class _Base>
    using __byte_allocator_interface_t = __any::interface<
      __byte_allocator,
      _Base,
      __any::__extends<__any::__icopyable, __any::__iequality_comparable>
    >;

    // NOLINTBEGIN(modernize-use-override)
    template <class _Base>
    struct __byte_allocator : __byte_allocator_interface_t<_Base> {
      using __byte_allocator_interface_t<_Base>::interface::interface;

      [[nodiscard]]
      constexpr virtual auto allocate(size_t __n) -> std::byte* {
        return __any::__value(*this).allocate(__n);
      }

      constexpr virtual void deallocate(std::byte* __byte, size_t __n) noexcept {
        __any::__value(*this).deallocate(__byte, __n);
      }
    };
    // NOLINTEND(modernize-use-override)
  } // namespace __detail

  template <class _Ty>
  struct __any_allocator {
    using value_type = _Ty;

    __any_allocator() = default;

    template <__not_same_as<__any_allocator> _Alloc>
      requires __is_not_instance_of<_Alloc, __any_allocator> && __simple_allocator<_Alloc>
    __any_allocator(_Alloc __alloc) noexcept {
      using __value_t = std::allocator_traits<_Alloc>::value_type;
      static_assert(
        __same_as<_Ty, __value_t>,
        "__any_allocator<T> must be constructed with an allocator of the same value type");
      __alloc_.emplace(STDEXEC::__rebind_allocator<std::byte>(__alloc));
    }

    template <__not_same_as<_Ty> _Uy>
    __any_allocator(__any_allocator<_Uy> __other) noexcept
      : __alloc_(std::move(__other.__alloc_)) {
    }

    [[nodiscard]]
    constexpr bool has_value() const noexcept {
      return !__any::__empty(__alloc_);
    }

    [[nodiscard]]
    constexpr auto type() const noexcept -> __type_index const & {
      return __any::__type(__alloc_);
    }

    [[nodiscard]]
    constexpr auto allocate(size_t __n) -> _Ty* {
      void* __void_ptr = __alloc_.allocate(__n * sizeof(_Ty));
      return static_cast<_Ty*>(__void_ptr);
    }

    constexpr virtual void deallocate(_Ty* __ptr, size_t __n) noexcept {
      void* __void_ptr = static_cast<void*>(__ptr);
      __alloc_.deallocate(static_cast<std::byte*>(__void_ptr), __n * sizeof(_Ty));
    }

   private:
    __any::__any<__detail::__byte_allocator> __alloc_{};
  };

  template <class _Alloc>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __any_allocator(_Alloc) -> __any_allocator<typename _Alloc::value_type>;

  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __any_allocator(std::allocator<void>) -> __any_allocator<std::byte>;
} // namespace STDEXEC
