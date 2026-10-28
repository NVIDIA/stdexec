/* Copyright (c) 2026 Ian Petersen
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

#include "../stdexec/__detail/__concepts.hpp"

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <type_traits>

#include "../stdexec/__detail/__prologue.hpp"

//! Defines template <class _Delegate> exec::__frame_allocator_t
//!
//! The intended use for __frame_allocator_t is the dynamic allocation of
//! operation states and/or coroutine frames. __frame_allocator_t<_Delegate> is
//! an allocator of std::byte that allocates and deallocates through _Delegate,
//! which may be one of:
//! - another allocator, which allocates and deallocates as itself rebound to
//!   the requested type;
//! - std::pmr::memory_resource*; or
//! - T* for some T that inherits from std::pmr::memory_resource, in which case
//!   __frame_allocator_t knows the concrete type of its memory_resource and so
//!   may be able to avoid virtual dispatch.
//!
//! A frame allocator only provides storage: __frame_allocator_t declares no
//! construct or destroy members, so allocator_traits constructs objects with
//! construct_at and destroys them with destroy_at. In particular, it never does
//! uses-allocator construction, and a delegate allocator's own construct and
//! destroy members, if it has any, are never called.
namespace experimental::execution
{
  namespace __fa
  {
    using namespace STDEXEC;

    //! An allocator that allocates and deallocates through _Alloc but declares
    //! no construct or destroy members
    template <class _Alloc>
    class __allocator_adaptor
    {
      using __traits_t = std::allocator_traits<_Alloc>;

      template <class>
      friend class __allocator_adaptor;

      STDEXEC_ATTRIBUTE(no_unique_address) _Alloc __alloc_;

     public:
      using value_type      = __traits_t::value_type;
      using pointer         = __traits_t::pointer;
      using size_type       = __traits_t::size_type;
      using is_always_equal = __traits_t::is_always_equal;

      template <class _Ty>
      struct rebind
      {
        using other = __allocator_adaptor<typename __traits_t::template rebind_alloc<_Ty>>;
      };

      /*implicit*/ constexpr __allocator_adaptor(_Alloc const &__alloc)
        noexcept(std::is_nothrow_copy_constructible_v<_Alloc>)
        : __alloc_(__alloc)
      {}

      template <class _OtherAlloc>
      /*implicit*/ constexpr __allocator_adaptor(__allocator_adaptor<_OtherAlloc> const &__other)
        noexcept(std::is_nothrow_constructible_v<_Alloc, _OtherAlloc const &>)
        : __alloc_(__other.__alloc_)
      {}

      constexpr auto
      allocate(size_type __n) noexcept(noexcept(std::declval<_Alloc &>().allocate(__n))) -> pointer
      {
        return __alloc_.allocate(__n);
      }

      constexpr void deallocate(pointer __p, size_type __n) noexcept
      {
        __alloc_.deallocate(__p, __n);
      }

      template <class _OtherAlloc>
      constexpr auto
      operator==(__allocator_adaptor<_OtherAlloc> const &__other) const noexcept -> bool
      {
        return __alloc_ == __other.__alloc_;
      }
    };

    //! An allocator that allocates from and deallocates to a memory resource of
    //! type _Resource, which is std::pmr::memory_resource or derives from it
    template <class _Resource, class _Ty>
    class __resource_allocator
    {
      template <class, class>
      friend class __resource_allocator;

      _Resource *__resource_;

     public:
      using value_type = _Ty;
      using pointer    = value_type *;

      template <class _Uy>
      struct rebind
      {
        using other = __resource_allocator<_Resource, _Uy>;
      };

      // unlike polymorphic_allocator, there's no default constructor, because
      // there's no default resource of type _Resource

      /*implicit*/ constexpr __resource_allocator(_Resource *__resource) noexcept
        : __resource_(__resource)
      {}

      template <class _Uy>
      /*implicit*/ constexpr __resource_allocator(
        __resource_allocator<_Resource, _Uy> const &__other) noexcept
        : __resource_(__other.__resource_)
      {}

      constexpr auto allocate(std::size_t __n) -> pointer
      {
        return static_cast<pointer>(
          __resource_->allocate(__n * sizeof(value_type), alignof(value_type)));
      }

      constexpr void deallocate(pointer __p, std::size_t __n) noexcept
      {
        __resource_->deallocate(__p, __n * sizeof(value_type), alignof(value_type));
      }

      constexpr auto resource() const noexcept -> _Resource *
      {
        return __resource_;
      }

      template <class _Uy>
      constexpr auto
      operator==(__resource_allocator<_Resource, _Uy> const &__other) const noexcept -> bool
      {
        return *__resource_ == *__other.__resource_;
      }
    };

    template <class _Delegate>
    struct __frame_allocator;

    //! Handle the case that _Delegate is an allocator already
    template <class _Delegate>
      requires __simple_allocator<_Delegate>
    struct __frame_allocator<_Delegate>
    {
      using type = __allocator_adaptor<
        typename std::allocator_traits<_Delegate>::template rebind_alloc<std::byte>>;
    };

    //! Handle the case that _Delegate is a pointer to std::pmr::memory_resource
    //! or to a type that derives from it
    template <class _Resource>
      requires __std::derived_from<_Resource, std::pmr::memory_resource>
    struct __frame_allocator<_Resource *>
    {
      using type = __resource_allocator<_Resource, std::byte>;
    };
  }  // namespace __fa

  template <class _Delegate>
  using __frame_allocator_t = __fa::__frame_allocator<std::remove_cvref_t<_Delegate>>::type;
}  // namespace experimental::execution

namespace exec = experimental::execution;

#include "../stdexec/__detail/__epilogue.hpp"
