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
//! The intended use for __frame_allocator_t is the dynamic allocation of operation
//! states and/or coroutine frames. __frame_allocator_t models the Allocator named
//! requirement in terms of its type parameter, _Delegate.
//!
//! _Delegate may be one of:
//!  - Another allocator, in which case __frame_allocator_t<_Delegate> is just _Delegate
//!    rebound to std::byte;
//!  - std::pmr::memory_resource*, in which case __frame_allocator_t<_Delegate> is just
//!    std::pmr::polymorphic_allocator<std::byte>; or
//!  - T* for some T that inherits from std::pmr::memor_resource, in which case
//!    __frame_allocator_t is an allocator that behaves like
//!    std::pmr::polymorphic_allocator<std::byte>, but knows the concrete type of its
//!    memory_resource and so therefore may be able to avoid virtual dispatch.
//!
//! Given that __frame_allocator_t<_Delegate> is just an alias to _Delegate when _Delegate
//! is an allocator type, it's up to that type whether its construct member function does
//! "uses-allocator construction". When _Delegate is a pointer to a memory_resource,
//! __frame_allocator_t<_Delegate>::construct does "uses-allocator construction".
namespace experimental::execution
{
  namespace __fa
  {
    using namespace STDEXEC;

    template <class _Delegate>
    struct __frame_allocator;

    //! Handle the case that _Delegate is an allocator already
    //!
    //! In this case, __frame_allocator_t<Delegate> is just an alias to _Delegate but
    //! rebound to std::byte.
    template <class _Delegate>
      requires __simple_allocator<_Delegate>
    struct __frame_allocator<_Delegate>
    {
      template <class _Ty>
      using type = std::allocator_traits<_Delegate>::template rebind_alloc<_Ty>;
    };

    //! Handle the case that _Delegate is exactly std::pmr::memory_resource *
    //!
    //! In this case, __frame_allocator_t<_Delegate> is exactly
    //! std::pmr::polymorphic_allocator<std::byte>.
    template <>
    struct __frame_allocator<std::pmr::memory_resource *>
    {
      template <class _Ty>
      using type = std::pmr::polymorphic_allocator<_Ty>;
    };

    //! Handle the case that _Delegate is a pointer to a type that derives from
    //! std::pmr::memory_resource
    //!
    //! In this case, __frame_allocator_t<_Delegate> is an allocator that behaves like
    //! std::pmr::polymorphic_allocator<std::byte> except that it knows the concrete type
    //! of its memory resource. In other words, allocation and deallocation are delegated
    //! to the given resource, construct does uses-allocator construction, etc.
    template <class _Delegate>
      requires __std::derived_from<_Delegate, std::pmr::memory_resource>
    struct __frame_allocator<_Delegate *>
    {
      template <class _Ty>
      struct type
      {
        using value_type = _Ty;
        using pointer    = value_type *;

        // polymorphic_allocator's default constructor grabs the default memory resource,
        // which we can't do because there's no default for _Delegate

        /*implicit*/ constexpr type(_Delegate *__resource) noexcept
          : __resource_(__resource)
        {}

        constexpr type(type const &) noexcept = default;

        template <class _Uy>
        /*implicit*/ constexpr type(type<_Uy> const &other) noexcept
          : __resource_(other.resource())
        {}

        constexpr ~type() = default;

        constexpr type &operator=(type const &) noexcept = default;

        constexpr pointer allocate(std::size_t __n)
        {
          return static_cast<pointer>(
            __resource_->allocate(__n * sizeof(value_type), alignof(value_type)));
        }

        constexpr void deallocate(void *__p, std::size_t __n) noexcept
        {
          __resource_->deallocate(__p, __n * sizeof(value_type), alignof(value_type));
        }

        template <class _Uy, class... _Args>
        constexpr void construct(_Uy *__p, _Args &&...__args)
        {
          (void) std::uninitialized_construct_using_allocator(__p,
                                                              *this,
                                                              static_cast<_Args &&>(__args)...);
        }

        template <class _Uy>
        constexpr void destroy(_Uy *__p) noexcept
        {
          __p->~_Uy();
        }

        constexpr _Delegate *resource() const noexcept
        {
          return __resource_;
        }

       private:
        _Delegate *__resource_;
      };
    };
  }  // namespace __fa

  template <class _Delegate>
  using __frame_allocator_t =
    __fa::__frame_allocator<std::remove_cvref_t<_Delegate>>::template type<std::byte>;
}  // namespace experimental::execution

namespace exec = experimental::execution;

#include "../stdexec/__detail/__epilogue.hpp"
