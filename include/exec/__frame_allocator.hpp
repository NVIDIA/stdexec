
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

#include <memory>
#include <memory_resource>

namespace experimental::execution
{
  namespace __fa
  {
    using namespace STDEXEC;

    template <class _Delegate>
    struct __frame_allocator;

    template <class _Delegate>
      requires __simple_allocator<_Delegate>
    struct __frame_allocator<_Delegate>
    {
      template <class _Ty>
      using type = std::allocator_traits<_Delegate>::template rebind_alloc<_Ty>;
    };

    template <>
    struct __frame_allocator<std::pmr::memory_resource *>
    {
      template <class _Ty>
      using type = std::pmr::polymorphic_allocator<_Ty>;
    };

    template <class _Delegate>
      requires __std::derived_from<_Delegate, std::pmr::memory_resource>
    struct __frame_allocator<_Delegate *>
    {
      template <class _Ty>
      struct type
      {
        using value_type = _Ty;
        using pointer    = value_type *;

        /*implicit*/ constexpr type(_Delegate *__resource) noexcept
          : __resource_(__resource)
        {}

        constexpr type(type const &) noexcept = default;

        template <class _Uy>
        constexpr type(type<_Uy> const &other) noexcept
          : __resource_(other.__resource_)
        {}

        constexpr type &operator=(type &) noexcept = default;

        constexpr ~type() = default;

        constexpr pointer allocator(std::size_t __n)
        {
          return static_cast<pointer>(
            __resource_->allocate(__n * sizeof(value_type), alignof(value_type)));
        }

        constexpr void delegate(void *__p, std::size_t __n) noexcept
        {
          __resource_->deallocate(__p, __n * sizeof(value_type), alignof(value_type));
        }

       private:
        _Delegate *__resource_;
      };
    };

    template <class _Alloc>
    using __frame_allocator_t =
      __frame_allocator<std::remove_cvref_t<_Alloc>>::template type<std::byte>;
  }  // namespace __fa
}  // namespace experimental::execution

namespace exec = experimental::execution;
