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

#include "../stdexec/__detail/__prologue.hpp"

namespace experimental::execution
{
  namespace __mem_rsc_adpt
  {
    using namespace STDEXEC;

    template <class _Adaptee>
    struct __memory_resource_adaptor;

    //! Handle the case that _Adaptee is an allocator of std::bytes
    //!
    //! Implement do_allocate and do_deallocate in terms of _Adaptee's allocate and
    //! deallocate, respectively. Implement do_is_equal in terms of _Adaptee's
    //! operator==.
    template <class _Adaptee>
      requires __simple_allocator<_Adaptee>
            && __same_as<std::byte, typename std::allocator_traits<_Adaptee>::value_type>
    struct __memory_resource_adaptor<_Adaptee>
    {
      //! Implement memory_resource in terms of an allocator<std::byte>
      class type : public std::pmr::memory_resource
      {
        using __traits = std::allocator_traits<_Adaptee>;
        static_assert(__same_as<std::byte, typename __traits::value_type>);
        typename __traits::allocator_type __alloc_;

       public:
        template <class _Alloc>
          requires(!__same_as<_Alloc, type>)
        constexpr explicit type(_Alloc const &__alloc) noexcept
          : __alloc_(__alloc)
        {
          using __rebound_traits = std::allocator_traits<_Alloc>::template rebind_traits<std::byte>;
          static_assert(__same_as<__traits, __rebound_traits>);
        }

        constexpr void *do_allocate(std::size_t __bytes, std::size_t __align) override
        {
          // TODO: we're not using __align, which is probably a bug
          return __traits::allocate(__alloc_, __bytes);
        }

        constexpr void do_deallocate(void *__p, std::size_t __bytes, std::size_t __align) override
        {
          // TODO: we're not using __align, which is probably a bug
          __traits::deallocate(__alloc_, new (__p) std::byte[__bytes], __bytes);
        }

        constexpr bool do_is_equal(std::pmr::memory_resource const &__other) const noexcept override
        {
          if (auto *__ptr = dynamic_cast<type const *>(&__other))
          {
            return __alloc_ == __ptr->__alloc_;
          }

          return false;
        }
      };
    };

    //! Handle the case that _Adaptee is an allocator of some type other than std::byte
    //!
    //! We just rebind _Adaptee to be an allocator of std::bytes and inherit our nested
    //! alias from the adaptor for that type. This strategy ensures that there's only one
    //! adaptor for an entire family of adapted allocator types, reducing template bloat
    //! and making the do_is_equals implementation sensible.
    template <class _Adaptee>
      requires __simple_allocator<_Adaptee>
    struct __memory_resource_adaptor<_Adaptee>
      : __memory_resource_adaptor<
          typename std::allocator_traits<_Adaptee>::template rebind_alloc<std::byte>>
    {};

    //! Handle the case that _Adaptee is a pointer to a type that derives from
    //! std::pmr::memory_resource
    //!
    //! In this case, there's nothing to "adapt" so we just alias _Adaptee.
    template <class _Adaptee>
      requires __std::constructible_from<std::pmr::polymorphic_allocator<std::byte>, _Adaptee>
    struct __memory_resource_adaptor<_Adaptee>
    {
      using type = _Adaptee;
    };
  }  // namespace __mem_rsc_adpt

  //! Adapt _Adaptee to be a std::pmr::memory_resource
  //!
  //! This alias is the identity when _Adaptee is a pointer to a type that derives from
  //! std::pmr::memory_resource. When _Adaptee is an allocator type, it is a type that
  //! derives from std::pmr::memory_resource and implements its pure-virtual member
  //! functions in terms of that allocator type rebound to std::byte.
  template <class _Adaptee>
  using __memory_resource_adaptor_t =
    __mem_rsc_adpt::__memory_resource_adaptor<std::remove_cvref_t<_Adaptee>>::type;
}  // namespace experimental::execution

namespace exec = experimental::execution;

#include "../stdexec/__detail/__epilogue.hpp"
