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
#include <cstring>
#include <memory>
#include <memory_resource>
#include <new>

#include "../stdexec/__detail/__prologue.hpp"

//! Defines template <class _Adaptee> exec::__memory_resource_adaptor_t
//!
//! A "memory resource adaptor" adapts a "thing that can allocate memory" to the
//! std::pmr::memory_resource interface. It's used in exec::function when the function's
//! type parameters do not require that its eventual receiver have an environment that's
//! queryable with exec::get_frame_allocator. In those circumstances, function will ensure
//! that the receiver to which its erased sender is connected *does* have an environment
//! that responds to get_frame_allocator with a type-erased frame allocator. The
//! type-erased frame allocator is a std::pmr::polymorphic_allocator<>, and the
//! memory_resource given to it is a __memory_resoure_adaptor<_Adaptee>, where _Adaptee is
//! the type of "allocator" used to allocate the function's operation state. _Adaptee may
//! be one of:
//!  - a type that models Allocator;
//!  - std::pmr::memory_resource*; or
//!  - T*, where T derives from std::pmr::memory_resource.
//!
//! Given an appropriate type, _Adaptee, __memory_resource_adaptor_t<_Adaptee> is a type
//! T, such that:
//!  - T is constructible from an lvalue reference to an object of type _Adaptee; and
//!  - given an object rsrc of type T, std::pmr::polymorphic_allocator<>(&rsrc) is a valid
//!    expression.
namespace experimental::execution
{
  namespace __mem_rsc_adpt
  {

// some old versions of Clang, GCC, and MSVC have non-constexpr new/delete
#if defined(__cpp_lib_constexpr_new) && __cpp_lib_constexpr_new >= 202406L
#  define STDEXEC_CONSTEXPR_ALLOC constexpr
#else
#  define STDEXEC_CONSTEXPR_ALLOC
#endif

    template <class... _Args>
    concept __can_globally_delete =  //
      requires(_Args... __args) {
        { ::operator delete(__args...) } -> std::same_as<void>;
      };

    struct __global_delete_fn
    {
      template <class _Void, class _Size, class _Align>
        requires __can_globally_delete<_Void, _Size, _Align>
      STDEXEC_CONSTEXPR_ALLOC void
      operator()(_Void __p, _Size __size, _Align __align) const noexcept
      {
        ::operator delete(__p, __size, __align);
      }

      template <class _Void, class _Size, class _Align>
      STDEXEC_CONSTEXPR_ALLOC void operator()(_Void __p, _Size, _Align __align) const noexcept
      {
        ::operator delete(__p, __align);
      }
    };

    inline constexpr __global_delete_fn __global_delete{};

    using namespace STDEXEC;

    template <class _Adaptee>
    struct __memory_resource_adaptor;

    //! Handle the case that _Adaptee is exactly std::allocator<std::byte>
    //!
    //! Implement do_allocate and do_deallocate in terms of ::operator new and
    //! ::operator delete rather than conservatively reimplementing aligned allocation
    //! on top of an arbitrary allocator.
    template <>
    struct __memory_resource_adaptor<std::allocator<std::byte>>
    {
      struct type : std::pmr::memory_resource
      {
        template <class _Ty>
        constexpr explicit type(std::allocator<_Ty> const &) noexcept
        {}

        STDEXEC_CONSTEXPR_ALLOC
        void *allocate(std::size_t __bytes, std::size_t __align = alignof(std::max_align_t)) const
        {
          return ::operator new(__bytes, static_cast<std::align_val_t>(__align));
        }

        STDEXEC_CONSTEXPR_ALLOC void
        deallocate(void       *__p,
                   std::size_t __bytes,
                   std::size_t __align = alignof(std::max_align_t)) const noexcept
        {
          __global_delete(__p, __bytes, static_cast<std::align_val_t>(__align));
        }

       private:
        STDEXEC_CONSTEXPR_ALLOC void *do_allocate(std::size_t __bytes, std::size_t __align) final
        {
          return allocate(__bytes, __align);
        }

        STDEXEC_CONSTEXPR_ALLOC void
        do_deallocate(void *__p, std::size_t __bytes, std::size_t __align) noexcept final
        {
          deallocate(__p, __bytes, __align);
        }

        STDEXEC_CONSTEXPR_ALLOC bool
        do_is_equal(std::pmr::memory_resource const &__other) const noexcept final
        {
          return !!dynamic_cast<type const *>(&__other);
        }
      };
    };

#undef STDEXEC_CONSTEXPR_ALLOC

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
      struct type : std::pmr::memory_resource
      {
        template <class _Alloc>
          requires(!__same_as<_Alloc, type>)
        constexpr explicit type(_Alloc const &__alloc) noexcept
          : __alloc_(__alloc)
        {
          using __rebound_traits = std::allocator_traits<_Alloc>::template rebind_traits<std::byte>;
          static_assert(__same_as<__traits, __rebound_traits>);
        }

        constexpr void *
        allocate(std::size_t __bytes, std::size_t __align = alignof(std::max_align_t))
        {
          // When asking __alloc_ for __bytes number of bytes, the worst case is that
          // the resulting address is byte-aligned, and we need to adjust right by
          // (__align - 1) bytes to get a properly aligned buffer; since we might have to
          // make that shift, we need to allocate too many bytes, possibly make the shift,
          // and return the resulting address. Since we're going to return an address that
          // might be offset from what we got back from __alloc_, we need a way to
          // retrieve from the offset address what the original address was so can pass to
          // deallocate a pointer that actually originally came from allocate. To do that,
          // we store a copy of the source address at the end of the buffer. To make room
          // for the possible rightward shift and the copy of a pointer, we need to
          // allocate extra space, and __bytes + __align - 1 + sizeof(void *) is the most
          // we might need.
          std::size_t __upstreamSize = __bytes + __align - 1 + sizeof(void *);
          void *const __buffer       = __traits::allocate(__alloc_, __upstreamSize);

          void *__ptr = __buffer;

          void *__ret = std::align(__align, __bytes, __ptr, __upstreamSize);

          // by asking for as much extra storage as we did, std::align ought to succeed
          STDEXEC_ASSERT(__ret != nullptr);
          // this is a postcondition of a successful call to std::align
          STDEXEC_ASSERT(__ret == __ptr);
          // we're going to store the value of __buffer in the first sizeof(void*) bytes
          // after the end of the returned buffer so there had better be room for that
          STDEXEC_ASSERT(__upstreamSize >= (__bytes + sizeof(void *)));

          // put the address we got from __alloc_ at the end of the buffer
          // we're going to return
          auto *__as_bytes = new (__ret) std::byte[__upstreamSize];
          std::memcpy(__as_bytes + __bytes, &__buffer, sizeof(__buffer));

          return __ret;
        }

        constexpr void deallocate(void       *__p,
                                  std::size_t __bytes,
                                  std::size_t __align = alignof(std::max_align_t)) noexcept
        {
          // we have to undo the bit-banging we did in allocate

          void *__address_to_free;
          std::memcpy(&__address_to_free, static_cast<std::byte *>(__p) + __bytes, sizeof(void *));

          std::size_t __size_to_free = __bytes + __align - 1 + sizeof(void *);

          [=]() mutable noexcept
          {
            // std::align mutates its final two by-reference arguments so run this
            // assertion inside an immediately-invoked lambda that captures the inputs by
            // value
            STDEXEC_ASSERT(std::align(__align, __bytes, __address_to_free, __size_to_free) == __p);
          }();

          __traits::deallocate(__alloc_,
                               new (__address_to_free) std::byte[__size_to_free],
                               __size_to_free);
        }

       private:
        using __traits = std::allocator_traits<_Adaptee>;
        static_assert(__same_as<std::byte, typename __traits::value_type>);
        typename __traits::allocator_type __alloc_;

        constexpr void *do_allocate(std::size_t __bytes, std::size_t __align) final
        {
          return allocate(__bytes, __align);
        }

        constexpr void
        do_deallocate(void *__p, std::size_t __bytes, std::size_t __align) noexcept final
        {
          deallocate(__p, __bytes, __align);
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
    {
      // This class is the reason we have a nested type alias named type inside a
      // constrained class template rather than just a constrained class template. We
      // are not deriving a resource adaptor from another adaptor; we're deriving one
      // meta-function from another so that we collapse the number of actual adaptor types
      // to the minimum.
    };

    //! Handle the case that _Adaptee is a pointer to a type that derives from
    //! std::pmr::memory_resource
    //!
    //! In this case, there's nothing to "adapt" but we need a type constructible
    //! from _Adaptee*, and whose operator& returns _Adaptee*.
    template <class _Adaptee>
      requires __std::constructible_from<std::pmr::polymorphic_allocator<std::byte>, _Adaptee *>
    struct __memory_resource_adaptor<_Adaptee *>
    {
      struct type
      {
        explicit constexpr type(_Adaptee *__resource) noexcept
          : __resource_(__resource)
        {}

        constexpr _Adaptee *operator&() const noexcept
        {
          return __resource_;
        }

       private:
        _Adaptee *__resource_;
      };
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
