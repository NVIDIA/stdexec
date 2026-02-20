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

#include "__concepts.hpp"
#include "__config.hpp"
#include "__type_traits.hpp"
#include "__typeinfo.hpp"
#include "__utility.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <bit>
#include <exception>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace STDEXEC::__any
{

  //////////////////////////////////////////////////////////////////////////////////////////
  //! any: a library for ad hoc polymorphism with value semantics
  //!
  //! @par Terminology:
  //!
  //! - "root":
  //!
  //!   A type satisfying the @c root concept that is used as the nucleus of a "model".
  //!   There are 5 root types:
  //!
  //!   - @c __iroot:                 the abstract root
  //!   - @c __value_root:            holds a concrete value
  //!   - @c __reference_root:        holds a concrete reference
  //!   - @c __value_proxy_root:      holds a type-erased value model
  //!   - @c __reference_proxy_root:  holds a type-erased reference model
  //!
  //!   Aside from @c __iroot, all root types inherit from @c __iabstract<Interface>, where
  //!   @c Interface is the interface that the root type implements.
  //!
  //!   The @c root concept is defined as:
  //!
  //!   @code
  //!   template <class Root>
  //!   concept root = requires (Root& root)
  //!   {
  //!     __any::__value(root);
  //!     { __any::reset(root); } -> std::same_as<void>;
  //!     { __any::type(root) } -> std::same_as<const __type_index &>;
  //!     { __any::data(root) } -> std::same_as<void *>;
  //!     { __any::empty(root) } -> std::same_as<bool>;
  //!   };
  //!   @endcode
  //!
  //! - "model":
  //!
  //!   A polymorphic wrapper around a root that is constructed by recursively applying a
  //!   given interface and its base interfaces to the root type. For example, given an
  //!   interface @c Derived that __extends @c Base, the value proxy model is a type derived
  //!   from @c Derived<Base<__value_proxy_root<Derived>>>. Model types implement their given
  //!   interfaces in terms of the root type. There are 5 model types:
  //!
  //!   - @c __iabstract:             akin to an abstract base class for the
  //!                                 interface
  //!   - @c __value_model:           implements the interface for a concrete value
  //!   - @c __reference_model:       implements the interface for a concrete
  //!                                 reference
  //!   - @c __value_proxy_model:     implements the interface over a type-erased
  //!                                 value model
  //!   - @c __reference_proxy_model: implements the interface over a type-erased
  //!                                 reference model
  //!
  //! - "proxy":
  //!
  //!   A level of indirection that stores either a type-erased model in a small buffer or a
  //!   pointer to an object stored elsewhere. The @c __value_proxy_root and @c
  //!   __reference_proxy_root types model the @c root concept and contain an array of bytes
  //!   in which they stores either a polymorphic model in-situ or a (tagged) pointer to a
  //!   heap-allocated model. The @c __value_proxy_model and @c __reference_proxy_model types
  //!   implement the given interface in terms of the root type.
  //!
  //! @par Notes:
  //!
  //! - @c Interface<Base> inherits directly from @c any::interface<Interface,Base>, which
  //!   inherits directly from @c Base.
  //!
  //! - Given an interface template @c Derived that __extends @c Base, the type
  //!   @c __iabstract<Derived> is derived from @c __iabstract<Base>.
  //!
  //! - In the case of multiple interface extension, the inheritance is forced to be linear.
  //!   As a result, for an interface @c C that __extends @c A and @c B (in that order),
  //!   @c __iabstract<C> will have a linear inheritance hierarchy; it will be an alias for
  //!   @c C<B<A<__iroot>>>. The result is that @c __iabstract<C> inherits from @c __iabstract<A>
  //!   but not from @c __iabstract<B>.
  //!
  //! - The "`__proxy_root`" types both implement an @c emplace function that accepts a
  //!   concrete value or reference, wraps it in the appropriate "`__model`" type, and stores
  //!   it either in-situ or on the heap depending on its size and whether it is nothrow
  //!   moveable.
  //!
  //! - The @c __root types (excluding @c __iroot) all inherit from @c __iabstract<Interface>.
  //!   The @c __model types implement the interface in terms of the root type.
  //!
  //! - @c any<Derived> inherits from @c __value_proxy_model<Derived>, which in turn inherits
  //!   from @c Derived<Base<__value_proxy_root<Derived>>>, which in turn inherits from
  //!   @c Derived<Base<__iroot>> (aka @c __iabstract<Derived> ).
  //!
  //! - @c __any_ptr<Derived> is implemented in terms of a mutable private
  //!   @c __reference_proxy_model<Derived> data member, which in turn inherits from
  //!   @c Derived<Base<__reference_proxy_root<Derived>>>.
  //!
  //! - For every @c any<Interface> instantiation, there are 5 instantiations of
  //!   @c Interface:
  //!
  //!   1. @c Interface<...Bases...<__iroot>>>...>
  //!   2. @c Interface<...Bases...<__value_root<Value,Interface>>...>
  //!   3. @c Interface<...Bases...<__reference_root<Value,Interface>>...>
  //!   4. @c Interface<...Bases...<__value_proxy_root<Interface>>...>
  //!   5. @c Interface<...Bases...<__reference_proxy_root<Interface>>...>

  constexpr size_t      __default_buffer_size = 3 * sizeof(void *);
  constexpr char const *__pure_virt_msg       = "internal error: pure virtual %s() called\n";

  //////////////////////////////////////////////////////////////////////////////////////////
  // forward declarations

  // any types
  template <template <class> class _Interface>
  struct __any;

  template <template <class> class _Interface>
  struct __any_ptr_base;

  template <template <class> class _Interface>
  struct __any_ptr;

  template <template <class> class _Interface>
  struct __any_const_ptr;

  template <template <class> class... _BaseInterfaces>
  struct __extends;

  // semiregular interfaces
  template <class _Base>
  struct __imovable;

  template <class _Base>
  struct __icopyable;

  template <class _Base>
  struct __iequality_comparable;

  template <class _Base>
  struct __isemiregular;

  struct __iroot;

  template <template <class> class _Interface>
  using __bases_of = _Interface<__iroot>::__bases_type;

  template <template <class> class _Interface,
            class _Base,
            class _BaseInterfaces   = __extends<>,
            size_t _BufferSize      = __default_buffer_size,
            size_t _BufferAlignment = alignof(std::max_align_t)>
  struct interface;

  //////////////////////////////////////////////////////////////////////////////////////////
  // __interface_cast
  template <template <class> class _Interface, class _Base>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline)
  constexpr _Interface<_Base> &__interface_cast(_Interface<_Base> &__arg) noexcept
  {
    return __arg;
  }

  template <template <class> class _Interface, class _Base>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline)
  constexpr _Interface<_Base> const &__interface_cast(_Interface<_Base> const &__arg) noexcept
  {
    return __arg;  // NOLINT(bugprone-return-const-ref-from-parameter)
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // accessors
  struct __access
  {
    struct __value_t
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto &operator()(_Ty &&__arg) const noexcept
      {
        return __arg.__value_(static_cast<_Ty &&>(__arg));
      }
    };

    struct __empty_t
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr bool operator()(_Ty const &__arg) const noexcept
      {
        return __arg.__empty_();
      }
    };

    struct __reset_t
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr void operator()(_Ty &__arg) const noexcept
      {
        __arg.__reset_();
      }
    };

    struct __type_t
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr __type_index const &operator()(_Ty const &__arg) const noexcept
      {
        return __arg.__type_();
      }
    };

    struct __data_t
    {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto *operator()(_Ty &__arg) const noexcept
      {
        return __arg.__data_();
      }
    };

    struct __caddressof_t
    {
      template <template <class> class _Interface, class _Base>
      [[nodiscard]]
      constexpr auto operator()(_Interface<_Base> const &__arg) const noexcept
      {
        return __any_const_ptr<_Interface>(std::addressof(__arg));
      }
    };

    struct __addressof_t : __caddressof_t
    {
      using __caddressof_t::operator();

      template <template <class> class _Interface, class _Base>
      [[nodiscard]]
      constexpr auto operator()(_Interface<_Base> &__arg) const noexcept
      {
        return __any_ptr<_Interface>(std::addressof(__arg));
      }
    };
  };

  [[maybe_unused]]
  inline constexpr auto __value = __access::__value_t{};
  [[maybe_unused]]
  inline constexpr auto __empty = __access::__empty_t{};
  [[maybe_unused]]
  inline constexpr auto __reset = __access::__reset_t{};
  [[maybe_unused]]
  inline constexpr auto __type = __access::__type_t{};
  [[maybe_unused]]
  inline constexpr auto __data = __access::__data_t{};
  [[maybe_unused]]
  inline constexpr auto __addressof = __access::__addressof_t{};
  [[maybe_unused]]
  inline constexpr auto __caddressof = __access::__caddressof_t{};

  // __value_of_t
  template <class _Ty>
  using __value_of_t = std::decay_t<decltype(__value(__declval<_Ty &>()))>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // __extension_of
  template <class _Interface, template <class> class _BaseInterface>
  concept __extension_of = requires(_Interface const &__arg) {
    STDEXEC::__any::__interface_cast<_BaseInterface>(__arg);
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __is_small: Model is Interface<_Ty> for some concrete _Ty
  template <class _Model>
  [[nodiscard]]
  constexpr bool __is_small(size_t __buffer_size) noexcept
  {
    constexpr bool __nothrow_movable = !__extension_of<_Model, __imovable>
                                    || std::is_nothrow_move_constructible_v<_Model>;
    return sizeof(_Model) <= __buffer_size && __nothrow_movable;
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // __tagged_ptr
  struct __tagged_ptr
  {
    STDEXEC_ATTRIBUTE(always_inline)
    /*implicit*/ constexpr __tagged_ptr() noexcept
      : __data_(std::uintptr_t(1))
    {}

    STDEXEC_ATTRIBUTE(always_inline)
    /*implicit*/ __tagged_ptr(void *__ptr, bool __tag = true) noexcept
      : __data_(reinterpret_cast<std::uintptr_t>(__ptr) | std::uintptr_t(__tag))
    {}

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    void *__get() const noexcept
    {
      return reinterpret_cast<void *>(__data_ & ~std::uintptr_t(1));
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr bool __is_tagged() const noexcept
    {
      return bool(__data_ & std::uintptr_t(1));
    }

    [[nodiscard]]
    constexpr bool operator==(std::nullptr_t) const noexcept
    {
      return __data_ <= std::uintptr_t(1);
    }

   private:
    std::uintptr_t __data_;
  };

  // template <class _Root>
  // concept root = requires (_Root& root)
  // {
  //   _Root::__value_(root);
  //   root.__reset_();
  //   { root.__type_() } -> std::same_as<const __type_index &>;
  //   { root.__data_() } -> std::same_as<void *>;
  //   { root.__empty_() } -> std::same_as<bool>;
  // };

  //! @c __iabstract must be an alias in order for @c __iabstract<_Derived> to be
  //! derived from
  //! @c __iabstract<_Base>. @c __iabstract<_Derived> is an alias for @c
  //! Derived<Base<__iroot>>.
  template <template <class> class _Interface, class _BaseInterfaces = __bases_of<_Interface>>
  using __iabstract = _Interface<__mcall1<_BaseInterfaces, __iroot>>;

  enum class __box_kind
  {
    __abstract,
    __object,
    __proxy
  };

  enum class __root_kind
  {
    __value,
    __reference
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __iroot
  struct __iroot
  {
    static constexpr STDEXEC::__any::__box_kind __box_kind = STDEXEC::__any::__box_kind::__abstract;
    static constexpr size_t __buffer_size                  = sizeof(__tagged_ptr);  // minimum size
    static constexpr size_t __buffer_alignment = alignof(__tagged_ptr);  // minimum alignment
    using __bases_type                         = __extends<>;

    // needed by MSVC for EBO to work for some reason:
    constexpr virtual ~__iroot() = default;

   private:
    template <template <class> class, class, class, size_t, size_t>
    friend struct interface;
    friend struct __access;

    template <class _Self>
    static constexpr auto __value_(_Self &&) noexcept -> _Self &&
    {
      return STDEXEC::__die<_Self &&>(__pure_virt_msg, "__value");
    }

    [[nodiscard]]
    constexpr virtual bool __empty_() const noexcept
    {
      return STDEXEC::__die<bool>(__pure_virt_msg, "empty");
    }

    constexpr virtual void __reset_() noexcept
    {
      STDEXEC::__die(__pure_virt_msg, "reset");
    }

    [[nodiscard]]
    constexpr virtual __type_index const &__type_() const noexcept
    {
      return STDEXEC::__die<__type_index const &>(__pure_virt_msg, "type");
    }

    [[nodiscard]]
    constexpr virtual void *__data_() const noexcept
    {
      return STDEXEC::__die<void *>(__pure_virt_msg, "data");
    }

    void __slice_to_() noexcept            = delete;  // NOLINT(modernize-use-equals-delete)
    void __indirect_bind_() const noexcept = delete;  // NOLINT(modernize-use-equals-delete)
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __box
  template <class _Value>
  struct __box
  {
    constexpr explicit __box(_Value __value) noexcept
      : __val_(std::move(__value))
    {}

    template <class... _Args>
    constexpr explicit __box(_Args &&...__args) noexcept
      : __val_(static_cast<_Args &&>(__args)...)
    {}

    template <class _Self>
    [[nodiscard]]
    static constexpr auto __value_(_Self &&__self) noexcept -> auto &&
    {
      return static_cast<_Self &&>(__self).__val_;
    }

   private:
    _Value __val_;
  };

  // A specialization of __box to take advantage of EBO (empty base optimization):
  template <class _Value>
    requires std::is_empty_v<_Value> && (!std::is_final_v<_Value>)
  struct STDEXEC_ATTRIBUTE(empty_bases) __box<_Value> : private _Value
  {
    constexpr explicit __box(_Value __value) noexcept
      : _Value(std::move(__value))
    {}

    template <class... _Args>
    constexpr explicit __box(_Args &&...__args) noexcept
      : _Value(static_cast<_Args &&>(__args)...)
    {}

    template <class _Self>
    [[nodiscard]]
    static constexpr auto __value_(_Self &&__self) noexcept -> auto &&
    {
      return std::forward<__copy_cvref_t<_Self, _Value>>(__self);
    }
  };

  template <class _Interface, __box_kind _BoxKind>
  concept __has_box_kind = std::remove_reference_t<_Interface>::__box_kind == _BoxKind;

  // Without the check against __has_box_kind, this concept would always be
  // satisfied when building an object model or a proxy model because of the
  // abstract implementation of Interface in the __iabstract layer.
  //
  // any<Derived>
  //   : __value_proxy_model<Derived, V>
  //       : Derived<Base<__value_proxy_root<Derived, V>>>    // __box_kind == proxy
  //         ^^^^^^^        : Derived<Base<__iroot>>          // __box_kind == abstract
  //                          ^^^^^^^

  template <class _Derived, template <class> class _Interface>
  concept __already_implements = requires(_Derived const &__arg) {
    { STDEXEC::__any::__interface_cast<_Interface>(__arg) } -> __has_box_kind<_Derived::__box_kind>;
  };

  // If we are slicing into a buffer that is smaller than our own, then slicing
  // may throw.
  template <class _Interface, class _Base, size_t _BufferSize>
  concept __nothrow_slice = (_Base::__box_kind != __box_kind::__abstract)
                         && (_Base::__root_kind == __root_kind::__value)
                         && (_Interface::__buffer_size >= _BufferSize);

  //////////////////////////////////////////////////////////////////////////////////////////
  // __extends
  template <>
  struct __extends<>
  {
    template <class _Base>
    using __f = _Base;
  };

  template <template <class> class _BaseInterface, template <class> class... _BaseInterfaces>
  struct __extends<_BaseInterface, _BaseInterfaces...>
  {
    template <class _Base, class _BasesOfBase = __mcall1<__bases_of<_BaseInterface>, _Base>>
    using __f = __mcall1<__extends<_BaseInterfaces...>,
                         // If Base already implements BaseInterface, do not re-apply it.
                         __if_c<__already_implements<_Base, _BaseInterface>,
                                _BasesOfBase,
                                _BaseInterface<_BasesOfBase>>>;
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __emplace_into
  template <class _Model, class... _Args>
  constexpr _Model &__emplace_into([[maybe_unused]] __iroot            *&__root_ptr,
                                   [[maybe_unused]] std::span<std::byte> __buff,
                                   _Args &&...__args)
  {
    static_assert(__decays_to<_Model, _Model>);
    STDEXEC_IF_CONSTEVAL
    {
      __root_ptr = ::new _Model(static_cast<_Args &&>(__args)...);
      return *static_cast<_Model *>(__root_ptr);
    }
    else
    {
      if (STDEXEC::__any::__is_small<_Model>(__buff.size()))
      {
        return *std::construct_at(reinterpret_cast<_Model *>(__buff.data()),
                                  static_cast<_Args &&>(__args)...);
      }
      else
      {
        auto *const __model = ::new _Model(static_cast<_Args &&>(__args)...);
        *__std::start_lifetime_as<__tagged_ptr>(__buff.data()) = __tagged_ptr(__model);
        return *__model;
      }
    }
  }

  template <int = 0, class _CvRefValue, class _Value = std::decay_t<_CvRefValue>>
  STDEXEC_ATTRIBUTE(always_inline)
  constexpr _Value &__emplace_into(__iroot            *&__root_ptr,
                                   std::span<std::byte> __buff,
                                   _CvRefValue        &&__value)
  {
    return STDEXEC::__any::__emplace_into<_Value>(__root_ptr,
                                                  __buff,
                                                  static_cast<_CvRefValue &&>(__value));
  }

  // reference
  template <template <class> class _Interface,
            class _Value,
            class _Extension = __iabstract<_Interface>>
  struct __reference_root;

  template <template <class> class _Interface,
            class _Value,
            class _Extension = __iabstract<_Interface>>
  struct __reference_model
    : _Interface<__mcall1<__bases_of<_Interface>, __reference_root<_Interface, _Value, _Extension>>>
  {
    using __base_t = _Interface<
      __mcall1<__bases_of<_Interface>, __reference_root<_Interface, _Value, _Extension>>>;
    using __base_t::__base_t;
  };

  // reference proxy
  template <template <class> class _Interface>
  struct __reference_proxy_root;

  template <template <class> class _Interface>
  struct __reference final  // __reference_proxy_model
    : _Interface<__mcall1<__bases_of<_Interface>, __reference_proxy_root<_Interface>>>
  {};

  template <template <class> class _Interface>
  using __reference_proxy_model = __reference<_Interface>;

  // __value
  template <template <class> class _Interface, class _Value>
  struct __value_root;

  template <template <class> class _Interface, class _Value>
  struct __value_model final
    : _Interface<__mcall1<__bases_of<_Interface>, __value_root<_Interface, _Value>>>
  {
    using __base_t = _Interface<__mcall1<__bases_of<_Interface>, __value_root<_Interface, _Value>>>;
    using __base_t::__base_t;

    // This is a virtual override if _Interface extends __imovable
    //! @pre __is_small<__value_model>(__buff.size())
    constexpr void __move_to(__iroot *&__ptr, std::span<std::byte> __buff) noexcept
    {
      static_assert(__extension_of<__iabstract<_Interface>, __imovable>);
      STDEXEC_ASSERT(STDEXEC::__any::__is_small<__value_model>(__buff.size()));
      STDEXEC::__any::__emplace_into(__ptr, __buff, std::move(*this));
      __reset(*this);
    }

    // This is a virtual override if _Interface extends __icopyable
    constexpr void __copy_to(__iroot *&__ptr, std::span<std::byte> __buff) const
    {
      static_assert(__extension_of<__iabstract<_Interface>, __icopyable>);
      STDEXEC_ASSERT(!__empty(*this));
      STDEXEC::__any::__emplace_into(__ptr, __buff, *this);
    }
  };

  // value proxy
  template <template <class> class _Interface>
  struct __value_proxy_root;

  template <template <class> class _Interface>
  struct __value_proxy_model
    : _Interface<__mcall1<__bases_of<_Interface>, __value_proxy_root<_Interface>>>
  {};

  //////////////////////////////////////////////////////////////////////////////////////////
  //! interface
  template <template <class> class _Interface,
            class _Base,
            class _BaseInterfaces,
            size_t _BufferSize,
            size_t _BufferAlignment>
  struct interface : _Base
  {
    static_assert(std::popcount(_BufferAlignment) == 1, "BufferAlignment must be a power of two");
    using __bases_type          = _BaseInterfaces;
    using __this_interface_type = __iabstract<_Interface, _BaseInterfaces>;
    using _Base::__indirect_bind_;
    using _Base::__slice_to_;
    using _Base::_Base;

    static constexpr size_t __buffer_size = _BufferSize > _Base::__buffer_size
                                            ? _BufferSize
                                            : _Base::__buffer_size;

    static constexpr size_t __buffer_alignment = _BufferAlignment > _Base::__buffer_alignment
                                                 ? _BufferAlignment
                                                 : _Base::__buffer_alignment;

    static constexpr bool __nothrow_slice =
      STDEXEC::__any::__nothrow_slice<__this_interface_type, _Base, __buffer_size>;

    //! @pre !empty(*this)
    constexpr virtual void
    __slice_to_(__value_proxy_root<_Interface> &__result) noexcept(__nothrow_slice)
    {
      STDEXEC_ASSERT(!__empty(*this));
      if constexpr (_Base::__box_kind != __box_kind::__abstract)
      {
        using __root_interface_t = _Base::__interface_type;
        constexpr bool __is_root_interface =
          std::same_as<__root_interface_t, __this_interface_type>;
        STDEXEC_ASSERT(!__is_root_interface);
        if constexpr (!__is_root_interface)
        {
          if constexpr (_Base::__box_kind == __box_kind::__proxy)
          {
            __value(*this).__slice_to_(__result);
            __reset(*this);
          }
          else  // if constexpr (_Base::__box_kind == __box_kind::__object)
          {
            // Move from type-erased values, but not from type-erased references
            constexpr bool __is_value = (_Base::__root_kind == __root_kind::__value);
            // potentially throwing:
            __result.emplace(STDEXEC_DECAY_COPY(STDEXEC::__move_if<__is_value>(__value(*this))));
          }
        }
      }
    }

    //! @pre !empty(*this)
    constexpr virtual void __indirect_bind_(__reference_proxy_root<_Interface> &__result) noexcept
    {
      STDEXEC_ASSERT(!__empty(*this));
      if constexpr (_Base::__box_kind == __box_kind::__proxy)
        __value(*this).__indirect_bind_(__result);
      else if constexpr (_Base::__box_kind == __box_kind::__object)
        __result.__object_bind_(*this);
    }

    //! @pre !empty(*this)
    constexpr virtual void
    __indirect_bind_(__reference_proxy_root<_Interface> &__result) const noexcept
    {
      STDEXEC_ASSERT(!__empty(*this));
      if constexpr (_Base::__box_kind == __box_kind::__proxy)
        __value(*this).__indirect_bind_(__result);
      else if constexpr (_Base::__box_kind == __box_kind::__object)
        __result.__object_bind_(*this);
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __box_root_kind
  template <__box_kind _BoxKind, __root_kind _RootKind>
  struct __box_root_kind
  {
    static constexpr STDEXEC::__any::__box_kind  __box_kind  = _BoxKind;
    static constexpr STDEXEC::__any::__root_kind __root_kind = _RootKind;
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __value_root
  template <template <class> class _Interface, class _Value>
  struct STDEXEC_ATTRIBUTE(empty_bases) __value_root
    : __iabstract<_Interface>
    , __box_root_kind<__box_kind::__object, __root_kind::__value>
    , __box<_Value>
  {
    using __value_type     = _Value;
    using __interface_type = __iabstract<_Interface>;
    using __value_root::__box_root_kind::__box_kind;
    using __box<_Value>::__box;
    using __box<_Value>::__value_;

    [[nodiscard]]
    constexpr bool __empty_() const noexcept final
    {
      return false;
    }

    constexpr void __reset_() noexcept final
    {
      // no-op
    }

    [[nodiscard]]
    constexpr __type_index const &__type_() const noexcept final
    {
      return __mtypeid<_Value>;
    }

    [[nodiscard]]
    constexpr void *__data_() const noexcept final
    {
      return const_cast<void *>(static_cast<void const *>(std::addressof(__value(*this))));
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __value_proxy_root
  template <template <class> class _Interface>
  struct STDEXEC_ATTRIBUTE(empty_bases) __value_proxy_root
    : __iabstract<_Interface>
    , __box_root_kind<__box_kind::__proxy, __root_kind::__value>
  {
    using __interface_type = __iabstract<_Interface>;
    using __iabstract<_Interface>::__buffer_size;
    using __iabstract<_Interface>::__buffer_alignment;
    using __value_proxy_root::__box_root_kind::__box_kind;

    static constexpr bool __movable  = __extension_of<__iabstract<_Interface>, __imovable>;
    static constexpr bool __copyable = __extension_of<__iabstract<_Interface>, __icopyable>;

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr __value_proxy_root() noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        __root_ptr_ = nullptr;
      }
      else
      {
        *__std::start_lifetime_as<__tagged_ptr>(__buff_) = __tagged_ptr();
      }
    }

    constexpr __value_proxy_root(__value_proxy_root &&__other) noexcept
      requires __movable
      : __value_proxy_root()
    {
      swap(__other);
    }

    constexpr __value_proxy_root(__value_proxy_root const &__other)
      requires __copyable
      : __value_proxy_root()
    {
      if (!__empty(__other))
        __value(__other).__copy_to(__root_ptr_, __buff_);
    }

    constexpr ~__value_proxy_root()
    {
      __reset_();
    }

    constexpr __value_proxy_root &operator=(__value_proxy_root &&__other) noexcept
      requires __movable
    {
      if (this != std::addressof(__other))
      {
        __reset_();
        swap(__other);
      }
      return *this;
    }

    constexpr __value_proxy_root &operator=(__value_proxy_root const &__other)
      requires __copyable
    {
      if (this != std::addressof(__other))
        __value_proxy_root(__other).swap(*this);
      return *this;
    }

    constexpr void swap(__value_proxy_root &__other) noexcept
      requires __movable
    {
      STDEXEC_IF_CONSTEVAL
      {
        std::swap(__root_ptr_, __other.__root_ptr_);
      }
      else
      {
        if (this == std::addressof(__other))
          return;

        auto &__this_ptr = *__std::start_lifetime_as<__tagged_ptr>(__buff_);
        auto &__that_ptr = *__std::start_lifetime_as<__tagged_ptr>(__other.__buff_);

        // This also covers the case where both __this_ptr and __that_ptr are null.
        if (__this_ptr.__is_tagged() && __that_ptr.__is_tagged())
          return std::swap(__this_ptr, __that_ptr);

        if (__this_ptr == nullptr)
          return __value(__other).__move_to(__root_ptr_, __buff_);

        if (__that_ptr == nullptr)
          return __value(*this).__move_to(__other.__root_ptr_, __other.__buff_);

        auto temp = std::move(*this);
        __value(__other).__move_to(__root_ptr_, __buff_);
        __value(temp).__move_to(__other.__root_ptr_, __other.__buff_);
      }
    }

    template <class _Value, class... _Args>
    constexpr _Value &emplace(_Args &&...__args)
    {
      __reset_();
      return __emplace_<_Value>(static_cast<_Args &&>(__args)...);
    }

    template <int = 0, class _CvRefValue, class _Value = std::decay_t<_CvRefValue>>
    constexpr _Value &emplace(_CvRefValue &&__value)
    {
      __reset_();
      return __emplace_<_Value>(static_cast<_CvRefValue &&>(__value));
    }

    [[nodiscard]]
    constexpr bool __in_situ_() const noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        return false;
      }
      else
      {
        return !(*__std::start_lifetime_as<__tagged_ptr>(__buff_)).__is_tagged();
      }
    }

   private:
    template <template <class> class>
    friend struct __any;
    friend struct __access;

    template <class _Value, class... _Args>
    constexpr _Value &__emplace_(_Args &&...__args)
    {
      static_assert(__decays_to<_Value, _Value>, "Value must be an object type.");
      using __model_type = __value_model<_Interface, _Value>;
      auto &__model      = STDEXEC::__any::__emplace_into<__model_type>(__root_ptr_,
                                                                   __buff_,
                                                                   static_cast<_Args &&>(
                                                                     __args)...);
      return __value(__model);
    }

    template <int = 0, class _CvRefValue, class _Value = std::decay_t<_CvRefValue>>
    constexpr _Value &__emplace_(_CvRefValue &&__value)
    {
      return __emplace_<_Value>(static_cast<_CvRefValue &&>(__value));
    }

    template <class _Self>
    [[nodiscard]]
    static constexpr auto __value_(_Self &&__self) noexcept -> auto &&
    {
      using __root_ptr_t      = std::add_pointer_t<__copy_cvref_t<_Self, __iroot>>;
      using __interface_ref_t = __copy_cvref_t<_Self &&, __iabstract<_Interface>>;
      using __interface_ptr_t = std::add_pointer_t<__interface_ref_t>;
      STDEXEC_IF_CONSTEVAL
      {
        return static_cast<__interface_ref_t>(
          *STDEXEC::__polymorphic_downcast<__interface_ptr_t>(__self.__root_ptr_));
      }
      else
      {
        auto const __ptr = *__std::start_lifetime_as<__tagged_ptr>(__self.__buff_);
        STDEXEC_ASSERT(__ptr != nullptr);
        auto *__root_ptr = static_cast<__root_ptr_t>(__ptr.__is_tagged() ? __ptr.__get()
                                                                         : __self.__buff_);
        return static_cast<__interface_ref_t>(
          *STDEXEC::__polymorphic_downcast<__interface_ptr_t>(__root_ptr));
      }
    }

    [[nodiscard]]
    constexpr bool __empty_() const noexcept final
    {
      STDEXEC_IF_CONSTEVAL
      {
        return __root_ptr_ == nullptr;
      }
      else
      {
        return *__std::start_lifetime_as<__tagged_ptr>(__buff_) == nullptr;
      }
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void __reset_() noexcept final
    {
      STDEXEC_IF_CONSTEVAL
      {
        delete std::exchange(__root_ptr_, nullptr);
      }
      else
      {
        auto &__ptr = *__std::start_lifetime_as<__tagged_ptr>(__buff_);
        if (__ptr == nullptr)
          return;
        else if (!__ptr.__is_tagged())
          std::destroy_at(std::addressof(__value(*this)));
        else
          delete std::addressof(__value(*this));

        __ptr = __tagged_ptr();
      }
    }

    [[nodiscard]]
    constexpr __type_index const &__type_() const noexcept final
    {
      return __empty_() ? __mtypeid<void> : __type(__value(*this));
    }

    [[nodiscard]]
    constexpr void *__data_() const noexcept final
    {
      return __empty_() ? nullptr : __data(__value(*this));
    }

    union
    {
      __iroot *__root_ptr_ = nullptr;                                //!< Used in consteval context
      alignas(__buffer_alignment) std::byte __buff_[__buffer_size];  //!< Used in runtime context
    };
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __reference_union
  template <class _Value>
  struct __reference_union
  {
    union
    {
      _Value  *__value_ptr_ = nullptr;
      __iroot *__root_ptr_;  // points to a __value_root<_Extension, _Value>
    };
    bool __which_ = false;  // true if root, false if value
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __reference_root
  template <template <class> class _Interface, class _Value>
  struct STDEXEC_ATTRIBUTE(empty_bases) __reference_root<_Interface, _Value>
    : __iabstract<_Interface>
    , __box_root_kind<__box_kind::__object, __root_kind::__reference>
  {
    using __value_type     = _Value;
    using __interface_type = __iabstract<_Interface>;
    using __reference_root::__box_root_kind::__box_kind;

    __reference_root(__reference_root &&)            = delete;
    __reference_root &operator=(__reference_root &&) = delete;

    constexpr explicit __reference_root(_Value *__value_ptr, __iroot *__root_ptr) noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        if (__value_ptr != nullptr)
          __union_ = ::new __reference_union<_Value>{.__value_ptr_ = __value_ptr,
                                                     .__which_     = false};
        else
          __union_ = ::new __reference_union<_Value>{.__root_ptr_ = __root_ptr, .__which_ = true};
      }
      else
      {
        if (__value_ptr != nullptr)
          __void_ptr_ = __tagged_ptr(__value_ptr, false);
        else
          __void_ptr_ = __tagged_ptr(__root_ptr, true);
      }
    }

    constexpr ~__reference_root()
    {
      STDEXEC_IF_CONSTEVAL
      {
        delete __union_;
      }
    }

    // Returns true if the reference is indirect (i.e., via a root)
    [[nodiscard]]
    constexpr bool __is_indirect_() const noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        return __union_->__which_;
      }
      else
      {
        return __void_ptr_.__is_tagged();
      }
    }

    [[nodiscard]]
    constexpr _Value *__get_value_ptr_() const noexcept
    {
      if (__is_indirect_())
        return nullptr;

      STDEXEC_IF_CONSTEVAL
      {
        return __union_->__value_ptr_;
      }
      else
      {
        return static_cast<_Value *>(__void_ptr_.__get());
      }
    }

    [[nodiscard]]
    constexpr __iroot *__get_root_ptr_() const noexcept
    {
      if (!__is_indirect_())
        return nullptr;

      STDEXEC_IF_CONSTEVAL
      {
        return __union_->__root_ptr_;
      }
      else
      {
        return static_cast<__iroot *>(__void_ptr_.__get());
      }
    }

    template <class _Self>
    [[nodiscard]]
    static constexpr auto __value_(_Self &&__self) noexcept -> auto &&
    {
      using __value_ref_t = __copy_cvref_t<_Self &&, __value_type>;

      STDEXEC_IF_NOT_CONSTEVAL
      {
        STDEXEC_ASSERT((std::is_convertible_v<_Value &, __value_ref_t>)
                       && "attempt to get a mutable reference from a const reference, or an rvalue "
                          "from an "
                          "lvalue");
      }

      if (__self.__is_indirect_())
        return static_cast<__value_ref_t>(const_cast<__value_ref_t &>(__self.__dereference_()));
      else
        return static_cast<__value_ref_t>(const_cast<__value_ref_t &>(*__self.__get_value_ptr_()));
    }

    [[nodiscard]]
    constexpr bool __empty_() const noexcept final
    {
      return false;
    }

    constexpr void __reset_() noexcept final
    {
      // no-op
    }

    [[nodiscard]]
    constexpr __type_index const &__type_() const noexcept final
    {
      return __mtypeid<__value_type>;
    }

    [[nodiscard]]
    constexpr void *__data_() const noexcept final
    {
      return const_cast<void *>(static_cast<void const *>(std::addressof(__value(*this))));
    }

   private:
    static_assert(!__extension_of<_Value, _Interface>,
                  "_Value must be a concrete type, not an _Interface type.");

    [[nodiscard]]
    constexpr virtual _Value &__dereference_() const noexcept
    {
      auto *__root_ptr     = __get_root_ptr_();
      using __value_root_t = __value_root<_Interface, __value_type>;
      return __value(*STDEXEC::__polymorphic_downcast<__value_root_t *>(__root_ptr));
    }

    union
    {
      __reference_union<_Value> *__union_;
      __tagged_ptr               __void_ptr_;
    };
  };

  template <template <class> class _Interface, class _Value, template <class> class _Extension>
  struct __reference_root<_Interface, _Value, __iabstract<_Extension>>
    : __reference_root<_Interface, _Value>
  {
    using __value_type = _Value;
    using __reference_root<_Interface, _Value>::__reference_root;

    [[nodiscard]]
    constexpr _Value &__dereference_() const noexcept final
    {
      auto *__root_ptr     = (*this).__get_root_ptr_();
      using __value_root_t = __value_root<_Extension, __value_type>;
      return __value(*STDEXEC::__polymorphic_downcast<__value_root_t *>(__root_ptr));
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __reference_proxy_root
  template <template <class> class _Interface>
  struct STDEXEC_ATTRIBUTE(empty_bases) __reference_proxy_root
    : __iabstract<_Interface>
    , __box_root_kind<__box_kind::__proxy, __root_kind::__reference>
  {
    using __interface_type = __iabstract<_Interface>;
    using __reference_proxy_root::__box_root_kind::__box_kind;

    constexpr __reference_proxy_root() noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        __root_ptr_ = nullptr;
      }
      else
      {
        *__std::start_lifetime_as<__tagged_ptr>(__buff_) = __tagged_ptr();
      }
    }

    __reference_proxy_root(__reference_proxy_root &&)            = delete;
    __reference_proxy_root &operator=(__reference_proxy_root &&) = delete;

    constexpr void __copy(__reference_proxy_root const &__other) noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        __value(__other).__indirect_bind_(*this);
      }
      else
      {
        std::memcpy(__buff_, __other.__buff_, sizeof(__buff_));
      }
    }

    constexpr ~__reference_proxy_root()
    {
      STDEXEC_IF_CONSTEVAL
      {
        __reset_();
      }
    }

    constexpr void swap(__reference_proxy_root &__other) noexcept
    {
      if (this == std::addressof(__other))
        return;

      STDEXEC_IF_CONSTEVAL
      {
        std::swap(__root_ptr_, __other.__root_ptr_);
      }
      else
      {
        std::swap(__buff_, __other.__buff_);
      }
    }

    template <__extension_of<_Interface> _CvModel>
    constexpr void __model_bind_(_CvModel &__model) noexcept
    {
      static_assert(__extension_of<_CvModel, _Interface>, "CvModel must implement Interface");
      STDEXEC_IF_CONSTEVAL
      {
        __model.__indirect_bind_(*this);
      }
      else
      {
        if constexpr (std::derived_from<_CvModel, __iabstract<_Interface>>)
        {
          //! Optimize for when Base derives from __iabstract<_Interface>. _Store the
          //! address of __value(__other) directly in __result as a tagged ptr instead of
          //! introducing an indirection.
          //! @post __is_tagged() == true
          auto &__ptr = *__std::start_lifetime_as<__tagged_ptr>(__buff_);
          __ptr       = static_cast<__iabstract<_Interface> *>(
            std::addressof(STDEXEC::__unconst(__model)));
        }
        else
        {
          //! @post __is_tagged() == false
          __model.__indirect_bind_(*this);
        }
      }
    }

    template <class _CvModel>
    constexpr void __object_bind_(_CvModel &__model) noexcept
    {
      static_assert(__extension_of<_CvModel, _Interface>);
      using __extension_type = _CvModel::__interface_type;
      using __value_type     = _CvModel::__value_type;
      using __model_type     = __reference_model<_Interface, __value_type, __extension_type>;
      if constexpr (_CvModel::__root_kind == __root_kind::__reference)
      {
        STDEXEC::__any::__emplace_into<__model_type>(__root_ptr_,
                                                     __buff_,
                                                     __model.__get_value_ptr_(),
                                                     __model.__get_root_ptr_());
      }
      else
      {
        __iroot *__root_ptr = std::addressof(STDEXEC::__unconst(__model));
        STDEXEC::__any::__emplace_into<__model_type>(__root_ptr_,
                                                     __buff_,
                                                     static_cast<__value_type *>(nullptr),
                                                     STDEXEC_DECAY_COPY(__root_ptr));
      }
    }

    template <class _CvValue>
    constexpr void __value_bind_(_CvValue &__value) noexcept
    {
      static_assert(!__extension_of<_CvValue, _Interface>);
      using __model_type = __reference_model<_Interface, _CvValue>;
      STDEXEC::__any::__emplace_into<__model_type>(__root_ptr_,
                                                   __buff_,
                                                   std::addressof(__value),
                                                   static_cast<__iroot *>(nullptr));
    }

    template <class _Self>
    [[nodiscard]]
    static constexpr auto __value_(_Self &&__self) noexcept -> auto &&
    {
      using __root_ptr_t      = std::add_pointer_t<__copy_cvref_t<_Self, __iroot>>;
      using __interface_ref_t = __copy_cvref_t<_Self &&, __iabstract<_Interface>>;
      using __interface_ptr_t = std::add_pointer_t<__interface_ref_t>;
      STDEXEC_IF_CONSTEVAL
      {
        return static_cast<__interface_ref_t>(
          *STDEXEC::__polymorphic_downcast<__interface_ptr_t>(__self.__root_ptr_));
      }
      else
      {
        STDEXEC_ASSERT(!__empty(__self));
        auto const  __ptr      = *__std::start_lifetime_as<__tagged_ptr>(__self.__buff_);
        auto *const __root_ptr = static_cast<__root_ptr_t>(__ptr.__is_tagged() ? __ptr.__get()
                                                                               : __self.__buff_);
        return static_cast<__interface_ref_t>(
          *STDEXEC::__polymorphic_downcast<__interface_ptr_t>(__root_ptr));
      }
    }

    [[nodiscard]]
    constexpr bool __empty_() const noexcept final
    {
      STDEXEC_IF_CONSTEVAL
      {
        return __root_ptr_ == nullptr;
      }
      else
      {
        return *__std::start_lifetime_as<__tagged_ptr>(__buff_) == nullptr;
      }
    }

    constexpr void __reset_() noexcept final
    {
      STDEXEC_IF_CONSTEVAL
      {
        delete std::exchange(__root_ptr_, nullptr);
      }
      else
      {
        *__std::start_lifetime_as<__tagged_ptr>(__buff_) = __tagged_ptr();
      }
    }

    [[nodiscard]]
    constexpr __type_index const &__type_() const noexcept final
    {
      return __empty_() ? __mtypeid<void> : __type(__value(*this));
    }

    [[nodiscard]]
    constexpr void *__data_() const noexcept final
    {
      return __empty_() ? nullptr : __data(__value(*this));
    }

    [[nodiscard]]
    constexpr bool __is_indirect_() const noexcept
    {
      STDEXEC_IF_CONSTEVAL
      {
        return true;
      }
      else
      {
        return !(*__std::start_lifetime_as<__tagged_ptr>(__buff_)).__is_tagged();
      }
    }

   private:
    union
    {
      __iroot *__root_ptr_ = nullptr;  //!< Used in consteval context
      // storage for one vtable __ptr and one pointer for the referant
      mutable std::byte __buff_[2 * sizeof(void *)];  //!< Used in runtime context
    };
  };  // struct __reference_proxy_root

  //////////////////////////////////////////////////////////////////////////////////////////
  // __bad_any_cast
  struct __bad_any_cast : std::exception
  {
    [[nodiscard]]
#if __cpp_lib_constexpr_exceptions >= 2025'02L  // constexpr support for std::exception
    constexpr
#endif
      char const *what() const noexcept override
    {
      return "__bad_any_cast";
    }
  };

#if defined(__cpp_exceptions) && __cpp_exceptions >= 1997'11L
  [[noreturn]]
  inline void __throw_bad_any_cast()
  {
    throw __bad_any_cast();
  }
#else
  [[noreturn]]
  inline constexpr void __throw_bad_any_cast() noexcept
  {
    STDEXEC::__die("__bad_any_cast\n");
  }
#endif

  //////////////////////////////////////////////////////////////////////////////////////////
  //! __any_static_cast
  template <class _Value>
  struct __any_static_cast_impl_t
  {
    template <template <class> class _Interface, class _Base>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto *operator()(_Interface<_Base> *__proxy_ptr) const noexcept
    {
      return __cast_<_Interface>(__proxy_ptr);
    }

    template <template <class> class _Interface, class _Base>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto *operator()(_Interface<_Base> const *__proxy_ptr) const noexcept
    {
      return __cast_<_Interface>(__proxy_ptr);
    }

   private:
    static_assert(__decays_to<_Value, _Value>, "Value must be a decayed type.");

    template <class _CvModel>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    static constexpr auto *__value_ptr_(_CvModel *__model_ptr) noexcept
    {
      return __model_ptr != nullptr ? std::addressof(__value(*__model_ptr)) : nullptr;
    }

    template <class _CvProxy>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    static constexpr bool __is_reference_(_CvProxy *__proxy_ptr) noexcept
    {
      if constexpr (_CvProxy::__root_kind == __root_kind::__reference)
        return (*__proxy_ptr).__is_indirect_();
      else
        return false;
    }

    template <template <class> class _Interface, class _CvProxy>
    [[nodiscard]]
    static constexpr auto *__cast_(_CvProxy *__proxy_ptr) noexcept
    {
      static_assert(_CvProxy::__box_kind == __box_kind::__proxy, "CvProxy must be a proxy type.");
      static_assert(!__extension_of<_Value, _Interface>,
                    "Cannot dynamic cast to an _Interface type.");
      using __value_model     = __copy_cvref_t<_CvProxy, __value_root<_Interface, _Value>>;
      using __reference_model = __copy_cvref_t<_CvProxy, __reference_root<_Interface, _Value>>;

      // get the address of the model from the proxy:
      auto *__model_ptr = std::addressof(__value(*__proxy_ptr));

      // If CvProxy is a reference proxy that stores the model indirectly, then __model_ptr
      // points to a reference model. Otherwise, it points to a value model.
      return __is_reference_(__proxy_ptr)
             ? __value_ptr_(STDEXEC::__polymorphic_downcast<__reference_model *>(__model_ptr))
             : __value_ptr_(STDEXEC::__polymorphic_downcast<__value_model *>(__model_ptr));
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  //! __any_static_cast
  template <class _Value>
  struct __any_dynamic_cast_t
  {
    template <class _CvProxy>
    [[nodiscard]]
    constexpr auto *operator()(_CvProxy *__proxy_ptr) const noexcept
    {
      return __type(*__proxy_ptr) == __mtypeid<_Value>
             ? __any_static_cast_impl_t<_Value>{}(__proxy_ptr)
             : nullptr;
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __basic_cast_t
  template <class _Value, template <class> class _Cast>
  struct __basic_cast_t
  {
    template <template <class> class _Interface, class _Base>
    [[nodiscard]]
    constexpr auto *operator()(_Interface<_Base> *__ptr) const noexcept
    {
      if constexpr (__extension_of<_Value, _Interface>)
        return __ptr;
      else if (__ptr == nullptr || __empty(*__ptr))
        return static_cast<_Value *>(nullptr);
      else
        return __cast(__ptr);
    }

    template <template <class> class _Interface, class _Base>
    [[nodiscard]]
    constexpr auto *operator()(_Interface<_Base> const *__ptr) const noexcept
    {
      if constexpr (__extension_of<_Value, _Interface>)
        return __ptr;
      else if (__ptr == nullptr || __empty(*__ptr))
        return static_cast<_Value const *>(nullptr);
      else
        return __cast(__ptr);
    }

    template <template <class> class _Interface, class _Base>
    [[nodiscard]]
    constexpr auto &&operator()(_Interface<_Base> &&__object) const
    {
      auto *__ptr = (*this)(std::addressof(__object));
      if (__ptr == nullptr)
        __throw_bad_any_cast();
      if constexpr (_Base::__root_kind == __root_kind::__reference)
        return *__ptr;
      else
        return std::move(*__ptr);
    }

    template <template <class> class _Interface, class _Base>
    [[nodiscard]]
    constexpr auto &operator()(_Interface<_Base> &__object) const
    {
      auto *__ptr = (*this)(std::addressof(__object));
      if (__ptr == nullptr)
        __throw_bad_any_cast();
      return *__ptr;
    }

    template <template <class> class _Interface, class _Base>
    [[nodiscard]]
    constexpr auto &operator()(_Interface<_Base> const &__object) const
    {
      auto *__ptr = (*this)(std::addressof(__object));
      if (__ptr == nullptr)
        __throw_bad_any_cast();
      return *__ptr;
    }

    template <template <class> class _Interface>
    [[nodiscard]]
    constexpr auto *operator()(__any_ptr<_Interface> const &__ptr) const
    {
      return (*this)(__ptr.operator->());
    }

    template <template <class> class _Interface>
    [[nodiscard]]
    constexpr auto *operator()(__any_const_ptr<_Interface> const &__ptr) const
    {
      return (*this)(__ptr.operator->());
    }

   private:
    static_assert(__decays_to<_Value, _Value>);
    // The cast is either checked (dynamic) or unchecked (static)
    static constexpr _Cast<_Value> __cast{};
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __any_cast
  template <class _Value>
  struct __any_cast_t : __basic_cast_t<_Value, __any_dynamic_cast_t>
  {};

  template <class _Value>
  constexpr __any_cast_t<_Value> __any_cast{};

  //////////////////////////////////////////////////////////////////////////////////////////
  // __any_static_cast
  template <class _Value>
  struct __any_static_cast_t : __basic_cast_t<_Value, __any_static_cast_impl_t>
  {};

  template <class _Value>
  constexpr __any_static_cast_t<_Value> __any_static_cast{};

  //////////////////////////////////////////////////////////////////////////////////////////
  // __imovable
  template <class _Base>
  struct __imovable : interface<__imovable, _Base>
  {
    using __imovable::interface::interface;

    constexpr virtual void __move_to(__iroot *&, std::span<std::byte>) noexcept
    {
      STDEXEC::__die(__pure_virt_msg, "__move_to");
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __icopyable
  template <class _Base>
  struct __icopyable : interface<__icopyable, _Base, __extends<__imovable>>
  {
    using __icopyable::interface::interface;

    constexpr virtual void __copy_to(__iroot *&, std::span<std::byte>) const
    {
      STDEXEC::__die(__pure_virt_msg, "__copy_to");
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // utils
  template <class _Value, template <class> class _Interface>
  concept __model_of = __decays_to<_Value, _Value> && !std::derived_from<_Value, __iroot>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // any
  template <template <class> class _Interface>
  struct __any final : __value_proxy_model<_Interface>
  {
   private:
    template <class _Other>
    static constexpr bool __as_large_as =
      (__iabstract<_Interface>::__buffer_size >= _Interface<_Other>::__buffer_size)
      && (_Interface<_Other>::__buffer_alignment % __iabstract<_Interface>::__buffer_alignment == 0)
      && (_Other::__root_kind == __root_kind::__value);

   public:
    __any() = default;

    // Construct from an object that implements the interface (and is not an any<>
    // itself)
    template <__model_of<_Interface> _Value>
    constexpr __any(_Value __value)
      : __any()
    {
      (*this).__emplace_(std::move(__value));
    }

    template <class _Type, class... _Args>
    constexpr explicit __any(std::in_place_type_t<_Type>, _Args &&...__args)
      : __any()
    {
      (*this).template __emplace_<_Type>(static_cast<_Args &&>(__args)...);
    }

    // Implicit derived-to-base conversion constructor
    template <class _Other>
      requires __extension_of<_Interface<_Other>, __imovable>
            && (_Other::__root_kind == __root_kind::__value)
    constexpr __any(_Interface<_Other> __other) noexcept(__as_large_as<_Other>)
    {
      (*this).__assign(std::move(__other));
    }

    template <class _Other>
      requires __extension_of<_Interface<_Other>, __icopyable>
            && (_Other::__root_kind == __root_kind::__reference)
    constexpr __any(_Interface<_Other> const &__other)
    {
      _Interface<_Other> __tmp;
      __tmp.__copy(__other);
      (*this).__assign(std::move(__tmp));
    }

    template <__model_of<_Interface> _Value>
    constexpr __any &operator=(_Value __value)
    {
      __reset(*this);
      (*this).__emplace_(std::move(__value));
      return *this;
    }

    // Implicit derived-to-base conversion constructor
    template <class _Other>
      requires __extension_of<_Interface<_Other>, __imovable>
            && (_Other::__root_kind == __root_kind::__value)
    constexpr __any &operator=(_Interface<_Other> __other) noexcept(__as_large_as<_Other>)
    {
      __reset(*this);
      (*this).__assign(std::move(__other));
      return *this;
    }

    template <class _Other>
      requires __extension_of<_Interface<_Other>, __icopyable>
            && (_Other::__root_kind == __root_kind::__reference)
    constexpr __any &operator=(_Interface<_Other> const &__other)
    {
      // Guard against __self-assignment when __other is a reference to *this
      if (__data(__other) == __data(*this))
        return *this;

      _Interface<_Other> __tmp;
      __tmp.__copy(__other);

      __reset(*this);
      (*this).__assign(std::move(__tmp));
      return *this;
    }

    friend constexpr void swap(__any &__lhs, __any &__rhs) noexcept
      requires __any::__movable
    {
      __lhs.swap(__rhs);
    }

   private:
    // Assigning from a type that __extends _Interface. _Its buffer may be larger than
    // ours, or it may be a reference type, so we can be only conditionally
    // noexcept.
    template <class _Other>
      requires __extension_of<_Interface<_Other>, __imovable>
    constexpr void __assign(_Interface<_Other> &&__other) noexcept(__as_large_as<_Other>)
    {
      constexpr bool __ptr_convertible = std::derived_from<_Other, __iabstract<_Interface>>;

      if (__empty(__other))
      {
        return;
        // NOLINTNEXTLINE(bugprone-branch-clone)
      }
      else if constexpr (_Other::__root_kind == __root_kind::__reference || !__ptr_convertible)
      {
        return __other.__slice_to_(*this);
      }
      else if (__other.__in_situ_())
      {
        return __other.__slice_to_(*this);
      }
      else
        STDEXEC_IF_CONSTEVAL
        {
          (*this).__root_ptr_ = std::exchange(__other.__root_ptr_, nullptr);
        }
      else
      {
        auto &__this_ptr = *__std::start_lifetime_as<__tagged_ptr>((*this).__buff_);
        auto &__that_ptr = *__std::start_lifetime_as<__tagged_ptr>(__other.__buff_);
        __this_ptr       = std::exchange(__that_ptr, nullptr);
      }
    }

    static_assert(sizeof(__iabstract<_Interface>) == sizeof(void *));  // sanity check
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __any_ptr_base
  template <template <class> class _Interface>
  struct __any_ptr_base
  {
    __any_ptr_base() = default;

    constexpr __any_ptr_base(std::nullptr_t) noexcept
      : __reference_()
    {}

    constexpr __any_ptr_base(__any_ptr_base const &__other) noexcept
      : __reference_()
    {
      (*this).__proxy_assign(std::addressof(__other.__reference_));
    }

    template <template <class> class _OtherInterface>
      requires __extension_of<__iabstract<_OtherInterface>, _Interface>
    constexpr __any_ptr_base(__any_ptr_base<_OtherInterface> const &__other) noexcept
      : __reference_()
    {
      (*this).__proxy_assign(std::addressof(__other.__reference_));
    }

    constexpr __any_ptr_base &operator=(__any_ptr_base const &__other) noexcept
    {
      __reset(__reference_);
      (*this).__proxy_assign(std::addressof(__other.__reference_));
      return *this;
    }

    constexpr __any_ptr_base &operator=(std::nullptr_t) noexcept
    {
      __reset(__reference_);
      return *this;
    }

    template <template <class> class _OtherInterface>
      requires __extension_of<__iabstract<_OtherInterface>, _Interface>
    constexpr __any_ptr_base &operator=(__any_ptr_base<_OtherInterface> const &__other) noexcept
    {
      __reset(__reference_);
      (*this).__proxy_assign(std::addressof(__other.__reference_));
      return *this;
    }

    friend constexpr void swap(__any_ptr_base &__lhs, __any_ptr_base &__rhs) noexcept
    {
      __lhs.__reference_.swap(__rhs.__reference_);
    }

    [[nodiscard]]
    constexpr bool operator==(__any_ptr_base const &__other) const noexcept
    {
      return __data(__reference_) == __data(__other.__reference_);
    }

   private:
    static_assert(sizeof(__iabstract<_Interface>) == sizeof(void *));  // sanity check

    template <template <class> class>
    friend struct __any_ptr_base;

    friend struct __any_ptr<_Interface>;
    friend struct __any_const_ptr<_Interface>;

    //! @param __other A pointer to a value proxy model implementing _Interface.
    template <__extension_of<_Interface> _CvValueProxy>
    constexpr void __proxy_assign(_CvValueProxy *__proxy_ptr) noexcept
    {
      static_assert(_CvValueProxy::__box_kind == __box_kind::__proxy);
      constexpr bool __is_const_ = std::is_const_v<_CvValueProxy>;

      if (__proxy_ptr == nullptr || __empty(*__proxy_ptr))
        return;
      // _Optimize for when _CvValueProxy derives from __iabstract<_Interface>. _Store the address
      // of __value(__other) directly in __result as a tagged ptr instead of introducing an
      // indirection.
      else if constexpr (std::derived_from<_CvValueProxy, __iabstract<_Interface>>)
        __reference_.__model_bind_(STDEXEC::__as_const_if<__is_const_>(__value(*__proxy_ptr)));
      else
        __value(*__proxy_ptr).__indirect_bind_(__reference_);
    }

    //! @param __other A pointer to a reference proxy model implementing _Interface.
    template <__extension_of<_Interface> _CvReferenceProxy>
      requires(_CvReferenceProxy::__root_kind == __root_kind::__reference)
    constexpr void __proxy_assign(_CvReferenceProxy *__proxy_ptr) noexcept
    {
      static_assert(_CvReferenceProxy::__box_kind == __box_kind::__proxy);
      using __model_type         = __reference_proxy_model<_Interface>;
      constexpr bool __is_const_ = std::is_const_v<_CvReferenceProxy>;

      if (__proxy_ptr == nullptr || __empty(*__proxy_ptr))
        return;
      // in the case where _CvReferenceProxy is a base class of __model_type, we can simply
      // downcast and copy the model directly.
      else if constexpr (std::derived_from<__model_type, _CvReferenceProxy>)
        __reference_.__copy(*STDEXEC::__polymorphic_downcast<__model_type const *>(__proxy_ptr));
      // _Otherwise, we are assigning from a derived reference to a base reference, and the
      // __other reference is indirect (i.e., it holds a __reference_model in its buffer). We
      // need to copy the referant model.
      else if ((*__proxy_ptr).__is_indirect_())
        __value(*__proxy_ptr).__indirect_bind_(__reference_);
      else
        __reference_.__model_bind_(STDEXEC::__as_const_if<__is_const_>(__value(*__proxy_ptr)));
    }

    template <class _CvValue>
    constexpr void __value_assign(_CvValue *__value_ptr) noexcept
    {
      if (__value_ptr != nullptr)
        __reference_.__value_bind_(*__value_ptr);
    }

    // the proxy model is mutable so that a const __any_ptr can return non-const
    // references from operator-> and operator*.
    mutable __reference_proxy_model<_Interface> __reference_;
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __any_ptr
  template <template <class> class _Interface>
  struct __any_ptr : __any_ptr_base<_Interface>
  {
    using __any_ptr_base<_Interface>::__any_ptr_base;
    using __any_ptr_base<_Interface>::operator=;

    // Disable const-to-mutable conversions:
    template <template <class> class _Other>
    __any_ptr(__any_const_ptr<_Other> const &) = delete;
    template <template <class> class _Other>
    __any_ptr &operator=(__any_const_ptr<_Other> const &) = delete;

    template <__model_of<_Interface> _Value>
    constexpr __any_ptr(_Value *__value_ptr) noexcept
      : __any_ptr_base<_Interface>()
    {
      (*this).__value_assign(__value_ptr);
    }

    template <__extension_of<_Interface> _Proxy>
    constexpr __any_ptr(_Proxy *__proxy_ptr) noexcept
      : __any_ptr_base<_Interface>()
    {
      (*this).__proxy_assign(__proxy_ptr);
    }

    template <__extension_of<_Interface> _Proxy>
    __any_ptr(_Proxy const *) = delete;

    template <__model_of<_Interface> _Value>
    constexpr __any_ptr &operator=(_Value *__value_ptr) noexcept
    {
      __reset((*this).__reference_);
      (*this).__value_assign(__value_ptr);
      return *this;
    }

    template <__extension_of<_Interface> _Proxy>
    constexpr __any_ptr &operator=(_Proxy *__proxy_ptr) noexcept
    {
      __reset((*this).__reference_);
      (*this).__proxy_assign(__proxy_ptr);
      return *this;
    }

    template <__extension_of<_Interface> _Proxy>
    __any_ptr &operator=(_Proxy const *__proxy_ptr) = delete;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto *operator->() const noexcept
    {
      return std::addressof((*this).__reference_);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto &operator*() const noexcept
    {
      return (*this).__reference_;
    }
  };

  template <template <class> class _Interface, class _Base>
  __any_ptr(_Interface<_Base> *) -> __any_ptr<_Interface>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // __any_const_ptr
  template <template <class> class _Interface>
  struct __any_const_ptr : __any_ptr_base<_Interface>
  {
    using __any_ptr_base<_Interface>::__any_ptr_base;
    using __any_ptr_base<_Interface>::operator=;

    template <__model_of<_Interface> _Value>
    constexpr __any_const_ptr(_Value const *__value_ptr) noexcept
      : __any_ptr_base<_Interface>()
    {
      (*this).__value_assign(__value_ptr);
    }

    template <__extension_of<_Interface> _Proxy>
    constexpr __any_const_ptr(_Proxy const *__proxy_ptr) noexcept
      : __any_ptr_base<_Interface>()
    {
      (*this).__proxy_assign(__proxy_ptr);
    }

    template <__model_of<_Interface> _Value>
    constexpr __any_const_ptr &operator=(_Value const *__value_ptr) noexcept
    {
      __reset((*this).__reference_);
      (*this).__value_assign(__value_ptr);
      return *this;
    }

    template <__extension_of<_Interface> _Proxy>
    constexpr __any_const_ptr &operator=(_Proxy const *__proxy_ptr) noexcept
    {
      __reset((*this).__reference_);
      (*this).__proxy_assign(__proxy_ptr);
      return *this;
    }

    friend constexpr void swap(__any_const_ptr &__lhs, __any_const_ptr &__rhs) noexcept
    {
      __lhs.__reference_.swap(__rhs.__reference_);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto const *operator->() const noexcept
    {
      return std::addressof((*this).__reference_);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto const &operator*() const noexcept
    {
      return (*this).__reference_;
    }
  };

  template <template <class> class _Interface, class _Base>
  __any_const_ptr(_Interface<_Base> const *) -> __any_const_ptr<_Interface>;

  //////////////////////////////////////////////////////////////////////////////////////////
  // __iequality_comparable
  template <class _Base>
  struct __iequality_comparable : interface<__iequality_comparable, _Base>
  {
    using __iequality_comparable::interface::interface;

    template <class _Other>
    [[nodiscard]]
    constexpr bool operator==(__iequality_comparable<_Other> const &__other) const
    {
      return __equal_to(STDEXEC::__any::__caddressof(__other));
    }

   private:
    [[nodiscard]]
    // NOLINTNEXTLINE(modernize-use-override)
    constexpr virtual bool __equal_to(__any_const_ptr<__iequality_comparable> __other) const
    {
      auto const &type = STDEXEC::__any::__type(*this);

      if (type != STDEXEC::__any::__type(*__other))
        return false;

      if (type == __mtypeid<void>)
        return true;

      using __value_type = __value_of_t<__iequality_comparable>;
      return __value(*this) == STDEXEC::__any::__any_static_cast<__value_type>(*__other);
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __isemiregular
  template <class _Base>
  struct __isemiregular
    : interface<__isemiregular, _Base, __extends<__icopyable, __iequality_comparable>>
  {
    using __isemiregular::interface::interface;
  };

}  // namespace STDEXEC::__any
