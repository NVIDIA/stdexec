/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__detail/__config.hpp"
#include "__detail/__utility.hpp"

#include "concepts.hpp" // IWYU pragma: keep

#include <cstddef>
#include <functional>

namespace STDEXEC {
  template <class _Fun0, class _Fun1>
  struct __composed {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fun0 __t0_;
    STDEXEC_ATTRIBUTE(no_unique_address) _Fun1 __t1_;

    template <class... _Ts>
      requires __callable<_Fun1, _Ts...> && __callable<_Fun0, __call_result_t<_Fun1, _Ts...>>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto operator()(_Ts&&... __ts) && noexcept(
      __callable<_Fun1, _Ts...> && __callable<_Fun0, __call_result_t<_Fun1, _Ts...>>)
      -> __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>> {
      return static_cast<_Fun0&&>(__t0_)(static_cast<_Fun1&&>(__t1_)(static_cast<_Ts&&>(__ts)...));
    }

    template <class... _Ts>
      requires __callable<const _Fun1&, _Ts...>
            && __callable<const _Fun0&, __call_result_t<const _Fun1&, _Ts...>>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto operator()(_Ts&&... __ts) const & noexcept(
      __callable<const _Fun1&, _Ts...>
      && __callable<const _Fun0&, __call_result_t<const _Fun1&, _Ts...>>)
      -> __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>> {
      return __t0_(__t1_(static_cast<_Ts&&>(__ts)...));
    }
  };

  inline constexpr struct __compose_t {
    template <class _Fun0, class _Fun1>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto operator()(_Fun0 __fun0, _Fun1 __fun1) const
      noexcept(__nothrow_move_constructible<_Fun0, _Fun1>) -> __composed<_Fun0, _Fun1> {
      return {static_cast<_Fun0&&>(__fun0), static_cast<_Fun1&&>(__fun1)};
    }
  } __compose{};

  namespace __invoke_ {
    template <class>
    inline constexpr bool __is_refwrap = false;
    template <class _Up>
    inline constexpr bool __is_refwrap<std::reference_wrapper<_Up>> = true;
    template <class _Up>
    inline constexpr bool __is_refwrap<const std::reference_wrapper<_Up>> = true;

    template <bool, bool, bool>
    struct __invoke_fn;

    template <>
    struct __invoke_fn<false, true, false> {
      template <class _MbrPtr, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_MbrPtr __mem_ptr, _Ty&& __ty) const noexcept
        -> decltype(((static_cast<_Ty&&>(__ty)).*__mem_ptr)) {
        return ((static_cast<_Ty&&>(__ty)).*__mem_ptr);
      }
    };

    template <>
    struct __invoke_fn<false, false, true> {
      template <class _MbrPtr, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_MbrPtr __mem_ptr, _Ty __ty) const noexcept
        -> decltype((__ty.get().*__mem_ptr)) {
        return (__ty.get().*__mem_ptr);
      }
    };

    template <>
    struct __invoke_fn<false, false, false> {
      template <class _MbrPtr, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_MbrPtr __mem_ptr, _Ty&& __ty) const noexcept
        -> decltype(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)) {
        return ((*static_cast<_Ty&&>(__ty)).*__mem_ptr);
      }
    };

    template <>
    struct __invoke_fn<true, true, false> {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const
        noexcept(noexcept(((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)))
          -> decltype(((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return ((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    template <>
    struct __invoke_fn<true, false, true> {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty __ty, _Args&&... __args) const
        noexcept(noexcept((__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...)))
          -> decltype((__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return (__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    template <>
    struct __invoke_fn<true, false, false> {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const noexcept(
        noexcept(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)))
        -> decltype(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return ((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Mbr, class _Class, class _Ty>
    using __invoke_fn_t = __invoke_fn<
      STDEXEC_IS_FUNCTION(_Mbr),
      STDEXEC_IS_BASE_OF(_Class, STDEXEC_REMOVE_REFERENCE(_Ty)),
      __is_refwrap<STDEXEC_REMOVE_REFERENCE(_Ty)>
    >;

    struct __invoke_t {
      template <class _Fun, class... _Args>
        requires __callable<_Fun, _Args...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Fun&& __fun, _Args&&... __args) const
        noexcept(__nothrow_callable<_Fun, _Args...>) -> __call_result_t<_Fun, _Args...> {
        return static_cast<_Fun&&>(__fun)(static_cast<_Args&&>(__args)...);
      }

      template <class _Mbr, class _Class, class _Ty, class... _Args>
        requires __callable<__invoke_fn_t<_Mbr, _Class, _Ty>, _Mbr _Class::*, _Ty, _Args...>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto operator()(_Mbr _Class::* __fun, _Ty&& __ty, _Args&&... __args) const noexcept(
        __nothrow_callable<__invoke_fn_t<_Mbr, _Class, _Ty>, _Mbr _Class::*, _Ty, _Args...>)
        -> __call_result_t<
          __mmangle_t<__invoke_fn_t, _Mbr, _Class, _Ty>, // to avoid GCC builtin mangling issues
          _Mbr _Class::*,
          _Ty,
          _Args...
        > {
        return __invoke_fn_t<_Mbr, _Class, _Ty>()(
          __fun, static_cast<_Ty&&>(__ty), static_cast<_Args&&>(__args)...);
      }
    };
  } // namespace __invoke_

  inline constexpr __invoke_::__invoke_t __invoke{};

  template <class _Fun, class... _As>
  concept __invocable = requires(_Fun&& __f, _As&&... __as) {
    __invoke(static_cast<_Fun&&>(__f), static_cast<_As&&>(__as)...);
  };

  template <class _Fun, class... _As>
  concept __nothrow_invocable = __invocable<_Fun, _As...> && requires(_Fun&& __f, _As&&... __as) {
    { __invoke(static_cast<_Fun&&>(__f), static_cast<_As&&>(__as)...) } noexcept;
  };

  template <class _Fun, class... _As>
  using __invoke_result_t = decltype(__invoke(__declval<_Fun>(), __declval<_As>()...));

  template <class _Fn, class _Default>
  struct __with_default : _Fn {
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr __with_default(_Fn __fn, _Default __default)
      noexcept(__nothrow_move_constructible<_Fn>)
      : _Fn(static_cast<_Fn&&>(__fn))
      , __default_(static_cast<_Default&&>(__default)) {
    }

    using _Fn::operator();

    template <class... _As>
      requires(!__callable<const _Fn&, _As...>)
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto
      operator()(_As&&...) const noexcept(__nothrow_copy_constructible<_Default>) -> _Default {
      return __default_;
    }

    _Default __default_{};
  };

  template <class _Fn, class _Default>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __with_default(_Fn, _Default) -> __with_default<_Fn, _Default>;

  template <class _Fn>
  struct __for_each {
    template <class... _Ts>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr void operator()(_Ts&&... __ts) noexcept((__nothrow_callable<_Fn&, _Ts> && ...)) {
      (static_cast<void>(__fn_(static_cast<_Ts&&>(__ts))), ...);
    }

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr void
      operator()(_Ts&&... __ts) const noexcept((__nothrow_callable<const _Fn&, _Ts> && ...)) {
      (static_cast<void>(__fn_(static_cast<_Ts&&>(__ts))), ...);
    }

    _Fn __fn_;
  };

  template <class _Fn>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __for_each(_Fn) -> __for_each<_Fn>;

  template <class _Ty>
  struct __construct {
    template <class... _As>
      requires __std::constructible_from<_Ty, _As...>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto operator()(_As&&... __as) const noexcept( //
      __nothrow_constructible_from<_Ty, _As...>) -> _Ty {
      return _Ty(static_cast<_As&&>(__as)...);
    }
  };
} // namespace STDEXEC
