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
#include "__detail/__meta.hpp"

#include "concepts.hpp" // IWYU pragma: keep

#include <functional>
#include <tuple>
#include <type_traits>
#include <cstddef>

namespace stdexec {
  template <class _Fun0, class _Fun1>
  struct __composed {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fun0 __t0_;
    STDEXEC_ATTRIBUTE(no_unique_address) _Fun1 __t1_;

    template <class... _Ts>
      requires __callable<_Fun1, _Ts...> && __callable<_Fun0, __call_result_t<_Fun1, _Ts...>>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    auto operator()(_Ts&&... __ts) && -> __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>> {
      return static_cast<_Fun0&&>(__t0_)(static_cast<_Fun1&&>(__t1_)(static_cast<_Ts&&>(__ts)...));
    }

    template <class... _Ts>
      requires __callable<const _Fun1&, _Ts...>
            && __callable<const _Fun0&, __call_result_t<const _Fun1&, _Ts...>>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    auto
      operator()(_Ts&&... __ts) const & -> __call_result_t<_Fun0, __call_result_t<_Fun1, _Ts...>> {
      return __t0_(__t1_(static_cast<_Ts&&>(__ts)...));
    }
  };

  inline constexpr struct __compose_t {
    template <class _Fun0, class _Fun1>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    auto operator()(_Fun0 __fun0, _Fun1 __fun1) const -> __composed<_Fun0, _Fun1> {
      return {static_cast<_Fun0&&>(__fun0), static_cast<_Fun1&&>(__fun1)};
    }
  } __compose{};

  namespace __invoke_ {
    template <class>
    inline constexpr bool __is_refwrap = false;
    template <class _Up>
    inline constexpr bool __is_refwrap<std::reference_wrapper<_Up>> = true;

    struct __funobj {
      template <class _Fun, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Fun&& __fun, _Args&&... __args) const
        noexcept(noexcept((static_cast<_Fun&&>(__fun))(static_cast<_Args&&>(__args)...)))
          -> decltype((static_cast<_Fun&&>(__fun))(static_cast<_Args&&>(__args)...)) {
        return static_cast<_Fun&&>(__fun)(static_cast<_Args&&>(__args)...);
      }
    };

    struct __memfn {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const
        noexcept(noexcept(((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)))
          -> decltype(((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return ((static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    struct __memfn_refwrap {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty __ty, _Args&&... __args) const
        noexcept(noexcept((__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...)))
          -> decltype((__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return (__ty.get().*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    struct __memfn_smartptr {
      template <class _Memptr, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Memptr __mem_ptr, _Ty&& __ty, _Args&&... __args) const noexcept(
        noexcept(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)))
        -> decltype(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...)) {
        return ((*static_cast<_Ty&&>(__ty)).*__mem_ptr)(static_cast<_Args&&>(__args)...);
      }
    };

    struct __memobj {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Mbr _Class::* __mem_ptr, _Ty&& __ty) const noexcept
        -> decltype(((static_cast<_Ty&&>(__ty)).*__mem_ptr)) {
        return ((static_cast<_Ty&&>(__ty)).*__mem_ptr);
      }
    };

    struct __memobj_refwrap {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Mbr _Class::* __mem_ptr, _Ty __ty) const noexcept
        -> decltype((__ty.get().*__mem_ptr)) {
        return (__ty.get().*__mem_ptr);
      }
    };

    struct __memobj_smartptr {
      template <class _Mbr, class _Class, class _Ty>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Mbr _Class::* __mem_ptr, _Ty&& __ty) const noexcept
        -> decltype(((*static_cast<_Ty&&>(__ty)).*__mem_ptr)) {
        return ((*static_cast<_Ty&&>(__ty)).*__mem_ptr);
      }
    };

    STDEXEC_ATTRIBUTE(host, device)
    auto __invoke_selector(__ignore, __ignore) noexcept -> __funobj;

    template <class _Mbr, class _Class, class _Ty>
    STDEXEC_ATTRIBUTE(host, device)
    auto __invoke_selector(_Mbr _Class::*, const _Ty&) noexcept {
      if constexpr (STDEXEC_IS_FUNCTION(_Mbr)) {
        // member function ptr case
        if constexpr (STDEXEC_IS_BASE_OF(_Class, _Ty)) {
          return __memfn{};
        } else if constexpr (__is_refwrap<_Ty>) {
          return __memfn_refwrap{};
        } else {
          return __memfn_smartptr{};
        }
      } else {
        // member object ptr case
        if constexpr (STDEXEC_IS_BASE_OF(_Class, _Ty)) {
          return __memobj{};
        } else if constexpr (__is_refwrap<_Ty>) {
          return __memobj_refwrap{};
        } else {
          return __memobj_smartptr{};
        }
      }
    }

    struct __invoke_t {
      template <class _Fun>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Fun&& __fun) const noexcept(noexcept(static_cast<_Fun&&>(__fun)()))
        -> decltype(static_cast<_Fun&&>(__fun)()) {
        return static_cast<_Fun&&>(__fun)();
      }

      template <class _Fun, class _Ty, class... _Args>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Fun&& __fun, _Ty&& __ty, _Args&&... __args) const
        noexcept(noexcept(__invoke_::__invoke_selector(__fun, __ty)(
          static_cast<_Fun&&>(__fun),
          static_cast<_Ty&&>(__ty),
          static_cast<_Args&&>(__args)...)))
          -> decltype(__invoke_::__invoke_selector(__fun, __ty)(
            static_cast<_Fun&&>(__fun),
            static_cast<_Ty&&>(__ty),
            static_cast<_Args&&>(__args)...)) {
        return decltype(__invoke_::__invoke_selector(__fun, __ty))()(
          static_cast<_Fun&&>(__fun), static_cast<_Ty&&>(__ty), static_cast<_Args&&>(__args)...);
      }
    };
  } // namespace __invoke_

  inline constexpr __invoke_::__invoke_t __invoke{};

  template <class _Fun, class... _As>
  concept __invocable = requires(_Fun&& __f, _As&&... __as) {
    __invoke(static_cast<_Fun &&>(__f), static_cast<_As &&>(__as)...);
  };

  template <class _Fun, class... _As>
  concept __nothrow_invocable = __invocable<_Fun, _As...> && requires(_Fun&& __f, _As&&... __as) {
    { __invoke(static_cast<_Fun &&>(__f), static_cast<_As &&>(__as)...) } noexcept;
  };

  template <class _Fun, class... _As>
  using __invoke_result_t = decltype(__invoke(__declval<_Fun>(), __declval<_As>()...));

  namespace __apply_ {
    using std::get;

    template <std::size_t... _Is, class _Fn, class _Tup>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto __impl(__indices<_Is...>, _Fn&& __fn, _Tup&& __tup) noexcept(
      noexcept(__invoke(static_cast<_Fn&&>(__fn), get<_Is>(static_cast<_Tup&&>(__tup))...)))
      -> decltype(__invoke(static_cast<_Fn&&>(__fn), get<_Is>(static_cast<_Tup&&>(__tup))...)) {
      return __invoke(static_cast<_Fn&&>(__fn), get<_Is>(static_cast<_Tup&&>(__tup))...);
    }

    template <class _Tup>
    using __tuple_indices = __make_indices<std::tuple_size<std::remove_cvref_t<_Tup>>::value>;

    template <class _Fn, class _Tup>
    using __result_t =
      decltype(__apply_::__impl(__tuple_indices<_Tup>(), __declval<_Fn>(), __declval<_Tup>()));
  } // namespace __apply_

  template <class _Fn, class _Tup>
  concept __applicable = __mvalid<__apply_::__result_t, _Fn, _Tup>;

  template <class _Fn, class _Tup>
  concept __nothrow_applicable =
    __applicable<_Fn, _Tup>
    && noexcept(
      __apply_::__impl(__apply_::__tuple_indices<_Tup>(), __declval<_Fn>(), __declval<_Tup>()));

  template <class _Fn, class _Tup>
    requires __applicable<_Fn, _Tup>
  using __apply_result_t = __apply_::__result_t<_Fn, _Tup>;

  struct __apply_t {
    template <class _Fn, class _Tup>
      requires __applicable<_Fn, _Tup>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Fn&& __fn, _Tup&& __tup) const
      noexcept(__nothrow_applicable<_Fn, _Tup>) -> __apply_result_t<_Fn, _Tup> {
      return __apply_::__impl(
        __apply_::__tuple_indices<_Tup>(), static_cast<_Fn&&>(__fn), static_cast<_Tup&&>(__tup));
    }
  };

  inline constexpr __apply_t __apply{};
} // namespace stdexec
