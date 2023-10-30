/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "__config.hpp"

namespace stdexec {

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // A very simple std::declval replacement that doesn't handle void
  template <class _Tp>
  _Tp&& __declval() noexcept;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __decay_t: An efficient implementation for std::decay
#if STDEXEC_HAS_BUILTIN(__decay)

  template <class _Ty>
  using __decay_t = __decay(_Ty);

#elif STDEXEC_NVHPC()

  template <class _Ty>
  using __decay_t = std::decay_t<_Ty>;

#else

  namespace __tt {
    struct __decay_object {
      template <class _Ty>
      static _Ty __g(_Ty const &);
      template <class _Ty>
      using __f = decltype(__g(__declval<_Ty>()));
    };

    struct __decay_default {
      template <class _Ty>
      static _Ty __g(_Ty);
      template <class _Ty>
      using __f = decltype(__g(__declval<_Ty>()));
    };

    struct __decay_abominable {
      template <class _Ty>
      using __f = _Ty;
    };

    struct __decay_void {
      template <class _Ty>
      using __f = void;
    };

    template <class _Ty>
    extern __decay_object __mdecay;

    template <class _Ty, class... Us>
    extern __decay_default __mdecay<_Ty(Us...)>;

    template <class _Ty, class... Us>
    extern __decay_default __mdecay<_Ty(Us...) noexcept>;

    template <class _Ty, class... Us>
    extern __decay_default __mdecay<_Ty (&)(Us...)>;

    template <class _Ty, class... Us>
    extern __decay_default __mdecay<_Ty (&)(Us...) noexcept>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const noexcept>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const &>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const & noexcept>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const &&>;

    template <class _Ty, class... Us>
    extern __decay_abominable __mdecay<_Ty(Us...) const && noexcept>;

    template <class _Ty>
    extern __decay_default __mdecay<_Ty[]>;

    template <class _Ty, std::size_t N>
    extern __decay_default __mdecay<_Ty[N]>;

    template <class _Ty, std::size_t N>
    extern __decay_default __mdecay<_Ty (&)[N]>;

    template <>
    inline __decay_void __mdecay<void>;

    template <>
    inline __decay_void __mdecay<void const>;
  } // namespace __tt

  template <class _Ty>
  using __decay_t = typename decltype(__tt::__mdecay<_Ty>)::template __f<_Ty>;

#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __copy_cvref_t: For copying cvref from one type to another
  struct __cp {
    template <class _Tp>
    using __f = _Tp;
  };

  struct __cpc {
    template <class _Tp>
    using __f = const _Tp;
  };

  struct __cplr {
    template <class _Tp>
    using __f = _Tp&;
  };

  struct __cprr {
    template <class _Tp>
    using __f = _Tp&&;
  };

  struct __cpclr {
    template <class _Tp>
    using __f = const _Tp&;
  };

  struct __cpcrr {
    template <class _Tp>
    using __f = const _Tp&&;
  };

  template <class>
  extern __cp __cpcvr;
  template <class _Tp>
  extern __cpc __cpcvr<const _Tp>;
  template <class _Tp>
  extern __cplr __cpcvr<_Tp&>;
  template <class _Tp>
  extern __cprr __cpcvr<_Tp&&>;
  template <class _Tp>
  extern __cpclr __cpcvr<const _Tp&>;
  template <class _Tp>
  extern __cpcrr __cpcvr<const _Tp&&>;
  template <class _Tp>
  using __copy_cvref_fn = decltype(__cpcvr<_Tp>);

  template <class _From, class _To>
  using __copy_cvref_t = typename __copy_cvref_fn<_From>::template __f<_To>;

#if !STDEXEC_HAS_BUILTIN(__is_const)
  template <class>
  inline constexpr bool __is_const = false;
  template <class _Up>
  inline constexpr bool __is_const<_Up const> = true;
#endif

} // namespace stdexec
