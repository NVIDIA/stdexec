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

#include <exception>    // IWYU pragma: keep for std::terminate
#include <type_traits>  // IWYU pragma: export
#include <utility>      // IWYU pragma: keep

namespace STDEXEC
{

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // A very simple std::declval replacement that doesn't handle void
  template <class _Tp, bool _Noexcept = true>
  using __declfn_t = auto (*)() noexcept(_Noexcept) -> _Tp;

  template <class _Tp, class...>
  extern __declfn_t<_Tp &&> __declval;

  template <class... _NoneSuch>
  extern __declfn_t<void> __declval<void, _NoneSuch...>;

#if STDEXEC_MSVC()
  template <class _Tp, bool _Noexcept = true>
  _Tp __declfn_() noexcept(_Noexcept)
  {
    STDEXEC_ASSERT(false && +"__declfn() should never be called" == nullptr);
    STDEXEC_TERMINATE();
  }
  template <class _Tp, bool _Noexcept = true>
  inline constexpr __declfn_t<_Tp, _Noexcept> __declfn() noexcept
  {
    return &__declfn_<_Tp, _Noexcept>;
  }
#else
  template <class _Tp, bool _Noexcept = true>
  using __declfn = __declfn_t<_Tp, _Noexcept>;
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __decay_t: An efficient implementation for std::decay
#if STDEXEC_HAS_BUILTIN(__decay) && (!STDEXEC_CLANG() || STDEXEC_CLANG_VERSION >= 2100)
  namespace __tt
  {
    template <class>
    struct __wrap;

    template <bool>
    struct __decay_
    {
      template <class _Ty>
      using __f = __decay(_Ty);
    };
  }  // namespace __tt

  template <class _Ty>
  using __decay_t = __tt::__decay_<bool(sizeof(__declfn_t<_Ty>))>::template __f<_Ty>;
#else
  template <class _Ty>
  using __decay_t = std::decay_t<_Ty>;
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __copy_cvref_t: For copying cvref from one type to another
  struct __cp
  {
    template <class _Tp>
    using __f = _Tp;
  };

  struct __cpc
  {
    template <class _Tp>
    using __f = _Tp const;
  };

  struct __cplr
  {
    template <class _Tp>
    using __f = _Tp &;
  };

  struct __cprr
  {
    template <class _Tp>
    using __f = _Tp &&;
  };

  struct __cpclr
  {
    template <class _Tp>
    using __f = _Tp const &;
  };

  struct __cpcrr
  {
    template <class _Tp>
    using __f = _Tp const &&;
  };

  template <class>
  extern __cp __cpcvr;
  template <class _Tp>
  extern __cpc __cpcvr<_Tp const>;
  template <class _Tp>
  extern __cplr __cpcvr<_Tp &>;
  template <class _Tp>
  extern __cprr __cpcvr<_Tp &&>;
  template <class _Tp>
  extern __cpclr __cpcvr<_Tp const &>;
  template <class _Tp>
  extern __cpcrr __cpcvr<_Tp const &&>;
  template <class _Tp>
  using __copy_cvref_fn = decltype(__cpcvr<_Tp>);

  template <class _From, class _To>
  using __copy_cvref_t = __copy_cvref_fn<_From>::template __f<_To>;

  template <class>
  inline constexpr bool __is_const_ = false;
  template <class _Up>
  inline constexpr bool __is_const_<_Up const> = true;

  namespace __tt
  {
    template <class _Ty>
    constexpr auto __remove_rvalue_reference_fn(_Ty &&) -> _Ty;
  }  // namespace __tt

  template <class _Ty>
  using __remove_rvalue_reference_t = decltype(__tt::__remove_rvalue_reference_fn(
    __declval<_Ty>()));

  // Implemented as a class instead of a free function
  // because of a bizarre nvc++ compiler bug:
  struct __cref_fn
  {
    template <class _Ty>
    constexpr auto operator()(_Ty const &) -> _Ty const &;
  };
  template <class _Ty>
  using __cref_t = decltype(__cref_fn{}(__declval<_Ty>()));

  // Because of nvc++ nvbugs#4679848, we can't make __mbool a simple alias for __mconstant,
  // and because of nvc++ nvbugs#4668709 it can't be a simple alias for std::bool_constant,
  // either. :-(
  // template <bool _Bp>
  // using __mbool = __mconstant<_Bp>;

  template <bool _Bp>
  struct __mbool : std::bool_constant<_Bp>
  {};

  using __mtrue  = __mbool<true>;
  using __mfalse = __mbool<false>;

}  // namespace STDEXEC
