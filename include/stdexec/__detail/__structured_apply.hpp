/*
 * Copyright (c) 2021-2026 NVIDIA Corporation
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

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "__meta.hpp"
#  include "__structured_binding_size.hpp"
#  include "__tuple.hpp"  // IWYU pragma: keep for __is_tuple and __tuple_size_v
#  include "__type_traits.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>  // IWYU pragma: keep for std::terminate
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
  namespace __detail
  {
    template <auto _Apply>
    struct __static_const
    {
      using type                  = decltype(_Apply);
      static constexpr type value = _Apply;
    };

#  if defined(__cpp_structured_bindings) && __cpp_structured_bindings >= 202411L

    // Structured bindings can introduce a pack, so the implementation of
    // __structured_apply is simple.
    template <bool _Nothrow>
    inline constexpr auto __structured_apply_impl =
      []<class _Fn, class _Type, class... _Us>(_Fn &&__fn, _Type &&__obj, _Us &&...__us)  //
      noexcept(_Nothrow) -> decltype(auto)
    {
      using __cpcv     = __copy_cvref_fn<_Type>;
      auto &[... __as] = __obj;
      return static_cast<_Fn &&>(__fn)(static_cast<_Us &&>(__us)...,
                                       static_cast<__mcall1<__cpcv, decltype(__as)> &&>(__as)...);
    };

    template <int _Ny, bool _Nothrow = true>
    inline constexpr auto const &__structured_apply_v = __structured_apply_impl<_Nothrow>;

#  else

    // Structured bindings *cannot* introduce a pack, so we explicitly handle
    // structures with up to 10 members.
    template <int _Ny, bool _Nothrow = true>
    extern __undefined<__msize_t<_Ny>> __structured_apply_v;

    template <bool _Nothrow>
    inline constexpr auto const &__structured_apply_v<0, _Nothrow> = __static_const<(      //
      []<class _Fn, class... _Us>(_Fn &&__fn, __ignore, _Us &&...__us) noexcept(_Nothrow)  //
      -> decltype(auto)                                                                    //
      {                                                                                    //
        return static_cast<_Fn &&>(__fn)(static_cast<_Us &&>(__us)...);                    //
      })>::value;                                                                          //

#    define STDEXEC_STRUCTURED_APPLY_ELEM_ID(_NY)   STDEXEC_PP_IF(_NY, STDEXEC_PP_COMMA, STDEXEC_PP_EAT)() __a##_NY
#    define STDEXEC_STRUCTURED_APPLY_ELEM(_NY)      , static_cast<__mcall1<__cpcv, decltype(__a##_NY)> &&>(__a##_NY)
#    define STDEXEC_STRUCTURED_APPLY_ITERATE(_IDX)                                              \
    template <bool _Nothrow>                                                                    \
    inline constexpr auto const& __structured_apply_v<_IDX, _Nothrow> = __static_const<(        \
      []<class _Fn, class _Type, class... _Us>(_Fn &&__fn, _Type &&__obj, _Us &&...__us)        \
        noexcept(_Nothrow) -> decltype(auto)                                                    \
      {                                                                                         \
        using __cpcv                                                = __copy_cvref_fn<_Type>;   \
        auto &[STDEXEC_PP_REPEAT(_IDX, STDEXEC_STRUCTURED_APPLY_ELEM_ID)] = __obj;              \
        return static_cast<_Fn &&>(__fn)(                                                       \
          static_cast<_Us &&>(__us)... STDEXEC_PP_REPEAT(_IDX, STDEXEC_STRUCTURED_APPLY_ELEM)); \
      })>::value

    STDEXEC_STRUCTURED_APPLY_ITERATE(1);
    STDEXEC_STRUCTURED_APPLY_ITERATE(2);
    STDEXEC_STRUCTURED_APPLY_ITERATE(3);
    STDEXEC_STRUCTURED_APPLY_ITERATE(4);
    STDEXEC_STRUCTURED_APPLY_ITERATE(5);
    STDEXEC_STRUCTURED_APPLY_ITERATE(6);
    STDEXEC_STRUCTURED_APPLY_ITERATE(7);
    STDEXEC_STRUCTURED_APPLY_ITERATE(8);
    STDEXEC_STRUCTURED_APPLY_ITERATE(9);
    STDEXEC_STRUCTURED_APPLY_ITERATE(10);
#    undef STDEXEC_STRUCTURED_APPLY_ELEM
#    undef STDEXEC_STRUCTURED_APPLY_ELEM_ID
#    undef STDEXEC_STRUCTURED_APPLY_ITERATE

#  endif

    struct __structured_apply
    {
     private:
      template <class _Fn>
      struct __get_declfn
      {
        template <class... _As>
        constexpr auto operator()(_As &&...) const noexcept
        {
          if constexpr (__callable<_Fn, _As...>)
          {
            return __declfn<__call_result_t<_Fn, _As...>, __nothrow_callable<_Fn, _As...>>();
          }
        }
      };

#  if STDEXEC_EDG()
      template <class _Fn, class _Type, class... _Us>
      static auto __declfn_fn()                                                             //
        -> decltype((__detail::__structured_apply_v<__structured_binding_size_v<_Type>>) (  //
          __get_declfn<_Fn>{},
          __declval<_Type>(),
          __declval<_Us>()...));

      template <class _Fn, class _Type, class... _Us>
        requires(__structured_binding_size_v<_Type> >= 0)
      using __declfn_t = decltype(__declfn_fn<_Fn, _Type, _Us...>());
#  else
      template <class _Fn, class _Type, class... _Us>
        requires(__structured_binding_size_v<_Type> >= 0)
      using __declfn_t = __result_of<__structured_apply_v<__structured_binding_size_v<_Type>>,
                                     __get_declfn<_Fn>,
                                     _Type,
                                     _Us...>;
#  endif

     public:
      template <class _Fn, class _Type, class... _Us, class _DeclFn = __declfn_t<_Fn, _Type, _Us...>>
        requires __callable<_DeclFn>
      constexpr auto operator()(_Fn &&__fn, _Type &&__obj, _Us &&...__us) const
        noexcept(__nothrow_callable<_DeclFn>) -> __call_result_t<_DeclFn>
      {
        return __structured_apply_v<__structured_binding_size_v<_Type>,
                                    __nothrow_callable<_DeclFn>>(static_cast<_Fn &&>(__fn),
                                                                 static_cast<_Type &&>(__obj),
                                                                 static_cast<_Us &&>(__us)...);
      }
    };
  }  // namespace __detail

  using __structured_apply_t = __detail::__structured_apply;
  inline constexpr __structured_apply_t __structured_apply{};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
