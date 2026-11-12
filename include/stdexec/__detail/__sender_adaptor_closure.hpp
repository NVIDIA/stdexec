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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "../functional.hpp"
#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__tuple.hpp"
#  include "__type_traits.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Closure>
  concept __sender_adaptor_closure =
    __std::derived_from<__decay_t<_Closure>, sender_adaptor_closure<__decay_t<_Closure>>>
    && __movable_value<_Closure>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Closure, class _Sender>
  concept __sender_adaptor_closure_for = __sender_adaptor_closure<_Closure> && sender<_Sender>
                                      && __callable<_Closure, _Sender>
                                      && sender<__call_result_t<_Closure, _Sender>>;

  namespace __clsur
  {
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Closure0, class _Closure1>
    struct STDEXEC_ATTRIBUTE(empty_bases) __composed
      : sender_adaptor_closure<__composed<_Closure0, _Closure1>>
      , STDEXEC::__composed<_Closure0, _Closure1>
    {};

    struct __sender_adaptor_closure_base
    {
      template <sender _Sender, __sender_adaptor_closure_for<_Sender> _Closure>
      STDEXEC_ATTRIBUTE(always_inline)
      friend constexpr auto operator|(_Sender&& __sndr, _Closure&& __clsur)
        noexcept(__nothrow_callable<_Closure, _Sender>) -> __call_result_t<_Closure, _Sender>
      {
        return static_cast<_Closure&&>(__clsur)(static_cast<_Sender&&>(__sndr));
      }

      template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
      STDEXEC_ATTRIBUTE(always_inline)
      friend constexpr auto operator|(_T0&& __t0, _T1&& __t1)
        noexcept(__nothrow_decay_copyable<_T0, _T1>) -> __composed<__decay_t<_T1>, __decay_t<_T0>>
      {
        return {
          {},
          {static_cast<_T1&&>(__t1), static_cast<_T0&&>(__t0)}
        };
      }
    };
  }  // namespace __clsur

  STDEXEC_MODULE_EXPORT
  template <__class _Dp>
  struct sender_adaptor_closure : __clsur::__sender_adaptor_closure_base
  {};

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Fn, class... _As>
  struct __closure : sender_adaptor_closure<__closure<_Fn, _As...>>
  {
    static_assert(__decayed<_Fn, _As...>);

    template <__not_decays_to<__closure> _FnFwd = _Fn, class... _AsFwd>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr explicit __closure(_FnFwd&& __fn, _AsFwd&&... __as)
      noexcept(__nothrow_decay_copyable<_FnFwd, _AsFwd...>)
      : __fn_{static_cast<_FnFwd&&>(__fn)}
      , __args_{static_cast<_AsFwd&&>(__as)...}
    {}

    template <class _Self, sender _Sender>
      requires __callable<__copy_cvref_t<_Self, _Fn>, _Sender, __copy_cvref_t<_Self, _As>...>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto operator())(this _Self&& __self, _Sender&& __sndr)
      noexcept(
        __nothrow_callable<__copy_cvref_t<_Self, _Fn>, _Sender, __copy_cvref_t<_Self, _As>...>)
    {
      return STDEXEC::__apply(static_cast<_Self&&>(__self).__fn_,
                              static_cast<_Self&&>(__self).__args_,
                              static_cast<_Sender&&>(__sndr));
    }
    STDEXEC_EXPLICIT_THIS_END(operator())

   private:
    STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
    STDEXEC_ATTRIBUTE(no_unique_address) __tuple<_As...> __args_;
  };

  template <class _Fn, class... _As>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __closure(_Fn, _As...) -> __closure<_Fn, _As...>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
