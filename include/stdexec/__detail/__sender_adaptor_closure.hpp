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

#include "__execution_fwd.hpp"

#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

namespace STDEXEC {
  // NOT TO SPEC:
  namespace __clsur {
    template <__class _Dp>
    struct sender_adaptor_closure;
  } // namespace __clsur

  using __clsur::sender_adaptor_closure;

  template <class _Tp>
  concept __sender_adaptor_closure =
    __std::derived_from<__decay_t<_Tp>, sender_adaptor_closure<__decay_t<_Tp>>>
    && __std::move_constructible<__decay_t<_Tp>> && __std::constructible_from<__decay_t<_Tp>, _Tp>;

  template <class _Tp, class _Sender>
  concept __sender_adaptor_closure_for = __sender_adaptor_closure<_Tp> && sender<__decay_t<_Sender>>
                                      && __callable<_Tp, __decay_t<_Sender>>
                                      && sender<__call_result_t<_Tp, __decay_t<_Sender>>>;

  namespace __clsur {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      template <sender _Sender>
        requires __callable<_T0, _Sender> && __callable<_T1, __call_result_t<_T0, _Sender>>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Sender&& __sndr) && -> __call_result_t<_T1, __call_result_t<_T0, _Sender>> {
        return static_cast<_T1&&>(__t1_)(static_cast<_T0&&>(__t0_)(static_cast<_Sender&&>(__sndr)));
      }

      template <sender _Sender>
        requires __callable<const _T0&, _Sender>
              && __callable<const _T1&, __call_result_t<const _T0&, _Sender>>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Sender&& __sndr)
        const & -> __call_result_t<const _T1&, __call_result_t<const _T0&, _Sender>> {
        return __t1_(__t0_(static_cast<_Sender&&>(__sndr)));
      }

      STDEXEC_ATTRIBUTE(no_unique_address) _T0 __t0_;
      STDEXEC_ATTRIBUTE(no_unique_address) _T1 __t1_;
    };

    template <__class _Dp>
    struct sender_adaptor_closure { };

    template <sender _Sender, __sender_adaptor_closure_for<_Sender> _Closure>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto
      operator|(_Sender&& __sndr, _Closure&& __clsur)
      noexcept(__nothrow_callable<_Closure, _Sender>) -> __call_result_t<_Closure, _Sender> {
      return static_cast<_Closure&&>(__clsur)(static_cast<_Sender&&>(__sndr));
    }

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator|(_T0&& __t0, _T1&& __t1) -> __compose<__decay_t<_T0>, __decay_t<_T1>> {
      return {{}, static_cast<_T0&&>(__t0), static_cast<_T1&&>(__t1)};
    }

    template <class _Fn, class... _As>
    struct __closure : sender_adaptor_closure<__closure<_Fn, _As...>> {
      template <class _FnFwd = _Fn, class... _AsFwd>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr explicit __closure(_FnFwd&& __fn, _AsFwd&&... __as)
        noexcept(__nothrow_move_constructible<_Fn, _As...>)
        : __fn_{static_cast<_FnFwd&&>(__fn)}
        , __args_{static_cast<_AsFwd&&>(__as)...} {
      }

      template <sender _Sender>
        requires __callable<_Fn, _Sender, _As...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto
        operator()(_Sender&& __sndr) && noexcept(__nothrow_callable<_Fn, _Sender, _As...>) {
        return STDEXEC::__apply(
          static_cast<_Fn&&>(__fn_),
          static_cast<__tuple<_As...>&&>(__args_),
          static_cast<_Sender&&>(__sndr));
      }

      template <sender _Sender>
        requires __callable<const _Fn&, _Sender, const _As&...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr auto operator()(_Sender&& __sndr) const & noexcept(
        __nothrow_callable<const _Fn&, _Sender, const _As&...>) {
        return STDEXEC::__apply(__fn_, __args_, static_cast<_Sender&&>(__sndr));
      }

     private:
      STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
      STDEXEC_ATTRIBUTE(no_unique_address) __tuple<_As...> __args_;
    };

    template <class _Fn, class... _As>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __closure(_Fn, _As...) -> __closure<_Fn, _As...>;
  } // namespace __clsur

  using __clsur::__closure;
} // namespace STDEXEC
