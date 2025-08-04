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

#include "__concepts.hpp"
#include "__senders_core.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  // NOT TO SPEC:
  namespace __closure {
    template <__class _Dp>
    struct sender_adaptor_closure;
  } // namespace __closure

  using __closure::sender_adaptor_closure;

  template <class _Tp>
  concept __sender_adaptor_closure =
    derived_from<__decay_t<_Tp>, sender_adaptor_closure<__decay_t<_Tp>>>
    && move_constructible<__decay_t<_Tp>> && constructible_from<__decay_t<_Tp>, _Tp>;

  template <class _Tp, class _Sender>
  concept __sender_adaptor_closure_for = __sender_adaptor_closure<_Tp> && sender<__decay_t<_Sender>>
                                      && __callable<_Tp, __decay_t<_Sender>>
                                      && sender<__call_result_t<_Tp, __decay_t<_Sender>>>;

  namespace __closure {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      STDEXEC_ATTRIBUTE(no_unique_address) _T0 __t0_;
      STDEXEC_ATTRIBUTE(no_unique_address) _T1 __t1_;

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
    };

    template <__class _Dp>
    struct sender_adaptor_closure { };

    template <sender _Sender, __sender_adaptor_closure_for<_Sender> _Closure>
    STDEXEC_ATTRIBUTE(always_inline)
    auto operator|(_Sender&& __sndr, _Closure&& __clsur) -> __call_result_t<_Closure, _Sender> {
      return static_cast<_Closure&&>(__clsur)(static_cast<_Sender&&>(__sndr));
    }

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    STDEXEC_ATTRIBUTE(always_inline)
    auto operator|(_T0&& __t0, _T1&& __t1) -> __compose<__decay_t<_T0>, __decay_t<_T1>> {
      return {{}, static_cast<_T0&&>(__t0), static_cast<_T1&&>(__t1)};
    }

    template <class _Fun, class... _As>
    struct __binder_back
      : __tuple_for<_As...>
      , sender_adaptor_closure<__binder_back<_Fun, _As...>> {
      STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_ { };

#if STDEXEC_INTELLISENSE()
      // MSVCBUG https://developercommunity.visualstudio.com/t/rejects-valid-EDG-invocation-of-lambda/10786020

      template <class _Sender>
      struct __lambda_rvalue {
        __binder_back& __self_;
        _Sender& __sndr_;

        STDEXEC_ATTRIBUTE(host, device, always_inline)
        auto operator()(_As&... __as) const noexcept(__nothrow_callable<_Fun, _Sender, _As...>)
          -> __call_result_t<_Fun, _Sender, _As...> {
          return static_cast<_Fun&&>(
            __self_.__fun_)(static_cast<_Sender&&>(__sndr_), static_cast<_As&&>(__as)...);
        }
      };

      template <class _Sender>
      struct __lambda_lvalue {
        __binder_back const & __self_;
        _Sender& __sndr_;

        STDEXEC_ATTRIBUTE(host, device, always_inline)
        auto operator()(const _As&... __as) const
          noexcept(__nothrow_callable<const _Fun&, _Sender, const _As&...>)
            -> __call_result_t<const _Fun&, _Sender, const _As&...> {
          return __self_.__fun_(static_cast<_Sender&&>(__sndr_), __as...);
        }
      };
#endif

      template <sender _Sender>
        requires __callable<_Fun, _Sender, _As...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto operator()(_Sender&& __sndr) && noexcept(__nothrow_callable<_Fun, _Sender, _As...>)
        -> __call_result_t<_Fun, _Sender, _As...> {
#if STDEXEC_INTELLISENSE()
        return this->apply(__lambda_rvalue<_Sender>{*this, __sndr}, *this);
#else
        return this->apply(
          [&__sndr, this](_As&... __as) noexcept(
            __nothrow_callable<_Fun, _Sender, _As...>) -> __call_result_t<_Fun, _Sender, _As...> {
            return static_cast<_Fun&&>(
              __fun_)(static_cast<_Sender&&>(__sndr), static_cast<_As&&>(__as)...);
          },
          *this);
#endif
      }

      template <sender _Sender>
        requires __callable<const _Fun&, _Sender, const _As&...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      auto operator()(_Sender&& __sndr) const & noexcept(
        __nothrow_callable<const _Fun&, _Sender, const _As&...>)
        -> __call_result_t<const _Fun&, _Sender, const _As&...> {
#if STDEXEC_INTELLISENSE()
        return this->apply(__lambda_lvalue<_Sender>{*this, __sndr}, *this);
#else
        return this->apply(
          [&__sndr, this](const _As&... __as) noexcept(
            __nothrow_callable<const _Fun&, _Sender, const _As&...>)
            -> __call_result_t<const _Fun&, _Sender, const _As&...> {
            return __fun_(static_cast<_Sender&&>(__sndr), __as...);
          },
          *this);
#endif
      }
    };
  } // namespace __closure

  using __closure::__binder_back;
} // namespace stdexec
