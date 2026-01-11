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

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__domain.hpp"
#include "__meta.hpp"
#include "__type_traits.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  namespace __detail {
    template <class _Domain, class _OpTag>
    struct __transform_sender_t {
      template <class _Sndr, class _Env>
      using __domain_for_t =
        __if_c<__has_transform_sender<_Domain, _OpTag, _Sndr, _Env>, _Domain, default_domain>;

      template <class _Sndr, class _Env, bool _Nothrow = true>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      static consteval auto __get_declfn() noexcept {
        using __domain_t = __domain_for_t<_Sndr, _Env>;
        using __result_t = __transform_sender_result_t<__domain_t, _OpTag, _Sndr, _Env>;

        constexpr bool __is_nothrow =
          __has_nothrow_transform_sender<__domain_t, _OpTag, _Sndr, _Env>;

        if constexpr (__merror<__result_t>) {
          return __declfn<__result_t>();
        } else if constexpr (__same_as<__result_t, _Sndr>) {
          return __declfn<__result_t, __is_nothrow>();
        } else if constexpr (__same_as<_OpTag, start_t>) {
          return __get_declfn<__result_t, _Env, (_Nothrow && __is_nothrow)>();
        } else {
          using __transform_recurse_t =
            __transform_sender_t<__completing_domain_t<void, __result_t, _Env>, set_value_t>;
          return __transform_recurse_t::template __get_declfn<
            __result_t,
            _Env,
            (_Nothrow && __is_nothrow)
          >();
        }
      }

      template <class _Sndr>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto
        operator()(_Sndr&& __sndr) const noexcept(__nothrow_move_constructible<_Sndr>) -> _Sndr {
        return static_cast<_Sndr&&>(__sndr);
      }

      template <class _Sndr, class _Env, auto _DeclFn = __get_declfn<_Sndr, _Env>()>
        requires __callable<__mtypeof<_DeclFn>>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
        noexcept(noexcept(_DeclFn())) -> decltype(_DeclFn()) {
        using __domain_t = __domain_for_t<_Sndr, _Env>;
        using __result_t = __transform_sender_result_t<__domain_t, _OpTag, _Sndr, _Env>;

        if constexpr (__same_as<__result_t, _Sndr>) {
          return __domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env);
        } else if constexpr (__same_as<_OpTag, start_t>) {
          return (*this)(
            __domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env), __env);
        } else {
          using __transform_recurse_t =
            __transform_sender_t<__completing_domain_t<void, __result_t, _Env>, set_value_t>;
          return __transform_recurse_t()(
            __domain_t().transform_sender(_OpTag(), static_cast<_Sndr&&>(__sndr), __env), __env);
        }
      }
    };
  } // namespace __detail

  struct transform_sender_t {
   private:
    template <class _Fn1, class _Fn2>
    struct __compose {
      template <class _Sndr, class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto operator()(_Sndr&& __sndr, const _Env& __env) const
        noexcept(noexcept(_Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env)))
          -> decltype(_Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env)) {
        return _Fn1()(_Fn2()(static_cast<_Sndr&&>(__sndr), __env), __env);
      }
    };

    // Two-phase transformation per P3826R0
    // 1. Completing domain transformation (where the sender completes)
    // 2. Starting domain transformation (where the operation state starts)
    template <class _Sndr, class _Env>
    using __impl_fn_t = __compose<
      __detail::__transform_sender_t<__detail::__starting_domain_t<_Env>, start_t>,
      __detail::__transform_sender_t<__detail::__completing_domain_t<void, _Sndr, _Env>, set_value_t>
    >;

   public:
    // NOT TO SPEC:
    template <class _Sndr>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto operator()(_Sndr&& __sndr) const noexcept(__nothrow_move_constructible<_Sndr>) //
      -> _Sndr {
      return static_cast<_Sndr&&>(__sndr);
    }

    template <class _Sndr, class _Env, auto _ImplFn = __impl_fn_t<_Sndr, _Env>{}>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto operator()(_Sndr && __sndr, const _Env & __env) const
      noexcept(noexcept(_ImplFn(static_cast<_Sndr&&>(__sndr), __env)))
        -> decltype(_ImplFn(static_cast<_Sndr&&>(__sndr), __env)) {
      return _ImplFn(static_cast<_Sndr&&>(__sndr), __env);
    }
  };

  inline constexpr transform_sender_t transform_sender{};

  /////////////////////////////////////////////////////////////////////////////
  template <class _Tag, class _Domain, class _Sender, class... _Args>
  concept __has_implementation_for =
    __detail::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>
    || __detail::__has_apply_sender<default_domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.apply_sender]
  inline constexpr struct apply_sender_t {
    template <class _Domain, class _Tag, class _Sender, class... _Args>
      requires __has_implementation_for<_Tag, _Domain, _Sender, _Args...>
    STDEXEC_ATTRIBUTE(always_inline)
    auto
      operator()(_Domain __dom, _Tag, _Sender&& __sndr, _Args&&... __args) const -> decltype(auto) {
      if constexpr (__detail::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>) {
        return __dom
          .apply_sender(_Tag(), static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      } else {
        return default_domain()
          .apply_sender(_Tag(), static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      }
    }
  } apply_sender{};

  template <class _Domain, class _Tag, class _Sender, class... _Args>
  using apply_sender_result_t = __call_result_t<apply_sender_t, _Domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env, class _Tag = set_value_t>
  concept __completes_on = __decays_to<
    __call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>, _Env>,
    _Scheduler
  >;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
