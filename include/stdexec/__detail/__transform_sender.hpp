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
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__sender_introspection.hpp"
#include "__type_traits.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  namespace __detail {
    struct __transform_env {
      template <class _Domain, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      auto
        operator()(_Domain __dom, _Sender&& __sndr, _Env&& __env) const noexcept -> decltype(auto) {
        if constexpr (__detail::__has_transform_env<_Domain, _Sender, _Env>) {
          return __dom.transform_env(static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
        } else {
          return default_domain()
            .transform_env(static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
        }
      }
    };

    struct __transform_sender_1 {
      template <class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __is_nothrow() noexcept -> bool {
        if constexpr (__detail::__has_transform_sender<_Domain, _Sender, _Env...>) {
          return noexcept(__declval<_Domain&>()
                            .transform_sender(__declval<_Sender>(), __declval<const _Env&>()...));
        } else {
          return noexcept(default_domain()
                            .transform_sender(__declval<_Sender>(), __declval<const _Env&>()...));
        }
      }

      template <class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const
        noexcept(__is_nothrow<_Domain, _Sender, const _Env&...>()) -> decltype(auto) {
        if constexpr (__detail::__has_transform_sender<_Domain, _Sender, _Env...>) {
          return __dom.transform_sender(static_cast<_Sender&&>(__sndr), __env...);
        } else {
          return default_domain().transform_sender(static_cast<_Sender&&>(__sndr), __env...);
        }
      }
    };

    template <class _Ty, class _Uy>
    concept __decay_same_as = same_as<__decay_t<_Ty>, __decay_t<_Uy>>;

    struct __transform_sender {
      template <class _Self = __transform_sender, class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const
        noexcept(__nothrow_callable<__transform_sender_1, _Domain, _Sender, const _Env&...>)
          -> decltype(auto) {
        using _Sender2 = __call_result_t<__transform_sender_1, _Domain, _Sender, const _Env&...>;
        // If the transformation doesn't change the sender's type, then do not
        // apply the transform recursively.
        if constexpr (__decay_same_as<_Sender, _Sender2>) {
          return __transform_sender_1()(__dom, static_cast<_Sender&&>(__sndr), __env...);
        } else {
          // We transformed the sender and got back a different sender. Transform that one too.
          return _Self()(
            __dom,
            __transform_sender_1()(__dom, static_cast<_Sender&&>(__sndr), __env...),
            __env...);
        }
      }
    };

    struct __transform_dependent_sender {
      // If we are doing a lazy customization of a type whose domain is value-dependent (e.g.,
      // let_value), first transform the sender to determine the domain. Then continue transforming
      // the sender with the requested domain.
      template <class _Domain, sender_expr _Sender, class _Env>
        requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
      auto operator()(_Domain __dom, _Sender&& __sndr, const _Env& __env) const
        noexcept(noexcept(__transform_sender()(
          __dom,
          dependent_domain().transform_sender(static_cast<_Sender&&>(__sndr), __env),
          __env))) -> decltype(auto) {
        static_assert(__none_of<_Domain, dependent_domain>);
        return __transform_sender()(
          __dom, dependent_domain().transform_sender(static_cast<_Sender&&>(__sndr), __env), __env);
      }
    };
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  inline constexpr struct transform_sender_t
    : __detail::__transform_sender
    , __detail::__transform_dependent_sender {
    using __detail::__transform_sender::operator();
    using __detail::__transform_dependent_sender::operator();
  } transform_sender{};

  inline constexpr __detail::__transform_env transform_env{};

  struct _CHILD_SENDERS_WITH_DIFFERENT_DOMAINS_ { };

  template <class _Sender, class _Env>
  constexpr auto dependent_domain::__is_nothrow_transform_sender() noexcept -> bool {
    using _Env2 = __call_result_t<__detail::__transform_env, dependent_domain&, _Sender, _Env>;
    return __v<decltype(__sexpr_apply(
      __declval<_Sender>(),
      []<class _Tag, class _Data, class... _Childs>(_Tag, _Data&&, _Childs&&...) {
        constexpr bool __first_transform_is_nothrow = noexcept(__make_sexpr<_Tag>(
          __declval<_Data>(),
          __detail::__transform_sender()(
            __declval<dependent_domain&>(), __declval<_Childs>(), __declval<const _Env2&>())...));
        using _Sender2 = decltype(__make_sexpr<_Tag>(
          __declval<_Data>(),
          __detail::__transform_sender()(
            __declval<dependent_domain&>(), __declval<_Childs>(), __declval<const _Env2&>())...));
        using _Domain2 =
          decltype(__sexpr_apply(__declval<_Sender2&>(), __detail::__common_domain_fn()));
        constexpr bool __second_transform_is_nothrow = noexcept(__detail::__transform_sender()(
          __declval<_Domain2&>(), __declval<_Sender2>(), __declval<const _Env&>()));
        return __mbool<__first_transform_is_nothrow && __second_transform_is_nothrow>();
      }))>;
  }

  template <sender_expr _Sender, class _Env>
    requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
  auto dependent_domain::transform_sender(_Sender&& __sndr, const _Env& __env) const
    noexcept(__is_nothrow_transform_sender<_Sender, _Env>()) -> decltype(auto) {
    // apply any algorithm-specific transformation to the environment
    const auto& __env2 = transform_env(*this, static_cast<_Sender&&>(__sndr), __env);

    // recursively transform the sender to determine the domain
    return __sexpr_apply(
      static_cast<_Sender&&>(__sndr),
      [&]<class _Tag, class _Data, class... _Childs>(_Tag, _Data&& __data, _Childs&&... __childs) {
        // TODO: propagate meta-exceptions here:
        auto __sndr2 = __make_sexpr<_Tag>(
          static_cast<_Data&&>(__data),
          __detail::__transform_sender()(*this, static_cast<_Childs&&>(__childs), __env2)...);
        using _Sender2 = decltype(__sndr2);

        auto __domain2 = __sexpr_apply(__sndr2, __detail::__common_domain_fn());
        using _Domain2 = decltype(__domain2);

        if constexpr (same_as<_Domain2, __none_such>) {
          return __mexception<_CHILD_SENDERS_WITH_DIFFERENT_DOMAINS_, _WITH_SENDER_<_Sender2>>();
        } else {
          return __detail::__transform_sender()(__domain2, std::move(__sndr2), __env);
        }
      });
  }

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
  template <class _Sender, class _Scheduler, class _Tag = set_value_t>
  concept __completes_on =
    __decays_to<__call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;
} // namespace stdexec

STDEXEC_PRAGMA_POP()
