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

#include "__execution_fwd.hpp" // IWYU pragma: keep

#include "__config.hpp"
#include "__concepts.hpp"
#include "__sender_introspection.hpp"
#include "__env.hpp"
#include "__meta.hpp"

#include "../functional.hpp"
#include "__utility.hpp"

namespace stdexec {

  struct default_domain;
  struct dependent_domain;

  namespace __domain {
    template <class _Tag>
    using __legacy_c11n_for = typename _Tag::__legacy_customizations_t;

    template <class _Tag, class... _Args>
    using __legacy_c11n_fn = //
      __make_dispatcher<__legacy_c11n_for<_Tag>, __none_such, _Args...>;

    template <class _Tag, class... _Args>
    concept __has_legacy_c11n = //
      __callable<__legacy_c11n_fn<_Tag, _Args...>, _Args...>;

    struct __legacy_customization {
      template <class _Tag, class _Data, class... _Children>
        requires __has_legacy_c11n<_Tag, _Data, _Children...>
      auto operator()(_Tag, _Data&& __data, _Children&&... __children) const -> decltype(auto) {
        return __legacy_c11n_fn<_Tag, _Data, _Children...>()(
          static_cast<_Data&&>(__data), static_cast<_Children&&>(__children)...);
      }
    };

    template <class _DomainOrTag, class _Sender, class... _Env>
    concept __has_transform_sender =
      requires(_DomainOrTag __tag, _Sender&& __sender, const _Env&... __env) {
        __tag.transform_sender(static_cast<_Sender&&>(__sender), __env...);
      };

    template <class _Sender, class... _Env>
    concept __has_default_transform_sender = //
      sender_expr<_Sender>                   //
      && __has_transform_sender<tag_of_t<_Sender>, _Sender, _Env...>;

    template <class _Type, class _Sender, class _Env>
    concept __has_transform_env = requires(_Type __obj, _Sender&& __sender, _Env&& __env) {
      __obj.transform_env(static_cast<_Sender&&>(__sender), static_cast<_Env&&>(__env));
    };

    template <class _Sender, class _Env>
    concept __has_default_transform_env = //
      sender_expr<_Sender>                //
      && __has_transform_env<tag_of_t<_Sender>, _Sender, _Env>;

    template <class _DomainOrTag, class... _Args>
    concept __has_apply_sender = requires(_DomainOrTag __tag, _Args&&... __args) {
      __tag.apply_sender(static_cast<_Args&&>(__args)...);
    };

    template <class _Sender>
    constexpr bool __is_nothrow_transform_sender() noexcept {
      if constexpr (__callable<__sexpr_apply_t, _Sender, __domain::__legacy_customization>) {
        return __nothrow_callable<__sexpr_apply_t, _Sender, __domain::__legacy_customization>;
      } else if constexpr (__domain::__has_default_transform_sender<_Sender>) {
        return noexcept(tag_of_t<_Sender>().transform_sender(__declval<_Sender>()));
      } else {
        return __nothrow_constructible_from<_Sender, _Sender>;
      }
    }

    template <class _Sender, class _Env>
    constexpr bool __is_nothrow_transform_sender() noexcept {
      if constexpr (__domain::__has_default_transform_sender<_Sender, _Env>) {
        return //
          noexcept(
            tag_of_t<_Sender>().transform_sender(__declval<_Sender>(), __declval<const _Env&>()));
      } else {
        return __nothrow_constructible_from<_Sender, _Sender>;
      }
    }
  } // namespace __domain

  namespace __detail {
    ///////////////////////////////////////////////////////////////////////////
    template <class _Env, class _Tag>
    using __completion_scheduler_for =
      __meval_or<__call_result_t, __none_such, get_completion_scheduler_t<_Tag>, _Env>;

    template <class _Env, class _Tag>
    using __completion_domain_for =
      __meval_or<__call_result_t, __none_such, get_domain_t, __completion_scheduler_for<_Env, _Tag>>;

    // Check the value, error, and stopped channels for completion schedulers.
    // Of the completion schedulers that are known, they must all have compatible
    // domains. This computes that domain, or else returns __none_such if there
    // are no completion schedulers or if they don't specify a domain.
    template <class _Env>
    struct __completion_domain_or_none_
      : __mdefer_<
          __mtransform<
            __mbind_front_q<__completion_domain_for, _Env>,
            __mremove<__none_such, __munique<__msingle_or<__none_such>>>>,
          set_value_t,
          set_error_t,
          set_stopped_t> { };

    template <class _Sender>
    using __completion_domain_or_none = __t<__completion_domain_or_none_<env_of_t<_Sender>>>;

    template <class _Sender>
    concept __consistent_completion_domains = __mvalid<__completion_domain_or_none, _Sender>;

    template <class _Sender>
    concept __has_completion_domain = (!same_as<__completion_domain_or_none<_Sender>, __none_such>);

    template <__has_completion_domain _Sender>
    using __completion_domain_of = __completion_domain_or_none<_Sender>;
  } // namespace __detail

  struct default_domain {
    default_domain() = default;

    // Called without the environment during eager customization
    template <class _Sender>
    STDEXEC_ATTRIBUTE((always_inline)) decltype(auto) transform_sender(_Sender&& __sndr) const
      noexcept(__domain::__is_nothrow_transform_sender<_Sender>()) {
      // Look for a legacy customization for the given tag, and if found, apply it.
      if constexpr (__callable<__sexpr_apply_t, _Sender, __domain::__legacy_customization>) {
        return stdexec::__sexpr_apply(
          static_cast<_Sender&&>(__sndr), __domain::__legacy_customization());
      } else if constexpr (__domain::__has_default_transform_sender<_Sender>) {
        return tag_of_t<_Sender>().transform_sender(static_cast<_Sender&&>(__sndr));
      } else {
        return static_cast<_Sender>(static_cast<_Sender&&>(__sndr));
      }
    }

    // Called with an environment during lazy customization
    template <class _Sender, class _Env>
    STDEXEC_ATTRIBUTE((always_inline)) decltype(auto) transform_sender(_Sender&& __sndr, const _Env& __env) const
      noexcept(__domain::__is_nothrow_transform_sender<_Sender, _Env>()) {
      if constexpr (__domain::__has_default_transform_sender<_Sender, _Env>) {
        return tag_of_t<_Sender>().transform_sender(static_cast<_Sender&&>(__sndr), __env);
      } else {
        return static_cast<_Sender>(static_cast<_Sender&&>(__sndr));
      }
    }

    template <class _Tag, class _Sender, class... _Args>
      requires __domain::__has_legacy_c11n<_Tag, _Sender, _Args...>
            || __domain::__has_apply_sender<_Tag, _Sender, _Args...>
    STDEXEC_ATTRIBUTE((always_inline)) decltype(auto) apply_sender(_Tag, _Sender&& __sndr, _Args&&... __args) const {
      // Look for a legacy customization for the given tag, and if found, apply it.
      if constexpr (__domain::__has_legacy_c11n<_Tag, _Sender, _Args...>) {
        return __domain::__legacy_c11n_fn<_Tag, _Sender, _Args...>()(
          static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      } else {
        return _Tag().apply_sender(static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      }
    }

    template <class _Sender, class _Env>
    auto transform_env(_Sender&& __sndr, _Env&& __env) const noexcept -> decltype(auto) {
      if constexpr (__domain::__has_default_transform_env<_Sender, _Env>) {
        return tag_of_t<_Sender>().transform_env(
          static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
      } else {
        return static_cast<_Env>(static_cast<_Env&&>(__env));
      }
    }
  };

  /////////////////////////////////////////////////////////////////////////////
  //! Function object implementing `get-domain-early(snd)`
  //! from [exec.snd.general] item 3.9. It is the first well-formed expression of
  //! a) `get_domain(get_env(sndr))`
  //! b) `completion-domain(sndr)`
  //! c) `default_domain()`
  inline constexpr struct __get_early_domain_t {
    template <class _Sender, class _Default = default_domain>
    auto operator()(const _Sender&, _Default __def = {}) const noexcept {
      if constexpr (__callable<get_domain_t, env_of_t<_Sender>>) {
        return __domain_of_t<env_of_t<_Sender>>();
      } else if constexpr (__detail::__has_completion_domain<_Sender>) {
        return __detail::__completion_domain_of<_Sender>();
      } else {
        return __def;
      }
    }
  } __get_early_domain{};

  template <class _Sender, class _Default = default_domain>
  using __early_domain_of_t = __call_result_t<__get_early_domain_t, _Sender, _Default>;

  /////////////////////////////////////////////////////////////////////////////
  inline constexpr struct __get_late_domain_t {
    // When connect is looking for a customization, it first checks the sender's
    // domain. If the sender knows the domain in which it completes, then that is
    // where the subsequent task will execute. Otherwise, look to the receiver for
    // late-bound information about the current execution context.
    template <class _Sender, class _Env>
    auto operator()(const _Sender& __sndr, const _Env& __env) const noexcept {
      if constexpr (!same_as<dependent_domain, __early_domain_of_t<_Sender, dependent_domain>>) {
        return __get_early_domain(__sndr);
      } else if constexpr (__callable<get_domain_t, const _Env&>) {
        return get_domain(__env);
      } else if constexpr (__callable<__composed<get_domain_t, get_scheduler_t>, const _Env&>) {
        return get_domain(get_scheduler(__env));
      } else {
        return default_domain();
      }
    }

    // The continues_on algorithm is the exception to the rule. It ignores the
    // domain of the predecessor, and dispatches based on the domain of the
    // scheduler to which execution is being transferred.
    template <sender_expr_for<continues_on_t> _Sender, class _Env>
    auto operator()(const _Sender& __sndr, const _Env&) const noexcept {
      return __sexpr_apply(__sndr, [](__ignore, auto& __data, __ignore) noexcept {
        auto __sched = get_completion_scheduler<set_value_t>(__data);
        return query_or(get_domain, __sched, default_domain());
      });
    }
  } __get_late_domain{};

  template <class _Sender, class _Env>
  using __late_domain_of_t = __call_result_t<__get_late_domain_t, _Sender, _Env>;

  /////////////////////////////////////////////////////////////////////////////
  // dependent_domain
  struct dependent_domain {
    // defined in __transform_sender.hpp
    template <class _Sender, class _Env>
    static constexpr auto __is_nothrow_transform_sender() noexcept -> bool;

    // defined in __transform_sender.hpp
    template <sender_expr _Sender, class _Env>
      requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
    STDEXEC_ATTRIBUTE((always_inline)) decltype(auto) transform_sender(_Sender&& __sndr, const _Env& __env) const
      noexcept(__is_nothrow_transform_sender<_Sender, _Env>());
  };

  namespace __domain {
    struct __common_domain_fn {
      template <class... _Domains>
      static auto __common_domain(_Domains...) noexcept {
        if constexpr (sizeof...(_Domains) == 0) {
          return default_domain();
        } else if constexpr (__one_of<dependent_domain, _Domains...>) {
          return dependent_domain();
        } else if constexpr (stdexec::__mvalid<std::common_type_t, _Domains...>) {
          return std::common_type_t<_Domains...>();
        } else {
          return __none_such();
        }
      }

      auto operator()(__ignore, __ignore, const auto&... __sndrs) const noexcept {
        return __common_domain(__get_early_domain(__sndrs)...);
      }
    };

    template <class... _Senders>
    using __common_domain_t = //
      __call_result_t<__common_domain_fn, int, int, _Senders...>;

    template <class... _Senders>
    concept __has_common_domain = //
      __none_of<__none_such, __common_domain_t<_Senders...>>;
  } // namespace __domain
} // namespace stdexec
