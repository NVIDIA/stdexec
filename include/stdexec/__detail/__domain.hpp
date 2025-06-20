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

#include "__config.hpp"
#include "__concepts.hpp"
#include "__env.hpp"
#include "__sender_introspection.hpp"
#include "__meta.hpp"

#include "../functional.hpp"
#include "__utility.hpp"

namespace stdexec {

  struct default_domain;
  struct dependent_domain;

  namespace __detail {
    template <class _DomainOrTag, class _Sender, class... _Env>
    concept __has_transform_sender =
      requires(_DomainOrTag __tag, _Sender&& __sender, const _Env&... __env) {
        __tag.transform_sender(static_cast<_Sender &&>(__sender), __env...);
      };

    template <class _DomainOrTag, class _Sender, class... _Env>
    concept __has_nothrow_transform_sender =
      requires(_DomainOrTag __tag, _Sender&& __sender, const _Env&... __env) {
        { __tag.transform_sender(static_cast<_Sender &&>(__sender), __env...) } noexcept;
      };

    template <class _Sender, class... _Env>
    concept __has_default_transform_sender =
      sender_expr<_Sender> && __has_transform_sender<tag_of_t<_Sender>, _Sender, _Env...>;

    template <class _DomainOrTag, class _Sender, class... _Env>
    using __transform_sender_result_t =
      decltype(_DomainOrTag{}.transform_sender(__declval<_Sender>(), __declval<const _Env&>()...));

    template <class _DomainOrTag, class _Sender, class _Env>
    concept __has_transform_env = requires(_DomainOrTag __tag, _Sender&& __sender, _Env&& __env) {
      __tag.transform_env(static_cast<_Sender &&>(__sender), static_cast<_Env &&>(__env));
    };

    template <class _Sender, class _Env>
    concept __has_default_transform_env = sender_expr<_Sender>
                                       && __has_transform_env<tag_of_t<_Sender>, _Sender, _Env>;

    template <class _DomainOrTag, class _Sender, class _Env>
    using __transform_env_result_t =
      decltype(_DomainOrTag{}.transform_env(__declval<_Sender>(), __declval<_Env>()));

    template <class _DomainOrTag, class... _Args>
    concept __has_apply_sender = requires(_DomainOrTag __tag, _Args&&... __args) {
      __tag.apply_sender(static_cast<_Args &&>(__args)...);
    };

    template <class _Tag, class... _Args>
    using __apply_sender_result_t = decltype(_Tag{}.apply_sender(__declval<_Args>()...));

    ////////////////////////////////////////////////////////////////////////////////////////////////
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
            __mremove<__none_such, __munique<__msingle_or<__none_such>>>
          >,
          set_value_t,
          set_error_t,
          set_stopped_t
        > { };

    template <class _Sender>
    using __completion_domain_or_none = __t<__completion_domain_or_none_<env_of_t<_Sender>>>;

    template <class _Sender>
    concept __consistent_completion_domains = __mvalid<__completion_domain_or_none, _Sender>;

    template <class _Sender>
    concept __has_completion_domain = (!same_as<__completion_domain_or_none<_Sender>, __none_such>);

    template <__has_completion_domain _Sender>
    using __completion_domain_of = __completion_domain_or_none<_Sender>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //! Function object implementing `get-domain-early(snd)`
    //! from [exec.snd.general] item 3.9. It is the first well-formed expression of
    //! a) `get_domain(get_env(sndr))`
    //! b) `completion-domain(sndr)`
    //! c) `default_domain()`
    struct __get_early_domain_t {
      template <class _Sender, class _Default = default_domain>
      auto operator()(const _Sender&, _Default = {}) const noexcept {
        if constexpr (__callable<get_domain_t, env_of_t<_Sender>>) {
          return __domain_of_t<env_of_t<_Sender>>();
        } else if constexpr (__has_completion_domain<_Sender>) {
          return __completion_domain_of<_Sender>();
        } else {
          return _Default();
        }
      }
    };

    template <class _Sender, class _Default = default_domain>
    using __early_domain_of_t = __call_result_t<__get_early_domain_t, _Sender, _Default>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //! Function object implementing `get-domain-late(snd)`
    struct __get_late_domain_t {
      // When connect is looking for a customization, it first checks if the sender has a
      // late domain override. If so, that is the domain that is used to transform the
      // sender. Otherwise, look to the receiver for information about where the resulting
      // operation state will be started.
      template <class _Sender, class _Env, class _Default = default_domain>
      auto operator()(const _Sender& __sndr, const _Env& __env, _Default = {}) const noexcept {
        // The schedule_from algorithm is the exception to the rule. It ignores the domain
        // of the predecessor, and dispatches based on the domain of the scheduler to
        // which execution is being transferred.
        if constexpr (__callable<get_domain_override_t, env_of_t<_Sender>>) {
          return get_domain_override(get_env(__sndr));
        } else if constexpr (__callable<get_domain_t, const _Env&>) {
          return get_domain(__env);
        } else if constexpr (__callable<__composed<get_domain_t, get_scheduler_t>, const _Env&>) {
          return get_domain(get_scheduler(__env));
        } else {
          return _Default();
        }
      }
    };

    template <class _Sender, class _Env, class _Default = default_domain>
    using __late_domain_of_t = __call_result_t<__get_late_domain_t, _Sender, _Env, _Default>;

    struct __common_domain_fn {
      template <
        class _Default = default_domain,
        class _Dependent = dependent_domain,
        class... _Domains
      >
      static auto __common_domain(_Domains...) noexcept {
        if constexpr (sizeof...(_Domains) == 0) {
          return _Default();
        } else if constexpr (__one_of<_Dependent, _Domains...>) {
          return _Dependent();
        } else if constexpr (stdexec::__mvalid<std::common_type_t, _Domains...>) {
          return std::common_type_t<_Domains...>();
        } else {
          return __none_such();
        }
      }

      auto operator()(__ignore, __ignore, const auto&... __sndrs) const noexcept {
        return __common_domain(__get_early_domain_t{}(__sndrs)...);
      }
    };
  } // namespace __detail

  struct default_domain {
    template <class _Sender, class... _Env>
      requires __detail::__has_default_transform_sender<_Sender, _Env...>
    STDEXEC_ATTRIBUTE(always_inline)
    auto transform_sender(_Sender&& __sndr, _Env&&... __env) const
      noexcept(__detail::__has_nothrow_transform_sender<tag_of_t<_Sender>, _Sender, _Env...>)
        -> __detail::__transform_sender_result_t<tag_of_t<_Sender>, _Sender, _Env...> {
      return tag_of_t<_Sender>().transform_sender(static_cast<_Sender&&>(__sndr), __env...);
    }

    template <class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    auto transform_sender(_Sender&& __sndr, _Env&&...) const
      noexcept(__nothrow_constructible_from<_Sender, _Sender>) -> _Sender {
      return static_cast<_Sender>(static_cast<_Sender&&>(__sndr));
    }

    template <class _Sender, class _Env>
      requires __detail::__has_default_transform_env<_Sender, _Env>
    auto transform_env(_Sender&& __sndr, _Env&& __env) const noexcept
      -> __detail::__transform_env_result_t<tag_of_t<_Sender>, _Sender, _Env> {
      return tag_of_t<_Sender>()
        .transform_env(static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env));
    }

    template <class _Env>
    auto transform_env(__ignore, _Env&& __env) const noexcept -> _Env {
      return static_cast<_Env>(static_cast<_Env&&>(__env));
    }

    template <class _Tag, class... _Args>
      requires __detail::__has_apply_sender<_Tag, _Args...>
    STDEXEC_ATTRIBUTE(always_inline)
    auto apply_sender(_Tag, _Args&&... __args) const
      -> __detail::__apply_sender_result_t<_Tag, _Args...> {
      return _Tag().apply_sender(static_cast<_Args&&>(__args)...);
    }
  };

  inline constexpr __detail::__get_early_domain_t __get_early_domain{};
  inline constexpr __detail::__get_late_domain_t __get_late_domain{};
  using __detail::__early_domain_of_t;
  using __detail::__late_domain_of_t;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // dependent_domain
  struct dependent_domain {
    // defined in __transform_sender.hpp
    template <class _Sender, class _Env>
    static constexpr auto __is_nothrow_transform_sender() noexcept -> bool;

    // defined in __transform_sender.hpp
    template <sender_expr _Sender, class _Env>
      requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
    STDEXEC_ATTRIBUTE(always_inline)
    auto transform_sender(_Sender&& __sndr, const _Env& __env) const
      noexcept(__is_nothrow_transform_sender<_Sender, _Env>()) -> decltype(auto);
  };

  template <class... _Senders>
  using __common_domain_t = __call_result_t<__detail::__common_domain_fn, int, int, _Senders...>;

  template <class... _Senders>
  concept __has_common_domain = __none_of<__none_such, __common_domain_t<_Senders...>>;
} // namespace stdexec
