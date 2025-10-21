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

  namespace __detail {
    template <class _Env, class _Tag>
    using __starting_domain =
      __meval_or<__call_result_t, default_domain, get_domain_t, const _Env&>;

    template <class _Sch>
    auto __get_scheduler_domain() -> __call_result_t<get_completion_domain_t<set_value_t>, _Sch>;

    template <class _Sch, class _Env>
    auto __get_scheduler_domain()
      -> __meval_or<__call_result_t, default_domain, get_completion_domain_t<set_value_t>, _Sch, _Env>;

    template <class _Sch, class... _Env>
    using __scheduler_domain_t = __decay_t<decltype(__detail::__get_scheduler_domain<_Sch, _Env...>())>;

    constexpr auto __find_pos(bool const* const __begin, bool const* const __end) noexcept -> size_t {
      for (bool const* __where = __begin; __where != __end; ++__where) {
        if (*__where) {
          return static_cast<size_t>(__where - __begin);
        }
      }
      return __npos;
    }

    template <class... _Fns>
    struct __first_callable {
    private:
      //! @brief Returns the first function that is callable with a given set of arguments.
      template <class... _Args, class _Self>
      static constexpr auto __get_1st(_Self&& __self) noexcept -> decltype(auto)
      {
        // NOLINTNEXTLINE (modernize-avoid-c-arrays)
        constexpr bool __flags[] = {__callable<__copy_cvref_t<_Self, _Fns>, _Args...>..., false};
        constexpr size_t __idx   = __find_pos(__flags, __flags + sizeof...(_Fns));
        if constexpr (__idx != __npos) {
          return std::get<__idx>(static_cast<_Self&&>(__self).__fns_);
        }
      }

      //! @brief Alias for the type of the first function that is callable with a given set of arguments.
      template <class _Self, class... _Args>
      using __1st_fn_t = decltype(__first_callable::__get_1st<_Args...>(__declval<_Self>()));

    public:
      //! @brief Calls the first function that is callable with a given set of arguments.
      template <class... _Args>
      constexpr auto operator()(_Args&&... __args) && noexcept(__nothrow_callable<__1st_fn_t<__first_callable, _Args...>, _Args...>)
        -> __call_result_t<__1st_fn_t<__first_callable, _Args...>, _Args...>
      {
        return __first_callable::__get_1st<_Args...>(static_cast<__first_callable&&>(*this))(
          static_cast<_Args&&>(__args)...);
      }

      //! @overload
      template <class... _Args>
      constexpr auto operator()(_Args&&... __args) const& noexcept(
        __nothrow_callable<__1st_fn_t<__first_callable const&, _Args...>, _Args...>)
        -> __call_result_t<__1st_fn_t<__first_callable const&, _Args...>, _Args...>
      {
        return __first_callable::__get_1st<_Args...>(*this)(static_cast<_Args&&>(__args)...);
      }

      std::tuple<_Fns...> __fns_;
    };

    template <class _Sender, class _Env>
    using __completing_domain = __call_result_t<__first_callable<get_domain_override_t, get_completion_domain_t<set_value_t>>, env_of_t<_Sender>, const _Env&>;
  } // namespace __detail

  namespace __queries {
    //////////////////////////////////////////////////////////////////////////////////////////
    //! @brief A query type for asking a sender's attributes for the domain on which that
    //! sender will complete. As with @c get_domain, it is used in tag dispatching to find a
    //! custom implementation of a sender algorithm.
    //!
    //! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
    template <__completion_tag _Tag>
    struct get_completion_domain_t {
      // This function object reads the completion domain from an attribute object or a
      // scheduler, accounting for the fact that the query member function may or may not
      // accept an environment.
      struct __read_query_t {
        template <class _Attrs>
          requires __queryable_with<_Attrs, get_completion_domain_t>
        constexpr auto operator()(const _Attrs& __attrs, __ignore = {}) const noexcept
          -> __decay_t<__query_result_t<_Attrs, get_completion_domain_t>>;

        template <class _Attrs, class _Env>
          requires __queryable_with<_Attrs, get_completion_domain_t, const _Env&>
        constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
          -> __decay_t<__query_result_t<_Attrs, get_completion_domain_t, const _Env&>>;
      };

    private:
      template <class _Attrs, class... _Env, class _Domain>
      static consteval auto __check_domain(_Domain) noexcept -> _Domain {
        // Sanity check: if a completion scheduler can be determined, then its domain must match
        // the domain returned by the attributes.
        if constexpr (__callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>) {
          using __sch_t = __call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>;
          if constexpr (!std::is_same_v<__sch_t, _Attrs>) // prevent infinite recursion
          {
            static_assert(std::is_same_v<_Domain, __detail::__scheduler_domain_t<__sch_t, const _Env&...>>,
                          "the sender claims to complete on a domain that is not the domain of its completion scheduler");
          }
        }
        return {};
      }

      template <class _Attrs, class... _Env>
      static constexpr auto __get_domain() noexcept
      {
        // If __attrs has a completion domain, then return it:
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
        {
          using __domain_t = __call_result_t<__read_query_t, const _Attrs&, const _Env&...>;
          return __check_domain<_Attrs, _Env...>(__domain_t{});
        }
        // Otherwise, if __attrs has a completion scheduler, we can ask that scheduler for its
        // completion domain.
        else if constexpr (__callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>)
        {
          using __sch_t        = __call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>;
          using __read_query_t = typename get_completion_domain_t<set_value_t>::__read_query_t;

          if constexpr (__callable<__read_query_t, __sch_t, const _Env&...>)
          {
            using __domain_t = __call_result_t<__read_query_t, __sch_t, const _Env&...>;
            return __domain_t{};
          }
          // Otherwise, if the scheduler's sender indicates that it completes inline, we can ask
          // the environment for its domain.
          else if constexpr (__completes_inline<env_of_t<__call_result_t<schedule_t, __sch_t>>, _Env...>
                            && __callable<get_domain_t, const _Env&...>)
          {
            return __call_result_t<get_domain_t, const _Env&...>{};
          }
          // Otherwise, if we are asking "late" (with an environment), return the default_domain
          else if constexpr (sizeof...(_Env) != 0)
          {
            return default_domain{};
          }
        }
        // Otherwise, if the attributes indicates that the sender completes inline, we can ask
        // the environment for its domain.
        else if constexpr (__completes_inline<_Attrs, _Env...> && __callable<get_domain_t, const _Env&...>)
        {
          return __call_result_t<get_domain_t, const _Env&...>{};
        }
        // Otherwise, if we are asking "late" (with an environment), return the default_domain
        else if constexpr (sizeof...(_Env) != 0)
        {
          return default_domain{};
        }
        // Otherwise, no completion domain can be determined. Return void.
      }

      template <class _Attrs, class... _Env>
      using __result_t = __unless_one_of_t<decltype(__get_domain<_Attrs, _Env...>()), void>;

    public:
      template <class _Attrs, class... _Env>
      constexpr auto operator()(const _Attrs&, const _Env&...) const noexcept -> __result_t<_Attrs, _Env...> {
        return {};
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };
  }

  using __queries::get_completion_domain_t;

#if !STDEXEC_GCC() || defined(__OPTIMIZE_SIZE__)
  template <__completion_tag _Query>
  inline constexpr get_completion_domain_t<_Query> get_completion_domain{};
#else
  template <>
  inline constexpr get_completion_domain_t<set_value_t> get_completion_domain<set_value_t>{};
  template <>
  inline constexpr get_completion_domain_t<set_error_t> get_completion_domain<set_error_t>{};
  template <>
  inline constexpr get_completion_domain_t<set_stopped_t> get_completion_domain<set_stopped_t>{};
#endif
} // namespace stdexec
