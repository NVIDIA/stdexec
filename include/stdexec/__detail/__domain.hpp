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

#include "__completion_behavior.hpp"
#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__execution_fwd.hpp"
#include "__meta.hpp"
#include "__sender_introspection.hpp"
#include "__utility.hpp"

#include <type_traits>

namespace STDEXEC {
  namespace __detail {
    template <class _DomainOrTag, class _OpTag, class _Sender, class... _Env>
    concept __has_transform_sender =
      requires(_DomainOrTag __tag, _Sender&& __sender, const _Env&... __env) {
        __tag.transform_sender(_OpTag(), static_cast<_Sender&&>(__sender), __env...);
      };

    template <class _DomainOrTag, class _OpTag, class _Sender, class... _Env>
    concept __has_nothrow_transform_sender =
      requires(_DomainOrTag __tag, _Sender&& __sender, const _Env&... __env) {
        { __tag.transform_sender(_OpTag(), static_cast<_Sender&&>(__sender), __env...) } noexcept;
      };

    template <class _DomainOrTag, class _OpTag, class _Sender, class... _Env>
    using __transform_sender_result_t =
      decltype(_DomainOrTag{}
                 .transform_sender(_OpTag(), __declval<_Sender>(), __declval<const _Env&>()...));

    template <class _DomainOrTag, class... _Args>
    concept __has_apply_sender = requires(_DomainOrTag __tag, _Args&&... __args) {
      __tag.apply_sender(static_cast<_Args&&>(__args)...);
    };

    template <class _Tag, class... _Args>
    using __apply_sender_result_t = decltype(_Tag{}.apply_sender(__declval<_Args>()...));
  } // namespace __detail

  ////////////////////////////////////////////////////////////////////////////////////////////////
  struct default_domain {
    template <class _OpTag, class _Sender, class _Env>
      requires __detail::__has_transform_sender<tag_of_t<_Sender>, _OpTag, _Sender, _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto transform_sender(_OpTag, _Sender&& __sndr, const _Env& __env) const
      noexcept(__detail::__has_nothrow_transform_sender<tag_of_t<_Sender>, _OpTag, _Sender, _Env>)
        -> __detail::__transform_sender_result_t<tag_of_t<_Sender>, _OpTag, _Sender, _Env> {
      return tag_of_t<_Sender>().transform_sender(_OpTag(), static_cast<_Sender&&>(__sndr), __env);
    }

    template <class _OpTag, class _Sender, class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto transform_sender(_OpTag, _Sender&& __sndr, const _Env&) const
      noexcept(__nothrow_move_constructible<_Sender>) -> _Sender {
      return static_cast<_Sender>(static_cast<_Sender&&>(__sndr));
    }

    template <class _Tag, class... _Args>
      requires __detail::__has_apply_sender<_Tag, _Args...>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto apply_sender(_Tag, _Args&&... __args) const
      -> __detail::__apply_sender_result_t<_Tag, _Args...> {
      return _Tag().apply_sender(static_cast<_Args&&>(__args)...);
    }
  };

  //! @brief Concept that checks whether a domain's sender transform behaves like that of
  //! @c default_domain when passed the same arguments. The concept is modeled when either
  //! of the following is
  template <class _Domain, class _OpTag, class _Sndr, class _Env>
  concept __default_domain_like = __same_as<
    __decay_t<__detail::__transform_sender_result_t<default_domain, _OpTag, _Sndr, _Env>>,
    __decay_t<__mcall<
      __mtry_catch_q<
        __detail::__transform_sender_result_t,
        __mconst<__detail::__transform_sender_result_t<default_domain, _OpTag, _Sndr, _Env>>
      >,
      _Domain,
      _OpTag,
      _Sndr,
      _Env
    >>
  >;

  template <class... _Domains>
  struct indeterminate_domain {
    constexpr indeterminate_domain() = default;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr indeterminate_domain(__ignore) noexcept {
    }

    //! @brief Transforms a sender with an optional environment.
    //!
    //! @tparam _OpTag Either start_t or set_value_t.
    //! @tparam _Sndr The type of the sender.
    //! @tparam _Env The type of the environment.
    //! @param __sndr The sender to be transformed.
    //! @param __env The environment used for the transformation.
    //! @return `default_domain{}.transform_sender(_OpTag{}, std::forward<_Sndr>(__sndr), __env)`
    //! @pre Every type in @c _Domains... must behave like @c default_domain when passed the
    //! same arguments. If this check fails, the @c static_assert triggers with: "ERROR:
    //! indeterminate domains: cannot pick an algorithm customization"
    template <class _OpTag, class _Sndr, class _Env>
      requires __detail::__has_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>
    [[nodiscard]]
    static constexpr auto transform_sender(_OpTag, _Sndr&& __sndr, const _Env& __env)
      noexcept(__detail::__has_nothrow_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>)
        -> __detail::__transform_sender_result_t<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env> {
      static_assert(
        (__default_domain_like<_Domains, _OpTag, _Sndr, _Env> && ...),
        "ERROR: indeterminate domains: cannot pick an algorithm customization");
      return tag_of_t<_Sndr>{}.transform_sender(_OpTag{}, static_cast<_Sndr&&>(__sndr), __env);
    }
  };

  namespace __detail {
    template <class... _Domains>
    using __indeterminate_domain_t = __if_c<
      sizeof...(_Domains) == 1,
      decltype((__(), ..., _Domains())),
      indeterminate_domain<_Domains...>
    >;

    template <class _DomainSet>
    using __domain_from_set_t = __mapply<
      __if_c<
        __mset_contains<_DomainSet, indeterminate_domain<>>,
        __mconst<indeterminate_domain<>>,
        __qq<__indeterminate_domain_t>
      >,
      _DomainSet
    >;

    template <class... _Domains>
    using __make_domain_t = __domain_from_set_t<__mmake_set<_Domains...>>;

    // Common domain for a set of domains
    template <class... _Domains>
    struct __common_domain {
      using __t = __minvoke<__mtry_catch_q<std::common_type_t, __qq<__make_domain_t>>, _Domains...>;
    };
  } // namespace __detail

  ////////////////////////////////////////////////////////////////////////////////////////////////
  template <class _Tag, sender _Sender, class... _Env>
    requires __sends<_Tag, _Sender, _Env...>
  using __completion_domain_of_t = __call_result_or_t<
    get_completion_domain_t<_Tag>,
    indeterminate_domain<>,
    env_of_t<_Sender>,
    const _Env&...
  >;

  template <class... _Domains>
  using __common_domain_t = __t<__detail::__common_domain<_Domains...>>;

  template <class... _Domains>
  concept __has_common_domain =
    __not_same_as<indeterminate_domain<>, __common_domain_t<_Domains...>>;

  template <class _Domain>
  using __ensure_valid_domain_t = __unless_one_of_t<_Domain, indeterminate_domain<>>;

  namespace __detail {
    template <class _Env>
    using __starting_domain_t = __call_result_t<get_domain_t, const _Env&>;

    template <class _Tag, class _Sender, class... _Env>
    using __completing_domain_t =
      __call_result_t<get_completion_domain_t<_Tag>, env_of_t<_Sender>, const _Env&...>;

    template <class _Sch, class... _Env>
    using __scheduler_domain_t =
      __call_result_t<get_completion_domain_t<set_value_t>, _Sch, _Env...>;

    constexpr auto
      __find_pos(bool const * const __begin, bool const * const __end) noexcept -> size_t {
      for (bool const * __where = __begin; __where != __end; ++__where) {
        if (*__where) {
          return static_cast<size_t>(__where - __begin);
        }
      }
      return __npos;
    }
  } // namespace __detail

  namespace __queries {
    //! @brief A wrapper around an environment that hides a set of queries.
    template <class _Env, class... _Queries>
    struct __hide_query {
      constexpr explicit __hide_query(_Env&& __env, _Queries...) noexcept
        : __env_{static_cast<_Env&&>(__env)} {
      }

      template <__none_of<_Queries...> _Query, class... _As>
        requires __queryable_with<_Env, _Query, _As...>
      constexpr auto operator()(_Query, _As&&... __as) const
        noexcept(__nothrow_queryable_with<_Env, _Query, _As...>)
          -> __query_result_t<_Env, _Query, _As...> {
        return __query<_Query>()(__env_, static_cast<_As&&>(__as)...);
      }

     private:
      _Env __env_;
    };

    template <class _Env, class... _Queries>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __hide_query(_Env&&, _Queries...) -> __hide_query<_Env, _Queries...>;

    //! @brief A wrapper around an environment that hides the get_scheduler and get_domain
    //! queries.
    template <class _Env>
    struct __hide_scheduler : __hide_query<_Env, get_scheduler_t, get_domain_t> {
      constexpr explicit __hide_scheduler(_Env&& __env) noexcept
        : __hide_query<_Env, get_scheduler_t, get_domain_t>{static_cast<_Env&&>(__env), {}, {}} {
      }
    };

    template <class _Env>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __hide_scheduler(_Env&&) -> __hide_scheduler<_Env>;

    //////////////////////////////////////////////////////////////////////////////////////////
    //! @brief A query type for asking a sender's attributes for the domain on which that
    //! sender will complete. As with @c get_domain, it is used in tag dispatching to find a
    //! custom implementation of a sender algorithm.
    //!
    //! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
    template <class _Tag>
    struct get_completion_domain_t {
      template <class Sig>
      static inline constexpr get_completion_domain_t<_Tag> (*signature)(Sig) = nullptr;

      // This function object reads the completion domain from an attribute object or a
      // scheduler, accounting for the fact that the query member function may or may not
      // accept an environment.
      struct __read_query_t {
        template <class _Attrs>
          requires __queryable_with<_Attrs, get_completion_domain_t>
        constexpr auto operator()(const _Attrs&, __ignore = {}) const noexcept {
          return __decay_t<__query_result_t<_Attrs, get_completion_domain_t>>{};
        }

        template <class _Attrs, class _Env>
          requires __queryable_with<_Attrs, get_completion_domain_t, const _Env&>
        constexpr auto operator()(const _Attrs&, const _Env&) const noexcept {
          return __decay_t<__query_result_t<_Attrs, get_completion_domain_t, const _Env&>>{};
        }
      };

     private:
      template <class _Sch, class... _Env, class _Domain>
      static consteval auto __check_domain_(_Domain) noexcept {
        static_assert(
          __same_as<_Domain, __detail::__scheduler_domain_t<_Sch, const _Env&...>>,
          "the sender claims to complete on a domain that is not the domain of its completion "
          "scheduler");
      }

      template <class _Attrs, class... _Env, class _Domain>
      static consteval auto __check_domain(_Domain) noexcept -> _Domain {
        // Sanity check: if a completion scheduler can be determined from the attributes
        // (not the environment), then its domain must match the domain returned by the attributes.
        if constexpr (!__same_as<_Tag, void>) {
          if constexpr (__callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>) {
            using __sch_t =
              __call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>;
            // Skip check if the "scheduler" is the same as the domain or the attributes
            // (this can happen with __prop_like which answers any query with the same type)
            if constexpr (!__same_as<__sch_t, _Attrs>) {
              __check_domain_<__sch_t, _Env...>(_Domain{});
            }
          }
        }
        return {};
      }

      template <class _Attrs, class... _Env>
      static constexpr auto __get_domain() noexcept {
        // If __attrs has a completion domain, then return it:
        if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>) {
          using __domain_t = __call_result_t<__read_query_t, const _Attrs&, const _Env&...>;
          return __check_domain<_Attrs, _Env...>(__domain_t{});
          // Otherwise, if _Tag is void, fall back to querying for the set_value_t completion domain:
        } else if constexpr (__same_as<_Tag, void>) {
          if constexpr (
            __callable<get_completion_domain_t<set_value_t>, const _Attrs&, const _Env&...>) {
            using __domain_t =
              __call_result_t<get_completion_domain_t<set_value_t>, const _Attrs&, const _Env&...>;
            return __check_domain<_Attrs, _Env...>(__domain_t{});
          } else {
            return void();
          }
        }
        // Otherwise, if __attrs has a completion scheduler, we can ask that scheduler for its
        // completion domain.
        else if constexpr (
          __callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>) {
          using __sch_t =
            __call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>;
          using X [[maybe_unused]] = decltype(__declval<__sch_t>().schedule());
          using __read_query_t = typename get_completion_domain_t<set_value_t>::__read_query_t;

          if constexpr (__callable<__read_query_t, __sch_t, const _Env&...>) {
            using __domain_t = __call_result_t<__read_query_t, __sch_t, const _Env&...>;
            return __domain_t{};
          }
          // Otherwise, if the scheduler's sender indicates that it completes inline, we can ask
          // the environment for its domain.
          else if constexpr (
            __completes_inline<_Tag, env_of_t<__call_result_t<schedule_t, __sch_t>>, _Env...>
            && __callable<get_domain_t, const _Env&...>) {
            return __call_result_t<get_domain_t, const _Env&...>{};
          }
          // Otherwise, if we are asking "late" (with an environment), return the default_domain
          else if constexpr (sizeof...(_Env) != 0) {
            return default_domain{};
          }
        }
        // Otherwise, if the attributes indicates that the sender completes inline, we can ask
        // the environment for its domain.
        else if constexpr (
          __completes_inline<_Tag, _Attrs, _Env...> && __callable<get_domain_t, const _Env&...>) {
          return __call_result_t<get_domain_t, const _Env&...>{};
        }
        // Otherwise, if we are asking "late" (with an environment), return the default_domain
        else if constexpr (sizeof...(_Env) != 0) {
          return default_domain{};
        }
        // Otherwise, no completion domain can be determined. Return void.
      }

      template <class _Attrs, class... _Env>
      using __result_t = __unless_one_of_t<decltype(__get_domain<_Attrs, _Env...>()), void>;

     public:
      template <class _Attrs, class... _Env>
      constexpr auto
        operator()(const _Attrs&, const _Env&...) const noexcept -> __result_t<_Attrs, _Env...> {
        return {};
      }

      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct get_domain_t {
      template <class _Sig>
      static inline constexpr get_domain_t (*signature)(_Sig) = nullptr;

      // Query with a .query member function:
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _Env&) const noexcept -> auto {
        if constexpr (__member_queryable_with<const _Env&, get_domain_t>) {
          return __decay_t<__member_query_result_t<_Env, get_domain_t>>{};
        } else if constexpr (__callable<get_scheduler_t, const _Env&>) {
          using __sch_t = __call_result_t<get_scheduler_t, const _Env&>;
          using __env_t = __hide_scheduler<const _Env&>;
          using __cmpl_sch_t =
            __call_result_t<get_completion_scheduler_t<set_value_t>, __sch_t, __env_t>;
          return __detail::__scheduler_domain_t<__cmpl_sch_t, __env_t>{};
        } else {
          return default_domain{};
        }
      }

      // Query with tag_invoke (legacy):
      template <class _Env>
        requires __tag_invocable<get_domain_t, const _Env&>
      [[deprecated("use a query member function instead of tag_invoke for queries")]]
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device) //
        constexpr auto operator()(const _Env&) const noexcept {
        return __decay_t<__tag_invoke_result_t<get_domain_t, const _Env&>>{};
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };
  } // namespace __queries

  using __queries::get_completion_domain_t;
  using __queries::get_domain_t;

#if !STDEXEC_GCC() || defined(__OPTIMIZE_SIZE__)
  template <class _Tag>
  inline constexpr get_completion_domain_t<_Tag> get_completion_domain{};
#else
  template <>
  inline constexpr get_completion_domain_t<> get_completion_domain<>{};
  template <>
  inline constexpr get_completion_domain_t<set_value_t> get_completion_domain<set_value_t>{};
  template <>
  inline constexpr get_completion_domain_t<set_error_t> get_completion_domain<set_error_t>{};
  template <>
  inline constexpr get_completion_domain_t<set_stopped_t> get_completion_domain<set_stopped_t>{};
#endif

  inline constexpr get_domain_t get_domain{};
} // namespace STDEXEC

// Specializations of std::common_type for STDEXEC::indeterminate_domain
namespace std {

  template <class... _Ds, class _Domain>
  struct common_type<::STDEXEC::indeterminate_domain<_Ds...>, _Domain> {
    using type = ::STDEXEC::__detail::__make_domain_t<_Ds..., _Domain>;
  };

  template <class _Domain, class... _Ds>
  struct common_type<_Domain, ::STDEXEC::indeterminate_domain<_Ds...>> {
    using type = ::STDEXEC::__detail::__make_domain_t<_Ds..., _Domain>;
  };

  template <class... _As, class... _Bs>
  struct common_type<
    ::STDEXEC::indeterminate_domain<_As...>,
    ::STDEXEC::indeterminate_domain<_Bs...>
  > {
    using type = ::STDEXEC::__detail::__make_domain_t<_As..., _Bs...>;
  };

} // namespace std
