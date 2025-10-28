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

    template <class _Tag, class _Sndr, class... _Env>
    using __completion_domain_of_t = __call_result_t<get_completion_domain_t<_Tag>, env_of_t<_Sndr>, const _Env&...>;

    template <class _Tag, class _Sndr, class... _Env>
    using __completion_domain_or_none_t = __meval_or<
      __call_result_t,
      __none_such,
      get_completion_domain_t<_Tag>,
      env_of_t<_Sndr>,
      const _Env&...>;
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

  namespace __detail {
    struct __common_domain_fn {
      struct __none_t{};

      template <class... _Domains>
      static auto __common_domain(_Domains...) noexcept {
        if constexpr (sizeof...(_Domains) == 0) {
          return default_domain{};
        } else if constexpr (stdexec::__mvalid<std::common_type_t, _Domains...>) {
          return std::common_type_t<_Domains...>();
        } else {
          return __none_such();
        }
      }

    private:
      // Helper: returns a single-element tuple if domain is valid, empty tuple otherwise
      template <class _Domain>
      static constexpr auto __maybe_tuple(_Domain __d) noexcept {
        if constexpr (same_as<_Domain, __none_t>) {
          return std::tuple<>{};
        } else {
          return std::tuple<_Domain>{__d};
        }
      }

      // Helper: filters out __none_such values and calls __common_domain with the rest
      template <class... _Domains>
      static constexpr auto __filter_and_common(_Domains... __doms) noexcept {
        auto __valid = std::tuple_cat(__maybe_tuple(__doms)...);
        return std::apply([]<class... _Valid>(_Valid... __v) {
          return __common_domain(__v...);
        }, __valid);
      }

      // Helper: tries to get completion domain, returns __none_such if not available
      template <class _Sndr>
      static constexpr auto __try_get_domain(const _Sndr& __sndr) noexcept {
        if constexpr (__callable<get_completion_domain_t<set_value_t>, env_of_t<_Sndr>>) {
          return get_completion_domain<set_value_t>(get_env(__sndr));
        } else {
          return __none_t{};
        }
      }

    public:
      auto operator()(__ignore, __ignore, const auto&... __sndrs) const noexcept {
        // Query each sender for its completion domain, filter out those that can't answer,
        // then compute the common domain of the remaining ones
        return __filter_and_common(__try_get_domain(__sndrs)...);
      }
    };
  } // namespace __detail

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
      template <class Sig>
      static inline constexpr get_completion_domain_t<_Tag> (*signature)(Sig) = nullptr;

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
        // TODO(gevtushenko): re-enable domain checking once we have a way to test it
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
