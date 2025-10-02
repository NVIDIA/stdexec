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
#include "__completion_behavior.hpp"
#include "__meta.hpp"
#include "__query.hpp"
#include "__stop_token.hpp"
#include "__tag_invoke.hpp"
// #include "__tuple.hpp"

#include <exception>  // IWYU pragma: keep for std::terminate
#include <functional> // IWYU pragma: keep for unwrap_reference_t
#include <type_traits>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.queries]
  namespace __queries {
    struct execute_may_block_caller_t : __query<execute_may_block_caller_t, true> {
      template <class _Attrs>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(same_as<bool, __call_result_t<execute_may_block_caller_t, const _Attrs&>>);
        static_assert(__nothrow_callable<execute_may_block_caller_t, const _Attrs&>);
      }
    };

    struct get_forward_progress_guarantee_t
      : __query<
          get_forward_progress_guarantee_t,
          forward_progress_guarantee::weakly_parallel,
          __q1<__decay_t>
        > {
      template <class _Attrs>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        using __result_t = __call_result_t<get_forward_progress_guarantee_t, const _Attrs&>;
        static_assert(same_as<forward_progress_guarantee, __result_t>);
        static_assert(__nothrow_callable<get_forward_progress_guarantee_t, const _Attrs&>);
      }
    };

    // TODO: implement allocator concept
    template <class _T0>
    concept __allocator_c = true;

    struct get_scheduler_t : __query<get_scheduler_t> {
      using __query<get_scheduler_t>::operator();

      template <class _Query = get_scheduler_t>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto
        operator()() const noexcept; // defined in __read_env.hpp // defined in __read_env.hpp

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept; // defined in __schedulers.hpp

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    //! The type for `get_delegation_scheduler` [exec.get.delegation.scheduler]
    //! A query object that asks for a scheduler that can be used to delegate
    //! work to for the purpose of forward progress delegation ([intro.progress]).
    struct get_delegation_scheduler_t : __query<get_delegation_scheduler_t> {
      using __query<get_delegation_scheduler_t>::operator();

      template <class _Query = get_delegation_scheduler_t>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept; // defined in __read_env.hpp

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept; // defined in __schedulers.hpp

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct get_allocator_t : __query<get_allocator_t> {
      using __query<get_allocator_t>::operator();

      template <class _Query = get_allocator_t>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept; // defined in __read_env.hpp

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(__nothrow_callable<get_allocator_t, const _Env&>);
        static_assert(__allocator_c<__call_result_t<get_allocator_t, const _Env&>>);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    using __get_stop_token_t = __query<get_stop_token_t, never_stop_token{}, __q1<__decay_t>>;

    struct get_stop_token_t : __get_stop_token_t {
      using __get_stop_token_t::operator();

      template <class _Query = get_stop_token_t>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept; // defined in __read_env.hpp

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(__nothrow_callable<get_stop_token_t, const _Env&>);
        static_assert(stoppable_token<__call_result_t<get_stop_token_t, const _Env&>>);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    template <__completion_tag _Query>
    struct get_completion_scheduler_t : __query<get_completion_scheduler_t<_Query>> {
      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept; // defined in __schedulers.hpp

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct get_domain_t : __query<get_domain_t, __no_default, __q1<__decay_t>> {
      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(
          __nothrow_callable<get_domain_t, const _Env&>,
          "Customizations of get_domain must be noexcept.");
        static_assert(
          __class<__call_result_t<get_domain_t, const _Env&>>,
          "Customizations of get_domain must return a class type.");
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct get_domain_override_t : __query<get_domain_override_t, __no_default, __q1<__decay_t>> {
      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(
          __nothrow_callable<get_domain_override_t, const _Env&>,
          "Customizations of get_domain_override must be noexcept.");
        static_assert(
          __class<__call_result_t<get_domain_override_t, const _Env&>>,
          "Customizations of get_domain_override must return a class type.");
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };

    struct __is_scheduler_affine_t {
      template <class _Result>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto __ensure_bool_constant() noexcept {
        if constexpr (__is_bool_constant<_Result>) {
          return static_cast<bool>(_Result::value);
        } else {
          static_assert(
            __is_bool_constant<_Result>,
            "The __is_scheduler_affine query must be one of the following forms:\n"
            "  static constexpr bool query(__is_scheduler_affine_t) noexcept;\n"
            "  bool_constant<Bool> query(__is_scheduler_affine_t) const noexcept;\n"
            "  bool_constant<Bool> query(__is_scheduler_affine_t, const Env&) const noexcept;\n");
        }
      }

      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        return __completes_inline<_Attrs, const _Env&...>;
      }

      template <__queryable_with<__is_scheduler_affine_t> _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        if constexpr (__statically_queryable_with<_Attrs, __is_scheduler_affine_t>) {
          return _Attrs::query(__is_scheduler_affine_t());
        } else {
          return __ensure_bool_constant<__query_result_t<_Attrs, __is_scheduler_affine_t>>();
        }
      }

      template <class _Attrs, class _Env>
        requires __queryable_with<_Attrs, __is_scheduler_affine_t, const _Env&>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        using __result_t = __query_result_t<_Attrs, __is_scheduler_affine_t, const _Env&>;
        return __ensure_bool_constant<__result_t>();
      }

      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()(const _Attrs&, const _Env&...) const noexcept -> bool {
        return operator()<_Attrs, _Env...>();
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };

    struct __root_t : __query<__root_t> {
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };

    struct __root_env {
      using __t = __root_env;
      using __id = __root_env;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(__root_t) noexcept -> bool {
        return true;
      }
    };
  } // namespace __queries

  using __queries::execute_may_block_caller_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_allocator_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegation_scheduler_t;
  using get_delegatee_scheduler_t
    [[deprecated("get_delegatee_scheduler_t has been renamed get_delegation_scheduler_t")]] =
      get_delegation_scheduler_t;
  using __queries::get_stop_token_t;
  using __queries::get_completion_scheduler_t;
  using __queries::get_domain_t;
  using __queries::get_domain_override_t;
  using __queries::__is_scheduler_affine_t;
  using __queries::__root_t;
  using __queries::__root_env;

  inline constexpr execute_may_block_caller_t execute_may_block_caller{};
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
  inline constexpr get_scheduler_t get_scheduler{};
  inline constexpr get_delegation_scheduler_t get_delegation_scheduler{};
  inline constexpr auto& get_delegatee_scheduler
    [[deprecated("get_delegatee_scheduler has been renamed get_delegation_scheduler")]]
    = get_delegation_scheduler;
  inline constexpr get_allocator_t get_allocator{};
  inline constexpr get_stop_token_t get_stop_token{};
#if !STDEXEC_GCC() || defined(__OPTIMIZE_SIZE__)
  template <__completion_tag _Query>
  inline constexpr get_completion_scheduler_t<_Query> get_completion_scheduler{};
#else
  template <>
  inline constexpr get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_stopped_t>
    get_completion_scheduler<set_stopped_t>{};
#endif

  inline constexpr get_domain_t get_domain{};
  inline constexpr get_domain_override_t get_domain_override{};

  template <class _Query, class _Queryable, class _Default>
  using __query_result_or_t = __call_result_t<query_or_t, _Query, _Queryable, _Default>;

  namespace __env {
    template <class _Tp, class _Promise>
    concept __has_as_awaitable_member = requires(_Tp&& __t, _Promise& __promise) {
      static_cast<_Tp &&>(__t).as_awaitable(__promise);
    };

    template <class _Promise>
    struct __with_await_transform {
      template <class _Ty>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      auto await_transform(_Ty&& __value) noexcept -> _Ty&& {
        return static_cast<_Ty&&>(__value);
      }

      template <class _Ty>
        requires __has_as_awaitable_member<_Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(noexcept(__declval<_Ty>().as_awaitable(__declval<_Promise&>())))
          -> decltype(__declval<_Ty>().as_awaitable(__declval<_Promise&>())) {
        return static_cast<_Ty&&>(__value).as_awaitable(static_cast<_Promise&>(*this));
      }

      template <class _Ty>
        requires(!__has_as_awaitable_member<_Ty, _Promise&>)
             && tag_invocable<as_awaitable_t, _Ty, _Promise&>
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto await_transform(_Ty&& __value)
        noexcept(nothrow_tag_invocable<as_awaitable_t, _Ty, _Promise&>)
          -> tag_invoke_result_t<as_awaitable_t, _Ty, _Promise&> {
        return tag_invoke(as_awaitable, static_cast<_Ty&&>(__value), static_cast<_Promise&>(*this));
      }
    };

    template <class _Env>
    struct __promise : __with_await_transform<__promise<_Env>> {
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto get_env() const noexcept -> const _Env&;
    };

    template <class ValueType>
    struct __prop_like {
      template <class _Query>
      STDEXEC_ATTRIBUTE(noreturn, nodiscard, host, device)
      constexpr auto query(_Query) const noexcept -> const ValueType& {
        STDEXEC_TERMINATE();
      }
    };

    // A singleton environment from a query/value pair
    template <class _Query, class _Value>
    struct prop {
      using __t = prop;
      using __id = prop;

      static_assert(__callable<_Query, __prop_like<_Value>>);

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query) const noexcept -> const _Value& {
        return __value;
      }

      STDEXEC_ATTRIBUTE(no_unique_address) _Query __query;
      STDEXEC_ATTRIBUTE(no_unique_address) _Value __value;
    };

    template <class _Query, class _Value>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      prop(_Query, _Value) -> prop<_Query, std::unwrap_reference_t<_Value>>;

    template <class _Query, auto _Value>
    struct cprop {
      using __t = cprop;
      using __id = cprop;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(_Query) noexcept {
        return _Value;
      }
    };

    //////////////////////////////////////////////////////////////////////
    // env
    template <class... Envs>
    struct env;

    template <>
    struct env<> {
      using __t = env;
      using __id = env;

      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto query() const = delete;
    };

    template <class Env>
    struct env<Env> : Env {
      using __t = env;
      using __id = env;
    };

    template <class Env>
    struct env<Env&> {
      using __t = env;
      using __id = env;

      template <class Query, class... _Args>
        requires __queryable_with<Env, Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(Query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<Env, Query, _Args...>)
          -> __query_result_t<Env, Query, _Args...> {
        return __query<Query>()(env_, static_cast<_Args&&>(__args)...);
      }

      Env& env_;
    };

    template <class Env>
    using __env_base = __if_c<std::is_reference_v<Env>, env<Env>, Env>;

    template <class Env1, class Env2>
    struct env<Env1, Env2> : __env_base<Env1> {
      using __t = env;
      using __id = env;

      using __env_base<Env1>::query;

      template <class Query, class... _Args>
        requires(!__queryable_with<Env1, Query, _Args...>)
             && __queryable_with<Env2, Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(Query, _Args&&... __args) const
        noexcept(__nothrow_queryable_with<Env2, Query, _Args...>)
          -> __query_result_t<Env2, Query, _Args...> {
        return __query<Query>()(env2_, static_cast<_Args&&>(__args)...);
      }

      STDEXEC_ATTRIBUTE(no_unique_address) Env2 env2_;
    };

    template <class Env1, class Env2, class... Envs>
    struct env<Env1, Env2, Envs...> : env<env<Env1, Env2>, Envs...> {
      using __t = env;
      using __id = env;
    };

    template <class... _Envs>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE env(_Envs...) -> env<std::unwrap_reference_t<_Envs>...>;

    template <class _EnvId>
    struct __fwd {
      using _Env = __cvref_t<_EnvId>;
      static_assert(__nothrow_move_constructible<_Env>);

      struct __t {
        using __id = __fwd;
        using __fwd_env_t = __t;

        template <__forwarding_query _Query, class... _Args>
          requires __queryable_with<_Env, _Query, _Args...>
        STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
        constexpr auto query(_Query, _Args&&... __args) const
          noexcept(__nothrow_queryable_with<_Env, _Query, _Args...>)
            -> __query_result_t<_Env, _Query, _Args...> {
          return __query<_Query>()(__env_, static_cast<_Args&&>(__args)...);
        }

        STDEXEC_ATTRIBUTE(no_unique_address)
        _Env __env_;
      };
    };

    template <class _Env>
    concept __is_fwd_env = __same_as<_Env, typename _Env::__fwd_env_t>;

    struct __fwd_fn {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env&& __env) const -> decltype(auto) {
        if constexpr (__decays_to<_Env, env<>> || __is_fwd_env<__decay_t<_Env>>) {
          return static_cast<_Env>(static_cast<_Env&&>(__env));
        } else {
          return __t<__fwd<__cvref_id<_Env>>>{static_cast<_Env&&>(__env)};
        }
      }
    };

    template <class _Env>
    using __fwd_env_t = __call_result_t<__fwd_fn, _Env>;

    template <class _EnvId, class _Query>
    struct __without_ {
      using _Env = __cvref_t<_EnvId>;
      static_assert(__nothrow_move_constructible<_Env>);

      struct __t : __env_base<_Env> {
        using __id = __without_;
        using __env_base<_Env>::query;

        STDEXEC_ATTRIBUTE(nodiscard, host, device)
        auto query(_Query) const noexcept = delete;
      };
    };

    struct __without_fn {
      template <class _Env, class _Query>
      constexpr auto operator()(_Env&& __env, _Query) const noexcept -> auto {
        if constexpr (__queryable_with<_Env, _Query>) {
          using _Without = __t<__without_<__cvref_id<_Env>, _Query>>;
          return _Without{static_cast<_Env&&>(__env)};
        } else {
          return static_cast<_Env&&>(__env);
        }
      }
    };

    inline constexpr __without_fn __without{};

    template <class _Env, class _Query, class... _Tags>
    using __without_t = __result_of<__without, _Env, _Query, _Tags...>;

    template <__nothrow_move_constructible _Fun>
    struct __from {
      using __t = __from;
      using __id = __from;
      STDEXEC_ATTRIBUTE(no_unique_address) _Fun __fun_;

      template <class _Query, class... _Args>
        requires __callable<const _Fun&, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      auto query(_Query, _Args&&... __args) const
        noexcept(__nothrow_callable<const _Fun&, _Query, _Args...>)
          -> __call_result_t<const _Fun&, _Query, _Args...> {
        return __fun_(_Query(), static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Fun>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __from(_Fun) -> __from<_Fun>;

    struct __join_fn {
      template <class _Env1, class _Env2>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env1&& __env1, _Env2&& __env2) const noexcept -> decltype(auto) {
        if constexpr (__decays_to<_Env1, env<>>) {
          return __fwd_fn()(static_cast<_Env2&&>(__env2));
        } else if constexpr (__decays_to<_Env2, env<>>) {
          return static_cast<_Env1>(static_cast<_Env1&&>(__env1));
        } else {
          return env<_Env1, __fwd_env_t<_Env2>>{
            {static_cast<_Env1&&>(__env1)}, __fwd_fn()(static_cast<_Env2&&>(__env2))};
        }
      }
    };

    inline constexpr __join_fn __join{};

    template <class _First, class... _Second>
    using __join_env_t = __result_of<__join, _First, _Second...>;

    struct __as_root_env_fn {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env __env) const noexcept
        -> __join_env_t<__root_env, std::unwrap_reference_t<_Env>> {
        return __join(__root_env{}, static_cast<std::unwrap_reference_t<_Env>&&>(__env));
      }
    };

    inline constexpr __as_root_env_fn __as_root_env{};

    template <class _Env>
    using __as_root_env_t = __result_of<__as_root_env, _Env>;
  } // namespace __env

  using __env::__join_env_t;
  using __env::__fwd_env_t;

  /////////////////////////////////////////////////////////////////////////////
  namespace __get_env {
    template <class _EnvProvider>
    concept __has_get_env = requires(const _EnvProvider& __env_provider) {
      __env_provider.get_env();
    };

    // For getting an execution environment from a receiver or the attributes from a sender.
    struct get_env_t {
      template <class _EnvProvider>
        requires __has_get_env<_EnvProvider>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _EnvProvider& __env_provider) const noexcept
        -> decltype(__env_provider.get_env()) {
        static_assert(queryable<decltype(__env_provider.get_env())>);
        static_assert(noexcept(__env_provider.get_env()), "get_env() members must be noexcept");
        return __env_provider.get_env();
      }

      template <class _EnvProvider>
        requires(!__has_get_env<_EnvProvider>) && tag_invocable<get_env_t, const _EnvProvider&>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _EnvProvider& __env_provider) const noexcept
        -> tag_invoke_result_t<get_env_t, const _EnvProvider&> {
        static_assert(queryable<tag_invoke_result_t<get_env_t, const _EnvProvider&>>);
        static_assert(nothrow_tag_invocable<get_env_t, const _EnvProvider&>);
        return tag_invoke(*this, __env_provider);
      }

      template <class _EnvProvider>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _EnvProvider&) const noexcept -> env<> {
        return {};
      }
    };
  } // namespace __get_env

  using __get_env::get_env_t;
  inline constexpr get_env_t get_env{};

  template <class _EnvProvider>
  concept environment_provider = requires(_EnvProvider& __ep) {
    { get_env(std::as_const(__ep)) } -> queryable;
  };

  template <class _Scheduler, class _LateDomain = __none_such>
  struct __sched_attrs {
    using __t = __sched_attrs;
    using __id = __sched_attrs;

    using __scheduler_t = __decay_t<_Scheduler>;
    using __sched_domain_t = __query_result_or_t<get_domain_t, __scheduler_t, default_domain>;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> __scheduler_t {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_t) const noexcept -> __sched_domain_t {
      return {};
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_override_t) const noexcept -> _LateDomain
      requires(!same_as<_LateDomain, __none_such>)
    {
      return {};
    }

    _Scheduler __sched_;
    STDEXEC_ATTRIBUTE(no_unique_address) _LateDomain __late_domain_ { };
  };

  template <class _Scheduler, class _LateDomain = __none_such>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sched_attrs(_Scheduler, _LateDomain = {})
    -> __sched_attrs<std::unwrap_reference_t<_Scheduler>, _LateDomain>;

  template <class _Scheduler>
  struct __sched_env {
    using __t = __sched_env;
    using __id = __sched_env;

    using __scheduler_t = __decay_t<_Scheduler>;
    using __sched_domain_t = __query_result_or_t<get_domain_t, __scheduler_t, default_domain>;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_scheduler_t) const noexcept -> __scheduler_t {
      return __sched_;
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(get_domain_t) const noexcept -> __sched_domain_t {
      return {};
    }

    _Scheduler __sched_;
  };

  template <class _Scheduler>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __sched_env(_Scheduler) -> __sched_env<std::unwrap_reference_t<_Scheduler>>;

  using __env::__as_root_env_t;
  using __env::__as_root_env;

  template <class _Env>
  concept __is_root_env = requires(_Env&& __env) {
    { __root_t{}(__env) } -> same_as<bool>;
  };

  template <class _Sender, class... _Env>
  concept __is_scheduler_affine = requires {
    requires __is_scheduler_affine_t().operator()<env_of_t<_Sender>, _Env...>();
  };

  // The attributes of a sender adaptor that does not introduce asynchrony.
  template <class _Sender>
  struct __sync_attrs {
    using __t = __sync_attrs;
    using __id = __sync_attrs;

    [[nodiscard]]
    constexpr auto query(__is_scheduler_affine_t) const noexcept {
      return __mbool<__is_scheduler_affine<_Sender>>();
    }

    template <class... _Env>
    [[nodiscard]]
    constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept {
      return get_completion_behavior<_Sender, _Env...>();
    }

    template <__forwarding_query _Query, class... _Args>
      requires __queryable_with<env_of_t<_Sender>, _Query, _Args...>
    [[nodiscard]]
    constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sender>, _Query, _Args...>)
        -> __query_result_t<env_of_t<_Sender>, _Query, _Args...> {
      return __query<_Query>()(get_env(__sndr_), static_cast<_Args&&>(__args)...);
    }

    const _Sender& __sndr_;
  };

  template <class _Sender>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sync_attrs(const _Sender&) -> __sync_attrs<_Sender>;

} // namespace stdexec

STDEXEC_PRAGMA_POP()
