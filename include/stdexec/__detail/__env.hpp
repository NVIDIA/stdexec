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
#include "__meta.hpp"
#include "__query.hpp"
#include "__tag_invoke.hpp"

#include <exception>  // IWYU pragma: keep for std::terminate
#include <functional> // IWYU pragma: keep for unwrap_reference_t
#include <type_traits>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.envs]
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

    // A singleton environment from a query/value pair
    template <class _Query, class _Value>
    struct prop {
      using __t = prop;
      using __id = prop;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query, auto&&...) const noexcept -> const _Value& {
        return __value;
      }

      STDEXEC_ATTRIBUTE(no_unique_address) _Query __query;
      STDEXEC_ATTRIBUTE(no_unique_address) _Value __value;

     private:
      struct __prop_like {
        STDEXEC_ATTRIBUTE(noreturn, nodiscard, host, device)
        constexpr auto query(_Query) const noexcept -> const _Value& {
          STDEXEC_TERMINATE();
        }
      };

      static_assert(__callable<_Query, __prop_like>);
    };

    template <class _Query, class _Value>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      prop(_Query, _Value) -> prop<_Query, std::unwrap_reference_t<_Value>>;

    template <class _Query, auto _Value>
    struct cprop {
      using __t = cprop;
      using __id = cprop;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(_Query, auto&&...) noexcept {
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

    struct __as_root_env_fn {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env __env) const noexcept
        -> __join_env_t<__root_env, std::unwrap_reference_t<_Env>> {
        return __join(__root_env{}, static_cast<std::unwrap_reference_t<_Env>&&>(__env));
      }
    };

  } // namespace __env

  using __env::__join_env_t;
  using __env::__fwd_env_t;
  using __env::__root_t;
  using __env::__root_env;

  inline constexpr __env::__fwd_fn __fwd_env{};
  inline constexpr __env::__as_root_env_fn __as_root_env{};

  template <class _Env>
  using __as_root_env_t = __result_of<__as_root_env, _Env>;

  template <class _Env>
  concept __is_root_env = requires(_Env&& __env) {
    { __root_t{}(__env) } -> same_as<bool>;
  };


  /////////////////////////////////////////////////////////////////////////////
  namespace __get_env {
    template <class _EnvProvider>
    using __get_env_member_result_t = decltype(__declval<_EnvProvider>().get_env());

    template <class _EnvProvider>
    concept __has_get_env = requires { typename __get_env_member_result_t<_EnvProvider>; };

    // For getting an execution environment from a receiver or the attributes from a sender.
    struct get_env_t {
     private:
      template <class _EnvProvider>
      static constexpr auto __get_declfn() noexcept {
        constexpr __declfn_t<_EnvProvider> __env_provider{};
        if constexpr (__has_get_env<_EnvProvider>) {
          static_assert(__has_get_env<_EnvProvider>);
          using __result_t = __get_env_member_result_t<_EnvProvider>;
          static_assert(noexcept(__env_provider().get_env()), "get_env() members must be noexcept");
          return __declfn<__result_t>();
        } else if constexpr (tag_invocable<get_env_t, const _EnvProvider&>) {
          using __result_t = tag_invoke_result_t<get_env_t, const _EnvProvider&>;
          constexpr bool __is_nothrow = nothrow_tag_invocable<get_env_t, const _EnvProvider&>;
          static_assert(__is_nothrow, "get_env tag_invoke overloads must be noexcept");
          return __declfn<__result_t>();
        } else {
          return __declfn<env<>>();
        }
      }

     public:
      template <class _EnvProvider, auto _DeclFn = __get_declfn<_EnvProvider>()>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto
        operator()(const _EnvProvider& __env_provider) const noexcept -> decltype(_DeclFn()) {
        if constexpr (__has_get_env<_EnvProvider>) {
          return __env_provider.get_env();
        } else if constexpr (tag_invocable<get_env_t, const _EnvProvider&>) {
          return tag_invoke(*this, __env_provider);
        } else {
          return env<>{};
        }
      }
    };
  } // namespace __get_env

  using __get_env::get_env_t;
  inline constexpr get_env_t get_env{};

  template <class _EnvProvider>
  concept environment_provider = requires(_EnvProvider& __ep) {
    { get_env(std::as_const(__ep)) } -> queryable;
  };
} // namespace stdexec

STDEXEC_PRAGMA_POP()
