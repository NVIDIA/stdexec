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

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.envs]
  namespace __env {
    // A singleton environment from a query/value pair
    template <class _Query, class _Value>
    struct prop {
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
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      auto query() const = delete;
    };

    template <class Env>
    struct env<Env> : Env { };

    template <class Env>
    struct env<Env&> {
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
    using __env_base_t = __if_c<std::is_reference_v<Env>, env<Env>, Env>;

    template <class Env1, class Env2>
    struct env<Env1, Env2> : __env_base_t<Env1> {
      using __env_base_t<Env1>::query;

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
    struct env<Env1, Env2, Envs...> : env<env<Env1, Env2>, Envs...> { };

    template <class... _Envs>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE env(_Envs...) -> env<std::unwrap_reference_t<_Envs>...>;

    template <class _Env>
    struct __fwd {
      static_assert(__nothrow_move_constructible<_Env>);

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

    template <class _Env>
    concept __is_fwd_env = __is_instance_of<__decay_t<_Env>, __fwd>;

    struct __fwd_fn {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env&& __env) const -> decltype(auto) {
        if constexpr (__decays_to<_Env, env<>> || __is_fwd_env<_Env>) {
          return static_cast<_Env>(static_cast<_Env&&>(__env));
        } else {
          return __fwd<_Env>{static_cast<_Env&&>(__env)};
        }
      }
    };

    template <class _Env>
    using __fwd_env_t = __call_result_t<__fwd_fn, _Env>;

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
    { __root_t{}(__env) } -> __std::same_as<bool>;
  };


  /////////////////////////////////////////////////////////////////////////////
  namespace __get_env {
    template <class _EnvProvider>
    using __get_env_member_result_t = decltype(__declval<_EnvProvider>().get_env());

    template <class _EnvProvider>
    concept __has_get_env = requires { typename __get_env_member_result_t<_EnvProvider>; };

    // For getting an execution environment from a receiver or the attributes from a sender.
    struct get_env_t {
      template <class _EnvProvider>
        requires __has_get_env<const _EnvProvider&>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _EnvProvider& __env_provider) const noexcept
        -> __get_env_member_result_t<const _EnvProvider&> {
        static_assert(noexcept(__env_provider.get_env()), "get_env() members must be noexcept");
        return __env_provider.get_env();
      }

      template <class _EnvProvider>
        requires __has_get_env<const _EnvProvider&>
              || __tag_invocable<get_env_t, const _EnvProvider&>
      [[deprecated("the use of tag_invoke for get_env is deprecated")]]
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device) //
        constexpr auto operator()(const _EnvProvider& __env_provider) const noexcept
        -> __tag_invoke_result_t<get_env_t, const _EnvProvider&> {
        static_assert(
          __nothrow_tag_invocable<get_env_t, const _EnvProvider&>,
          "get_env __tag_invoke overloads must be noexcept");
        return __tag_invoke(*this, __env_provider);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(__ignore) const noexcept -> env<> {
        return {};
      }
    };
  } // namespace __get_env

  using __get_env::get_env_t;
  inline constexpr get_env_t get_env{};

  // template <class _EnvProvider>
  // concept environment_provider = requires(_EnvProvider& __ep) {
  //   { get_env(std::as_const(__ep)) } -> queryable;
  // };

  template <class _EnvProvider>
  concept environment_provider = __minvocable_q<__call_result_t, get_env_t, const _EnvProvider&>;

} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
