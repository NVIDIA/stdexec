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
#include "__tuple.hpp"

#include <exception>   // IWYU pragma: keep for std::terminate
#include <functional>  // IWYU pragma: keep for unwrap_reference_t
#include <type_traits>

#include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.envs]
  namespace __env
  {
    //////////////////////////////////////////////////////////////////////
    // cprop
    template <class _Query, auto _Value>
    struct cprop
    {
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(_Query, auto &&...) noexcept
      {
        return _Value;
      }
    };

    template <class _Env>
    struct __fwd
    {
      static_assert(__nothrow_move_constructible<_Env>);

      template <__forwarding_query _Query, class... _Args>
        requires __queryable_with<_Env, _Query, _Args...>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto query(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<_Env, _Query, _Args...>)
          -> __query_result_t<_Env, _Query, _Args...>
      {
        return __query<_Query>()(__env_, static_cast<_Args &&>(__args)...);
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      _Env __env_;
    };

    template <class _Env>
    concept __is_fwd_env = __is_instance_of<__decay_t<_Env>, __fwd>;

    struct __fwd_fn
    {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env &&__env) const -> decltype(auto)
      {
        if constexpr (__decays_to<_Env, env<>> || __is_fwd_env<_Env>)
        {
          return static_cast<_Env>(static_cast<_Env &&>(__env));
        }
        else
        {
          return __fwd<_Env>{static_cast<_Env &&>(__env)};
        }
      }
    };

    template <class _Env>
    using __fwd_env_t = __call_result_t<__fwd_fn, _Env>;

    struct __join_fn
    {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env &&__env) const noexcept -> _Env
      {
        return static_cast<_Env &&>(__env);
      }

      template <class _Env1, class _Env2>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env1 &&__env1, _Env2 &&__env2) const noexcept -> decltype(auto)
      {
        if constexpr (__decays_to<_Env1, env<>>)
        {
          return __fwd_fn()(static_cast<_Env2 &&>(__env2));
        }
        else if constexpr (__decays_to<_Env2, env<>>)
        {
          return static_cast<_Env1>(static_cast<_Env1 &&>(__env1));
        }
        else
        {
          return env<_Env1, __fwd_env_t<_Env2>>{static_cast<_Env1 &&>(__env1),
                                                __fwd_fn()(static_cast<_Env2 &&>(__env2))};
        }
      }
    };

    inline constexpr __join_fn __join{};

    template <class _First, class... _Second>
    using __join_env_t = __result_of<__join, _First, _Second...>;

    struct __root_t : __query<__root_t>
    {
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool
      {
        return false;
      }
    };

    struct __root_env
    {
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(__root_t) noexcept -> bool
      {
        return true;
      }
    };

    struct __as_root_env_fn
    {
      template <class _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_Env __env) const noexcept
        -> __join_env_t<__root_env, std::unwrap_reference_t<_Env>>
      {
        return __join(__root_env{}, static_cast<std::unwrap_reference_t<_Env> &&>(__env));
      }
    };
  }  // namespace __env

  using __env::__join_env_t;
  using __env::__fwd_env_t;
  using __env::__root_t;
  using __env::__root_env;

  inline constexpr __env::__fwd_fn         __fwd_env{};
  inline constexpr __env::__as_root_env_fn __as_root_env{};

  template <class _Env>
  using __as_root_env_t = __result_of<__as_root_env, _Env>;

  template <class _Env>
  concept __is_root_env = requires(_Env &&__env) {
    { __root_t{}(__env) } -> __std::same_as<bool>;
  };

  //////////////////////////////////////////////////////////////////////
  // A singleton environment from a query/value pair
  template <class _Query, class _Value>
  struct prop
  {
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(_Query, auto &&...) const noexcept -> _Value const &
    {
      return __value;
    }

    STDEXEC_ATTRIBUTE(no_unique_address) _Query __query;
    STDEXEC_ATTRIBUTE(no_unique_address) _Value __value;

   private:
    struct __prop_like
    {
      STDEXEC_ATTRIBUTE(noreturn, nodiscard, host, device)
      constexpr auto query(_Query) const noexcept -> _Value const &
      {
        STDEXEC_TERMINATE();
      }
    };

    static_assert(__callable<_Query, __prop_like>);
  };

  template <class _Query, class _Value>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    prop(_Query, _Value) -> prop<_Query, std::unwrap_reference_t<_Value>>;

  namespace __detail
  {
    template <class _Query, class... _Args>
    struct __get_1st_env
    {
      template <class _Env>
      using __has_query_t = __mbool<__queryable_with<_Env, _Query, _Args...>>;

      template <class... _Envs>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(env<_Envs...> const &__env) const noexcept -> decltype(auto)
      {
        // count of elements that includes the first env that supports the query and all
        // subsequent envs
        STDEXEC_CONSTEXPR_LOCAL auto __index =
          sizeof...(_Envs) - __mcall<__mfind_if<__q1<__has_query_t>, __msize>, _Envs...>::value;
        if constexpr (__index < sizeof...(_Envs))
          return STDEXEC::__get<__index>(__env);
      }
    };
  }  // namespace __detail

  //////////////////////////////////////////////////////////////////////
  // env
  template <class... _Envs>
  struct env : __tuple<_Envs...>
  {
    template <class _Query, class... _Args>
    using __1st_env_t = __call_result_t<__detail::__get_1st_env<_Query, _Args...>, env const &>;

    template <class _Query, class... _Args>
      requires __not_same_as<__1st_env_t<_Query, _Args...>, void>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto query(_Query, _Args &&...__args) const
      noexcept(__nothrow_queryable_with<__1st_env_t<_Query, _Args...>, _Query, _Args...>)
        -> __query_result_t<__1st_env_t<_Query, _Args...>, _Query, _Args...>
    {
      auto const &__env = __detail::__get_1st_env<_Query, _Args...>()(*this);
      return __query<_Query>()(__env, static_cast<_Args &&>(__args)...);
    }
  };

  template <class... _Envs>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE env(_Envs...) -> env<std::unwrap_reference_t<_Envs>...>;

  /////////////////////////////////////////////////////////////////////////////
  namespace __detail
  {
    template <class _EnvProvider>
    using __get_env_member_result_t = decltype(__declval<_EnvProvider>().get_env());

    template <class _EnvProvider>
    concept __has_get_env_member = requires { typename __get_env_member_result_t<_EnvProvider>; };
  }  // namespace __detail

  //! @brief Customization point object that obtains the *environment* of a
  //!        sender or receiver.
  //!
  //! Every sender and every receiver has an associated *environment* — an
  //! unordered, type-keyed bag of properties such as the stop token, the
  //! allocator, the preferred scheduler, the start scheduler, and any
  //! domain-specific properties a sender adaptor wants to expose. The
  //! environment is what makes the sender model *contextual*: a sender
  //! adapts its behavior based on the environment of the receiver it is
  //! connected to.
  //!
  //! @c get_env(provider) returns the environment of @c provider, where
  //! @c provider is either a receiver (yielding its environment, which
  //! the sender will introspect via queries) or a sender (yielding its
  //! *attributes*, which the framework consults to determine things like
  //! the sender's completion scheduler).
  //!
  //! See [exec.queries.get_env] in the C++26 working draft.
  //!
  //! **Customization.**
  //!
  //! Most receivers and senders simply expose a @c noexcept,
  //! const-callable `.get_env()` member returning their environment:
  //!
  //! @code{.cpp}
  //! struct my_receiver {
  //!   using receiver_concept = stdexec::receiver_tag;
  //!
  //!   auto get_env() const noexcept {
  //!     return stdexec::env{stdexec::prop{stdexec::get_stop_token, my_stop_token_}};
  //!   }
  //! };
  //! @endcode
  //!
  //! Many receivers don't have any properties to expose — for those, the
  //! @c get_env member can simply return an empty @c stdexec::env<> (or
  //! the @c get_env CPO will return one automatically via its @c __ignore
  //! overload).
  //!
  //! @c tag_invoke-based customization is supported via a deprecated
  //! overload, retained for backwards compatibility.
  //!
  //! **Environment queries.**
  //!
  //! Once you have an environment, you query it by calling the appropriate
  //! query CPO on it: <tt>get_stop_token(env)</tt>,
  //! <tt>get_allocator(env)</tt>, <tt>get_scheduler(env)</tt>, etc. Each
  //! query is a separate CPO; the environment dispatches based on the
  //! query's type. Inside a sender pipeline you almost always reach for
  //! @c stdexec::read_env (or its helpers like @c get_stop_token() with
  //! no argument) rather than calling @c get_env directly.
  //!
  //! @see stdexec::env             — the environment container type
  //! @see stdexec::read_env        — the sender factory that exposes env values to pipelines
  //! @see stdexec::get_stop_token  — example of an environment query CPO
  //! @see stdexec::get_allocator
  //! @see stdexec::get_scheduler
  struct get_env_t
  {
    //! @brief Obtain the environment of @c __env_provider.
    //!
    //! Dispatches to <tt>__env_provider.get_env()</tt>, statically
    //! asserting that the member is @c noexcept.
    //!
    //! @tparam _EnvProvider A type whose const-lvalue has a
    //!                      `.get_env() const` member.
    //! @param __env_provider The receiver or sender whose environment to
    //!                       retrieve.
    //! @returns The environment object (typed as defined by the provider).
    template <class _EnvProvider>
      requires __detail::__has_get_env_member<_EnvProvider const &>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(_EnvProvider const &__env_provider) const noexcept
      -> __detail::__get_env_member_result_t<_EnvProvider const &>
    {
      static_assert(noexcept(__env_provider.get_env()), "get_env() members must be noexcept");
      return __env_provider.get_env();
    }

    template <class _EnvProvider>
      requires __detail::__has_get_env_member<_EnvProvider const &>
            || __tag_invocable<get_env_t, _EnvProvider const &>
    [[deprecated("the use of tag_invoke for get_env is deprecated")]]
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)  //
      constexpr auto operator()(_EnvProvider const &__env_provider) const noexcept
      -> __tag_invoke_result_t<get_env_t, _EnvProvider const &>
    {
      static_assert(__nothrow_tag_invocable<get_env_t, _EnvProvider const &>,
                    "get_env __tag_invoke overloads must be noexcept");
      return __tag_invoke(*this, __env_provider);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(__ignore) const noexcept -> env<>
    {
      return {};
    }
  };

  //! @brief The customization point object for obtaining a sender's or
  //!        receiver's environment.
  //!
  //! @c get_env is an instance of @ref get_env_t. See @ref get_env_t for
  //! the full description and customization examples.
  //!
  //! @hideinitializer
  inline constexpr get_env_t get_env{};

  template <class _EnvProvider>
  concept __environment_provider = __minvocable_q<__call_result_t, get_env_t, _EnvProvider const &>;
}  // namespace STDEXEC

#include "__epilogue.hpp"
