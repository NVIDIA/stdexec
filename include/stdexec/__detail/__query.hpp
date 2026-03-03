/*
 * Copyright (c) 2025 NVIDIA Corporation
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

// // include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__meta.hpp"
#include "__tag_invoke.hpp"
#include "__utility.hpp"

#include <type_traits>

namespace STDEXEC
{
  // [exec.queries.queryable]
  template <class T>
  concept __queryable = __std::destructible<T>;

  template <class _Env, class _Query, class... _Args>
  concept __member_queryable_with = __queryable<_Env>
                                 && requires(_Env const   &__env,
                                             _Query const &__query,
                                             __declfn_t<_Args &&>... __args) {
                                      { __env.query(__query, __args()...) };
                                    };

  template <class _Env, class _Query, class... _Args>
  concept __nothrow_member_queryable_with = __member_queryable_with<_Env, _Query, _Args...>
                                         && requires(_Env const   &__env,
                                                     _Query const &__query,
                                                     __declfn_t<_Args &&>... __args) {
                                              { __env.query(__query, __args()...) } noexcept;
                                            };

  template <class _Env, class _Qy, class... _Args>
  using __member_query_result_t = decltype(__declval<_Env const &>().query(__declval<_Qy const &>(),
                                                                           __declval<_Args>()...));

  constexpr __none_such __no_default{};

  template <class _Query, class _Env, class... _Args>
  concept __has_validation = requires { _Query::template __validate<_Env, _Args...>(); };

  template <class _Query, auto _Default = __no_default, class _Transform = __q1<__midentity>>
  struct __query  // NOLINT(bugprone-crtp-constructor-accessibility)
    : __query<_Query, __no_default, _Transform>
  {
    using __base_t = __query<_Query, __no_default, _Transform>;
    using __base_t::operator();

    template <class... _Args>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(__ignore, _Args &&...) const noexcept  //
      -> __mcall1<_Transform, __mtypeof<_Default>>
    {
      return _Default;
    }
  };

  template <class _Query, class _Transform>
  struct __query<_Query, __no_default, _Transform>
  {
    template <class _Sig>
    inline static constexpr _Query (*signature)(_Sig) = nullptr;

    // Query with a .query member function:
    template <class _Qy = _Query, class _Env, class... _Args>
      requires __member_queryable_with<_Env const &, _Qy, _Args...>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(_Env const &__env, _Args &&...__args) const
      noexcept(__nothrow_member_queryable_with<_Env, _Qy, _Args...>)
        -> __mcall1<_Transform, __member_query_result_t<_Env, _Qy, _Args...>>
    {
      if constexpr (__has_validation<_Query, _Env, _Args...>)
      {
        _Query::template __validate<_Env, _Args...>();
      }
      return __env.query(_Query(), static_cast<_Args &&>(__args)...);
    }

    // Query with tag_invoke (legacy):
    template <class _Qy = _Query, class _Env, class... _Args>
      requires __tag_invocable<_Qy, _Env const &, _Args...>
    [[deprecated("the use of tag_invoke for queries is deprecated")]]
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)  //
      constexpr auto operator()(_Env const &__env, _Args &&...__args) const
      noexcept(__nothrow_tag_invocable<_Qy, _Env const &, _Args...>)
        -> __mcall1<_Transform, __tag_invoke_result_t<_Qy, _Env const &, _Args...>>
    {
      if constexpr (__has_validation<_Query, _Env, _Args...>)
      {
        _Query::template __validate<_Env, _Args...>();
      }
      return __tag_invoke(_Query(), __env, static_cast<_Args &&>(__args)...);
    }
  };

  template <class _Env, class _Query, class... _Args>
  concept __queryable_with = __callable<__query<_Query>, _Env &, _Args...>;

  template <class _Env, class _Query, class... _Args>
  concept __nothrow_queryable_with = __nothrow_callable<__query<_Query>, _Env &, _Args...>;

  template <class _Env, class _Query, class... _Args>
  using __query_result_t = __call_result_t<__query<_Query>, _Env &, _Args...>;

  template <class _Env, class _Query, class... _Args>
  concept __statically_queryable_with_impl = requires(_Query __q, _Args &&...__args) {
    std::remove_reference_t<_Env>::query(__q, static_cast<_Args &&>(__args)...);
  };

  template <class _Env, class _Query, class... _Args>
  concept __statically_queryable_with = __queryable_with<_Env, _Query, _Args...>
                                     && __statically_queryable_with_impl<_Env, _Query, _Args...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.queries]
  template <class _Tp>
  concept __is_bool_constant = requires { typename __mbool<_Tp::value>; };

  template <class _Tag>
  concept __forwarding_query = forwarding_query(_Tag{});

  //////////////////////////////////////////////////////////////////////////////////////////
  // __is_completion_query
  template <class _Query>
  inline constexpr bool __is_completion_query = false;
  template <class _Tag>
  inline constexpr bool __is_completion_query<get_completion_domain_t<_Tag>> = true;
  template <class _Tag>
  inline constexpr bool __is_completion_query<get_completion_scheduler_t<_Tag>> = true;
  template <class _Tag>
  inline constexpr bool __is_completion_query<__get_completion_behavior_t<_Tag>> = true;
}  // namespace STDEXEC

STDEXEC_P2300_NAMESPACE_BEGIN()
  struct forwarding_query_t
  {
    template <class _Query>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    consteval auto operator()(_Query) const noexcept -> bool
    {
      if constexpr (STDEXEC::__queryable_with<_Query, forwarding_query_t>)
      {
        return STDEXEC::__query<forwarding_query_t>()(_Query());
      }
      else
      {
        return STDEXEC::__std::derived_from<_Query, forwarding_query_t>;
      }
    }
  };

  inline constexpr forwarding_query_t forwarding_query{};
STDEXEC_P2300_NAMESPACE_END()
