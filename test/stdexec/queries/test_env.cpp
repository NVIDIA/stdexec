/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

namespace
{
  template <typename T>
  concept can_get_domain = requires(T const & t) { t.query(::ex::get_domain); };

  namespace zero
  {
    using env = ::ex::env<>;
    static_assert(std::is_same_v<::ex::never_stop_token, ::ex::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  }  // namespace zero

  namespace one
  {
    using env = ::ex::env<::ex::env<>>;
    static_assert(std::is_same_v<::ex::never_stop_token, ::ex::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  }  // namespace one

  namespace two
  {
    using env = ::ex::env<::ex::env<>, ::ex::env<>>;
    static_assert(std::is_same_v<::ex::never_stop_token, ::ex::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  }  // namespace two

  namespace three
  {
    using env = ::ex::env<::ex::env<>, ::ex::env<>, ::ex::env<>>;
    static_assert(std::is_same_v<::ex::never_stop_token, ::ex::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  }  // namespace three

  // https://github.com/NVIDIA/stdexec/issues/1840
  constexpr struct FwdFoo
    : ex::__query<FwdFoo>
    , ex::forwarding_query_t
  {
    using ex::__query<FwdFoo>::operator();
  } fwd_foo{};

  constexpr struct Foo : ex::__query<Foo>
  {
  } foo{};

  constexpr struct Bar : ex::__query<Bar>
  {
  } bar{};

  constexpr bool test()
  {
    auto env = ex::env{
      ex::env{ex::prop{fwd_foo, 42.}, ex::prop{foo, 'F'}},
      ex::prop{                   bar,              31415}
    };

    static_assert(ex::__queryable_with<decltype(env), Foo>);
    return fwd_foo(env) == 42. && bar(env) == 31415;
  }
  static_assert(test());

  struct EnvOfThree
    : ex::env<ex::env<ex::prop<FwdFoo, int>, ex::prop<Foo, int>>, ex::prop<Bar, int>>
  {};

  static_assert(ex::__queryable_with<EnvOfThree, Foo>);

  struct non_dependent_attrs
  {
    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept
    {
      return ex::inline_scheduler{};
    }
  };

  TEST_CASE("env forwards non-dependent queries to the root environment", "[queries][env]")
  {
    auto attrs = ex::env{
      ex::prop{fwd_foo, 'F'},
      non_dependent_attrs{}
    };
    auto sch = ex::get_completion_scheduler<ex::set_value_t>(attrs, ex::env{});
    CHECK(std::same_as<decltype(sch), ex::inline_scheduler>);
  }
}  // namespace
