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

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

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

  // Before v19, clang could not compile this test because of the large number of nested
  // envs.
#if !STDEXEC_CLANG() || STDEXEC_CLANG_VERSION >= 1900

#  define DEFINE_QUERY(name) constexpr struct name ## _t : ex::__query<name ## _t> {} name{}

  DEFINE_QUERY(query_0);
  DEFINE_QUERY(query_1);
  DEFINE_QUERY(query_2);
  DEFINE_QUERY(query_3);
  DEFINE_QUERY(query_4);
  DEFINE_QUERY(query_5);
  DEFINE_QUERY(query_6);
  DEFINE_QUERY(query_7);
  DEFINE_QUERY(query_8);
  DEFINE_QUERY(query_9);
  DEFINE_QUERY(query_10);
  DEFINE_QUERY(query_11);
  DEFINE_QUERY(query_12);

  TEST_CASE("env supports lots of child envs without exceeding compiler limits", "[queries][env]")
  {
    auto env = ex::env{
      ex::prop{ query_0,  0},
      ex::prop{ query_1,  1},
      ex::prop{ query_2,  2},
      ex::prop{ query_3,  3},
      ex::prop{ query_4,  4},
      ex::prop{ query_5,  5},
      ex::prop{ query_6,  6},
      ex::prop{ query_7,  7},
      ex::prop{ query_8,  8},
      ex::prop{ query_9,  9},
      ex::prop{query_10, 10},
      ex::prop{query_11, 11},
      ex::prop{query_12, 12}
    };

    CHECK(env.query(query_0) == 0);
    CHECK(env.query(query_1) == 1);
    CHECK(env.query(query_2) == 2);
    CHECK(env.query(query_3) == 3);
    CHECK(env.query(query_4) == 4);
    CHECK(env.query(query_5) == 5);
    CHECK(env.query(query_6) == 6);
    CHECK(env.query(query_7) == 7);
    CHECK(env.query(query_8) == 8);
    CHECK(env.query(query_9) == 9);
    CHECK(env.query(query_10) == 10);
    CHECK(env.query(query_11) == 11);
    CHECK(env.query(query_12) == 12);
  }

#endif
}  // namespace
