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

namespace {

  template <typename T>
  concept can_get_domain = requires(const T& t) { t.query(::STDEXEC::get_domain); };

  namespace zero {
    using env = ::STDEXEC::env<>;
    static_assert(std::is_same_v<::STDEXEC::never_stop_token, ::STDEXEC::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace zero

  namespace one {
    using env = ::STDEXEC::env<::STDEXEC::env<>>;
    static_assert(std::is_same_v<::STDEXEC::never_stop_token, ::STDEXEC::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace one

  namespace two {
    using env = ::STDEXEC::env<::STDEXEC::env<>, ::STDEXEC::env<>>;
    static_assert(std::is_same_v<::STDEXEC::never_stop_token, ::STDEXEC::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace two

  namespace three {
    using env = ::STDEXEC::env<::STDEXEC::env<>, ::STDEXEC::env<>, ::STDEXEC::env<>>;
    static_assert(std::is_same_v<::STDEXEC::never_stop_token, ::STDEXEC::stop_token_of_t<env>>);
    static_assert(!can_get_domain<env>);
  } // namespace three

  // https://github.com/NVIDIA/stdexec/issues/1840
  constexpr struct FwdFoo
    : STDEXEC::__query<FwdFoo>
    , STDEXEC::forwarding_query_t {
    using STDEXEC::__query<FwdFoo>::operator();
  } fwd_foo{};

  constexpr struct Foo : STDEXEC::__query<Foo> {
  } foo{};

  constexpr struct Bar : STDEXEC::__query<Bar> {
  } bar{};

  constexpr bool test() {
    auto env = STDEXEC::env{
      STDEXEC::env{STDEXEC::prop{fwd_foo, 42.}, STDEXEC::prop{foo, 'F'}},
      STDEXEC::prop{                        bar,                   31415}
    };

    static_assert(STDEXEC::__queryable_with<decltype(env), Foo>);
    return fwd_foo(env) == 42. && bar(env) == 31415;
  }
  static_assert(test());

  struct EnvOfThree
    : STDEXEC::env<
        STDEXEC::env<STDEXEC::prop<FwdFoo, int>, STDEXEC::prop<Foo, int>>,
        STDEXEC::prop<Bar, int>
      > { };

  static_assert(STDEXEC::__queryable_with<EnvOfThree, Foo>);
} // namespace
