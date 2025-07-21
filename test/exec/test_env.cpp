/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <exec/env.hpp>

namespace {
  // Two dummy properties:
  constexpr struct Foo : stdexec::forwarding_query_t {
    template <class Env>
      requires stdexec::tag_invocable<Foo, Env>
    auto operator()(const Env& e) const {
      return stdexec::tag_invoke(*this, e);
    }
  } foo{};

  constexpr struct Bar {
    friend constexpr auto tag_invoke(stdexec::forwarding_query_t, const Bar&) noexcept -> bool {
      return true;
    }

    template <class Env>
      requires stdexec::tag_invocable<Bar, Env>
    auto operator()(const Env& e) const {
      return stdexec::tag_invoke(*this, e);
    }
  } bar{};

  TEST_CASE("Test make_env works", "[env]") {
    auto e = stdexec::prop{foo, 42};
    CHECK(foo(e) == 42);

    auto e2 = exec::make_env(e, stdexec::prop{bar, 43});
    CHECK(foo(e2) == 42);
    CHECK(bar(e2) == 43);

    auto e3 = exec::make_env(e2, stdexec::prop{foo, 44});
    CHECK(foo(e3) == 44);
    CHECK(bar(e3) == 43);

    auto e4 = exec::without(e3, foo);
    STATIC_REQUIRE(!std::invocable<Foo, decltype(e4)>);
    CHECK(bar(e4) == 43);
  }
} // namespace
