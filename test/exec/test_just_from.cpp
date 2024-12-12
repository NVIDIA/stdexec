/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "exec/just_from.hpp"
#include "test_common/tuple.hpp"
#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

TEST_CASE("just_from is a sender", "[just_from]") {
  SECTION("potentially throwing") {
    auto s = exec::just_from([](auto sink) { return sink(42); });
    using S = decltype(s);
    STATIC_REQUIRE(ex::sender<S>);
    STATIC_REQUIRE(ex::sender_in<S>);
    ::check_val_types<ex::__mset<pack<int>>>(s);
    ::check_err_types<ex::__mset<std::exception_ptr>>(s);
    ::check_sends_stopped<false>(s);
  }

  SECTION("not potentially throwing") {
    auto s = exec::just_from([](auto sink) noexcept { return sink(42); });
    using S = decltype(s);
    STATIC_REQUIRE(ex::sender<S>);
    STATIC_REQUIRE(ex::sender_in<S>);
    ::check_val_types<ex::__mset<pack<int>>>(s);
    ::check_err_types<ex::__mset<>>(s);
    ::check_sends_stopped<false>(s);
  }
}

TEST_CASE("just_from basically works", "[just_from]") {
  auto s = exec::just_from([](auto sink) noexcept { return sink(42, 43, 44); });
  ::check_val_types<ex::__mset<pack<int, int, int>>>(s);
  ::check_err_types<ex::__mset<>>(s);
  ::check_sends_stopped<false>(s);

  auto [a, b, c] = ex::sync_wait(s).value();
  CHECK(a == 42);
  CHECK(b == 43);
  CHECK(c == 44);
}

TEST_CASE("just_from with multiple completions", "[just_from]") {
  auto fn = [](auto sink) noexcept {
    if (sizeof(sink) == ~0ul) {
      std::puts("sink(42)");
      sink(42);
    } else {
      std::puts("sink(43, 44)");
      sink(43, 44);
    }
    return ex::completion_signatures<ex::set_value_t(int), ex::set_value_t(int, int)>{};
  };
  auto s = exec::just_from(fn);
  ::check_val_types<ex::__mset<pack<int>, pack<int, int>>>(s);
  ::check_err_types<ex::__mset<>>(s);
  ::check_sends_stopped<false>(s);

  auto var = ex::sync_wait_with_variant(s).value();
  std::visit(
    []<class Tupl>(Tupl tupl) {
      constexpr auto N = std::tuple_size_v<Tupl>;
      CHECK(N == 2);
      if constexpr (N == 2) {
        CHECK_TUPLE(tupl == std::tuple{43, 44});
      }
    },
    var);
}
