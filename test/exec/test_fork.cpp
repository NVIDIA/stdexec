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

#include "exec/fork_join.hpp"
#include "exec/just_from.hpp"
#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;

namespace {
  TEST_CASE("fork_join is a sender", "[adaptors][fork_join]") {
    auto sndr = exec::fork_join(just(), then([] { }));
    STATIC_REQUIRE(sender<decltype(sndr)>);
  }

  TEST_CASE("fork_join is a sender in empty env", "[adaptors][fork_join]") {
    auto sndr = exec::fork_join(just(), then([] { }));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(sndr), env<>>,
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>
      >);
  }

  TEST_CASE("fork_join broadcasts results to multiple continuations", "[adaptors][fork_join]") {
    auto fn = [](auto sink) {
      sink(42);
      return completion_signatures<
        set_value_t(int),
        set_value_t(int, int),
        set_value_t(int, int, int)
      >{};
    };
    auto sndr = exec::fork_join(
      exec::just_from(fn),
      then([](auto&&... is) {
        CHECK(sizeof...(is) == 1);
        CHECK(((is == 42) && ...));
        STATIC_REQUIRE((std::is_same_v<decltype(is), const int&> && ...));
        return (is + ...);
      }),
      then([](auto&&... is) {
        CHECK(sizeof...(is) == 1);
        CHECK(((is == 42) && ...));
        STATIC_REQUIRE((std::is_same_v<decltype(is), const int&> && ...));
        return (is + ...);
      }));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(sndr), env<>>,
        completion_signatures<set_value_t(int, int), set_error_t(std::exception_ptr), set_stopped_t()>
      >);

    auto [i1, i2] = sync_wait(sndr).value();
    CHECK(i1 == 42);
    CHECK(i2 == 42);
  }
} // namespace
