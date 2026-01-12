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

#include "exec/finally.hpp"
#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

using namespace STDEXEC;

namespace {

  TEST_CASE("finally is a sender", "[adaptors][finally]") {
    auto s = exec::finally(just(), just());
    STATIC_REQUIRE(sender<decltype(s)>);
  }

  TEST_CASE("finally is a sender in empty env", "[adaptors][finally]") {
    auto s = exec::finally(just(), just());
    STATIC_REQUIRE(sender_in<decltype(s), ex::env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t()>
      >);
  }

  TEST_CASE("finally executes the final action", "[adaptors][finally]") {
    bool called = false;
    auto s = exec::finally(just(), just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t()>
      >);
    sync_wait(s);
    CHECK(called);
  }

  TEST_CASE("finally executes the final action and returns integer", "[adaptors][finally]") {
    bool called = false;
    auto s = exec::finally(just(42), just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t(int)>
      >);
    auto [i] = *sync_wait(s);
    CHECK(called);
    CHECK(i == 42);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("finally does not execute the final action and throws integer", "[adaptors][finally]") {
    bool called = false;

    auto s = exec::finally(
      just(21) | then([](int) -> int { throw 42; }),
      just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_error_t(std::exception_ptr), set_value_t(int)>
      >);
    CHECK_THROWS_AS(sync_wait(s), int);
    CHECK(called);
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE("finally includes the error types of the final action", "[adaptors][finally]") {
    auto s = exec::finally(just(), just_error(42));
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_error_t(int)>
      >);
  }

  TEST_CASE("finally includes the stopped signal of the final action", "[adaptors][finally]") {
    auto s = exec::finally(just(), just_stopped());
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(s), ex::env<>>,
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>
      >);
  }
} // namespace
