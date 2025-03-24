/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

namespace {

  TEST_CASE("Simple test for just_error", "[factories][just_error]") {
    auto op = ex::connect(ex::just_error(std::exception_ptr{}), expect_error_receiver{});
    ex::start(op);
    // receiver ensures that set_error() is called
  }

  TEST_CASE("just_error returns a sender", "[factories][just_error]") {
    using t = decltype(ex::just_error(std::exception_ptr{}));
    static_assert(ex::sender<t>, "ex::just_error must return a sender");
  }

  TEST_CASE("just_error returns a typed sender", "[factories][just_error]") {
    using t = decltype(ex::just_error(std::exception_ptr{}));
    static_assert(ex::sender_in<t, ex::env<>>, "ex::just_error must return a sender");
  }

  TEST_CASE("error types are properly set for just_error<int>", "[factories][just_error]") {
    check_err_types<ex::__mset<int>>(ex::just_error(1));
  }

  TEST_CASE(
    "error types are properly set for just_error<exception_ptr>",
    "[factories][just_error]") {
    // we should not get std::exception_ptr twice
    check_err_types<ex::__mset<std::exception_ptr>>(ex::just_error(std::exception_ptr()));
  }

  TEST_CASE("value types are properly set for just_error", "[factories][just_error]") {
    // there is no variant of calling `set_value(recv)`
    check_val_types<ex::__mset<>>(ex::just_error(1));
  }

  TEST_CASE("just_error cannot call set_stopped", "[factories][just_error]") {
    check_sends_stopped<false>(ex::just_error(1));
  }

  TEST_CASE("just_error removes cv qualifier for the given type", "[factories][just_error]") {
    std::string str{"hello"};
    const std::string& crefstr = str;
    auto snd = ex::just_error(crefstr);
    check_err_types<ex::__mset<std::string>>(snd);
  }
} // namespace
