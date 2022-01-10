/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

TEST_CASE("Simple test for just_done", "[factories][just_done]") {
  auto op = ex::connect(ex::just_done(), expect_done_receiver{});
  ex::start(op);
  // receiver ensures that set_done() is called
}

TEST_CASE("just_done returns a sender", "[factories][just_done]") {
  using t = decltype(ex::just_done());
  static_assert(ex::sender<t>, "ex::just_done must return a sender");
  REQUIRE(ex::sender<t>);
}

TEST_CASE("just_done returns a typed sender", "[factories][just_done]") {
  using t = decltype(ex::just_done());
  static_assert(ex::typed_sender<t, empty_env>, "ex::just_done must return a typed_sender");
}

TEST_CASE("value types are properly set for just_done", "[factories][just_done]") {
  check_val_types<type_array<>>(ex::just_done());
}
TEST_CASE("error types are properly set for just_done", "[factories][just_done]") {
  // no errors sent by just_done
  check_err_types<type_array<>>(ex::just_done());
}
TEST_CASE("just_done advertises that it can call set_done", "[factories][just_done]") {
  check_sends_done<true>(ex::just_done());
}
