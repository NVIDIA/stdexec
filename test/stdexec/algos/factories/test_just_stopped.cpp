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

namespace {

#if !STDEXEC_GCC() || STDEXEC_GCC_VERSION >= 12'00
  constexpr int test_constexpr() noexcept {
    struct receiver {
      using receiver_concept = ex::receiver_t;
      constexpr void set_stopped() && noexcept {
        invoked = true;
      }
      bool& invoked;
    };
    bool invoked = false;
    auto op = ex::connect(ex::just_stopped(), receiver{invoked});
    ex::start(op);
    return invoked;
  }
  static_assert(test_constexpr());
#endif

  TEST_CASE("Simple test for just_stopped", "[factories][just_stopped]") {
    auto op = ex::connect(ex::just_stopped(), expect_stopped_receiver{});
    ex::start(op);
    // receiver ensures that set_stopped() is called
  }

  TEST_CASE("just_stopped returns a sender", "[factories][just_stopped]") {
    using t = decltype(ex::just_stopped());
    static_assert(ex::sender<t>, "ex::just_stopped must return a sender");
    REQUIRE(ex::sender<t>);
  }

  TEST_CASE("just_stopped returns a typed sender", "[factories][just_stopped]") {
    using t = decltype(ex::just_stopped());
    static_assert(ex::sender_in<t, ex::env<>>, "ex::just_stopped must return a sender");
  }

  TEST_CASE("value types are properly set for just_stopped", "[factories][just_stopped]") {
    check_val_types<ex::__mset<>>(ex::just_stopped());
  }

  TEST_CASE("error types are properly set for just_stopped", "[factories][just_stopped]") {
    // no errors sent by just_stopped
    check_err_types<ex::__mset<>>(ex::just_stopped());
  }

  TEST_CASE("just_stopped advertises that it can call set_stopped", "[factories][just_stopped]") {
    check_sends_stopped<true>(ex::just_stopped());
  }
} // namespace
