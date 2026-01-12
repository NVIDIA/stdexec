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

#include "test_common/receivers.hpp"
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <climits>
#include <string>

namespace ex = STDEXEC;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")

namespace {

  struct recv_value {
    int* target_;

    void set_value(int val) noexcept {
      *target_ = val;
    }

    void set_error(int ec) noexcept {
      *target_ = -ec;
    }

    void set_stopped() noexcept {
      *target_ = INT_MAX;
    }
  };

  struct recv_rvalref {
    int* target_;

    void set_value(int val) noexcept {
      *target_ = val;
    }

    void set_error(int ec) noexcept {
      *target_ = -ec;
    }

    void set_stopped() noexcept {
      *target_ = INT_MAX;
    }
  };

  struct recv_ref {
    int* target_;

    void set_value(int val) noexcept {
      *target_ = val;
    }

    void set_error(int ec) noexcept {
      *target_ = -ec;
    }

    void set_stopped() noexcept {
      *target_ = INT_MAX;
    }
  };

  struct recv_cref {
    int* target_;

    void set_value(int val) noexcept {
      *target_ = val;
    }

    void set_error(int ec) noexcept {
      *target_ = -ec;
    }

    void set_stopped() noexcept {
      *target_ = INT_MAX;
    }
  };

  TEST_CASE("can call set_value on a void receiver", "[cpo][cpo_receiver]") {
    ex::set_value(expect_void_receiver{});
  }

  TEST_CASE("can call set_value on a int receiver", "[cpo][cpo_receiver]") {
    ex::set_value(expect_value_receiver{10}, 10);
  }

  TEST_CASE("can call set_value on a string receiver", "[cpo][cpo_receiver]") {
    ex::set_value(expect_value_receiver{std::string{"hello"}}, std::string{"hello"});
  }

  TEST_CASE("can call set_stopped on a receiver", "[cpo][cpo_receiver]") {
    ex::set_stopped(expect_stopped_receiver{});
  }

  TEST_CASE("can call set_error on a receiver", "[cpo][cpo_receiver]") {
    std::exception_ptr ex = std::make_exception_ptr(std::bad_alloc{});
    ex::set_error(expect_error_receiver{}, ex);
  }

  TEST_CASE("can call set_error with an error code on a receiver", "[cpo][cpo_receiver]") {
    std::error_code errCode{100, std::generic_category()};
    ex::set_error(expect_error_receiver{errCode}, errCode);
  }

  TEST_CASE("set_value with a value passes the value to the receiver", "[cpo][cpo_receiver]") {
    ex::set_value(expect_value_receiver{10}, 10);
  }

  TEST_CASE("can call set_value on a receiver with plain value type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_value_t, recv_value, int>, "cannot call set_value on recv_value");
    int val = 0;
    ex::set_value(recv_value{&val}, 10);
    REQUIRE(val == 10);
  }

  TEST_CASE("can call set_value on a receiver with r-value ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_value_t, recv_rvalref, int>, "cannot call set_value on recv_rvalref");
    int val = 0;
    ex::set_value(recv_rvalref{&val}, 10);
    REQUIRE(val == 10);
  }

  TEST_CASE("can call set_value on a receiver with ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_value_t, recv_ref&, int>, "cannot call set_value on recv_ref");
    int val = 0;
    recv_ref recv{&val};
    ex::set_value(recv, 10);
    REQUIRE(val == 10);
  }

  TEST_CASE("can call set_value on a receiver with const ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_value_t, recv_cref, int>, "cannot call set_value on recv_cref");
    int val = 0;
    ex::set_value(recv_cref{&val}, 10);
    REQUIRE(val == 10);
  }

  TEST_CASE("can call set_error on a receiver with plain value type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_error_t, recv_value, int>, "cannot call set_error on recv_value");
    int val = 0;
    ex::set_error(recv_value{&val}, 10);
    REQUIRE(val == -10);
  }

  TEST_CASE("can call set_error on a receiver with r-value ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_error_t, recv_rvalref, int>, "cannot call set_error on recv_rvalref");
    int val = 0;
    ex::set_error(recv_rvalref{&val}, 10);
    REQUIRE(val == -10);
  }

  TEST_CASE("can call set_error on a receiver with ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_error_t, recv_ref&, int>, "cannot call set_error on recv_ref");
    int val = 0;
    recv_ref recv{&val};
    ex::set_error(recv, 10);
    REQUIRE(val == -10);
  }

  TEST_CASE("can call set_error on a receiver with const ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_error_t, recv_cref, int>, "cannot call set_error on recv_cref");
    int val = 0;
    ex::set_error(recv_cref{&val}, 10);
    REQUIRE(val == -10);
  }

  TEST_CASE("can call set_stopped on a receiver with plain value type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_stopped_t, recv_value>, "cannot call set_stopped on recv_value");
    int val = 0;
    ex::set_stopped(recv_value{&val});
    REQUIRE(val == INT_MAX);
  }

  TEST_CASE("can call set_stopped on a receiver with r-value ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_stopped_t, recv_rvalref>, "cannot call set_stopped on recv_rvalref");
    int val = 0;
    ex::set_stopped(recv_rvalref{&val});
    REQUIRE(val == INT_MAX);
  }

  TEST_CASE("can call set_stopped on a receiver with ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_stopped_t, recv_ref&>, "cannot call set_stopped on recv_ref");
    int val = 0;
    recv_ref recv{&val};
    ex::set_stopped(recv);
    REQUIRE(val == INT_MAX);
  }

  TEST_CASE("can call set_stopped on a receiver with const ref type", "[cpo][cpo_receiver]") {
    static_assert(
      std::invocable<ex::set_stopped_t, recv_cref>, "cannot call set_stopped on recv_cref");
    int val = 0;
    ex::set_stopped(recv_cref{&val});
    REQUIRE(val == INT_MAX);
  }

  TEST_CASE(
    "tag types can be deduced from set_value, set_error and set_stopped",
    "[cpo][cpo_receiver]") {
    static_assert(std::is_same_v<const ex::set_value_t, decltype(ex::set_value)>, "type mismatch");
    static_assert(std::is_same_v<const ex::set_error_t, decltype(ex::set_error)>, "type mismatch");
    static_assert(
      std::is_same_v<const ex::set_stopped_t, decltype(ex::set_stopped)>, "type mismatch");
  }
} // namespace

STDEXEC_PRAGMA_POP()
