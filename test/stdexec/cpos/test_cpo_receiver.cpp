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
#include "test_common/receivers.hpp"

#include <string>
#include <climits>

namespace ex = stdexec;

struct recv_value {
  int* target_;

  friend void tag_invoke(ex::set_value_t, recv_value self, int val) noexcept { *self.target_ = val; }
  friend void tag_invoke(ex::set_error_t, recv_value self, int ec) noexcept { *self.target_ = -ec; }
  friend void tag_invoke(ex::set_stopped_t, recv_value self) noexcept { *self.target_ = INT_MAX; }
  friend empty_env tag_invoke(ex::get_env_t, const recv_value&) noexcept {
    return {};
  }
};
struct recv_rvalref {
  int* target_;

  friend void tag_invoke(ex::set_value_t, recv_rvalref&& self, int val) noexcept { *self.target_ = val; }
  friend void tag_invoke(ex::set_error_t, recv_rvalref&& self, int ec) noexcept { *self.target_ = -ec; }
  friend void tag_invoke(ex::set_stopped_t, recv_rvalref&& self) noexcept { *self.target_ = INT_MAX; }
  friend empty_env tag_invoke(ex::get_env_t, const recv_rvalref&) noexcept {
    return {};
  }
};
struct recv_ref {
  int* target_;

  friend void tag_invoke(ex::set_value_t, recv_ref& self, int val) noexcept { *self.target_ = val; }
  friend void tag_invoke(ex::set_error_t, recv_ref& self, int ec) noexcept { *self.target_ = -ec; }
  friend void tag_invoke(ex::set_stopped_t, recv_ref& self) noexcept { *self.target_ = INT_MAX; }
  friend empty_env tag_invoke(ex::get_env_t, const recv_ref&) noexcept {
    return {};
  }
};
struct recv_cref {
  int* target_;

  friend void tag_invoke(ex::set_value_t, const recv_cref& self, int val) noexcept { *self.target_ = val; }
  friend void tag_invoke(ex::set_error_t, const recv_cref& self, int ec) noexcept { *self.target_ = -ec; }
  friend void tag_invoke(ex::set_stopped_t, const recv_cref& self) noexcept { *self.target_ = INT_MAX; }
  friend empty_env tag_invoke(ex::get_env_t, const recv_cref&) noexcept {
    return {};
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
  std::exception_ptr ex;
  try {
    throw std::bad_alloc{};
  } catch (...) {
    ex = std::current_exception();
  }
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
  static_assert(std::invocable<ex::set_stopped_t, recv_value>, "cannot call set_stopped on recv_value");
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
  static_assert(std::invocable<ex::set_stopped_t, recv_ref&>, "cannot call set_stopped on recv_ref");
  int val = 0;
  recv_ref recv{&val};
  ex::set_stopped(recv);
  REQUIRE(val == INT_MAX);
}
TEST_CASE("can call set_stopped on a receiver with const ref type", "[cpo][cpo_receiver]") {
  static_assert(std::invocable<ex::set_stopped_t, recv_cref>, "cannot call set_stopped on recv_cref");
  int val = 0;
  ex::set_stopped(recv_cref{&val});
  REQUIRE(val == INT_MAX);
}

TEST_CASE("set_value can be called through tag_invoke", "[cpo][cpo_receiver]") {
  int val = 0;
  tag_invoke(ex::set_value, recv_value{&val}, 10);
  REQUIRE(val == 10);
}
TEST_CASE("set_error can be called through tag_invoke", "[cpo][cpo_receiver]") {
  int val = 0;
  tag_invoke(ex::set_error, recv_value{&val}, 10);
  REQUIRE(val == -10);
}
TEST_CASE("set_stopped can be called through tag_invoke", "[cpo][cpo_receiver]") {
  int val = 0;
  tag_invoke(ex::set_stopped, recv_value{&val});
  REQUIRE(val == INT_MAX);
}

TEST_CASE(
    "tag types can be deduced from set_value, set_error and set_stopped", "[cpo][cpo_receiver]") {
  static_assert(std::is_same_v<const ex::set_value_t, decltype(ex::set_value)>, "type mismatch");
  static_assert(std::is_same_v<const ex::set_error_t, decltype(ex::set_error)>, "type mismatch");
  static_assert(std::is_same_v<const ex::set_stopped_t, decltype(ex::set_stopped)>, "type mismatch");
}
