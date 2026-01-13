// /*
//  * Copyright (c) 2022 Lucian Radu Teodorescu
//  *
//  * Licensed under the Apache License Version 2.0 with LLVM Exceptions
//  * (the "License"); you may not use this file except in compliance with
//  * the License. You may obtain a copy of the License at
//  *
//  *   https://llvm.org/LICENSE.txt
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>

namespace ex = STDEXEC;

struct recv_no_set_value {
  using receiver_concept = STDEXEC::receiver_t;

  void set_stopped() noexcept {
  }
  void set_error(std::exception_ptr) noexcept {
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};

struct recv_set_value_except {
  using receiver_concept = STDEXEC::receiver_t;

  void set_value() {
  }
  void set_stopped() noexcept {
  }
  void set_error(std::exception_ptr) noexcept {
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};

struct recv_set_value_noexcept {
  using receiver_concept = STDEXEC::receiver_t;

  void set_value() noexcept {
  }
  void set_stopped() noexcept {
  }
  void set_error(std::exception_ptr) noexcept {
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};

struct recv_set_error_except {
  using receiver_concept = STDEXEC::receiver_t;

  void set_value() noexcept {
  }
  void set_stopped() noexcept {
  }
  void set_error(std::exception_ptr) {
    throw std::logic_error{"err"};
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};
struct recv_set_stopped_except {
  using receiver_concept = STDEXEC::receiver_t;

  void set_value() noexcept {
  }
  void set_stopped() {
    throw std::logic_error{"err"};
  }
  void set_error(std::exception_ptr) noexcept {
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};

struct recv_non_movable {
  using receiver_concept = STDEXEC::receiver_t;

  recv_non_movable() = default;
  ~recv_non_movable() = default;
  recv_non_movable(recv_non_movable&&) = delete;
  recv_non_movable& operator=(recv_non_movable&&) = delete;
  recv_non_movable(const recv_non_movable&) = default;
  recv_non_movable& operator=(const recv_non_movable&) = default;

  void set_value() noexcept {
  }
  void set_stopped() noexcept {
  }
  void set_error(std::exception_ptr) noexcept {
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept {
    return {};
  }
};

TEST_CASE("receiver types satisfy the receiver concept", "[concepts][receiver]") {
  using namespace empty_recv;

  REQUIRE(ex::receiver<recv0>);
  REQUIRE(ex::receiver<recv_int>);
  REQUIRE(ex::receiver<recv0_ec>);
  REQUIRE(ex::receiver<recv_int_ec>);
  REQUIRE(ex::receiver<expect_void_receiver<>>);
  REQUIRE(ex::receiver<expect_void_receiver_ex>);
  REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, int>>);
  REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, double>>);
  REQUIRE(ex::receiver<expect_stopped_receiver<>>);
  REQUIRE(ex::receiver<expect_stopped_receiver_ex<>>);
  REQUIRE(ex::receiver<expect_error_receiver<>>);
  REQUIRE(ex::receiver<expect_error_receiver_ex<std::error_code>>);
  REQUIRE(ex::receiver<logging_receiver>);
}

TEST_CASE("receiver types satisfy the receiver_of concept", "[concepts][receiver]") {
  using namespace empty_recv;

  REQUIRE(ex::receiver_of<recv_int, ex::completion_signatures<ex::set_value_t(int)>>);
  REQUIRE(ex::receiver_of<recv0_ec, ex::completion_signatures<ex::set_error_t(std::error_code)>>);
  REQUIRE(
    ex::receiver_of<recv_int_ec, ex::completion_signatures<ex::set_error_t(std::error_code)>>);
  REQUIRE(ex::receiver_of<recv_int_ec, ex::completion_signatures<ex::set_value_t(int)>>);
  REQUIRE(
    ex::receiver_of<
      expect_value_receiver<ex::env<>, int>,
      ex::completion_signatures<ex::set_value_t(int)>
    >);
  REQUIRE(
    ex::receiver_of<
      expect_value_receiver<ex::env<>, double>,
      ex::completion_signatures<ex::set_value_t(double)>
    >);
  REQUIRE(
    ex::receiver_of<expect_stopped_receiver<>, ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(
    ex::receiver_of<expect_stopped_receiver_ex<>, ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(
    ex::receiver_of<expect_error_receiver<>, ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(
    ex::receiver_of<
      expect_error_receiver_ex<std::error_code>,
      ex::completion_signatures<ex::set_value_t(char)>
    >);
  REQUIRE(ex::receiver_of<logging_receiver, ex::completion_signatures<ex::set_value_t(char)>>);
}

TEST_CASE(
  "receiver type w/o set_value models receiver but not receiver_of",
  "[concepts][receiver]") {
  REQUIRE(ex::receiver<recv_no_set_value>);
  REQUIRE(!ex::receiver_of<recv_no_set_value, ex::completion_signatures<ex::set_value_t()>>);
}

TEST_CASE("type with set_value noexcept is a receiver", "[concepts][receiver]") {
  REQUIRE(ex::receiver<recv_set_value_noexcept>);
  REQUIRE(ex::receiver_of<recv_set_value_noexcept, ex::completion_signatures<ex::set_value_t()>>);
}

TEST_CASE("non-movable type is not a receiver", "[concepts][receiver]") {
  REQUIRE(!ex::receiver<recv_non_movable>);
  REQUIRE(!ex::receiver_of<recv_non_movable, ex::completion_signatures<ex::set_value_t()>>);
}
