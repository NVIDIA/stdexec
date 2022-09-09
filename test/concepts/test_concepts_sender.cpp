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

namespace ex = std::execution;

struct oper {
  oper() = default;
  oper(oper&&) = delete;
  friend void tag_invoke(ex::start_t, oper&) noexcept {}
};

struct my_sender0 {
  using completion_signatures =
    ex::completion_signatures<             //
      ex::set_value_t(),                   //
      ex::set_error_t(std::exception_ptr), //
      ex::set_stopped_t()>;

  friend oper tag_invoke(ex::connect_t, my_sender0, empty_recv::recv0&& r) { return {}; }
};
TEST_CASE("type w/ proper types, is a sender & sender", "[concepts][sender]") {
  REQUIRE(ex::sender<my_sender0>);
  REQUIRE(ex::sender<my_sender0, empty_env>);

  REQUIRE(ex::sender_of<my_sender0, ex::set_value_t()>);
  REQUIRE(ex::sender_of<my_sender0, ex::set_error_t(std::exception_ptr)>);
  REQUIRE(ex::sender_of<my_sender0, ex::set_stopped_t()>);
  REQUIRE(ex::sender_of<my_sender0, ex::set_value_t(), empty_env>);
  REQUIRE(ex::sender_of<my_sender0, ex::set_error_t(std::exception_ptr), empty_env>);
  REQUIRE(ex::sender_of<my_sender0, ex::set_stopped_t(), empty_env>);
}
TEST_CASE(
    "sender that accepts a void sender models sender_to the given sender", "[concepts][sender]") {
  REQUIRE(ex::sender_to<my_sender0, empty_recv::recv0>);
}

struct my_sender_int {
  using completion_signatures =
    ex::completion_signatures<             //
      ex::set_value_t(int),                //
      ex::set_error_t(std::exception_ptr), //
      ex::set_stopped_t()>;

  friend oper tag_invoke(ex::connect_t, my_sender_int, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("my_sender_int is a sender & sender", "[concepts][sender]") {
  REQUIRE(ex::sender<my_sender_int>);
  REQUIRE(ex::sender<my_sender_int, empty_env>);
  REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int)>);
  REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int), empty_env>);
}
TEST_CASE("sender that accepts an int receiver models sender_to the given receiver",
    "[concepts][sender]") {
  REQUIRE(ex::sender_to<my_sender_int, empty_recv::recv_int>);
}

TEST_CASE("not all combinations of senders & receivers satisfy the sender_to concept",
    "[concepts][sender]") {
  REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv_int>);
  REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv0_ec>);
  REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv_int_ec>);
  REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv0>);
  REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv0_ec>);
  REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv_int_ec>);
}

TEST_CASE("can query completion signatures for a typed sender that sends nothing", "[concepts][sender]") {
  check_val_types<type_array<type_array<>>>(my_sender0{});
  check_err_types<type_array<std::exception_ptr>>(my_sender0{});
  check_sends_stopped<true>(my_sender0{});
  REQUIRE(ex::sender_of<my_sender0, ex::set_value_t()>);
}
TEST_CASE("can query completion signatures for a typed sender that sends int", "[concepts][sender]") {
  check_val_types<type_array<type_array<int>>>(my_sender_int{});
  check_err_types<type_array<std::exception_ptr>>(my_sender_int{});
  check_sends_stopped<true>(my_sender_int{});
  REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int)>);
}

struct multival_sender {
  using completion_signatures =
    ex::completion_signatures<      //
      ex::set_value_t(int, double), //
      ex::set_value_t(short, long), //
      ex::set_error_t(std::exception_ptr)>;

  friend oper tag_invoke(ex::connect_t, multival_sender, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("check completion signatures for sender that advertises multiple sets of values",
    "[concepts][sender]") {
  check_val_types<type_array<type_array<int, double>, type_array<short, long>>>(multival_sender{});
  check_err_types<type_array<std::exception_ptr>>(multival_sender{});
  check_sends_stopped<false>(multival_sender{});
  REQUIRE_FALSE(ex::sender_of<multival_sender, ex::set_value_t(int, double)>);
}

struct ec_sender {
  using completion_signatures =
    ex::completion_signatures<             //
      ex::set_value_t(),                   //
      ex::set_error_t(std::exception_ptr), //
      ex::set_error_t(int)>;

  friend oper tag_invoke(ex::connect_t, ec_sender, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("check completion signatures for sender that also supports error codes", "[concepts][sender]") {
  check_val_types<type_array<type_array<>>>(ec_sender{});
  check_err_types<type_array<std::exception_ptr, int>>(ec_sender{});
  check_sends_stopped<false>(ec_sender{});
  REQUIRE(ex::sender_of<ec_sender, ex::set_value_t()>);
}
