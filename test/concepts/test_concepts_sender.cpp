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
using set_value_t = std::decay_t<decltype(ex::set_value)>;
using set_error_t = std::decay_t<decltype(ex::set_error)>;
using set_stopped_t = std::decay_t<decltype(ex::set_stopped)>;

struct oper {
  friend void tag_invoke(ex::start_t, oper&) noexcept {}
};

struct empty_sender : ex::sender_base {};

TEST_CASE("type deriving from sender_base, w/o start, is a sender", "[concepts][sender]") {
  REQUIRE(ex::sender<empty_sender>);
}
TEST_CASE(
    "type deriving from sender_base, w/o start, doesn't model typed_sender", "[concepts][sender]") {
  REQUIRE_FALSE(ex::typed_sender<empty_sender, empty_env>);
}
TEST_CASE(
    "type deriving from sender_base, w/o start, doesn't model sender_to", "[concepts][sender]") {
  REQUIRE_FALSE(ex::sender_to<empty_sender, empty_recv::recv0>);
}

struct simple_sender : ex::sender_base {
  template <typename R>
  friend oper tag_invoke(ex::connect_t, simple_sender, R) {
    return {};
  }
};

TEST_CASE("type deriving from sender_base models sender and sender_to", "[concepts][sender]") {
  REQUIRE(ex::sender<simple_sender>);
  REQUIRE(ex::sender_to<simple_sender, empty_recv::recv0>);
}
TEST_CASE("type deriving from sender_base, doesn't model typed_sender", "[concepts][sender]") {
  REQUIRE_FALSE(ex::typed_sender<simple_sender, empty_env>);
}

struct my_sender0 : ex::completion_signatures<           //
                        set_value_t(),                   //
                        set_error_t(std::exception_ptr), //
                        set_stopped_t()> {

  friend oper tag_invoke(ex::connect_t, my_sender0, empty_recv::recv0&& r) { return {}; }
};
TEST_CASE("type w/ proper types, is a sender & typed_sender", "[concepts][sender]") {
  REQUIRE(ex::sender<my_sender0>);
  REQUIRE(ex::typed_sender<my_sender0, empty_env>);
}
TEST_CASE(
    "sender that accepts a void sender models sender_to the given sender", "[concepts][sender]") {
  REQUIRE(ex::sender_to<my_sender0, empty_recv::recv0>);
}

struct my_sender_int : ex::completion_signatures<           //
                           set_value_t(int),                //
                           set_error_t(std::exception_ptr), //
                           set_stopped_t()> {

  friend oper tag_invoke(ex::connect_t, my_sender_int, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("my_sender_int is a sender & typed_sender", "[concepts][sender]") {
  REQUIRE(ex::sender<my_sender_int>);
  REQUIRE(ex::typed_sender<my_sender_int, empty_env>);
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

TEST_CASE("can apply sender traits to invalid sender", "[concepts][sender]") {
  REQUIRE(sizeof(ex::sender_traits_t<empty_sender, empty_env>) <= sizeof(int));
}

TEST_CASE("can apply sender traits to senders deriving from sender_base", "[concepts][sender]") {
  REQUIRE(sizeof(ex::sender_traits_t<simple_sender, empty_env>) <= sizeof(int));
}

TEST_CASE("can query sender traits for a typed sender that sends nothing", "[concepts][sender]") {
  check_val_types<type_array<type_array<>>>(my_sender0{});
  check_err_types<type_array<std::exception_ptr>>(my_sender0{});
  check_sends_stopped<true>(my_sender0{});
}
TEST_CASE("can query sender traits for a typed sender that sends int", "[concepts][sender]") {
  check_val_types<type_array<type_array<int>>>(my_sender_int{});
  check_err_types<type_array<std::exception_ptr>>(my_sender_int{});
  check_sends_stopped<true>(my_sender_int{});
}

struct multival_sender : ex::completion_signatures<    //
                             set_value_t(int, double), //
                             set_value_t(short, long), //
                             set_error_t(std::exception_ptr)> {

  friend oper tag_invoke(ex::connect_t, multival_sender, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("check sender traits for sender that advertises multiple sets of values",
    "[concepts][sender]") {
  check_val_types<type_array<type_array<int, double>, type_array<short, long>>>(multival_sender{});
  check_err_types<type_array<std::exception_ptr>>(multival_sender{});
  check_sends_stopped<false>(multival_sender{});
}

struct ec_sender : ex::completion_signatures<           //
                       set_value_t(),                   //
                       set_error_t(std::exception_ptr), //
                       set_error_t(int)> {
  friend oper tag_invoke(ex::connect_t, ec_sender, empty_recv::recv_int&& r) { return {}; }
};
TEST_CASE("check sender traits for sender that also supports error codes", "[concepts][sender]") {
  check_val_types<type_array<type_array<>>>(ec_sender{});
  check_err_types<type_array<std::exception_ptr, int>>(ec_sender{});
  check_sends_stopped<false>(ec_sender{});
}
