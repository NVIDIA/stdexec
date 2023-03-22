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
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

TEST_CASE("transfer_just returns a sender", "[factories][transfer_just]") {
  auto snd = ex::transfer_just(inline_scheduler{}, 13);
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE("transfer_just with environment returns a sender", "[factories][transfer_just]") {
  auto snd = ex::transfer_just(inline_scheduler{}, 13);
  static_assert(ex::sender_in<decltype(snd), empty_env>);
  (void) snd;
}

TEST_CASE("transfer_just simple example", "[factories][transfer_just]") {
  inline_scheduler sched;
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE(
  "transfer_just calls the receiver when the scheduler dictates",
  "[factories][transfer_just]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(snd, expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE("transfer_just can be called with value type scheduler", "[factories][transfer_just]") {
  auto snd = ex::transfer_just(inline_scheduler{}, 13);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer_just can be called with rvalue ref scheduler", "[factories][transfer_just]") {
  auto snd = ex::transfer_just(inline_scheduler{}, 13);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer_just can be called with const ref scheduler", "[factories][transfer_just]") {
  const inline_scheduler sched;
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer_just can be called with ref scheduler", "[factories][transfer_just]") {
  inline_scheduler sched;
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer_just forwards set_error calls", "[factories][transfer_just]") {
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("transfer_just forwards set_error calls of other types", "[factories][transfer_just]") {
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("transfer_just forwards set_stopped calls", "[factories][transfer_just]") {
  stopped_scheduler sched{};
  auto snd = ex::transfer_just(sched, 13);
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE(
  "transfer_just has the values_type corresponding to the given values",
  "[factories][transfer_just]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(ex::transfer_just(sched, 1));
  check_val_types<type_array<type_array<int, double>>>(ex::transfer_just(sched, 3, 0.14));
  check_val_types<type_array<type_array<int, double, std::string>>>(
    ex::transfer_just(sched, 3, 0.14, std::string{"pi"}));
}

TEST_CASE("transfer_just keeps error_types from scheduler's sender", "[factories][transfer_just]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<>>(ex::transfer_just(sched1, 1));
  check_err_types<type_array<std::exception_ptr>>(ex::transfer_just(sched2, 2));
  check_err_types<type_array<int>>(ex::transfer_just(sched3, 3));
}

TEST_CASE(
  "transfer_just sends an exception_ptr if value types are potentially throwing when copied",
  "[factories][transfer_just]") {
  inline_scheduler sched{};

  check_err_types<type_array<std::exception_ptr>>(ex::transfer_just(sched, potentially_throwing{}));
}

TEST_CASE(
  "transfer_just keeps sends_stopped from scheduler's sender",
  "[factories][transfer_just]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(ex::transfer_just(sched1, 1));
  check_sends_stopped<true>(ex::transfer_just(sched2, 2));
  check_sends_stopped<true>(ex::transfer_just(sched3, 3));
}

TEST_CASE("transfer_just advertises its completion scheduler", "[factories][transfer_just]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  REQUIRE(
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::transfer_just(sched1, 1)))
    == sched1);
  REQUIRE(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(ex::transfer_just(sched1, 1)))
    == sched1);

  REQUIRE(
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::transfer_just(sched2, 2)))
    == sched2);
  REQUIRE(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(ex::transfer_just(sched2, 2)))
    == sched2);

  REQUIRE(
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::transfer_just(sched3, 3)))
    == sched3);
  REQUIRE(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(ex::transfer_just(sched3, 3)))
    == sched3);
}

// Modify the value when we invoke this custom defined transfer_just implementation
auto tag_invoke(decltype(ex::transfer_just), inline_scheduler sched, std::string value) {
  return ex::transfer(ex::just("Hello, " + value), sched);
}

TEST_CASE("transfer_just can be customized", "[factories][transfer_just]") {
  // The customization will alter the value passed in
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"world"});
  std::string res;
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
  ex::start(op);
  REQUIRE(res == "Hello, world");
}
