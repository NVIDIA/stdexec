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
#include <exec/static_thread_pool.hpp>

#include <chrono>

namespace ex = stdexec;

using namespace std::chrono_literals;

TEST_CASE("transfer returns a sender", "[adaptors][transfer]") {
  auto snd = ex::transfer(ex::just(13), inline_scheduler{});
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE("transfer with environment returns a sender", "[adaptors][transfer]") {
  auto snd = ex::transfer(ex::just(13), inline_scheduler{});
  static_assert(ex::sender_in<decltype(snd), empty_env>);
  (void) snd;
}

TEST_CASE("transfer simple example", "[adaptors][transfer]") {
  auto snd = ex::transfer(ex::just(13), inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer can be piped", "[adaptors][transfer]") {
  // Just transfer a value to the impulse scheduler
  ex::scheduler auto sched = impulse_scheduler{};
  ex::sender auto snd = ex::just(13) | ex::transfer(sched);
  // Start the operation
  int res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
  ex::start(op);

  // The value will be available when the scheduler will execute the next operation
  REQUIRE(res == 0);
  sched.start_next();
  REQUIRE(res == 13);
}

TEST_CASE("transfer calls the receiver when the scheduler dictates", "[adaptors][transfer]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(snd, expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE("transfer calls the given sender when the scheduler dictates", "[adaptors][transfer]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::then([&]() -> int {
                    called = true;
                    return 19;
                  });

  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::transfer(std::move(snd_base), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
  ex::start(op);
  // The sender is started, even if the scheduler hasn't yet triggered
  CHECK(called);
  // ... but didn't send the value to the receiver yet
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(called);
  CHECK(recv_value == 19);
}

TEST_CASE("transfer works when changing threads", "[adaptors][transfer]") {
  exec::static_thread_pool pool{2};
  std::atomic<bool> called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::transfer(ex::just(), pool.get_scheduler()) //
                        | ex::then([&] { called.store(true); });
    ex::start_detached(std::move(snd));
  }
  // wait for the work to be executed, with timeout
  // perform a poor-man's sync
  // NOTE: it's a shame that the `join` method in static_thread_pool is not public
  for (int i = 0; i < 1000 && !called.load(); i++) {
    std::this_thread::sleep_for(1ms);
  }
  // the work should be executed
  REQUIRE(called);
}

TEST_CASE("transfer can be called with rvalue ref scheduler", "[adaptors][transfer]") {
  auto snd = ex::transfer(ex::just(13), inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer can be called with const ref scheduler", "[adaptors][transfer]") {
  const inline_scheduler sched;
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer can be called with ref scheduler", "[adaptors][transfer]") {
  inline_scheduler sched;
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("transfer forwards set_error calls", "[adaptors][transfer]") {
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("transfer forwards set_error calls of other types", "[adaptors][transfer]") {
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("transfer forwards set_stopped calls", "[adaptors][transfer]") {
  stopped_scheduler sched{};
  auto snd = ex::transfer(ex::just(13), sched);
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE(
  "transfer has the values_type corresponding to the given values",
  "[adaptors][transfer]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(ex::transfer(ex::just(1), sched));
  check_val_types<type_array<type_array<int, double>>>(ex::transfer(ex::just(3, 0.14), sched));
  check_val_types<type_array<type_array<int, double, std::string>>>(
    ex::transfer(ex::just(3, 0.14, std::string{"pi"}), sched));
}

TEST_CASE("transfer keeps error_types from scheduler's sender", "[adaptors][transfer]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<>>(ex::transfer(ex::just(1), sched1));
  check_err_types<type_array<std::exception_ptr>>(ex::transfer(ex::just(2), sched2));
  check_err_types<type_array<int>>(ex::transfer(ex::just(3), sched3));
}

TEST_CASE(
  "transfer sends an exception_ptr if value types are potentially throwing when copied",
  "[adaptors][transfer]") {
  inline_scheduler sched{};

  check_err_types<type_array<std::exception_ptr>>(
    ex::transfer(ex::just(potentially_throwing{}), sched));
}

TEST_CASE("transfer keeps sends_stopped from scheduler's sender", "[adaptors][transfer]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(ex::transfer(ex::just(1), sched1));
  check_sends_stopped<true>(ex::transfer(ex::just(2), sched2));
  check_sends_stopped<true>(ex::transfer(ex::just(3), sched3));
}

struct val_type1 {
  int val_;
};

struct val_type2 {
  int val_;
};

struct val_type3 {
  int val_;
};

using just_val1_sender_t = decltype(ex::just(val_type1{0}));
using just_val2_sender_t = decltype(ex::just(val_type2{0}));
using just_val3_sender_t = decltype(ex::transfer_just(impulse_scheduler{}, val_type3{0}));

// Customization of transfer
// Return a different sender when we invoke this custom defined transfer implementation
auto tag_invoke(decltype(ex::transfer), just_val1_sender_t, inline_scheduler sched) {
  return ex::just(val_type1{53});
}

// Customization of schedule_from
// Return a different sender when we invoke this custom defined transfer implementation
auto tag_invoke(decltype(ex::schedule_from), inline_scheduler sched, just_val2_sender_t) {
  return ex::just(val_type2{59});
}

// Customization of transfer with scheduler
// Return a different sender when we invoke this custom defined transfer implementation
auto tag_invoke(
  decltype(ex::transfer),
  impulse_scheduler /*sched_src*/,
  just_val3_sender_t,
  inline_scheduler sched_dest) {
  return ex::transfer_just(sched_dest, val_type3{61});
}

TEST_CASE("transfer can be customized", "[adaptors][transfer]") {
  // The customization will return a different value
  auto snd = ex::transfer(ex::just(val_type1{1}), inline_scheduler{});
  val_type1 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
  ex::start(op);
  REQUIRE(res.val_ == 53);
}

TEST_CASE("transfer follows schedule_from customization", "[adaptors][transfer]") {
  // The schedule_from customization will return a different value
  auto snd = ex::transfer(ex::just(val_type2{2}), inline_scheduler{});
  val_type2 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
  ex::start(op);
  REQUIRE(res.val_ == 59);
}

TEST_CASE("transfer can be customized with two schedulers", "[adaptors][transfer]") {
  // The customization will return a different value
  ex::scheduler auto sched_src = impulse_scheduler{};
  ex::scheduler auto sched_dest = inline_scheduler{};
  auto snd = ex::transfer_just(sched_src, val_type3{1}) //
           | ex::transfer(sched_dest);
  val_type3 res{0};
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
  ex::start(op);
  // we are not using impulse_scheduler anymore, so the value should be available
  REQUIRE(res.val_ == 61);
}
