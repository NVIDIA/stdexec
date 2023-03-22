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

TEST_CASE("schedule_from returns a sender", "[adaptors][schedule_from]") {
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE("schedule_from with environment returns a sender", "[adaptors][schedule_from]") {
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender_in<decltype(snd), empty_env>);
  (void) snd;
}

TEST_CASE("schedule_from simple example", "[adaptors][schedule_from]") {
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE(
  "schedule_from calls the receiver when the scheduler dictates",
  "[adaptors][schedule_from]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(snd, expect_value_receiver_ex{recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE(
  "schedule_from calls the given sender when the scheduler dictates",
  "[adaptors][schedule_from]") {
  bool called{false};
  auto snd_base = ex::just() //
                | ex::then([&]() -> int {
                    called = true;
                    return 19;
                  });

  int recv_value{0};
  impulse_scheduler sched;
  auto snd = ex::schedule_from(sched, std::move(snd_base));
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

TEST_CASE("schedule_from works when changing threads", "[adaptors][schedule_from]") {
  exec::static_thread_pool pool{2};
  std::atomic<bool> called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::schedule_from(pool.get_scheduler(), ex::just()) //
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

struct non_default_constructible {
  int x;

  non_default_constructible(int x)
    : x(x) {
  }

  friend bool
    operator==(non_default_constructible const & lhs, non_default_constructible const & rhs) {
    return lhs.x == rhs.x;
  }
};

TEST_CASE("schedule_from can accept non-default constructible types", "[adaptors][schedule_from]") {
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(non_default_constructible{13}));
  auto op = ex::connect(std::move(snd), expect_value_receiver{non_default_constructible{13}});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("schedule_from can be called with rvalue ref scheduler", "[adaptors][schedule_from]") {
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("schedule_from can be called with const ref scheduler", "[adaptors][schedule_from]") {
  const inline_scheduler sched;
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("schedule_from can be called with ref scheduler", "[adaptors][schedule_from]") {
  inline_scheduler sched;
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("schedule_from forwards set_error calls", "[adaptors][schedule_from]") {
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("schedule_from forwards set_error calls of other types", "[adaptors][schedule_from]") {
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}

TEST_CASE("schedule_from forwards set_stopped calls", "[adaptors][schedule_from]") {
  stopped_scheduler sched{};
  auto snd = ex::schedule_from(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE(
  "schedule_from has the values_type corresponding to the given values",
  "[adaptors][schedule_from]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(ex::schedule_from(sched, ex::just(1)));
  check_val_types<type_array<type_array<int, double>>>(ex::schedule_from(sched, ex::just(3, 0.14)));
  check_val_types<type_array<type_array<int, double, std::string>>>(
    ex::schedule_from(sched, ex::just(3, 0.14, std::string{"pi"})));
}

TEST_CASE("schedule_from keeps error_types from scheduler's sender", "[adaptors][schedule_from]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<>>(ex::schedule_from(sched1, ex::just(1)));
  check_err_types<type_array<std::exception_ptr>>(ex::schedule_from(sched2, ex::just(2)));
  check_err_types<type_array<int>>(ex::schedule_from(sched3, ex::just(3)));
}

TEST_CASE(
  "schedule_from sends an exception_ptr if value types are potentially throwing when copied",
  "[adaptors][schedule_from]") {
  inline_scheduler sched{};

  check_err_types<type_array<std::exception_ptr>>(
    ex::schedule_from(sched, ex::just(potentially_throwing{})));
}

TEST_CASE(
  "schedule_from keeps sends_stopped from scheduler's sender",
  "[adaptors][schedule_from]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(ex::schedule_from(sched1, ex::just(1)));
  check_sends_stopped<true>(ex::schedule_from(sched2, ex::just(2)));
  check_sends_stopped<true>(ex::schedule_from(sched3, ex::just(3)));
}

using just_string_sender_t = decltype(ex::just(std::string{}));

// Customization of schedule_from
// Return a different sender
auto tag_invoke(decltype(ex::schedule_from), inline_scheduler sched, just_string_sender_t) {
  return ex::just(std::string{"hijacked"});
}

TEST_CASE("schedule_from can be customized", "[adaptors][schedule_from]") {
  // The customization will return a different value
  auto snd = ex::schedule_from(inline_scheduler{}, ex::just(std::string{"transfer"}));
  auto op = ex::connect(std::move(snd), expect_value_receiver(std::string{"hijacked"}));
  ex::start(op);
}
