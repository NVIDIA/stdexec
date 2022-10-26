/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2021-2022 NVIDIA Corporation
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
#include <exec/env.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;

using namespace std::chrono_literals;

TEST_CASE("split returns a sender", "[adaptors][split]") {
  auto snd = ex::split(ex::just(19));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("split with environment returns a sender", "[adaptors][split]") {
  auto snd = ex::split(ex::just(19));
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("split simple example", "[adaptors][split]") {
  auto snd = ex::split(ex::just(19));
  auto op1 = ex::connect(snd, expect_value_receiver{19});
  auto op2 = ex::connect(snd, expect_value_receiver{19});
  ex::start(op1);
  ex::start(op2);
  // The receiver will ensure that the right value is produced
}
TEST_CASE("split executes predecessor sender once", "[adaptors][split]") {
  SECTION("when parameters are passed") {
    int counter{};
    auto snd = ex::split(ex::just() | ex::then([&]{ counter++; return counter; }));
    auto op1 = ex::connect(snd, expect_value_receiver{1});
    auto op2 = ex::connect(snd, expect_value_receiver{1});
    ex::start(op1);
    ex::start(op2);
    // The receiver will ensure that the right value is produced

    REQUIRE( counter == 1 );
  }

  SECTION("without parameters") {
    int counter{};
    auto snd = ex::split(ex::just() | ex::then([&] { counter++; }));
    stdexec::sync_wait(snd | ex::then([]{}));
    stdexec::sync_wait(snd | ex::then([]{}));
    REQUIRE( counter == 1 );
  }
}
TEST_CASE("split passes lvalue references", "[adaptors][split]") {
  auto split = ex::split(ex::just(42));
  using split_t = decltype(split);
  using value_t = ex::value_types_of_t<split_t, stdexec::__empty_env, std::tuple>;
  static_assert(std::is_same_v<value_t, std::variant<std::tuple<const int&>>>);

  auto then = split | ex::then([] (const int &cval) {
    int &val = const_cast<int&>(cval);
    const int prev_val = val;
    val /= 2;
    return prev_val;
  });

  auto op1 = ex::connect(std::move(then), expect_value_receiver{42});
  auto op2 = ex::connect(split, expect_value_receiver{21});
  ex::start(op1);
  ex::start(op2);
  // The receiver will ensure that the right value is produced
}
TEST_CASE("split forwards errors", "[adaptors][split]") {
  SECTION("of exception_ptr type")
  {
    auto split = ex::split(ex::just_error(std::exception_ptr{}));
    using split_t = decltype(split);
    using error_t = ex::error_types_of_t<split_t, stdexec::__empty_env, std::variant>;
    static_assert(std::is_same_v<error_t, std::variant<const std::exception_ptr&>>);

    auto op = ex::connect(split, expect_error_receiver{});
    ex::start(op);
    // The receiver will ensure that the right value is produced
  }

  SECTION("of any type")
  {
    auto split = ex::split(ex::just_error(42));
    using split_t = decltype(split);
    using error_t = ex::error_types_of_t<split_t, stdexec::__empty_env, std::variant>;
    static_assert(std::is_same_v<error_t, std::variant<const std::exception_ptr&, const int&>>);

    auto op = ex::connect(split, expect_error_receiver<int>{});
    ex::start(op);
  }
}
TEST_CASE("split forwards stop signal", "[adaptors][split]") {
  auto split = ex::split(ex::just_stopped());
  using split_t = decltype(split);
  static_assert(ex::sends_stopped<split_t, stdexec::__empty_env>);

  auto op = ex::connect(split, expect_stopped_receiver{});
  ex::start(op);
  // The receiver will ensure that the right value is produced
}
TEST_CASE("split forwards external stop signal (1)", "[adaptors][split]") {
  stdexec::in_place_stop_source ssource;
  bool called = false;
  int counter{};
  auto split = ex::split(ex::just() | ex::then([&]{ called = true; }));
  auto sndr =
    exec::write(
      ex::upon_stopped(
        split,
        [&]{ ++counter; return 42; }),
      exec::with(ex::get_stop_token, ssource.get_token()));
  auto op1 = ex::connect(std::move(sndr), expect_value_receiver{42});
  auto op2 = ex::connect(sndr, expect_value_receiver{42});
  ssource.request_stop();
  REQUIRE( counter == 0 );
  ex::start(op1);
  REQUIRE( counter == 1 );
  REQUIRE( !called );
  ex::start(op2);
  REQUIRE( counter == 2 );
}
TEST_CASE("split forwards external stop signal (2)", "[adaptors][split]") {
  stdexec::in_place_stop_source ssource;
  bool called = false;
  int counter{};
  auto split = ex::split(ex::just() | ex::then([&]{ called = true; return 7; }));
  auto sndr =
    exec::write(
      ex::upon_stopped(
        split,
        [&]{ ++counter; return 42; }),
      exec::with(ex::get_stop_token, ssource.get_token()));
  auto op1 = ex::connect(sndr, expect_value_receiver{7});
  auto op2 = ex::connect(sndr, expect_value_receiver{7});
  REQUIRE( counter == 0 );
  ex::start(op1); // operation starts and finishes.
  REQUIRE( counter == 0 );
  REQUIRE( called );
  ssource.request_stop();
  ex::start(op2); // operation is done, result is delivered.
  REQUIRE( counter == 0 );
}
TEST_CASE("split forwards external stop signal (3)", "[adaptors][split]") {
  impulse_scheduler sched;
  stdexec::in_place_stop_source ssource;
  bool called = false;
  int counter{};
  auto split =
    ex::split(
      ex::on(
        sched,
        ex::just() | ex::then([&]{ called = true; return 7; })));
  auto sndr =
    exec::write(
      ex::upon_stopped(
        split,
        [&]{ ++counter; return 42; }),
      exec::with(ex::get_stop_token, ssource.get_token()));
  auto op1 = ex::connect(sndr, expect_value_receiver{42});
  auto op2 = ex::connect(sndr, expect_value_receiver{42});
  REQUIRE( counter == 0 );
  ex::start(op1); // puts a unit of work on the impulse_scheduler and
                  // op1 into the list of waiting operations.
  REQUIRE( counter == 0 );
  REQUIRE( !called );
  ssource.request_stop();
  ex::start(op2); // puts op2 in the list of waiting operations.
  REQUIRE( counter == 0 );
  REQUIRE( !called );
  sched.start_next(); // Impulse scheduler notices stop has been requested
                      // and completes op1 with "stopped", which notifies
                      // all waiting states.
  REQUIRE( counter == 2 );
  REQUIRE( !called );
}
TEST_CASE("split forwards external stop signal (4)", "[adaptors][split]") {
  impulse_scheduler sched;
  stdexec::in_place_stop_source ssource;
  bool called = false;
  int counter{};
  auto split =
    ex::split(ex::just() | ex::then([&]{ called = true; return 7; }));
  auto sndr1 =
    ex::on(
      sched,
      ex::upon_stopped(
        exec::write(
          split,
          exec::with(ex::get_stop_token, ssource.get_token())),
        [&]{ ++counter; return 42; }));
  auto sndr2 =
    exec::write(
      ex::on(
        sched,
        ex::upon_stopped(
          split,
          [&]{ ++counter; return 42; })),
      exec::with(ex::get_stop_token, ssource.get_token()));
  auto op1 = ex::connect(sndr1, expect_value_receiver{7});
  auto op2 = ex::connect(sndr2, expect_stopped_receiver{});
  REQUIRE( counter == 0 );
  ex::start(op1); // puts a unit of work on the impulse_scheduler.
  REQUIRE( counter == 0 );
  REQUIRE( !called );
  sched.start_next(); // Impulse scheduler starts split sender, which
                      // completes with 7.
  REQUIRE( counter == 0 );
  REQUIRE( called );
  ex::start(op2); // puts another unit of work on the impulse_scheduler
  REQUIRE( counter == 0 );
  ssource.request_stop();
  sched.start_next(); // Impulse scheduler notices stop has been
                      // requested and "stops" the work.
  REQUIRE( counter == 0 );
}
TEST_CASE("split forwards results from a different thread", "[adaptors][split]") {
  exec::static_thread_pool pool{1};
  auto split = ex::schedule(pool.get_scheduler()) | //
               ex::then([] {
                 std::this_thread::sleep_for(1ms);
                 return 42;
               }) | //
               ex::split();

  auto [val] = stdexec::sync_wait(split).value();
  REQUIRE( val == 42 );
}
TEST_CASE("split is thread-safe", "[adaptors][split]") {
  exec::static_thread_pool pool{1};

  std::mt19937_64 eng{std::random_device{}()};  // or seed however you want
  std::uniform_int_distribution<> dist{0, 1000};

  auto split = ex::transfer_just(pool.get_scheduler(), std::chrono::microseconds{dist(eng)}) | //
               ex::then([] (std::chrono::microseconds delay) {
                 std::this_thread::sleep_for(delay);
                 return 42;
               }) | //
               ex::split();

  const unsigned n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(n_threads);
  std::vector<int> thread_results(n_threads, 0);
  const std::vector<std::chrono::microseconds> delays = [&] {
    std::vector<std::chrono::microseconds> thread_delay(n_threads);
    for (unsigned tid = 0; tid < n_threads; tid++) {
      thread_delay[tid] = std::chrono::microseconds{dist(eng)};
    }
    return thread_delay;
  }();
  for (unsigned tid = 0; tid < n_threads; tid++) {
    threads[tid] = std::thread([&split, &delays, &thread_results, tid] {
      inline_scheduler scheduler{};

      std::this_thread::sleep_for(delays[tid]);
      auto [val] = stdexec::sync_wait(
          split |                   //
          ex::transfer(scheduler) | //
          ex::then([](int v) { return v; })).value();
      thread_results[tid] = val;
    });
  }
  for (unsigned tid = 0; tid < n_threads; tid++) {
    threads[tid].join();
    REQUIRE( thread_results[tid] == 42 );
  }
}
TEST_CASE("split can be an rvalue", "[adaptors][split]") {
  auto [val] = stdexec::sync_wait(
      ex::just(42) |
      ex::split() |
      ex::then([](int v) { return v; } )).value();

  REQUIRE( val == 42 );
}
TEST_CASE("split can nest", "[adaptors][split]") {
  auto split_1 = ex::just(42) | ex::split();
  auto split_2 = split_1 | ex::split();

  auto [v1] = stdexec::sync_wait(
      split_1 | //
      ex::then([](const int &cv) {
        int &v = const_cast<int&>(cv);
        return v = 1;
      })).value();

  auto [v2] = stdexec::sync_wait(
      split_2 | //
      ex::then([](const int &cv) {
        int &v = const_cast<int&>(cv);
        return v = 2;
      })).value();

  auto [v3] = stdexec::sync_wait(split_1).value();

  REQUIRE( v1 == 1 );
  REQUIRE( v2 == 2 );
  REQUIRE( v3 == 1 );
}
TEST_CASE("split doesn't advertise completion scheduler", "[adaptors][split]") {
  inline_scheduler sched;

  auto snd = ex::transfer_just(sched, 42) | ex::split();
  using snd_t = decltype(snd);
  static_assert(!stdexec::__has_completion_scheduler<snd_t, ex::set_value_t>);
  static_assert(!stdexec::__has_completion_scheduler<snd_t, ex::set_error_t>);
  static_assert(!stdexec::__has_completion_scheduler<snd_t, ex::set_stopped_t>);
  (void)snd;
}
