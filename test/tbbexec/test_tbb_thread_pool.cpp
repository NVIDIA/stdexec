/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
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
#include <exec/on.hpp>
#include <tbbexec/tbb_thread_pool.hpp>

#include <iostream>

#include <map>

namespace ex = stdexec;

template <ex::scheduler Sched = inline_scheduler>
inline auto _with_scheduler(Sched sched = {}) {
  return exec::write(exec::with(ex::get_scheduler, std::move(sched)));
}

template <ex::scheduler Sched = inline_scheduler>
inline auto _make_env_with_sched(Sched sched = {}) {
  return exec::make_env(exec::with(ex::get_scheduler, std::move(sched)));
}

using _env_with_sched_t = decltype(_make_env_with_sched());

TEST_CASE(
    "exec::on works when changing threads with tbbexec::tbb_thread_pool", "[adaptors][exec::on]") {
  tbbexec::tbb_thread_pool pool{2};
  bool called{false};
  // launch some work on the thread pool
  ex::sender auto snd = exec::on(pool.get_scheduler(), ex::just()) //
                        | ex::then([&] { called = true; }) | _with_scheduler();
  stdexec::sync_wait(std::move(snd));
  // the work should be executed
  REQUIRE(called);
}

namespace {

int compute(int x) { return x + 1; }

std::size_t legible_thread_id() {
  static auto [mutex, map] =
      std::pair<std::mutex, std::map<decltype(std::this_thread::get_id()), std::size_t>>{};
  std::lock_guard lock(mutex);
  return map.try_emplace(std::this_thread::get_id(), map.size()).first->second;
}

} // namespace

TEST_CASE("more tbb_thread_pool") {

  compute(1);
  // Declare a pool of 1 worker threads (godbolt won't let us have more):

  tbbexec::tbb_thread_pool pool(1);

  // Declare a pool of 8 worker threads:
  exec::static_thread_pool other_pool(1);
  // Get a handle to the thread pool:
  auto other_sched = other_pool.get_scheduler();

  // Get a handle to the thread pool:
  auto tbb_sched = pool.get_scheduler();

  [[maybe_unused]] exec::inline_scheduler inline_sched;

  // Describe some work
  std::mutex mutex;
  std::map<std::string, std::size_t> log;
  auto fun = [&](auto x) {
    return [&, x](int i) {
      {
        std::lock_guard lock{mutex};
        log.emplace(x, legible_thread_id());
      }
      std::cout << "thread " << legible_thread_id() << ": x = " << x << "\n";
      return compute(i);
    };
  };

  std::cout << "Main thread is thread " << legible_thread_id() << "\n\n";
  auto a = tbb::task_arena{tbb::task_arena::attach{}};
  a.enqueue([] {
    // std::ignore = legible_thread_id();
    std::cout << "One tbb thread is thread " << legible_thread_id() << "\n\n";
  });

  stdexec::sync_wait(stdexec::on(other_sched, stdexec::just()) | stdexec::then([] {
    // std::ignore = legible_thread_id();
    std::cout << "other thread is thread " << legible_thread_id() << "\n\n";
  }));

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(1ms);

  using namespace stdexec;

  // Something's weird: If I start one chain with schedule(inline_scheduler) it crashes.

  auto work = when_all( //
      schedule(tbb_sched) | then([] { return 1; }) | then(fun("a tbb_sched")) |
          then(fun("b tbb_sched")),
      schedule(other_sched) | then([] { return 0; }) | then(fun("c other_sched")) |
          transfer(tbb_sched) | then(fun("d tbb_sched")),
      schedule(tbb_sched) | then([] { return 2; }) | then(fun("e tbb_sched")) |
          transfer(other_sched) | then(fun("f other_sched")) | transfer(tbb_sched) |
          then(fun("g tbb_sched")));

  // Launch the work and wait for the result:
  auto [i, j, k] = stdexec::sync_wait(std::move(work)).value();
  CHECK(i == 3);
  CHECK(j == 2);
  CHECK(k == 5);
  std::cout << i << ", " << j << ", " << k;
  CHECK(
      log == decltype(log){{"a tbb_sched", 3}, {"b tbb_sched", 3}, {"c other_sched", 2},
                 {"d tbb_sched", 3}, {"e tbb_sched", 3}, {"f other_sched", 2}, {"g tbb_sched", 3}});
  // clang-format off
  //auto j = -1;
  /*(auto [i,j] = stdexec::sync_wait(
      when_all(schedule(tbb_sched)    | then([] { return 1; })
              ,schedule(tbb_sched) | then([] { return 2; })//,schedule(inline_sched) | then([] { return 0; })
              )
      ).value();
               //schedule(tbb_sched)    | then([] { return 2; }))).value();
  // clang-format on
  auto k = -1;
  std::cout << std::format("{}, {}, {}", i, j, k);*/
  // Print the results:
  // std::cout << std::format("{}, {}, {}", i, j, k);
}

/*
TEST_CASE("exec::on can be called with rvalue ref scheduler", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("exec::on can be called with const ref scheduler", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  const inline_scheduler sched;
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
  ex::start(op);
  // The receiver checks if we receive the right value
}
TEST_CASE("exec::on can be called with ref scheduler", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  inline_scheduler sched;
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("exec::on forwards set_error calls", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{env, std::exception_ptr{}});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("exec::on forwards set_error calls of other types", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  error_scheduler<std::string> sched{std::string{"error"}};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_error_receiver{env, std::string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}
TEST_CASE("exec::on forwards set_stopped calls", "[adaptors][exec::on]") {
  auto env = _make_env_with_sched();
  stopped_scheduler sched{};
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{env});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

TEST_CASE(
    "exec::on has the values_type corresponding to the given values", "[adaptors][exec::on]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(exec::on(sched, ex::just(1)) | _with_scheduler());
  check_val_types<type_array<type_array<int, double>>>(
      exec::on(sched, ex::just(3, 0.14)) | _with_scheduler());
  check_val_types<type_array<type_array<int, double, std::string>>>(
      exec::on(sched, ex::just(3, 0.14, std::string{"pi"})) | _with_scheduler());
}
TEST_CASE("exec::on keeps error_types from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>(
      exec::on(sched1, ex::just(1)) | _with_scheduler());
  check_err_types<type_array<std::exception_ptr>>(
      exec::on(sched2, ex::just(2)) | _with_scheduler());
  check_err_types<type_array<std::exception_ptr, int>>(
      exec::on(sched3, ex::just(3)) | _with_scheduler());
}
TEST_CASE("exec::on keeps sends_stopped from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(exec::on(sched1, ex::just(1)) | _with_scheduler());
  check_sends_stopped<true>(exec::on(sched2, ex::just(2)) | _with_scheduler());
  check_sends_stopped<true>(exec::on(sched3, ex::just(3)) | _with_scheduler());
}
*/