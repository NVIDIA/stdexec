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
#include <exec/env.hpp>
#include <exec/static_thread_pool.hpp>

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

TEST_CASE("exec::on returns a sender", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("exec::on with environment returns a sender", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  static_assert(ex::sender_in<decltype(snd), _env_with_sched_t>);
  (void)snd;
}
TEST_CASE("exec::on simple example", "[adaptors][exec::on]") {
  auto snd = exec::on(inline_scheduler{}, ex::just(13));
  auto op = ex::connect(
    std::move(snd),
    expect_value_receiver{env_tag{}, _make_env_with_sched(), 13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

TEST_CASE("exec::on calls the receiver when the scheduler dictates", "[adaptors][exec::on]") {
  int recv_value{0};
  impulse_scheduler sched;
  auto env = _make_env_with_sched();
  auto snd = exec::on(sched, ex::just(13));
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{env, recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CHECK(recv_value == 0);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CHECK(recv_value == 13);
}

TEST_CASE("exec::on calls the given sender when the scheduler dictates", "[adaptors][exec::on]") {
  bool called{false};
  auto snd_base = ex::just() | ex::then([&]() -> int {
    called = true;
    return 19;
  });

  int recv_value{0};
  impulse_scheduler sched;
  auto env = _make_env_with_sched();
  auto snd = exec::on(sched, std::move(snd_base));
  auto op = ex::connect(std::move(snd), expect_value_receiver_ex{env, recv_value});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task
  // The base sender shouldn't be started
  CHECK_FALSE(called);

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CHECK(called);
  CHECK(recv_value == 19);
}

TEST_CASE("exec::on works when changing threads", "[adaptors][exec::on]") {
  exec::static_thread_pool pool{2};
  bool called{false};
  // launch some work on the thread pool
  ex::sender auto snd = exec::on(pool.get_scheduler(), ex::just()) //
                        | ex::then([&] { called = true; })
                        | _with_scheduler();
  stdexec::sync_wait(std::move(snd));
  // the work should be executed
  REQUIRE(called);
}

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

TEST_CASE("exec::on has the values_type corresponding to the given values", "[adaptors][exec::on]") {
  inline_scheduler sched{};

  check_val_types<type_array<type_array<int>>>(exec::on(sched, ex::just(1)) | _with_scheduler());
  check_val_types<type_array<type_array<int, double>>>(exec::on(sched, ex::just(3, 0.14)) | _with_scheduler());
  check_val_types<type_array<type_array<int, double, std::string>>>(
      exec::on(sched, ex::just(3, 0.14, std::string{"pi"})) | _with_scheduler());
}
TEST_CASE("exec::on keeps error_types from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>(exec::on(sched1, ex::just(1)) | _with_scheduler());
  check_err_types<type_array<std::exception_ptr>>(exec::on(sched2, ex::just(2)) | _with_scheduler());
  check_err_types<type_array<std::exception_ptr, int>>(exec::on(sched3, ex::just(3)) | _with_scheduler());
}
TEST_CASE("exec::on keeps sends_stopped from scheduler's sender", "[adaptors][exec::on]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(exec::on(sched1, ex::just(1)) | _with_scheduler());
  check_sends_stopped<true>(exec::on(sched2, ex::just(2)) | _with_scheduler());
  check_sends_stopped<true>(exec::on(sched3, ex::just(3)) | _with_scheduler());
}
