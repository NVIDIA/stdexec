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
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

namespace ex = STDEXEC;

namespace {

  template <ex::scheduler Sched = inline_scheduler>
  inline auto _with_scheduler(Sched sched = {}) {
    return ex::write_env(ex::prop{ex::get_scheduler, std::move(sched)});
  }

  template <ex::scheduler Sched = inline_scheduler>
  inline auto _make_env_with_sched(Sched sched = {}) {
    return ex::prop{ex::get_scheduler, std::move(sched)};
  }

  using _env_with_sched_t = decltype(_make_env_with_sched());

  TEST_CASE("STDEXEC::on returns a sender", "[adaptors][on]") {
    auto snd = ex::on(inline_scheduler{}, ex::just(13));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("STDEXEC::on with environment returns a sender", "[adaptors][on]") {
    auto snd = ex::on(inline_scheduler{}, ex::just(13));
    static_assert(ex::sender_in<decltype(snd), _env_with_sched_t>);
    (void) snd;
  }

  TEST_CASE("STDEXEC::on simple example", "[adaptors][on]") {
    auto snd = ex::on(inline_scheduler{}, ex::just(13));
    auto op =
      ex::connect(std::move(snd), expect_value_receiver{env_tag{}, _make_env_with_sched(), 13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("STDEXEC::on calls the receiver when the scheduler dictates", "[adaptors][on]") {
    int recv_value{0};
    impulse_scheduler sched;
    auto env = _make_env_with_sched();
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{env, recv_value});
    ex::start(op);
    // Up until this point, the scheduler didn't start any task; no effect expected
    CHECK(recv_value == 0);

    // Tell the scheduler to start executing one task
    sched.start_next();
    CHECK(recv_value == 13);
  }

  TEST_CASE("STDEXEC::on calls the given sender when the scheduler dictates", "[adaptors][on]") {
    bool called{false};
    auto snd_base = ex::just() | ex::then([&]() -> int {
                      called = true;
                      return 19;
                    });

    int recv_value{0};
    impulse_scheduler sched;
    auto env = _make_env_with_sched();
    auto snd = ex::on(sched, std::move(snd_base));
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

  TEST_CASE("STDEXEC::on works when changing threads", "[adaptors][on]") {
    exec::static_thread_pool pool{2};
    bool called{false};
    // launch some work on the thread pool
    ex::sender auto snd = ex::on(pool.get_scheduler(), ex::just())
                        | ex::then([&] { called = true; }) | _with_scheduler();
    ex::sync_wait(std::move(snd));
    // the work should be executed
    REQUIRE(called);
  }

  TEST_CASE("STDEXEC::on can be called with rvalue ref scheduler", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    auto snd = ex::on(inline_scheduler{}, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("STDEXEC::on can be called with const ref scheduler", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    const inline_scheduler sched;
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("STDEXEC::on can be called with ref scheduler", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    inline_scheduler sched;
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{env_tag{}, env, 13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("STDEXEC::on forwards set_error calls", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_error_receiver{env, std::exception_ptr{}});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("STDEXEC::on forwards set_error calls of other types", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    error_scheduler<std::string> sched{std::string{"error"}};
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_error_receiver{env, std::string{"error"}});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("STDEXEC::on forwards set_stopped calls", "[adaptors][on]") {
    auto env = _make_env_with_sched();
    stopped_scheduler sched{};
    auto snd = ex::on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{env});
    ex::start(op);
    // The receiver checks if we receive the stopped signal
  }

  TEST_CASE("STDEXEC::on has the values_type corresponding to the given values", "[adaptors][on]") {
    inline_scheduler sched{};

    check_val_types<ex::__mset<pack<int>>>(ex::on(sched, ex::just(1)) | _with_scheduler());
    check_val_types<ex::__mset<pack<int, double>>>(
      ex::on(sched, ex::just(3, 0.14)) | _with_scheduler());
    check_val_types<ex::__mset<pack<int, double, std::string>>>(
      ex::on(sched, ex::just(3, 0.14, std::string{"pi"})) | _with_scheduler());
  }

  TEST_CASE("STDEXEC::on keeps error_types from scheduler's sender", "[adaptors][on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(ex::on(sched1, ex::just(1)) | _with_scheduler());
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::on(sched2, ex::just(2)) | _with_scheduler());
    check_err_types<ex::__mset<int>>(ex::on(sched3, ex::just(3)) | _with_scheduler());
  }

  TEST_CASE("STDEXEC::on keeps sends_stopped from scheduler's sender", "[adaptors][on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::on(sched1, ex::just(1)) | _with_scheduler());
    check_sends_stopped<true>(ex::on(sched2, ex::just(2)) | _with_scheduler());
    check_sends_stopped<true>(ex::on(sched3, ex::just(3)) | _with_scheduler());
  }
} // namespace
