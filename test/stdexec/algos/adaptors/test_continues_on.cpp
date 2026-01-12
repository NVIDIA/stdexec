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
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

#include <chrono> // IWYU pragma: keep for chrono_literals

namespace ex = STDEXEC;

using namespace std::chrono_literals;

namespace {

  TEST_CASE("continues_on returns a sender", "[adaptors][continues_on]") {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("continues_on with environment returns a sender", "[adaptors][continues_on]") {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("continues_on simple example", "[adaptors][continues_on]") {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be piped", "[adaptors][continues_on]") {
    // Just transfer a value to the impulse scheduler
    ex::scheduler auto sched = impulse_scheduler{};
    ex::sender auto snd = ex::just(13) | ex::continues_on(sched);
    // Start the operation
    int res{0};
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);

    // The value will be available when the scheduler will execute the next operation
    REQUIRE(res == 0);
    sched.start_next();
    REQUIRE(res == 13);
  }

  TEST_CASE(
    "continues_on calls the receiver when the scheduler dictates",
    "[adaptors][continues_on]") {
    int recv_value{0};
    impulse_scheduler sched;
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(snd, expect_value_receiver_ex{recv_value});
    ex::start(op);
    // Up until this point, the scheduler didn't start any task; no effect expected
    CHECK(recv_value == 0);

    // Tell the scheduler to start executing one task
    sched.start_next();
    CHECK(recv_value == 13);
  }

  TEST_CASE(
    "continues_on calls the given sender when the scheduler dictates",
    "[adaptors][continues_on]") {
    bool called{false};
    auto snd_base = ex::just() | ex::then([&]() -> int {
                      called = true;
                      return 19;
                    });

    int recv_value{0};
    impulse_scheduler sched;
    auto snd = ex::continues_on(std::move(snd_base), sched);
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

  TEST_CASE("continues_on works when changing threads", "[adaptors][continues_on]") {
    exec::static_thread_pool pool{2};
    std::atomic<bool> called{false};
    {
      // lunch some work on the thread pool
      ex::sender auto snd = ex::continues_on(ex::just(), pool.get_scheduler())
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

  TEST_CASE("continues_on can be called with rvalue ref scheduler", "[adaptors][continues_on]") {
    auto snd = ex::continues_on(ex::just(13), inline_scheduler{});
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be called with const ref scheduler", "[adaptors][continues_on]") {
    const inline_scheduler sched;
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on can be called with ref scheduler", "[adaptors][continues_on]") {
    inline_scheduler sched;
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("continues_on forwards set_error calls", "[adaptors][continues_on]") {
    error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("continues_on forwards set_error calls of other types", "[adaptors][continues_on]") {
    error_scheduler<std::string> sched{std::string{"error"}};
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("continues_on forwards set_stopped calls", "[adaptors][continues_on]") {
    stopped_scheduler sched{};
    auto snd = ex::continues_on(ex::just(13), sched);
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    // The receiver checks if we receive the stopped signal
  }

  TEST_CASE(
    "continues_on has the values_type corresponding to the given values",
    "[adaptors][continues_on]") {
    inline_scheduler sched{};

    check_val_types<ex::__mset<pack<int>>>(ex::continues_on(ex::just(1), sched));
    check_val_types<ex::__mset<pack<int, double>>>(ex::continues_on(ex::just(3, 0.14), sched));
    check_val_types<ex::__mset<pack<int, double, std::string>>>(
      ex::continues_on(ex::just(3, 0.14, std::string{"pi"}), sched));
  }

  TEST_CASE("continues_on keeps error_types from scheduler's sender", "[adaptors][continues_on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(ex::continues_on(ex::just(1), sched1));
    check_err_types<ex::__mset<std::exception_ptr>>(ex::continues_on(ex::just(2), sched2));
    check_err_types<ex::__mset<int>>(ex::continues_on(ex::just(3), sched3));
  }

  TEST_CASE(
    "continues_on sends an exception_ptr if value types are potentially throwing when copied",
    "[adaptors][continues_on]") {
    inline_scheduler sched{};

    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::continues_on(ex::just(potentially_throwing{}), sched));
  }

  TEST_CASE(
    "continues_on keeps sends_stopped from scheduler's sender",
    "[adaptors][continues_on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::continues_on(ex::just(1), sched1));
    check_sends_stopped<true>(ex::continues_on(ex::just(2), sched2));
    check_sends_stopped<true>(ex::continues_on(ex::just(3), sched3));
  }

  struct value_type {
    int val_;
  };

  template <class Sender>
  using value_type_of_t =
    ex::value_types_of_t<Sender, ex::env<>, std::type_identity_t, std::type_identity_t>;

  struct continues_on_test_domain {
    template <ex::sender_expr_for<ex::continues_on_t> Sender>
    static auto transform_sender(STDEXEC::set_value_t, Sender&&, const auto&...) {
      return ex::just(value_type{53});
    }
  };

  TEST_CASE("continues_on can be customized early", "[adaptors][continues_on]") {
    // The customization will return a different value
    basic_inline_scheduler<continues_on_test_domain> sched;
    auto snd = ex::continues_on(ex::just(value_type{1}), sched);
    value_type res{0};
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);
    REQUIRE(res.val_ == 53);
  }

  struct test_domain_A {
    template <ex::sender_expr_for<ex::continues_on_t> Sender, class Env>
    auto transform_sender(STDEXEC::set_value_t, Sender&&, Env&&) const {
      return ex::just(std::string("hello"));
    }
  };

  struct test_domain_B {
    template <ex::sender_expr_for<ex::continues_on_t> Sender, class Env>
    auto transform_sender(STDEXEC::set_value_t, Sender&&, Env&&) const {
      return ex::just(std::string("goodbye"));
    }
  };

  TEST_CASE(
    "continues_on late customization is based on the scheduler's domain",
    "[adaptors][continues_on]") {
    // The customization will return a different value
    ex::scheduler auto sched_A = basic_inline_scheduler<test_domain_A>{};
    ex::scheduler auto sched_B = basic_inline_scheduler<test_domain_B>{};
    auto snd = ex::just(std::string("meow")) //
             | ex::continues_on(sched_A)     //
             | ex::continues_on(sched_B);
    std::string res;
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);
    REQUIRE(res == "goodbye");
  }
} // namespace
