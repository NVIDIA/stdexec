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

#include <chrono> // IWYU pragma: keep for std::chrono_literals

namespace ex = STDEXEC;

using namespace std::chrono_literals;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunneeded-internal-declaration")

namespace {

  TEST_CASE("starts_on returns a sender", "[adaptors][starts_on]") {
    auto snd = ex::starts_on(inline_scheduler{}, ex::just(13));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("starts_on with environment returns a sender", "[adaptors][starts_on]") {
    auto snd = ex::starts_on(inline_scheduler{}, ex::just(13));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("starts_on simple example", "[adaptors][starts_on]") {
    auto snd = ex::starts_on(inline_scheduler{}, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("starts_on calls the receiver when the scheduler dictates", "[adaptors][starts_on]") {
    int recv_value{0};
    impulse_scheduler sched;
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
    ex::start(op);
    // Up until this point, the scheduler didn't start any task; no effect expected
    CHECK(recv_value == 0);

    // Tell the scheduler to start executing one task
    sched.start_next();
    CHECK(recv_value == 13);
  }

  TEST_CASE(
    "starts_on calls the given sender when the scheduler dictates",
    "[adaptors][starts_on]") {
    bool called{false};
    auto snd_base = ex::just() | ex::then([&]() -> int {
                      called = true;
                      return 19;
                    });

    int recv_value{0};
    impulse_scheduler sched;
    auto snd = ex::starts_on(sched, std::move(snd_base));
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
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

  TEST_CASE("starts_on works when changing threads", "[adaptors][starts_on]") {
    exec::static_thread_pool pool{2};
    std::atomic<bool> called{false};
    {
      // lunch some work on the thread pool
      ex::sender auto snd = ex::starts_on(pool.get_scheduler(), ex::just())
                          | ex::then([&] { called.store(true); });
      ex::start_detached(std::move(snd));
    }
    // wait for the work to be executed, with timeout
    // perform a poor-man's sync
    // NOTE: it's a shame that the `join` method in static_thread_pool is not public
    for (int i = 0; i < 1000 && !called.load(); i++)
      std::this_thread::sleep_for(1ms);
    // the work should be executed
    REQUIRE(called);
  }

  TEST_CASE("starts_on can be called with rvalue ref scheduler", "[adaptors][starts_on]") {
    auto snd = ex::starts_on(inline_scheduler{}, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("starts_on can be called with const ref scheduler", "[adaptors][starts_on]") {
    const inline_scheduler sched;
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("starts_on can be called with ref scheduler", "[adaptors][starts_on]") {
    inline_scheduler sched;
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_value_receiver{13});
    ex::start(op);
    // The receiver checks if we receive the right value
  }

  TEST_CASE("starts_on forwards set_error calls", "[adaptors][starts_on]") {
    error_scheduler<std::exception_ptr> sched{std::exception_ptr{}};
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("starts_on forwards set_error calls of other types", "[adaptors][starts_on]") {
    error_scheduler<std::string> sched{std::string{"error"}};
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"error"}});
    ex::start(op);
    // The receiver checks if we receive an error
  }

  TEST_CASE("starts_on forwards set_stopped calls", "[adaptors][starts_on]") {
    stopped_scheduler sched{};
    auto snd = ex::starts_on(sched, ex::just(13));
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    // The receiver checks if we receive the stopped signal
  }

  TEST_CASE(
    "starts_on has the values_type corresponding to the given values",
    "[adaptors][starts_on]") {
    inline_scheduler sched{};

    check_val_types<ex::__mset<pack<int>>>(ex::starts_on(sched, ex::just(1)));
    check_val_types<ex::__mset<pack<int, double>>>(ex::starts_on(sched, ex::just(3, 0.14)));
    check_val_types<ex::__mset<pack<int, double, std::string>>>(
      ex::starts_on(sched, ex::just(3, 0.14, std::string{"pi"})));
  }

  TEST_CASE("starts_on keeps error_types from scheduler's sender", "[adaptors][starts_on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(ex::starts_on(sched1, ex::just(1)));
    check_err_types<ex::__mset<std::exception_ptr>>(ex::starts_on(sched2, ex::just(2)));
    check_err_types<ex::__mset<int>>(ex::starts_on(sched3, ex::just(3)));
  }

  TEST_CASE("starts_on keeps sends_stopped from scheduler's sender", "[adaptors][starts_on]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::starts_on(sched1, ex::just(1)));
    check_sends_stopped<true>(ex::starts_on(sched2, ex::just(2)));
    check_sends_stopped<true>(ex::starts_on(sched3, ex::just(3)));
  }

  // Return a different sender when we invoke this custom defined starts_on implementation
  struct starts_on_test_domain {
    template <ex::sender_expr_for<ex::starts_on_t> Sender>
    static auto transform_sender(STDEXEC::set_value_t, Sender&&, const auto&...) {
      return ex::just(std::string{"Hello, world!"});
    }
  };

  TEST_CASE("starts_on can be customized", "[adaptors][starts_on]") {
    // The customization will return a different value
    basic_inline_scheduler<starts_on_test_domain> sched;
    auto snd = ex::starts_on(sched, ex::just(std::string{"world"}));
    std::string res;
    auto op = ex::connect(std::move(snd), expect_value_receiver_ex{res});
    ex::start(op);
    REQUIRE(res == "Hello, world!");
  }

  struct move_checker {
    move_checker() noexcept = default;

    move_checker(const move_checker& other) noexcept {
      REQUIRE(other.valid);
      valid = true;
    }

    auto operator=(const move_checker& other) noexcept -> move_checker& {
      REQUIRE(other.valid);
      valid = true;
      return *this;
    }

    move_checker(move_checker&& other) noexcept {
      other.valid = false;
      valid = true;
    }

    auto operator=(move_checker&& other) noexcept -> move_checker& {
      other.valid = false;
      valid = true;
      return *this;
    }

   private:
    bool valid{true};
  };

  struct move_checking_inline_scheduler {
    [[nodiscard]]
    auto schedule() const noexcept {
      return sender{};
    }

    auto operator==(const move_checking_inline_scheduler&) const noexcept -> bool {
      return true;
    }

   private:
    template <typename Receiver>
    struct opstate : immovable {
      void start() & noexcept {
        ex::set_value(static_cast<Receiver&&>(rcvr_));
      }

      Receiver rcvr_;
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      template <typename Receiver>
      [[nodiscard]]
      auto connect(Receiver rcvr) const -> opstate<Receiver> {
        return {{}, static_cast<Receiver&&>(rcvr)};
      }

      [[nodiscard]]
      auto get_env() const noexcept {
        return sched_attrs(move_checking_inline_scheduler(), ex::set_value);
      }
    };

    move_checker mc_;
  };

  TEST_CASE("starts_on does not reference a moved-from scheduler", "[adaptors][starts_on]") {
    move_checking_inline_scheduler is;
    ex::sender auto snd = ex::starts_on(is, ex::just()) | ex::then([] { });
    ex::sync_wait(std::move(snd));
  }
} // namespace

STDEXEC_PRAGMA_POP()
