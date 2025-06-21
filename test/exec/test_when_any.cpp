/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include <exec/when_any.hpp>
#include <exec/single_thread_context.hpp>
#include <numbers>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <catch2/catch.hpp>

namespace ex = stdexec;

using namespace stdexec;

namespace {

  TEST_CASE("when_ny returns a sender", "[adaptors][when_any]") {
    auto snd = exec::when_any(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("when_any with environment returns a sender", "[adaptors][when_any]") {
    auto snd = exec::when_any(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("when_any simple example", "[adaptors][when_any]") {
    auto snd = exec::when_any(ex::just(3.0));
    auto snd1 = std::move(snd) | ex::then([](double y) { return y + 0.1415; });
    const double expected = 3.0 + 0.1415;
    auto op = ex::connect(std::move(snd1), expect_value_receiver{expected});
    ex::start(op);
  }

  TEST_CASE("when_any completes with only one sender", "[adaptors][when_any]") {
    ex::sender auto snd = exec::when_any(
      completes_if{false} | ex::then([] { return 1; }),
      completes_if{true} | ex::then([] { return 42; }));
    wait_for_value(std::move(snd), 42);

    ex::sender auto snd2 = exec::when_any(
      completes_if{true} | ex::then([] { return 1; }),
      completes_if{false} | ex::then([] { return 42; }));
    wait_for_value(std::move(snd2), 1);
  }

  TEST_CASE("when_any with move-only types", "[adaptors][when_any]") {
    ex::sender auto snd = exec::when_any(
      completes_if{false} | ex::then([] { return movable(1); }), ex::just(movable(42)));
    wait_for_value(std::move(snd), movable(42));
  }

  TEST_CASE("when_any forwards stop signal", "[adaptors][when_any]") {
    stopped_scheduler stop;
    int result = 42;
    ex::sender auto snd = exec::when_any(completes_if{false}, ex::schedule(stop))
                        | ex::then([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("nested when_any is stoppable", "[adaptors][when_any]") {
    int result = 41;
    ex::sender auto snd = exec::when_any(
                            exec::when_any(completes_if{false}, completes_if{false}),
                            completes_if{false},
                            ex::just(),
                            completes_if{false})
                        | ex::then([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("stop is forwarded", "[adaptors][when_any]") {
    int result = 41;
    ex::sender auto snd = exec::when_any(ex::just_stopped(), completes_if{false})
                        | ex::upon_stopped([&result] { result += 1; });
    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("when_any is thread-safe", "[adaptors][when_any]") {
    exec::single_thread_context ctx1;
    exec::single_thread_context ctx2;
    exec::single_thread_context ctx3;

    auto sch1 = ex::schedule(ctx1.get_scheduler());
    auto sch2 = ex::schedule(ctx2.get_scheduler());
    auto sch3 = ex::schedule(ctx3.get_scheduler());

    int result = 41;

    ex::sender auto snd = exec::when_any(
      sch1 | ex::let_value([] { return exec::when_any(completes_if{false}); }),
      sch2 | ex::let_value([] { return completes_if{false}; }),
      sch3 | ex::then([&result] { result += 1; }),
      completes_if{false});

    ex::sync_wait(std::move(snd));
    REQUIRE(result == 42);
  }

  TEST_CASE("when_any completion signatures", "[adaptors][when_any]") {
    struct move_throws {
      move_throws() = default;

      move_throws(move_throws&&) noexcept(false) {
      }

      auto operator=(move_throws&&) noexcept(false) -> move_throws& {
        return *this;
      }
    };

    auto just = exec::when_any(ex::just());
    static_assert(sender<decltype(just)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just)>,
                  completion_signatures<set_value_t(), set_stopped_t()>
    >);

    auto just_string = exec::when_any(ex::just(std::string("foo")));
    static_assert(sender<decltype(just_string)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_string)>,
                  completion_signatures<set_value_t(std::string), set_stopped_t()>
    >);

    auto just_stopped = exec::when_any(ex::just_stopped());
    static_assert(sender<decltype(just_stopped)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_stopped)>,
                  completion_signatures<set_stopped_t()>
    >);

    auto just_then = exec::when_any(ex::just() | ex::then([] { return 42; }));
    static_assert(sender<decltype(just_then)>);
    static_assert(
      set_equivalent<
        completion_signatures_of_t<decltype(just_then)>,
        completion_signatures<set_value_t(int), set_stopped_t(), set_error_t(std::exception_ptr)>
      >);

    auto just_then_noexcept = exec::when_any(ex::just() | ex::then([]() noexcept { return 42; }));
    static_assert(sender<decltype(just_then_noexcept)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_then_noexcept)>,
                  completion_signatures<set_value_t(int), set_stopped_t()>
    >);

    auto just_move_throws = exec::when_any(ex::just(move_throws{}));
    static_assert(sender<decltype(just_move_throws)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(just_move_throws)>,
                  completion_signatures<
                    set_value_t(move_throws),
                    set_stopped_t(),
                    set_error_t(std::exception_ptr)
                  >
    >);

    auto mulitple_senders = exec::when_any(
      ex::just(std::numbers::pi),
      ex::just(std::string()),
      ex::just(std::string()),
      ex::just() | ex::then([] { return 42; }),
      ex::just() | ex::then([] { return 42; }));
    static_assert(sender<decltype(mulitple_senders)>);
    static_assert(set_equivalent<
                  completion_signatures_of_t<decltype(mulitple_senders)>,
                  completion_signatures<
                    set_value_t(double),
                    set_value_t(std::string),
                    set_value_t(int),
                    set_stopped_t(),
                    set_error_t(std::exception_ptr)
                  >
    >);
    // wait_for_value(std::move(snd), movable(42));
  }

  template <class Receiver>
  struct dup_op {
    Receiver rec;

    void start() & noexcept {
      stdexec::set_error(
        static_cast<Receiver&&>(rec), std::make_exception_ptr(std::runtime_error("dup")));
    }
  };

  struct dup_sender {
    using sender_concept = stdexec::sender_t;
    using completion_signatures = stdexec::completion_signatures<
      set_value_t(),
      set_error_t(std::exception_ptr),
      set_error_t(std::exception_ptr&&)
    >;

    template <class Receiver>
    friend auto tag_invoke(connect_t, dup_sender, Receiver rec) noexcept -> dup_op<Receiver> {
      return {static_cast<Receiver&&>(rec)};
    }
  };

#if !STDEXEC_STD_NO_EXCEPTIONS()
  TEST_CASE("when_any - with duplicate completions", "[adaptors][when_any]") {
    REQUIRE_THROWS(stdexec::sync_wait(exec::when_any(dup_sender{})));
  }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()
} // namespace
