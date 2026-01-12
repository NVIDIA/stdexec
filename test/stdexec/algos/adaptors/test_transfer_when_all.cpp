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
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

// For testing `transfer_when_all` we assume that, the main implementation is based on `transfer`
// and `when_all`. As both of these are tested independently, we provide fewer tests here.

// For testing `transfer_when_all_with_variant`, we just check a couple of examples, check
// customization, and we assume it's implemented in terms of `transfer_when_all`.

namespace {

  TEST_CASE("transfer_when_all returns a sender", "[adaptors][transfer_when_all]") {
    auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "transfer_when_all with environment returns a sender",
    "[adaptors][transfer_when_all]") {
    auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("transfer_when_all simple example", "[adaptors][transfer_when_all]") {
    auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
    auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
    auto op = ex::connect(
      std::move(snd1), expect_value_receiver{3.1415}); // NOLINT(modernize-use-std-numbers)
    ex::start(op);
  }

  TEST_CASE("transfer_when_all with no senders", "[adaptors][transfer_when_all]") {
    auto snd = ex::transfer_when_all(inline_scheduler{});
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "transfer_when_all transfers the result when the scheduler dictates",
    "[adaptors][transfer_when_all]") {
    impulse_scheduler sched;
    auto snd = ex::transfer_when_all(sched, ex::just(3), ex::just(0.1415));
    auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
    double res{0.0};
    auto op = ex::connect(std::move(snd1), expect_value_receiver_ex{res});
    ex::start(op);
    CHECK(res == 0.0);
    sched.start_next();
    CHECK(res == 3.1415);
  }

  TEST_CASE(
    "transfer_when_all with no senders transfers the result",
    "[adaptors][transfer_when_all]") {
    impulse_scheduler sched;
    auto snd = ex::transfer_when_all(sched);
    auto snd1 = std::move(snd) | ex::then([]() { return true; });
    bool res{false};
    auto op = ex::connect(std::move(snd1), expect_value_receiver_ex{res});
    ex::start(op);
    CHECK(!res);
    sched.start_next();
    CHECK(res);
  }

  TEST_CASE("transfer_when_all_with_variant returns a sender", "[adaptors][transfer_when_all]") {
    auto snd =
      ex::transfer_when_all_with_variant(inline_scheduler{}, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "transfer_when_all_with_variant with environment returns a sender",
    "[adaptors][transfer_when_all]") {
    auto snd =
      ex::transfer_when_all_with_variant(inline_scheduler{}, ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("transfer_when_all_with_variant basic example", "[adaptors][transfer_when_all]") {
    ex::sender auto snd =
      ex::transfer_when_all_with_variant(inline_scheduler{}, ex::just(2), ex::just(3.14));
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int>>{2}, std::variant<std::tuple<double>>{3.14});
  }

  namespace {
    template <class Tag, auto Fun>
    struct basic_domain {
      template <ex::sender_expr_for<Tag> Sender, class Env>
      auto transform_sender(STDEXEC::set_value_t, Sender&&, const Env&) const {
        return Fun();
      }
    };
  } // anonymous namespace

  TEST_CASE("transfer_when_all works with custom domain", "[adaptors][transfer_when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::transfer_when_all_t, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::transfer_when_all(scheduler(), ex::just(3), ex::just(0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::transfer_when_all_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd));
    }
  }

  TEST_CASE(
    "transfer_when_all_with_variant works with custom domain",
    "[adaptors][transfer_when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::transfer_when_all_with_variant_t, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::transfer_when_all_with_variant(scheduler(), ex::just(3), ex::just(0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::transfer_when_all_with_variant_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd));
    }

    SECTION("late customization") {
      using domain = basic_domain<ex::transfer_when_all_with_variant_t, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::starts_on(
        scheduler(),
        ex::transfer_when_all_with_variant(inline_scheduler(), ex::just(3), ex::just(0.1415)));
      wait_for_value(std::move(snd), std::string{"hello world"});
    }
  }

  TEST_CASE(
    "transfer_when_all_with_variant finds transfer_when_all customizations",
    "[adaptors][transfer_when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::transfer_when_all_t, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::transfer_when_all_with_variant(scheduler(), ex::just(3), ex::just(0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::transfer_when_all_with_variant_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd));
    }

    SECTION("late customization") {
      using domain = basic_domain<ex::transfer_when_all_t, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::starts_on(
        scheduler(),
        ex::transfer_when_all_with_variant(inline_scheduler(), ex::just(3), ex::just(0.1415)));
      wait_for_value(std::move(snd), std::string{"hello world"});
    }
  }
} // namespace
