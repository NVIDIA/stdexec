/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "exec/fork_join.hpp"
#include "exec/just_from.hpp"
#include "test_common/schedulers.hpp"
#include "test_common/type_helpers.hpp"

#include <catch2/catch.hpp>

#include <atomic>

using namespace STDEXEC;

namespace {
  TEST_CASE("fork_join is a sender", "[adaptors][fork_join]") {
    auto sndr = exec::fork_join(just(), then([] { }));
    STATIC_REQUIRE(sender<decltype(sndr)>);
  }

  TEST_CASE("fork_join is a sender in empty env", "[adaptors][fork_join]") {
    auto sndr = exec::fork_join(just(), then([] { }));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(sndr), env<>>,
        completion_signatures<set_value_t(), set_error_t(std::exception_ptr)>
      >);
  }

  struct ForwardingThen {
    template <typename Value>
    constexpr decltype(auto) operator()(Value&& value) const noexcept {
      return std::forward<Value>(value);
    }

    constexpr void operator()() const noexcept {
    }
  };

  template <char ID>
  struct identifiable_domain : public STDEXEC::default_domain { };

  TEST_CASE("fork_join completion domain and scheduler", "[adaptors][fork_join]") {
    basic_inline_scheduler<identifiable_domain<'A'>> sched_A;
    basic_inline_scheduler<identifiable_domain<'B'>> sched_B;
    basic_inline_scheduler<identifiable_domain<'C'>> sched_C;

    auto sndr = STDEXEC::schedule(sched_A) | STDEXEC::then(ForwardingThen{})
              | exec::fork_join(
                  STDEXEC::on(sched_B, STDEXEC::then(ForwardingThen{})),
                  STDEXEC::on(sched_C, STDEXEC::then(ForwardingThen{})))
              | STDEXEC::then(ForwardingThen{});

    auto domain = STDEXEC::get_completion_domain<STDEXEC::set_value_t>(STDEXEC::get_env(sndr));
    static_assert(std::same_as<decltype(domain), identifiable_domain<'A'>>);

    auto sched = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr));
    static_assert(std::same_as<decltype(sched), decltype(sched_A)>);
  }

  TEST_CASE("fork_join broadcasts results to multiple continuations", "[adaptors][fork_join]") {
    auto fn = [](auto sink) {
      sink(42);
      return completion_signatures<
        set_value_t(int),
        set_value_t(int, int),
        set_value_t(int, int, int)
      >{};
    };
    auto sndr = exec::fork_join(
      exec::just_from(fn),
      then([](auto&&... is) {
        CHECK(sizeof...(is) == 1);
        CHECK(((is == 42) && ...));
        STATIC_REQUIRE((std::is_same_v<decltype(is), const int&> && ...));
        return (is + ...);
      }),
      then([](auto&&... is) {
        CHECK(sizeof...(is) == 1);
        CHECK(((is == 42) && ...));
        STATIC_REQUIRE((std::is_same_v<decltype(is), const int&> && ...));
        return (is + ...);
      }));
    STATIC_REQUIRE(sender_in<decltype(sndr), env<>>);
    STATIC_REQUIRE(
      set_equivalent<
        completion_signatures_of_t<decltype(sndr), env<>>,
        completion_signatures<set_value_t(int, int), set_error_t(std::exception_ptr)>
      >);

    auto [i1, i2] = sync_wait(sndr).value();
    CHECK(i1 == 42);
    CHECK(i2 == 42);
  }

  TEST_CASE("fork_join with empty value channel", "[adaptors][fork_join]") {
    auto sndr = ::STDEXEC::just() | ::STDEXEC::then([]() noexcept -> void { })
              | exec::fork_join(
                  ::STDEXEC::then([]() noexcept -> void { }),
                  ::STDEXEC::then([]() noexcept -> void { }));

    ::STDEXEC::sync_wait(std::move(sndr));
  }

  TEST_CASE("fork_join can be nested", "[adaptors][fork_join]") {
    std::atomic<int> witness = 0;

    auto make_then = [&witness]() {
      return ::STDEXEC::then([&witness]() noexcept { ++witness; });
    };

    auto sndr = ::STDEXEC::just() | make_then()
              | exec::fork_join(make_then(), exec::fork_join(make_then(), make_then()));

    ::STDEXEC::sync_wait(std::move(sndr));

    CHECK(witness == 4);
  }

  struct customize_fork_join_domain : public STDEXEC::default_domain {
    template <STDEXEC::sender_expr_for<exec::fork_join_t> Sndr, class Env>
    constexpr auto transform_sender(STDEXEC::set_value_t, Sndr&&, const Env&) const noexcept {
      return STDEXEC::just(std::string("congrats on customizing fork_join_t"));
    }
  };

  TEST_CASE("fork_join is customizable", "[adaptors][fork_join]") {
    basic_inline_scheduler<customize_fork_join_domain> sched{};

    auto sndr =
      ::STDEXEC::schedule(sched) | ::STDEXEC::then(ForwardingThen{})
      | exec::fork_join(::STDEXEC::then(ForwardingThen{}), ::STDEXEC::then(ForwardingThen{}));

    CHECK(
      std::get<0>(STDEXEC::sync_wait(std::move(sndr)).value())
      == "congrats on customizing fork_join_t");
  }

  TEST_CASE("fork_join following a continues_on", "[adaptors][fork_join]") {
    std::atomic<int> witness = 0;

    auto make_then = [&witness]() {
      return ::STDEXEC::then([&witness]() noexcept { ++witness; });
    };

    auto sndr = ::STDEXEC::just() | ::STDEXEC::continues_on(::STDEXEC::inline_scheduler{})
              | exec::fork_join(make_then(), make_then());

    ::STDEXEC::sync_wait(std::move(sndr));

    CHECK(witness == 2);
  }
} // namespace
