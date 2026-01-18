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
#include <exec/async_scope.hpp>
#include <exec/env.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

// For testing `when_all_with_variant`, we just check a couple of examples, check customization, and
// we assume it's implemented in terms of `when_all`.

namespace {

  TEST_CASE("when_all returns a sender", "[adaptors][when_all]") {
    auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender<decltype(snd)>);
    static_assert(noexcept(ex::connect(snd, expect_error_receiver{})));
    (void) snd;
  }

  TEST_CASE("when_all with environment returns a sender", "[adaptors][when_all]") {
    auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("when_all simple example", "[adaptors][when_all]") {
    auto snd = ex::when_all(ex::just(3), ex::just(0.1415));
    auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
    auto op = ex::connect(
      std::move(snd1), expect_value_receiver{3.1415}); // NOLINT(modernize-use-std-numbers)
    ex::start(op);
  }

  TEST_CASE("when_all returning two values can we waited on", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all(ex::just(2), ex::just(3));
    wait_for_value(std::move(snd), 2, 3);
  }

  TEST_CASE("when_all with 5 senders", "[adaptors][when_all]") {
    ex::sender auto snd =
      ex::when_all(ex::just(2), ex::just(3), ex::just(5), ex::just(7), ex::just(11));
    wait_for_value(std::move(snd), 2, 3, 5, 7, 11);
  }

  TEST_CASE("when_all with 8 senders", "[adaptors][when_all]") {
    // 8 senders is the boundary case for the optimized tuple specializations
    ex::sender auto snd = ex::when_all(
      ex::just(2),
      ex::just(3),
      ex::just(5),
      ex::just(7),
      ex::just(11),
      ex::just(13),
      ex::just(17),
      ex::just(19));
    wait_for_value(std::move(snd), 2, 3, 5, 7, 11, 13, 17, 19);
  }

  TEST_CASE("when_all with 10 senders", "[adaptors][when_all]") {
    // 10 senders uses the generic tuple with __box storage
    ex::sender auto snd = ex::when_all(
      ex::just(2),
      ex::just(3),
      ex::just(5),
      ex::just(7),
      ex::just(11),
      ex::just(13),
      ex::just(17),
      ex::just(19),
      ex::just(23),
      ex::just(29));
    wait_for_value(std::move(snd), 2, 3, 5, 7, 11, 13, 17, 19, 23, 29);
  }

  TEST_CASE("when_all with just one sender", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all(ex::just(2));
    wait_for_value(std::move(snd), 2);
  }

  TEST_CASE("when_all with move-only types", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all(ex::just(movable(2)));
    wait_for_value(std::move(snd), movable(2));
  }

  TEST_CASE("when_all with no senders", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all();
    wait_for_value(std::move(snd));
  }

  TEST_CASE("when_all when one sender sends void", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all(ex::just(2), ex::just());
    wait_for_value(std::move(snd), 2);
  }

  TEST_CASE("when_all_with_variant basic example", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all_with_variant(ex::just(2), ex::just(3.14));
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int>>{2}, std::variant<std::tuple<double>>{3.14});
  }

  TEST_CASE("when_all_with_variant with same type", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all_with_variant(ex::just(2), ex::just(3));
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int>>{2}, std::variant<std::tuple<int>>{3});
  }

  TEST_CASE("when_all_with_variant with no senders", "[adaptors][when_all]") {
    ex::sender auto snd = ex::when_all_with_variant();
    wait_for_value(std::move(snd));
  }

  TEST_CASE("when_all completes when children complete", "[adaptors][when_all]") {
    impulse_scheduler sched;
    bool called{false};
    ex::sender auto snd = ex::when_all(
                            ex::transfer_just(sched, 11),
                            ex::transfer_just(sched, 13),
                            ex::transfer_just(sched, 17))
                        | ex::then([&](int a, int b, int c) {
                            called = true;
                            return a + b + c;
                          });
    auto op = ex::connect(std::move(snd), expect_value_receiver{41});
    ex::start(op);
    // The when_all scheduler will complete only after 3 impulses
    CHECK_FALSE(called);
    sched.start_next();
    CHECK_FALSE(called);
    sched.start_next();
    CHECK_FALSE(called);
    sched.start_next();
    CHECK(called);
  }

  TEST_CASE("when_all can be used with just_*", "[adaptors][when_all]") {
    ex::sender auto snd =
      ex::when_all(ex::just(2), ex::just_error(std::exception_ptr{}), ex::just_stopped());
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "when_all terminates with error if one child terminates with error",
    "[adaptors][when_all]") {
    error_scheduler sched;
    ex::sender auto snd = ex::when_all(ex::just(2), ex::transfer_just(sched, 5), ex::just(7));
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("when_all terminates with stopped if one child is cancelled", "[adaptors][when_all]") {
    stopped_scheduler sched;
    ex::sender auto snd = ex::when_all(ex::just(2), ex::transfer_just(sched, 5), ex::just(7));
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("when_all cancels remaining children if error is detected", "[adaptors][when_all]") {
    impulse_scheduler sched;
    error_scheduler err_sched;
    bool called1{false};
    bool called3{false};
    bool cancelled{false};
    ex::sender auto snd = ex::when_all(
      ex::starts_on(sched, ex::just()) | ex::then([&] { called1 = true; }),
      ex::starts_on(sched, ex::transfer_just(err_sched, 5)),
      ex::starts_on(sched, ex::just()) | ex::then([&] { called3 = true; }) | ex::let_stopped([&] {
        cancelled = true;
        return ex::just();
      }));
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
    // The first child will complete; the third one will be cancelled
    CHECK_FALSE(called1);
    CHECK_FALSE(called3);
    sched.start_next(); // start the first child
    CHECK(called1);
    sched.start_next(); // start the second child; this will generate an error
    CHECK_FALSE(called3);
    sched.start_next(); // start the third child
    CHECK_FALSE(called3);
    CHECK(cancelled);
  }

  TEST_CASE("when_all cancels remaining children if cancel is detected", "[adaptors][when_all]") {
    stopped_scheduler stopped_sched;
    impulse_scheduler sched;
    bool called1{false};
    bool called3{false};
    bool cancelled{false};
    ex::sender auto snd = ex::when_all(
      ex::starts_on(sched, ex::just()) | ex::then([&] { called1 = true; }),
      ex::starts_on(sched, ex::transfer_just(stopped_sched, 5)),
      ex::starts_on(sched, ex::just()) | ex::then([&] { called3 = true; }) | ex::let_stopped([&] {
        cancelled = true;
        return ex::just();
      }));
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    // The first child will complete; the third one will be cancelled
    CHECK_FALSE(called1);
    CHECK_FALSE(called3);
    sched.start_next(); // start the first child
    CHECK(called1);
    sched.start_next(); // start the second child; this will call set_stopped
    CHECK_FALSE(called3);
    sched.start_next(); // start the third child
    CHECK_FALSE(called3);
    CHECK(cancelled);
  }

  TEST_CASE(
    "when_all has the values_type based on the children, decayed and as rvalue references",
    "[adaptors][when_all]") {
    check_val_types<ex::__mset<pack<int>>>(ex::when_all(ex::just(13)));
    check_val_types<ex::__mset<pack<double>>>(ex::when_all(ex::just(3.14)));
    check_val_types<ex::__mset<pack<int, double>>>(ex::when_all(ex::just(3, 0.14)));

    check_val_types<ex::__mset<pack<>>>(ex::when_all(ex::just()));

    check_val_types<ex::__mset<pack<int, double>>>(ex::when_all(ex::just(3), ex::just(0.14)));
    check_val_types<ex::__mset<pack<int, double, int, double>>>(
      ex::when_all(ex::just(3), ex::just(0.14), ex::just(1, 0.4142)));

    // if one child returns void, then the value is simply missing
    check_val_types<ex::__mset<pack<int, double>>>(
      ex::when_all(ex::just(3), ex::just(), ex::just(0.14)));

    // if children send references, they get decayed
    check_val_types<ex::__mset<pack<int, double>>>(
      ex::when_all(ex::split(ex::just(3)), ex::split(ex::just(0.14))));
  }

  TEST_CASE("when_all has the error_types based on the children", "[adaptors][when_all]") {
    check_err_types<ex::__mset<int>>(ex::when_all(ex::just_error(13)));
    check_err_types<ex::__mset<double>>(ex::when_all(ex::just_error(3.14)));

    check_err_types<ex::__mset<>>(ex::when_all(ex::just()));

    check_err_types<ex::__mset<int, double>>(ex::when_all(ex::just_error(3), ex::just_error(0.14)));
    check_err_types<ex::__mset<int, double, std::string>>(
      ex::when_all(ex::just_error(3), ex::just_error(0.14), ex::just_error(std::string{"err"})));

    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::when_all(ex::just(13), ex::just_error(std::exception_ptr{}), ex::just_stopped()));
  }

  TEST_CASE(
    "when_all has sends_stopped == true if and only if at least one child sends stopped",
    "[adaptors][when_all]") {
    check_sends_stopped<false>(ex::when_all(ex::just(13)));
    check_sends_stopped<false>(ex::when_all(ex::just_error(-1)));
    check_sends_stopped<true>(ex::when_all(ex::just_stopped()));

    check_sends_stopped<false>(ex::when_all(ex::just(3), ex::just(0.14)));
    check_sends_stopped<true>(ex::when_all(ex::just(3), ex::just_error(-1), ex::just_stopped()));
  }

  struct test_domain1 { };

  struct test_domain2 : test_domain1 { };

  TEST_CASE("when_all propagates domain from children", "[adaptors][when_all]") {
    basic_inline_scheduler<test_domain1> sched1;
    basic_inline_scheduler<test_domain2> sched2;

    auto snd =
      ex::when_all(ex::starts_on(sched1, ex::just(13)), ex::starts_on(sched2, ex::just(3.14)));
    auto env = ex::get_env(snd);
    auto domain = ex::get_completion_domain<ex::set_value_t>(env, ex::env<>{});
    STATIC_REQUIRE(std::same_as<decltype(domain), test_domain1>);
  }

  namespace {
    enum customize : std::size_t {
      early,
      late,
      none
    };

    template <class Tag, customize C, auto Fun>
    struct basic_domain {
      template <ex::sender_expr_for<Tag> Sender, class... Env>
        requires(sizeof...(Env) == C)
      auto transform_sender(STDEXEC::set_value_t, Sender&&, Env&&...) const {
        return Fun();
      }
    };
  } // anonymous namespace

  TEST_CASE("when_all works with custom domain", "[adaptors][when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::when_all_t, customize::none, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd =
        ex::when_all(ex::transfer_just(scheduler(), 3), ex::transfer_just(scheduler(), 0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::when_all_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd), ex::env{});
    }

    SECTION("late customization") {
      using domain = basic_domain<ex::when_all_t, customize::late, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::starts_on(scheduler(), ex::when_all(ex::just(3), ex::just(0.1415)));
      wait_for_value(std::move(snd), std::string{"hello world"});
    }
  }

  TEST_CASE("when_all_with_variant works with custom domain", "[adaptors][when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::when_all_with_variant_t, customize::none, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::when_all_with_variant(
        ex::transfer_just(scheduler(), 3), ex::transfer_just(scheduler(), 0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::when_all_with_variant_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd), ex::env{});
    }

    SECTION("late customization") {
      using domain = basic_domain<ex::when_all_with_variant_t, customize::late, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd =
        ex::starts_on(scheduler(), ex::when_all_with_variant(ex::just(3), ex::just(0.1415)));
      wait_for_value(std::move(snd), std::string{"hello world"});
    }
  }

  TEST_CASE("when_all_with_variant finds when_all customizations", "[adaptors][when_all]") {
    constexpr auto hello = [] {
      return ex::just(std::string{"hello world"});
    };

    SECTION("sender has correct domain") {
      using domain = basic_domain<ex::when_all_t, customize::none, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd = ex::when_all_with_variant(
        ex::transfer_just(scheduler(), 3), ex::transfer_just(scheduler(), 0.1415));
      static_assert(ex::sender_expr_for<decltype(snd), ex::when_all_with_variant_t>);
      [[maybe_unused]]
      domain dom = ex::get_completion_domain<ex::set_value_t>(ex::get_env(snd), ex::env{});
    }

    SECTION("late customization") {
      using domain = basic_domain<ex::when_all_t, customize::late, hello>;
      using scheduler = basic_inline_scheduler<domain>;

      auto snd =
        ex::starts_on(scheduler(), ex::when_all_with_variant(ex::just(3), ex::just(0.1415)));
      wait_for_value(std::move(snd), std::string{"hello world"});
    }
  }

  TEST_CASE("when_all defers stop handling to its children", "[adaptors][when_all]") {
    ex::inplace_stop_source source;
    source.request_stop();
    auto snd = ex::when_all(ex::just(), ex::just());
    static_assert(set_equivalent<
                  ex::completion_signatures_of_t<decltype(snd), ex::env<>>,
                  ex::completion_signatures<ex::set_value_t()>
    >);
    auto env = ex::prop(ex::get_stop_token, source.get_token());
    static_assert(set_equivalent<
                  ex::completion_signatures_of_t<decltype(snd), decltype(env)>,
                  ex::completion_signatures<ex::set_value_t()>
    >);
    auto op = ex::connect(snd, expect_void_receiver{});
    ex::start(op);
  }


  TEST_CASE(
    "when_all handles stop requests from the environment correctly",
    "[adaptors][when_all") {
    auto snd = ex::when_all(completes_if(false), completes_if(false));

    exec::async_scope scope;
    scope.spawn(snd);
    scope.request_stop();
    ex::sync_wait(scope.on_empty());
  }
} // namespace
