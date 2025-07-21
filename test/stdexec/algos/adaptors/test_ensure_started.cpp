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
#include <exec/async_scope.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;
using exec::async_scope;

namespace {

  TEST_CASE("ensure_started returns a sender", "[adaptors][ensure_started]") {
    auto snd = ex::ensure_started(ex::just(19));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  static const auto env = ex::prop{ex::get_scheduler, inline_scheduler{}};

  TEST_CASE("ensure_started with environment returns a sender", "[adaptors][ensure_started]") {
    auto snd = ex::ensure_started(ex::just(19), env);
    static_assert(ex::sender_in<decltype(snd), decltype(env)>);
    (void) snd;
  }

  TEST_CASE("ensure_started void value early", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::just() | ex::then([&] { called = true; });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE("ensure_started void value late", "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called{false};
    auto snd1 = ex::starts_on(sch, ex::just()) | ex::then([&] { called = true; });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK_FALSE(called);
    // execute the next scheduled item
    sch.start_next();
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE("ensure_started single value early", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::just() | ex::then([&] {
                  called = true;
                  return 42;
                });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_value_receiver{42});
    ex::start(op);
  }

  TEST_CASE("ensure_started single value late", "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called{false};
    auto snd1 = ex::starts_on(sch, ex::just()) | ex::then([&] {
                  called = true;
                  return 42;
                });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK_FALSE(called);
    // execute the next scheduled item
    sch.start_next();
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_value_receiver{42});
    ex::start(op);
  }

  TEST_CASE("ensure_started multiple values early", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::let_value(ex::just(), [&] {
      called = true;
      return ex::just(42, movable{56});
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_value_receiver{42, movable{56}});
    ex::start(op);
  }

  TEST_CASE("ensure_started multiple values late", "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called{false};
    auto snd1 = ex::let_value(ex::starts_on(sch, ex::just()), [&] {
      called = true;
      return ex::just(42, movable{56});
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK_FALSE(called);
    // execute the next scheduled item
    sch.start_next();
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_value_receiver{42, movable{56}});
    ex::start(op);
  }

  TEST_CASE("ensure_started error early", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::let_value(ex::just(), [&] {
      called = true;
      return ex::just_error(42);
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_error_receiver{42});
    ex::start(op);
  }

  TEST_CASE("ensure_started error late", "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called{false};
    auto snd1 = ex::let_value(ex::starts_on(sch, ex::just()), [&] {
      called = true;
      return ex::just_error(42);
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK_FALSE(called);
    // execute the next scheduled item
    sch.start_next();
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_error_receiver{42});
    ex::start(op);
  }

  TEST_CASE("ensure_started stopped early", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::let_value(ex::just(), [&] {
      called = true;
      return ex::just_stopped();
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("ensure_started stopped late", "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called{false};
    auto snd1 = ex::let_value(ex::starts_on(sch, ex::just()), [&] {
      called = true;
      return ex::just_stopped();
    });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK_FALSE(called);
    // execute the next scheduled item
    sch.start_next();
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "stopping ensure_started before the source completes calls set_stopped",
    "[adaptors][ensure_started]") {
    ex::inplace_stop_source stop_source;
    impulse_scheduler sch;
    bool called{false};
    auto snd = ex::starts_on(sch, ex::just(19))
             | ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()})
             | ex::ensure_started();
    auto op = ex::connect(std::move(snd), expect_stopped_receiver_ex{called});
    ex::start(op);
    // request stop before the source yields a value
    stop_source.request_stop();
    // make the source yield the value
    sch.start_next();
    CHECK(called);
  }

  TEST_CASE(
    "stopping ensure_started before the lazy opstate is started calls set_stopped",
    "[adaptors][ensure_started]") {
    ex::inplace_stop_source stop_source;
    impulse_scheduler sch;
    int count = 0;
    bool called{false};
    auto snd = ex::let_value(
                 ex::just() | ex::then([&] { ++count; }),
                 [=] { return ex::starts_on(sch, ex::just(19)); })
             | ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()})
             | ex::ensure_started();
    CHECK(count == 1);
    auto op = ex::connect(std::move(snd), expect_stopped_receiver_ex{called});
    // request stop before the source yields a value
    stop_source.request_stop();
    ex::start(op);
    // make the source yield the value
    sch.start_next();
    CHECK(called);
  }

  TEST_CASE(
    "stopping ensure_started after the task has already completed doesn't change the result",
    "[adaptors][ensure_started]") {
    ex::inplace_stop_source stop_source;
    int count = 0;
    auto snd = ex::just() | ex::then([&] {
                 ++count;
                 return 42;
               })
             | ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()})
             | ex::ensure_started();
    CHECK(count == 1);
    auto op = ex::connect(std::move(snd), expect_value_receiver{42});
    // request stop before the source yields a value
    stop_source.request_stop();
    ex::start(op);
  }

  TEST_CASE(
    "Dropping the sender without connecting it calls set_stopped",
    "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    bool called = false;
    {
      auto snd = ex::starts_on(sch, ex::just()) | ex::upon_stopped([&] { called = true; })
               | ex::ensure_started();
    }
    // make the source yield the value
    sch.start_next();
    CHECK(called);
  }

  TEST_CASE(
    "Dropping the opstate without starting it calls set_stopped",
    "[adaptors][ensure_started]") {
    impulse_scheduler sch;
    int state = -1;
    bool called = false;
    {
      auto snd = ex::starts_on(sch, ex::just()) | ex::upon_stopped([&] { called = true; })
               | ex::ensure_started();
      auto op = ex::connect(std::move(snd), logging_receiver{state});
    }
    // make the source yield the value
    sch.start_next();
    CHECK(called);
    // make sure the logging_receiver was never called
    CHECK(state == -1);
  }

  TEST_CASE("Repeated ensure_started compiles", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::just() | ex::then([&] { called = true; });
    CHECK_FALSE(called);
    auto snd2 = ex::ensure_started(std::move(snd1));
    auto snd = ex::ensure_started(std::move(snd2));
    STATIC_REQUIRE(ex::same_as<decltype(snd2), decltype(snd)>);
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE("ensure_started with move only input sender", "[adaptors][ensure_started]") {
    bool called{false};
    auto snd1 = ex::just(movable(42)) | ex::then([&](movable&&) { called = true; });
    CHECK_FALSE(called);
    auto snd = ex::ensure_started(std::move(snd1));
    CHECK(called);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "Repeated ensure_started with operator| does not return reference to temporary",
    "[adaptors][ensure_started]") {
    auto snd = ex::ensure_started(ex::just()) | ex::ensure_started();
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  struct my_sender {
    using sender_concept = ex::sender_t;
    using is_sender = void;

    using completion_signatures = ex::completion_signatures_of_t<decltype(ex::just())>;

    template <class Recv>
    friend auto tag_invoke(ex::connect_t, my_sender&&, Recv&& recv) {
      return ex::connect(ex::just(), std::forward<Recv>(recv));
    }

    template <class Recv>
    friend auto tag_invoke(ex::connect_t, const my_sender&, Recv&& recv) {
      return ex::connect(ex::just(), std::forward<Recv>(recv));
    }
  };

  TEST_CASE("ensure_started accepts a custom sender", "[adaptors][ensure_started]") {
    auto snd1 = my_sender();
    auto snd2 = ex::ensure_started(std::move(snd1));
    static_assert(ex::__well_formed_sender<decltype(snd1)>);
    static_assert(ex::__well_formed_sender<decltype(snd2)>);
    using Snd = decltype(snd2);
    static_assert(ex::enable_sender<Snd>);
    static_assert(ex::sender<Snd>);
    static_assert(ex::same_as<ex::env_of_t<Snd>, ex::env<>>);
    (void) snd1;
    (void) snd2;
  }
} // namespace
