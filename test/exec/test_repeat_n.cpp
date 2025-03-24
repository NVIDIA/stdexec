/*
 * Copyright (c) 2023 Runner-2019
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "exec/repeat_n.hpp"
#include "exec/on.hpp"
#include "exec/trampoline_scheduler.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/concepts.hpp"
#include "stdexec/execution.hpp"
#include <exception>
#include <system_error>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>
#include <iostream>

#include <catch2/catch.hpp>
#include <type_traits>

using namespace stdexec;

namespace {
  TEST_CASE("repeat_n returns a sender", "[adaptors][repeat_n]") {
    auto snd = exec::repeat_n(ex::just() | then([] { }), 10);
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("repeat_n with environment returns a sender", "[adaptors][repeat_n]") {
    auto snd = exec::repeat_n(just() | then([] { }), 10);
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("repeat_n produces void value to downstream receiver", "[adaptors][repeat_n]") {
    sender auto source = just(1) | then([](int) { });
    sender auto snd = exec::repeat_n(std::move(source), 10);
    // The receiver checks if we receive the void value
    auto op = stdexec::connect(std::move(snd), expect_void_receiver{});
    stdexec::start(op);
  }

  TEST_CASE("simple example for repeat_n", "[adaptors][repeat_n]") {
    sender auto snd = exec::repeat_n(just(), 2);
    stdexec::sync_wait(std::move(snd));
  }

  TEST_CASE("repeat_n works with with zero repetitions", "[adaptors][repeat_n]") {
    std::size_t count = 0;
    ex::sender auto snd = just() //
                        | then([&count] { ++count; }) | exec::repeat_n(0) | then([] { return 1; });
    wait_for_value(std::move(snd), 1);
    CHECK(count == 0);
  }

  TEST_CASE("repeat_n works with a single repetition", "[adaptors][repeat_n]") {
    std::size_t count = 0;
    ex::sender auto snd = just() //
                        | then([&count] { ++count; }) | exec::repeat_n(1) | then([] { return 1; });
    wait_for_value(std::move(snd), 1);
    CHECK(count == 1);
  }

  TEST_CASE("repeat_n works with multiple repetitions", "[adaptors][repeat_n]") {
    std::size_t count = 0;
    ex::sender auto snd = just() //
                        | then([&count] { ++count; }) | exec::repeat_n(3) | then([] { return 1; });
    wait_for_value(std::move(snd), 1);
    CHECK(count == 3);
  }

  TEST_CASE("repeat_n forwards set_error calls of other types", "[adaptors][repeat_n]") {
    int count = 0;
    auto snd = let_value(
                 just(),
                 [&] {
                   ++count;
                   return just_error(std::string("error"));
                 })
             | exec::repeat_n(10);
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string("error")});
    stdexec::start(op);
    CHECK(count == 1);
  }

  TEST_CASE("repeat_n forwards set_stopped calls", "[adaptors][repeat_n]") {
    int count = 0;
    auto snd = let_value(
                 just(),
                 [&] {
                   ++count;
                   return just_stopped();
                 })
             | exec::repeat_n(10);
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    stdexec::start(op);
    CHECK(count == 1);
  }

  TEST_CASE(
    "running deeply recursing algo on repeat_n doesn't blow the stack",
    "[adaptors][repeat_n]") {
    int n = 0;
    sender auto snd = exec::repeat_n(just() | then([&n] { ++n; }), 1'000'000);
    stdexec::sync_wait(std::move(snd));
    CHECK(n == 1'000'000);
  }

  TEST_CASE("repeat_n works when changing threads", "[adaptors][repeat_n]") {
    exec::static_thread_pool pool{2};
    bool called{false};
    sender auto snd = exec::on(
      pool.get_scheduler(), //
      ex::just()            //
        | ex::then([&] { called = true; }) | exec::repeat_n(10));
    stdexec::sync_wait(std::move(snd));
    REQUIRE(called);
  }
} // namespace
