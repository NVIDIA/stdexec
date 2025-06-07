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
#include <numbers>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

namespace {

  TEST_CASE("stopped_as_optional returns a sender", "[adaptors][stopped_as_optional]") {
    auto snd = ex::stopped_as_optional(ex::just(11));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "stopped_as_optional with environment returns a sender",
    "[adaptors][stopped_as_optional]") {
    auto snd = ex::stopped_as_optional(ex::just(11));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("stopped_as_optional simple example", "[adaptors][stopped_as_optional]") {
    stopped_scheduler sched;
    auto snd = ex::stopped_as_optional(ex::transfer_just(sched, 11));
    auto op = ex::connect(std::move(snd), expect_value_receiver{std::optional<int>{}});
    ex::start(op);
  }

  TEST_CASE("stopped_as_optional can we waited on", "[adaptors][stopped_as_optional]") {
    ex::sender auto snd = ex::just(11) | ex::stopped_as_optional();
    wait_for_value(std::move(snd), std::optional<int>{11});
  }

  TEST_CASE(
    "stopped_as_optional shall not work with multi-value senders",
    "[adaptors][stopped_as_optional]") {
    auto snd = ex::just(3, 0.1415) | ex::stopped_as_optional();
    static_assert(!ex::sender_to<decltype(snd), expect_error_receiver<>>);
  }

  TEST_CASE(
    "stopped_as_optional shall not work with senders that have multiple alternatives",
    "[adaptors][stopped_as_optional]") {
    ex::sender auto in_snd = fallible_just{13} | ex::let_error([](std::exception_ptr) {
                               return ex::just(std::string{"err"});
                             });
    check_val_types<ex::__mset<pack<int>, pack<std::string>>>(in_snd);
    auto snd = std::move(in_snd) | ex::stopped_as_optional();
    static_assert(!ex::sender_to<decltype(snd), expect_error_receiver<>>);
  }

  TEST_CASE("stopped_as_optional forwards errors", "[adaptors][stopped_as_optional]") {
    error_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 13) | ex::stopped_as_optional();
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("stopped_as_optional doesn't forward cancellation", "[adaptors][stopped_as_optional]") {
    stopped_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 13) | ex::stopped_as_optional();
    wait_for_value(std::move(snd), std::optional<int>{});
  }

  TEST_CASE(
    "stopped_as_optional adds std::optional to values_type",
    "[adaptors][stopped_as_optional]") {
    check_val_types<ex::__mset<pack<std::optional<int>>>>(ex::just(23) | ex::stopped_as_optional());
    check_val_types<ex::__mset<pack<std::optional<double>>>>(
      ex::just(std::numbers::pi) | ex::stopped_as_optional());
  }

  TEST_CASE(
    "stopped_as_optional keeps error_types from input sender",
    "[adaptors][stopped_as_optional]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{-1};

    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched1, 11) | ex::stopped_as_optional());
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2, 13) | ex::stopped_as_optional());

    check_err_types<ex::__mset<std::exception_ptr, int>>(
      ex::transfer_just(sched3, 13) | ex::stopped_as_optional());
  }

  TEST_CASE(
    "stopped_as_optional overrides sends_stopped to false",
    "[adaptors][stopped_as_optional]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::transfer_just(sched1, 1) | ex::stopped_as_optional());
    check_sends_stopped<false>(ex::transfer_just(sched2, 2) | ex::stopped_as_optional());
    check_sends_stopped<false>(ex::transfer_just(sched3, 3) | ex::stopped_as_optional());
  }
} // namespace
