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
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace {

  TEST_CASE("stopped_as_error returns a sender", "[adaptors][stopped_as_error]") {
    auto snd = ex::stopped_as_error(ex::just(11), -1);
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("stopped_as_error with environment returns a sender", "[adaptors][stopped_as_error]") {
    auto snd = ex::stopped_as_error(ex::just(11), -1);
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("stopped_as_error simple example", "[adaptors][stopped_as_error]") {
    stopped_scheduler sched;
    auto snd = ex::stopped_as_error(ex::transfer_just(sched, 11), -1);
    auto op = ex::connect(std::move(snd), expect_error_receiver{-1});
    ex::start(op);
  }

  TEST_CASE("stopped_as_error can we piped", "[adaptors][stopped_as_error]") {
    inline_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 11) | ex::stopped_as_error(std::exception_ptr{});
    auto op = ex::connect(std::move(snd), expect_value_receiver{11});
    ex::start(op);
  }

  TEST_CASE("stopped_as_error can work with `just_stopped`", "[adaptors][stopped_as_error]") {
    ex::sender auto snd = ex::just_stopped() | ex::stopped_as_error(-1);
    auto op = ex::connect(std::move(snd), expect_error_receiver{-1});
    ex::start(op);
  }

  TEST_CASE("stopped_as_error can we waited on", "[adaptors][stopped_as_error]") {
    inline_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 11) | ex::stopped_as_error(std::exception_ptr{});
    wait_for_value(std::move(snd), 11);
  }

  TEST_CASE("stopped_as_error using error_code error type", "[adaptors][stopped_as_error]") {
    impulse_scheduler sched; // scheduler that can send stopped signals
    ex::sender auto snd = ex::transfer_just(sched, 11)
                        | ex::stopped_as_error(std::error_code(1, std::generic_category()));
    check_val_types<ex::__mset<pack<int>>>(snd);
    check_err_types<ex::__mset<std::error_code>>(snd);
    check_sends_stopped<false>(snd);

    auto op = ex::connect(std::move(snd), expect_value_receiver{11});
    ex::start(op);
    sched.start_next();
  }

  TEST_CASE(
    "stopped_as_error using error_code error type, for stopped signal",
    "[adaptors][stopped_as_error]") {
    stopped_scheduler sched;
    std::error_code errcode(1, std::generic_category());
    ex::sender auto snd = ex::transfer_just(sched, 11) | ex::stopped_as_error(errcode);
    check_val_types<ex::__mset<pack<int>>>(snd);
    check_err_types<ex::__mset<std::error_code>>(snd);
    check_sends_stopped<false>(snd);

    auto op = ex::connect(std::move(snd), expect_error_receiver{errcode});
    ex::start(op);
  }

  // `stopped_as_error` is implemented in terms of `let_error`, so the tests for `let_error` cover
  // more ground.

  TEST_CASE(
    "stopped_as_error keeps values_type from input sender",
    "[adaptors][stopped_as_error]") {
    inline_scheduler sched;
    check_val_types<ex::__mset<pack<int>>>(ex::transfer_just(sched, 23) | ex::stopped_as_error(-1));
    check_val_types<ex::__mset<pack<double>>>(
      ex::transfer_just(sched, std::numbers::pi) | ex::stopped_as_error(-1));
  }

  TEST_CASE(
    "stopped_as_error keeps error_types from input sender",
    "[adaptors][stopped_as_error]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{-1};

    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1, 11) | ex::stopped_as_error(std::exception_ptr{}));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2, 13) | ex::stopped_as_error(std::exception_ptr{}));

    check_err_types<ex::__mset<int, std::exception_ptr>>(
      ex::transfer_just(sched3, 13) | ex::stopped_as_error(std::exception_ptr{}));
  }

  TEST_CASE("stopped_as_error can add more types to error_types", "[adaptors][stopped_as_error]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{-1};

    check_err_types<ex::__mset<>>(ex::transfer_just(sched1, 11) | ex::stopped_as_error(-1));
    check_err_types<ex::__mset<std::exception_ptr, int>>(
      ex::transfer_just(sched2, 13) | ex::stopped_as_error(-1));

    check_err_types<ex::__mset<int>>(ex::transfer_just(sched3, 13) | ex::stopped_as_error(-1));

    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1, 11) | ex::stopped_as_error(-1)
      | ex::stopped_as_error(std::string{"err"}));
  }

  TEST_CASE("stopped_as_error overrides sends_stopped to false", "[adaptors][stopped_as_error]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};
    stopped_scheduler sched3{};

    check_sends_stopped<false>(ex::transfer_just(sched1, 1) | ex::stopped_as_error(-1));
    check_sends_stopped<false>(ex::transfer_just(sched2, 2) | ex::stopped_as_error(-1));
    check_sends_stopped<false>(ex::transfer_just(sched3, 3) | ex::stopped_as_error(-1));
  }
} // namespace
