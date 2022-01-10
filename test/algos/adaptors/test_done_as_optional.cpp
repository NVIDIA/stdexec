/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = std::execution;

TEST_CASE("done_as_optional returns a sender", "[adaptors][done_as_optional]") {
  auto snd = ex::done_as_optional(ex::just(11));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("done_as_optional returns a typed_sender", "[adaptors][done_as_optional]") {
  auto snd = ex::done_as_optional(ex::just(11));
  static_assert(ex::typed_sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("done_as_optional simple example", "[adaptors][done_as_optional]") {
  done_scheduler sched;
  auto snd = ex::done_as_optional(ex::transfer_just(sched, 11));
  auto op = ex::connect(std::move(snd), expect_value_receiver{std::optional<int>{}});
  ex::start(op);
}

TEST_CASE("done_as_optional can we waited on", "[adaptors][done_as_optional]") {
  ex::sender auto snd = ex::just(11) | ex::done_as_optional();
  wait_for_value(std::move(snd), std::optional<int>{11});
}

TEST_CASE(
    "done_as_optional shall not work with multi-value senders", "[adaptors][done_as_optional]") {
  auto snd = ex::just(3, 0.1415) | ex::done_as_optional();
  // static_assert(!ex::sender<decltype(snd)>); // TODO
  static_assert(!std::invocable<ex::connect_t, decltype(snd), expect_error_receiver>);
}

TEST_CASE("done_as_optional shall not work with senders that have multiple alternatives",
    "[adaptors][done_as_optional]") {
  ex::sender auto in_snd =
      ex::just(13) //
      | ex::let_error([](std::exception_ptr) { return ex::just(std::string{"err"}); });
  check_val_types<type_array<type_array<int>, type_array<std::string>>>(in_snd);
  auto snd = std::move(in_snd) | ex::done_as_optional();
  // static_assert(!ex::sender<decltype(snd)>); // TODO
  static_assert(!std::invocable<ex::connect_t, decltype(snd), expect_error_receiver>);
}

TEST_CASE("done_as_optional forwards errors", "[adaptors][done_as_optional]") {
  error_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) | ex::done_as_optional();
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("done_as_optional doesn't forward cancellation", "[adaptors][done_as_optional]") {
  done_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) | ex::done_as_optional();
  wait_for_value(std::move(snd), std::optional<int>{});
}

TEST_CASE("done_as_optional adds std::optional to values_type", "[adaptors][done_as_optional]") {
  check_val_types<type_array<type_array<std::optional<int>>>>(
      ex::just(23) | ex::done_as_optional());
  check_val_types<type_array<type_array<std::optional<double>>>>(
      ex::just(3.1415) | ex::done_as_optional());
}
TEST_CASE(
    "TODO: done_as_optional keeps error_types from input sender", "[adaptors][done_as_optional]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{-1};

  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched1, 11) | ex::done_as_optional());
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched2, 13) | ex::done_as_optional());

  // TODO: error types should be forwarded (transfer_just bug)
  // check_err_types<type_array<int, std::exception_ptr>>( //
  //     ex::transfer_just(sched3, 13) | ex::done_as_optional());
  // Invalid check:
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched3, 13) | ex::done_as_optional());
}
TEST_CASE("done_as_optional overrides send_done to false", "[adaptors][done_as_optional]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  done_scheduler sched3{};

  check_sends_done<false>( //
      ex::transfer_just(sched1, 1) | ex::done_as_optional());
  check_sends_done<false>( //
      ex::transfer_just(sched2, 2) | ex::done_as_optional());
  check_sends_done<false>( //
      ex::transfer_just(sched3, 3) | ex::done_as_optional());
}
