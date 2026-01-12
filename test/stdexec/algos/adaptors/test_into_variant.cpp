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
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace {

  TEST_CASE("into_variant returns a sender", "[adaptors][into_variant]") {
    auto snd = ex::into_variant(ex::just(11));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("into_variant with environment returns a sender", "[adaptors][into_variant]") {
    auto snd = ex::into_variant(ex::just(11));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("into_variant simple example", "[adaptors][into_variant]") {
    bool called{false};
    auto snd = ex::into_variant(ex::just(11)) | ex::then([&](std::variant<std::tuple<int>> x) {
                 called = true;
                 CHECK(std::get<0>(std::get<0>(x)) == 11);
               });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
    CHECK(called);
  }

  TEST_CASE("into_variant returning void can we waited on", "[adaptors][into_variant]") {
    ex::sender auto snd = ex::just(11) | ex::into_variant();
    wait_for_value(std::move(snd), std::variant<std::tuple<int>>{11});
  }

  TEST_CASE(
    "into_variant with senders that sends multiple values at once",
    "[adaptors][into_variant]") {
    ex::sender auto snd = ex::just(3, 0.1415) | ex::into_variant();
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int, double>>{std::make_tuple(3, 0.1415)});
  }

  TEST_CASE(
    "into_variant with senders that have multiple alternatives",
    "[adaptors][into_variant]") {
    ex::sender auto in_snd = fallible_just{13} | ex::let_error([](std::exception_ptr) {
                               return ex::just(std::string{"err"});
                             });
    check_val_types<ex::__mset<pack<int>, pack<std::string>>>(std::move(in_snd));

    ex::sender auto snd = std::move(in_snd) | ex::into_variant();
    wait_for_value(
      std::move(snd), std::variant<std::tuple<int>, std::tuple<std::string>>{std::make_tuple(13)});
  }

  TEST_CASE("into_variant can be used with just_error", "[adaptors][into_variant]") {
    ex::sender auto snd = ex::just_error(std::string{"err"}) | ex::into_variant();
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
    ex::start(op);
  }

  TEST_CASE("into_variant can be used with just_stopped", "[adaptors][into_variant]") {
    ex::sender auto snd = ex::just_stopped() | ex::into_variant();
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("into_variant forwards errors", "[adaptors][into_variant]") {
    error_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 13) | ex::into_variant();
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("into_variant forwards cancellation", "[adaptors][into_variant]") {
    stopped_scheduler sched;
    ex::sender auto snd = ex::transfer_just(sched, 13) | ex::into_variant();
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "into_variant has the values_type corresponding to the given values",
    "[adaptors][into_variant]") {
    check_val_types<ex::__mset<pack<std::variant<std::tuple<>>>>>(ex::just() | ex::into_variant());
    check_val_types<ex::__mset<pack<std::variant<std::tuple<int>>>>>(
      ex::just(23) | ex::into_variant());
    check_val_types<ex::__mset<pack<std::variant<std::tuple<double>>>>>(
      ex::just(std::numbers::pi) | ex::into_variant());

    check_val_types<ex::__mset<pack<std::variant<std::tuple<int, double>>>>>(
      ex::just(3, 0.1415) | ex::into_variant());

    check_val_types<ex::__mset<pack<std::variant<std::tuple<int>, std::tuple<std::string>>>>>(
      fallible_just{13}
      | ex::let_error([](std::exception_ptr) { return ex::just(std::string{"err"}); })
      // sender here can send either `int` or `std::string`
      | ex::into_variant());
  }

  TEST_CASE("into_variant keeps error_types from input sender", "[adaptors][into_variant]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};

    check_err_types<ex::__mset<std::exception_ptr>>(ex::transfer_just(sched1) | ex::into_variant());
    check_err_types<ex::__mset<std::exception_ptr>>(ex::transfer_just(sched2) | ex::into_variant());
    check_err_types<ex::__mset<std::exception_ptr, int>>(ex::just_error(-1) | ex::into_variant());
  }

  TEST_CASE("into_variant keeps sends_stopped from input sender", "[adaptors][into_variant]") {
    inline_scheduler sched1{};
    error_scheduler sched2{};

    check_sends_stopped<false>(ex::transfer_just(sched1) | ex::into_variant());
    check_sends_stopped<true>(ex::transfer_just(sched2) | ex::into_variant());
    check_sends_stopped<true>(ex::just_stopped() | ex::into_variant());
  }

} // namespace
