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
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

#include <chrono>

namespace ex = stdexec;

using namespace std::chrono_literals;

TEST_CASE("let_stopped returns a sender", "[adaptors][let_stopped]") {
  auto snd = ex::let_stopped(ex::just(), [] { return ex::just(); });
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE("let_stopped with environment returns a sender", "[adaptors][let_stopped]") {
  auto snd = ex::let_stopped(ex::just(), [] { return ex::just(); });
  static_assert(ex::sender_in<decltype(snd), empty_env>);
  (void) snd;
}

TEST_CASE("let_stopped simple example", "[adaptors][let_stopped]") {
  bool called{false};
  auto snd = ex::let_stopped(ex::just_stopped(), [&] {
    called = true;
    return ex::just();
  });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

TEST_CASE("let_stopped can be piped", "[adaptors][let_stopped]") {
  ex::sender auto snd = ex::just() | ex::let_stopped([] { return ex::just(); });
  (void) snd;
}

TEST_CASE(
  "let_stopped returning void can we waited on (cancel annihilation)",
  "[adaptors][let_stopped]") {
  ex::sender auto snd = ex::just_stopped() | ex::let_stopped([] { return ex::just(); });
  stdexec::sync_wait(std::move(snd));
}

TEST_CASE(
  "let_stopped can be used to produce values (cancel to value)",
  "[adaptors][let_stopped]") {
  ex::sender auto snd = ex::just_stopped() //
                      | ex::let_stopped([] { return ex::just(std::string{"cancelled"}); });
  wait_for_value(std::move(snd), std::string{"cancelled"});
}

TEST_CASE("let_stopped can throw, calling set_error", "[adaptors][let_stopped]") {
  auto snd = ex::just_stopped() //
           | ex::let_stopped([] {
               throw std::logic_error{"err"};
               return ex::just(1);
             });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("let_stopped can be used with just_error", "[adaptors][let_stopped]") {
  ex::sender auto snd = ex::just_error(1) //
                      | ex::let_stopped([] { return ex::just(17); });
  auto op = ex::connect(std::move(snd), expect_error_receiver{1});
  ex::start(op);
}

TEST_CASE("let_stopped function is not called on regular flow", "[adaptors][let_stopped]") {
  bool called{false};
  error_scheduler sched;
  ex::sender auto snd = ex::just(13) //
                      | ex::let_stopped([&] {
                          called = true;
                          return ex::just(0);
                        });
  auto op = ex::connect(std::move(snd), expect_value_receiver{13});
  ex::start(op);
  CHECK_FALSE(called);
}

TEST_CASE("let_stopped function is not called on error flow", "[adaptors][let_stopped]") {
  bool called{false};
  error_scheduler<int> sched{42};
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                      | ex::let_stopped([&] {
                          called = true;
                          return ex::just(0);
                        });
  auto op = ex::connect(std::move(snd), expect_error_receiver{42});
  ex::start(op);
  CHECK_FALSE(called);
}

TEST_CASE(
  "let_stopped has the values_type from the input sender if returning error",
  "[adaptors][let_stopped]") {
  check_val_types<type_array<type_array<int>>>(
    ex::just(7) //
    | ex::let_stopped([] { return ex::just_error(0); }));
  check_val_types<type_array<type_array<double>>>(
    ex::just(3.14) //
    | ex::let_stopped([] { return ex::just_error(0); }));
  check_val_types<type_array<type_array<std::string>>>(
    ex::just(std::string{"hello"}) //
    | ex::let_stopped([] { return ex::just_error(0); }));
}

TEST_CASE(
  "let_stopped adds to values_type the value types of the returned sender",
  "[adaptors][let_stopped]") {
  check_val_types<type_array<type_array<int>>>(
    ex::just(1) //
    | ex::let_stopped([] { return ex::just(11); }));
  check_val_types<type_array<type_array<int>>>(
    ex::just(1) //
    | ex::let_stopped([] { return ex::just(3.14); }));
  check_val_types<type_array<type_array<int>>>(
    ex::just(1) //
    | ex::let_stopped([] { return ex::just(std::string{"hello"}); }));
}

TEST_CASE(
  "let_stopped has the error_type from the input sender if returning value",
  "[adaptors][let_stopped]") {
  check_err_types<type_array<int>>(         //
    ex::just_error(7)                       //
    | ex::let_stopped([] { return ex::just(0); }));
  check_err_types<type_array<double>>(      //
    ex::just_error(3.14)                    //
    | ex::let_stopped([] { return ex::just(0); }));
  check_err_types<type_array<std::string>>( //
    ex::just_error(std::string{"hello"})    //
    | ex::let_stopped([] { return ex::just(0); }));
}

TEST_CASE("let_stopped adds to error_type of the input sender", "[adaptors][let_stopped]") {
  impulse_scheduler sched;
  ex::sender auto in_snd = ex::transfer_just(sched, 11);
  check_err_types<type_array<std::exception_ptr, int>>(         //
    in_snd                                                      //
    | ex::let_stopped([] { return ex::just_error(0); }));
  check_err_types<type_array<std::exception_ptr, double>>(      //
    in_snd                                                      //
    | ex::let_stopped([] { return ex::just_error(3.14); }));
  check_err_types<type_array<std::exception_ptr, std::string>>( //
    in_snd                                                      //
    | ex::let_stopped([] { return ex::just_error(std::string{"err"}); }));
}

TEST_CASE("let_stopped can be used instead of stopped_as_error", "[adaptors][let_stopped]") {
  impulse_scheduler sched;
  ex::sender auto in_snd = ex::transfer_just(sched, 11);
  check_val_types<type_array<type_array<int>>>(in_snd);
  check_err_types<type_array<>>(in_snd);
  check_sends_stopped<true>(in_snd);

  ex::sender auto snd = std::move(in_snd) | ex::let_stopped([] { return ex::just_error(-1); });

  check_val_types<type_array<type_array<int>>>(snd);
  check_err_types<type_array<std::exception_ptr, int>>(snd);
  check_sends_stopped<false>(snd);
}

TEST_CASE("let_stopped overrides sends_stopped from input sender", "[adaptors][let_stopped]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  // Returning ex::just
  check_sends_stopped<false>( //
    ex::transfer_just(sched1) //
    | ex::let_stopped([] { return ex::just(); }));
  check_sends_stopped<false>( //
    ex::transfer_just(sched2) //
    | ex::let_stopped([] { return ex::just(); }));
  check_sends_stopped<false>( //
    ex::transfer_just(sched3) //
    | ex::let_stopped([] { return ex::just(); }));

  // Returning ex::just_stopped
  check_sends_stopped<false>( //
    ex::transfer_just(sched1) //
    | ex::let_stopped([] { return ex::just_stopped(); }));
  check_sends_stopped<true>(  //
    ex::transfer_just(sched2) //
    | ex::let_stopped([] { return ex::just_stopped(); }));
  check_sends_stopped<true>(  //
    ex::transfer_just(sched3) //
    | ex::let_stopped([] { return ex::just_stopped(); }));
}

// Return a different sender when we invoke this custom defined on implementation
using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));

template <typename Fun>
auto tag_invoke(ex::let_stopped_t, inline_scheduler sched, my_string_sender_t, Fun) {
  return ex::just(std::string{"Don't stop me now"});
}

TEST_CASE("let_stopped can be customized", "[adaptors][let_stopped]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"}) //
           | ex::let_stopped([] { return ex::just(std::string{"stopped"}); });
  wait_for_value(std::move(snd), std::string{"Don't stop me now"});
}
