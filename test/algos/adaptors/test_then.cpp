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

TEST_CASE("then returns a sender", "[adaptors][then]") {
  auto snd = ex::then(ex::just(), [] {});
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("then with environment returns a sender", "[adaptors][then]") {
  auto snd = ex::then(ex::just(), [] {});
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("then simple example", "[adaptors][then]") {
  bool called{false};
  auto snd = ex::then(ex::just(), [&] { called = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

TEST_CASE("then can be piped", "[adaptors][then]") {
  ex::sender auto snd = ex::just() | ex::then([] {});
  (void)snd;
}

TEST_CASE("then returning void can we waited on", "[adaptors][then]") {
  ex::sender auto snd = ex::just() | ex::then([] {});
  std::this_thread::sync_wait(std::move(snd));
}

TEST_CASE("then can be used to transform the value", "[adaptors][then]") {
  auto snd = ex::just(13) | ex::then([](int x) -> int { return 2 * x + 1; });
  wait_for_value(std::move(snd), 27);
}

TEST_CASE("then can be used to change the value type", "[adaptors][then]") {
  auto snd = ex::just(3) | ex::then([](int x) -> double { return x + 0.1415; });
  wait_for_value(std::move(snd), 3.1415);
}

TEST_CASE("then can be used with multiple parameters", "[adaptors][then]") {
  auto snd = ex::just(3, 0.1415) | ex::then([](int x, double y) -> double { return x + y; });
  wait_for_value(std::move(snd), 3.1415);
}

TEST_CASE("then can throw, and set_error will be called", "[adaptors][then]") {
  auto snd = ex::just(13) //
             | ex::then([](int x) -> int {
                 throw std::logic_error{"err"};
                 return x + 5;
               });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("TODO: then can be used with just_error", "[adaptors][then]") {
  ex::sender auto snd = ex::just_error(std::string{"err"}) //
                        | ex::then([]() -> int { return 17; });
  // TODO: this should work
  // auto op = ex::connect(std::move(snd), expect_error_receiver{});
  // ex::start(op);
  // invalid check:
  static_assert(!std::invocable<ex::connect_t, decltype(snd), expect_error_receiver>);
}
TEST_CASE("then can be used with just_stopped", "[adaptors][then]") {
  ex::sender auto snd = ex::just_stopped() | //
                        ex::then([]() -> int { return 17; });
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("then function is not called on error", "[adaptors][then]") {
  bool called{false};
  error_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                        | ex::then([&](int x) -> int {
                            called = true;
                            return x + 5;
                          });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}
TEST_CASE("then function is not called when cancelled", "[adaptors][then]") {
  bool called{false};
  stopped_scheduler sched;
  ex::sender auto snd = ex::transfer_just(sched, 13) //
                        | ex::then([&](int x) -> int {
                            called = true;
                            return x + 5;
                          });
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}

TEST_CASE("then has the values_type corresponding to the given values", "[adaptors][then]") {
  check_val_types<type_array<type_array<int>>>(ex::just() | ex::then([] { return 7; }));
  check_val_types<type_array<type_array<double>>>(ex::just() | ex::then([] { return 3.14; }));
  check_val_types<type_array<type_array<std::string>>>(
      ex::just() | ex::then([] { return std::string{"hello"}; }));
}
TEST_CASE("TODO: then keeps error_types from input sender", "[adaptors][then]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched1) | ex::then([] {}));
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched2) | ex::then([] {}));
  // check_err_types<type_array<int, std::exception_ptr>>( //
  //     ex::transfer_just(sched3) | ex::then([] {}));
  // TODO: then should also forward the error types sent by the input sender
  // incorrect check:
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched3) | ex::then([] {}));
}
TEST_CASE("TODO: then keeps sends_stopped from input sender", "[adaptors][then]") {
  inline_scheduler sched1{};
  error_scheduler sched2{};
  stopped_scheduler sched3{};

  check_sends_stopped<false>( //
      ex::transfer_just(sched1) | ex::then([] {}));
  check_sends_stopped<false>( //
      ex::transfer_just(sched2) | ex::then([] {}));
  // check_sends_stopped<true>( //
  //     ex::transfer_just(sched3) | ex::then([] {}));
  // TODO: transfer should forward its "sends_stopped" info
  // incorrect check:
  check_sends_stopped<false>( //
      ex::transfer_just(sched3) | ex::then([] {}));
}

// Return a different sender when we invoke this custom defined on implementation
using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));
template <typename Fun>
auto tag_invoke(ex::then_t, inline_scheduler sched, my_string_sender_t, Fun) {
  return ex::just(std::string{"hallo"});
}

TEST_CASE("then can be customized", "[adaptors][then]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"}) //
             | ex::then([](std::string x) { return x + ", world"; });
  wait_for_value(std::move(snd), std::string{"hallo"});
}
