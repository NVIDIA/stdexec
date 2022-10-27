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

namespace ex = stdexec;

// For testing `transfer_when_all` we assume that, the main implementation is based on `transfer`
// and `when_all`. As both of these are tested independently, we provide fewer tests here.

// For testing `transfer_when_all_with_variant`, we just check a couple of examples, check
// customization, and we assume it's implemented in terms of `transfer_when_all`.

TEST_CASE("transfer_when_all returns a sender", "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("transfer_when_all with environment returns a sender", "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("transfer_when_all simple example", "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all(inline_scheduler{}, ex::just(3), ex::just(0.1415));
  auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
  auto op = ex::connect(std::move(snd1), expect_value_receiver{3.1415});
  ex::start(op);
}
TEST_CASE("transfer_when_all with no senders", "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all(inline_scheduler{});
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
}

TEST_CASE("transfer_when_all transfers the result when the scheduler dictates",
    "[adaptors][transfer_when_all]") {
  impulse_scheduler sched;
  auto snd = ex::transfer_when_all(sched, ex::just(3), ex::just(0.1415));
  auto snd1 = std::move(snd) | ex::then([](int x, double y) { return x + y; });
  double res{0.0};
  auto op = ex::connect(std::move(snd1), expect_value_receiver_ex{res});
  ex::start(op);
  CHECK(res == 0.0);
  sched.start_next();
  CHECK(res == 3.1415);
}
TEST_CASE("transfer_when_all with no senders transfers the result", "[adaptors][transfer_when_all]") {
  impulse_scheduler sched;
  auto snd = ex::transfer_when_all(sched);
  auto snd1 = std::move(snd) | ex::then([]() { return true; });
  bool res{false};
  auto op = ex::connect(std::move(snd1), expect_value_receiver_ex{res});
  ex::start(op);
  CHECK(!res);
  sched.start_next();
  CHECK(res);
}

TEST_CASE("transfer_when_all_with_variant returns a sender", "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all_with_variant(inline_scheduler{}, ex::just(3), ex::just(0.1415));
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}
TEST_CASE("transfer_when_all_with_variant with environment returns a sender",
    "[adaptors][transfer_when_all]") {
  auto snd = ex::transfer_when_all_with_variant(inline_scheduler{}, ex::just(3), ex::just(0.1415));
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}
TEST_CASE("transfer_when_all_with_variant basic example", "[adaptors][transfer_when_all]") {
  ex::sender auto snd = ex::transfer_when_all_with_variant( //
      inline_scheduler{},                                   //
      ex::just(2),                                          //
      ex::just(3.14)                                        //
  );
  wait_for_value(
      std::move(snd), std::variant<std::tuple<int>>{2}, std::variant<std::tuple<double>>{3.14});
}

using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));

auto tag_invoke(ex::transfer_when_all_t, inline_scheduler, my_string_sender_t, my_string_sender_t) {
  // Return a different sender when we invoke this custom defined on implementation
  return ex::just(std::string{"first program"});
}

TEST_CASE("transfer_when_all can be customized", "[adaptors][transfer_when_all]") {
  // The customization will return a different value
  auto snd = ex::transfer_when_all(                                 //
      inline_scheduler{},                                           //
      ex::transfer_just(inline_scheduler{}, std::string{"hello,"}), //
      ex::transfer_just(inline_scheduler{}, std::string{" world!"}) //
  );
  wait_for_value(std::move(snd), std::string{"first program"});
}

auto tag_invoke(ex::transfer_when_all_with_variant_t, inline_scheduler, my_string_sender_t,
    my_string_sender_t) {
  // Return a different sender when we invoke this custom defined on implementation
  return ex::just(std::string{"first program"});
}

TEST_CASE("transfer_when_all_with_variant can be customized", "[adaptors][transfer_when_all]") {
  // The customization will return a different value
  auto snd = ex::transfer_when_all_with_variant(                    //
      inline_scheduler{},                                           //
      ex::transfer_just(inline_scheduler{}, std::string{"hello,"}), //
      ex::transfer_just(inline_scheduler{}, std::string{" world!"}) //
  );
  wait_for_value(std::move(snd), std::string{"first program"});
}
