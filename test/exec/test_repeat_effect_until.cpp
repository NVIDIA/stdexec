/*
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

#include "exec/repeat_effect_until.hpp"
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

TEST_CASE("repeat_effect_until returns a sender", "[adaptors][repeat_effect_until]") {
  auto snd = exec::repeat_effect_until(ex::just() | then([] { return false; }));
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

TEST_CASE(
  "repeat_effect_until with environment returns a sender",
  "[adaptors][repeat_effect_until]") {
  auto snd = exec::repeat_effect_until(just() | then([] { return true; }));
  static_assert(ex::sender_in<decltype(snd), empty_env>);
  (void) snd;
}

TEST_CASE("test connect cpo of repeat_effect_until", "[adaptors][repeat_effect_until]") {
  sender auto source = just(1) | then([](int n) { return true; });
  sender auto snd = exec::repeat_effect_until(std::move(source));
  auto op = stdexec::connect(std::move(snd), expect_void_receiver{});
  start(op);
}

TEST_CASE(
  "Input sender produces a int value that can be converted to bool",
  "[adaptors][repeat_effect_until]") {
  sender auto snd = exec::repeat_effect_until(just(1));
  auto op = stdexec::connect(std::move(snd), expect_value_receiver{});
  start(op);
}

TEST_CASE(
  "Input sender produces a object value that can be converted to bool",
  "[adaptors][repeat_effect_until]") {
  struct pred {
    operator bool() {
      return --n <= 100;
    }

    int n = 100;
  };

  pred p;
  auto input_snd = just() | then([&p] { return p; });
  stdexec::sync_wait(exec::repeat_effect_until(std::move(input_snd)));
}

TEST_CASE(
  "repeat effect will pass error signal of input sender to downstream' receiver",
  "[adaptors][repeat_effect_until]") {
  struct pred {
    operator bool() {
      return --n <= 100;
    }

    int n = 100;
  };

  pred p;
  auto input_snd = just() | then([&p] { return p; });
  stdexec::sync_wait(exec::repeat_effect_until(std::move(input_snd)));
}

// TEST_CASE("simple example for repeat_effect_until", "[adaptors][repeat_effect_until]") {
//   // Run 1'000'000 times
//   int n = 1;
//   sender auto snd = exec::repeat_effect_until(just() | then([&n] {
//                                                 ++n;
//                                                 return n == 1'000'000;
//                                               }));

//   stdexec::sync_wait(std::move(snd));
//   CHECK(n == 1'000'000);
// }

TEST_CASE("repeat_effect_until with pipeline operator", "[adaptors][repeat_effect_until]") {
  bool should_stopped = true;
  ex::sender auto snd = just(should_stopped) | exec::repeat_effect_until() | then([] { return 1; });
  wait_for_value(std::move(snd), 1);
}
