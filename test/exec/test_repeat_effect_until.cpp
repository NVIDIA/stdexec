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

#include "exec/repeat_effect_until.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <catch2/catch.hpp>

#include <memory>
#include <utility>

namespace ex = stdexec;

namespace {

  struct boolean_sender {
    using sender_concept = ex::sender_t;
    using __t = boolean_sender;
    using __id = boolean_sender;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(bool)>;

    template <class Receiver>
    struct operation {
      Receiver rcvr_;
      int counter_;

      void start() & noexcept {
        if (counter_ == 0) {
          ex::set_value(static_cast<Receiver &&>(rcvr_), true);
        } else {
          ex::set_value(static_cast<Receiver &&>(rcvr_), false);
        }
      }
    };

    template <ex::receiver_of<completion_signatures> Receiver>
    friend auto
      tag_invoke(ex::connect_t, boolean_sender self, Receiver rcvr) -> operation<Receiver> {
      return {static_cast<Receiver &&>(rcvr), --*self.counter_};
    }

    std::shared_ptr<int> counter_ = std::make_shared<int>(1000);
  };

  TEST_CASE("repeat_effect_until returns a sender", "[adaptors][repeat_effect_until]") {
    auto snd = exec::repeat_effect_until(ex::just() | ex::then([] { return false; }));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "repeat_effect_until with environment returns a sender",
    "[adaptors][repeat_effect_until]") {
    auto snd = exec::repeat_effect_until(ex::just() | ex::then([] { return true; }));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE(
    "repeat_effect_until produces void value to downstream receiver",
    "[adaptors][repeat_effect_until]") {
    ex::sender auto source = ex::just(1) | ex::then([](int) { return true; });
    ex::sender auto snd = exec::repeat_effect_until(std::move(source));
    // The receiver checks if we receive the void value
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE("simple example for repeat_effect_until", "[adaptors][repeat_effect_until]") {
    ex::sender auto snd = exec::repeat_effect_until(boolean_sender{});
    ex::sync_wait(std::move(snd));
  }

  TEST_CASE("repeat_effect_until works with pipeline operator", "[adaptors][repeat_effect_until]") {
    bool should_stopped = true;
    ex::sender auto snd = ex::just(should_stopped) | exec::repeat_effect_until()
                        | ex::then([] { return 1; });
    wait_for_value(std::move(snd), 1);
  }

  TEST_CASE(
    "repeat_effect_until works when input sender produces an int value",
    "[adaptors][repeat_effect_until]") {
    ex::sender auto snd = exec::repeat_effect_until(ex::just(1));
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "repeat_effect_until works when input sender produces an object that can be converted to bool"
    "[adaptors][repeat_effect_until]") {
    struct pred {
      operator bool() {
        return --n <= 100;
      }

      int n = 100;
    };

    pred p;
    auto input_snd = ex::just() | ex::then([&p] { return p; });
    ex::sync_wait(exec::repeat_effect_until(std::move(input_snd)));
  }

  TEST_CASE(
    "repeat_effect_until forwards set_error calls of other types",
    "[adaptors][repeat_effect_until]") {
    auto snd = ex::just_error(std::string("error")) | exec::repeat_effect_until();
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string("error")});
    ex::start(op);
  }

  TEST_CASE("repeat_effect_until forwards set_stopped calls", "[adaptors][repeat_effect_until]") {
    auto snd = ex::just_stopped() | exec::repeat_effect_until();
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE(
    "running deeply recursing algo on repeat_effect_until doesn't blow the stack",
    "[adaptors][repeat_effect_until]") {
    int n = 1;
    ex::sender auto snd = exec::repeat_effect_until(ex::just() | ex::then([&n] {
                                                      ++n;
                                                      return n == 1'000'000;
                                                    }));
    ex::sync_wait(std::move(snd));
    CHECK(n == 1'000'000);
  }

  TEST_CASE("repeat_effect_until works when changing threads", "[adaptors][repeat_effect_until]") {
    exec::static_thread_pool pool{2};
    bool called{false};
    ex::sender auto snd = ex::on(pool.get_scheduler(), ex::just() | ex::then([&] {
                                                         called = true;
                                                         return called;
                                                       }) | exec::repeat_effect_until());
    ex::sync_wait(std::move(snd));

    REQUIRE(called);
  }

  TEST_CASE(
    "repeat_effect_until works with bulk on a static_thread_pool",
    "[adaptors][repeat_effect_until]") {
    exec::static_thread_pool pool{2};
    std::atomic<bool> failed{false};
    const auto tid = std::this_thread::get_id();
    bool called{false};
    ex::sender auto snd = stdexec::on(
      pool.get_scheduler(), ex::just() | ex::bulk(ex::par_unseq, 1024, [&](int) noexcept {
                              if (tid == std::this_thread::get_id()) {
                                failed = true;
                              }
                            }) | ex::then([&] {
                              called = true;
                              return called;
                            }) | exec::repeat_effect_until());
    stdexec::sync_wait(std::move(snd));
    REQUIRE(called);
  }

  TEST_CASE("repeat_effect repeats until an error is encountered", "[adaptors][repeat_effect]") {
    int counter = 0;
    ex::sender auto snd = exec::repeat_effect(
      succeed_n_sender(10, ex::set_error, std::string("error")) | ex::then([&] { ++counter; }));
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string("error")});
    ex::start(op);
    REQUIRE(counter == 10);
  }

  TEST_CASE("repeat_effect repeats until stopped is encountered", "[adaptors][repeat_effect]") {
    int counter = 0;
    ex::sender auto snd = exec::repeat_effect(
      succeed_n_sender(10, ex::set_stopped) | ex::then([&] { ++counter; }));
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
    REQUIRE(counter == 10);
  }
} // namespace
