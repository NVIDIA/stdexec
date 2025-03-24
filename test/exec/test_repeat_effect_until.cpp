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
#include "exec/on.hpp"
#include "exec/trampoline_scheduler.hpp"
#include "exec/static_thread_pool.hpp"
#include "stdexec/concepts.hpp"
#include "stdexec/execution.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <catch2/catch.hpp>

#include <memory>
#include <utility>

using namespace stdexec;

namespace {

  struct boolean_sender {
    using sender_concept = stdexec::sender_t;
    using __t = boolean_sender;
    using __id = boolean_sender;
    using completion_signatures = stdexec::completion_signatures<set_value_t(bool)>;

    template <class Receiver>
    struct operation {
      Receiver rcvr_;
      int counter_;

      void start() & noexcept {
        if (counter_ == 0) {
          stdexec::set_value(static_cast<Receiver &&>(rcvr_), true);
        } else {
          stdexec::set_value(static_cast<Receiver &&>(rcvr_), false);
        }
      }
    };

    template <receiver_of<completion_signatures> Receiver>
    friend auto tag_invoke(connect_t, boolean_sender self, Receiver rcvr) -> operation<Receiver> {
      return {static_cast<Receiver &&>(rcvr), --*self.counter_};
    }

    std::shared_ptr<int> counter_ = std::make_shared<int>(1000);
  };

  TEST_CASE("repeat_effect_until returns a sender", "[adaptors][repeat_effect_until]") {
    auto snd = exec::repeat_effect_until(ex::just() | then([] { return false; }));
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE(
    "repeat_effect_until with environment returns a sender",
    "[adaptors][repeat_effect_until]") {
    auto snd = exec::repeat_effect_until(just() | then([] { return true; }));
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE(
    "repeat_effect_until produces void value to downstream receiver",
    "[adaptors][repeat_effect_until]") {
    sender auto source = just(1) | then([](int) { return true; });
    sender auto snd = exec::repeat_effect_until(std::move(source));
    // The receiver checks if we receive the void value
    auto op = stdexec::connect(std::move(snd), expect_void_receiver{});
    stdexec::start(op);
  }

  TEST_CASE("simple example for repeat_effect_until", "[adaptors][repeat_effect_until]") {
    sender auto snd = exec::repeat_effect_until(boolean_sender{});
    stdexec::sync_wait(std::move(snd));
  }

  TEST_CASE("repeat_effect_until works with pipeline operator", "[adaptors][repeat_effect_until]") {
    bool should_stopped = true;
    ex::sender auto snd = just(should_stopped) | exec::repeat_effect_until()
                        | then([] { return 1; });
    wait_for_value(std::move(snd), 1);
  }

  TEST_CASE(
    "repeat_effect_until works when input sender produces an int value",
    "[adaptors][repeat_effect_until]") {
    sender auto snd = exec::repeat_effect_until(just(1));
    auto op = stdexec::connect(std::move(snd), expect_void_receiver{});
    stdexec::start(op);
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
    auto input_snd = just() | then([&p] { return p; });
    stdexec::sync_wait(exec::repeat_effect_until(std::move(input_snd)));
  }

  TEST_CASE(
    "repeat_effect_until forwards set_error calls of other types",
    "[adaptors][repeat_effect_until]") {
    auto snd = just_error(std::string("error")) | exec::repeat_effect_until();
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string("error")});
    stdexec::start(op);
  }

  TEST_CASE("repeat_effect_until forwards set_stopped calls", "[adaptors][repeat_effect_until]") {
    auto snd = just_stopped() | exec::repeat_effect_until();
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    stdexec::start(op);
  }

  TEST_CASE(
    "running deeply recursing algo on repeat_effect_until doesn't blow the stack",
    "[adaptors][repeat_effect_until]") {
    int n = 1;
    sender auto snd = exec::repeat_effect_until(just() | then([&n] {
                                                  ++n;
                                                  return n == 1'000'000;
                                                }));
    stdexec::sync_wait(std::move(snd));
    CHECK(n == 1'000'000);
  }

  TEST_CASE("repeat_effect_until works when changing threads", "[adaptors][repeat_effect_until]") {
    exec::static_thread_pool pool{2};
    bool called{false};
    sender auto snd = exec::on(
      pool.get_scheduler(), //
      ex::just()            //
        | ex::then([&] {
            called = true;
            return called;
          })
        | exec::repeat_effect_until());
    stdexec::sync_wait(std::move(snd));

    REQUIRE(called);
  }
} // namespace
