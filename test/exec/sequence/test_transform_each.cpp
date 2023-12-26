/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "exec/sequence/transform_each.hpp"

#include "exec/sequence/any_sequence_of.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/iterate.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include <catch2/catch.hpp>

#include <exec/env.hpp>
#include <exec/on.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

namespace {

  struct next_rcvr {
    using __id = next_rcvr;
    using __t = next_rcvr;
    using receiver_concept = stdexec::receiver_t;

    friend auto tag_invoke(exec::set_next_t, next_rcvr, auto item) {
      return item;
    }

    friend void tag_invoke(stdexec::set_value_t, next_rcvr) noexcept {
    }

    friend stdexec::empty_env tag_invoke(stdexec::get_env_t, next_rcvr) noexcept {
      return {};
    }
  };

  TEST_CASE(
    "transform_each - transform sender applies adaptor to no elements",
    "[sequence_senders][transform_each][empty_sequence]") {
    int counter = 0;
    auto transformed = exec::transform_each(
      exec::empty_sequence(), stdexec::then([&counter]() noexcept { ++counter; }));
    auto op = exec::subscribe(transformed, next_rcvr{});
    stdexec::start(op);
    CHECK(counter == 0);
  }

  TEST_CASE(
    "transform_each - transform sender applies adaptor to a sender",
    "[sequence_senders][transform_each]") {
    int value = 0;
    auto transformed = exec::transform_each(
      stdexec::just(42), stdexec::then([&value](int x) noexcept { value = x; }));
    auto op = exec::subscribe(transformed, next_rcvr{});
    stdexec::start(op);
    CHECK(value == 42);
  }

  TEST_CASE(
    "transform_each - transform sender applies adaptor to a sender and ignores all values",
    "[sequence_senders][transform_each][ignore_all_values]") {
    int value = 0;
    auto transformed =
      exec::transform_each(stdexec::just(42), stdexec::then([&value](int x) { value = x; }))
      | exec::ignore_all_values();
    stdexec::sync_wait(transformed);
    CHECK(value == 42);
  }

#if STDEXEC_HAS_STD_RANGES()
  TEST_CASE(
    "transform_each - transform sender applies adaptor to each item",
    "[sequence_senders][transform_each][iterate]") {
    auto range = [](auto from, auto to) {
      return exec::iterate(std::views::iota(from, to));
    };
    auto then_each = [](auto f) {
      return exec::transform_each(stdexec::then(f));
    };
    int total = 0;
    auto sum = range(0, 10) //
             | then_each([&total](int x) noexcept { total += x; });
    stdexec::sync_wait(exec::ignore_all_values(sum));
    CHECK(total == 45);
  }
#endif

  struct my_domain {
    template <ex::sender_expr_for<ex::then_t> Sender, class Env>
    static auto transform_sender(Sender&&, const Env&) {
      return ex::just(int{42});
    }
  };

  TEST_CASE("transform_each - can be customized late", "[transform_each][ignore_all_values]") {
    // The customization will return a different value
    basic_inline_scheduler<my_domain> sched;
    int result = 0;
    auto start = ex::just(std::string{"hello"});
    auto with_scheduler = exec::write(exec::with(ex::get_scheduler, inline_scheduler()));
    auto adaptor = exec::on(sched, ex::then([](std::string x) { return x + ", world"; }))
                 | with_scheduler;
    auto snd = start                                                      //
             | exec::transform_each(adaptor)                              //
             | exec::transform_each(ex::then([&](int x) { result = x; })) //
             | exec::ignore_all_values();
    stdexec::sync_wait(snd);
    CHECK(result == 42);
  }
}
