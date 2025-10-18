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

#include "exec/sequence/test_scheduler.hpp"

#include "exec/sequence/marbles.hpp"
#include "exec/sequence/merge.hpp"
#include "exec/sequence/transform_each.hpp"
#include "exec/sequence.hpp"
#include "stdexec/__detail/__just.hpp"
#include "stdexec/__detail/__meta.hpp"
#include <catch2/catch.hpp>

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/sequences.hpp>
#include <test_common/type_helpers.hpp>

namespace {

  // a sequence adaptor that schedules each item to complete
  // on the specified scheduler
  [[maybe_unused]] static constexpr auto continues_each_on = [](auto sched) {
    return exec::transform_each(ex::continues_on(sched));
  };
  // a sequence adaptor that schedules each item to complete
  // on the specified scheduler after the specified duration
  [[maybe_unused]] static constexpr auto delays_each_on = [](auto sched, duration_of_t<decltype(sched)> after) noexcept {
    return exec::transform_each(stdexec::let_value([sched, after](auto&&... vs) noexcept {
      auto at = sched.now() + after;
      return sequence(schedule_at(sched, at), stdexec::just(vs...));
    }));
  };

  using __marble_t = exec::marble_t<test_clock>;
  using __marbles_t = std::vector<__marble_t>;

# if STDEXEC_HAS_STD_RANGES()

  TEST_CASE(
  "test_scheduler - parse empty diagram",
  "[sequence_senders][test_scheduler][marbles]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    auto marbles = get_marbles_from(__clock, ""_mstr);
    auto expected = __marbles_t{};
    CHECK(marbles.size() == 0);
    CHECK(marbles == expected);
  }

  TEST_CASE(
  "test_scheduler - record marbles via test_context",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(__clock.now() == test_clock::time_point{0ms});
    auto __scheduler = __test.get_scheduler();
    auto __sequence = __scheduler.schedule() | stdexec::then([]() noexcept { return '0'; });
    auto actual = __test.get_marbles_from(__sequence);
    CHECK(__clock.now() == test_clock::time_point{0ms});
    auto expected = get_marbles_from(__clock,
      "=^(0|)"_mstr);
    CHECK(actual == expected);
  }

  TEST_CASE(
  "test_scheduler - test_context schedule_after advances test_clock",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(__clock.now() == test_clock::time_point{0ms});
    auto __scheduler = __test.get_scheduler();
    auto __sequence = schedule_after(__scheduler, 2ms) | stdexec::then([]() noexcept { return '0'; });
    auto expected = get_marbles_from(__clock,
      "=^--(0|)"_mstr);
    auto actual = __test.get_marbles_from(__sequence);
    CHECK(test_clock::time_point{2ms} == __clock.now());
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence advances test_clock",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -a--b---c|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-a--b---c|"_mstr);
    auto actual = __test.get_marbles_from(__sequence);
    CHECK(test_clock::time_point{9ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence never",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0-"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5 998ms $"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{1000ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence error",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0--#"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5--#$"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence error in middle",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0--#--1|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5--#$"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence stopped",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0--."_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5--.$"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence stopped in middle",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0--.--1|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5--.$"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence transform",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  -0--1---2|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^-5--6---7|"_mstr);
    auto actual = __test.get_marbles_from(__sequence | then_each([](char c){ return c+5; }));
    CHECK(test_clock::time_point{9ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence simple shift",
  "[sequence_senders][test_scheduler]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence = __test.get_marble_sequence_from(
      "  012--|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^--012|"_mstr);
    auto actual = __test.get_marbles_from(__sequence | delays_each_on(__test.get_scheduler(), 2ms));
    CHECK(test_clock::time_point{5ms} == __clock.now());
    CAPTURE(__sequence.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context multi-second marble-sequence shift",
  "[sequence_senders][test_scheduler]") {
    auto __real_time_now = std::chrono::steady_clock::now();
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());

    auto __sequence = __test.get_marble_sequence_from(
      "   5s       0 5s 1 5s 2 100ms |"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^ 5s 100ms 0 5s 1 5s 2       |"_mstr);

    auto actual = __test.get_marbles_from(__sequence | delays_each_on(__test.get_scheduler(), 100ms), 16s);

    CHECK(test_clock::time_point{5s + 5s + 5s + 100ms + 3ms} == __clock.now());

    auto __real_time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - __real_time_now);
    CAPTURE(__sequence.__marbles_);
    CAPTURE(__real_time_elapsed);
    CHECK(expected == actual);
  }

  TEST_CASE(
  "test_scheduler - test_context marble-sequence merge",
  "[sequence_senders][test_scheduler][merge]") {
    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from(
      "  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from(
      "  -1-3   -4|"_mstr);
    auto expected = get_marbles_from(__clock,
      "=^01-(23)-4|"_mstr);
    auto actual = __test.get_marbles_from(merge(__sequence0, __sequence1));
    CHECK(test_clock::time_point{6ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

#  endif // STDEXEC_HAS_STD_RANGES()
} // namespace
