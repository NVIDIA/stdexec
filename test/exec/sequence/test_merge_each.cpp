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

#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/merge_each.hpp"
#include "exec/sequence/merge.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/iterate.hpp"
#include "exec/sequence/test_scheduler.hpp"
#include "exec/sequence_senders.hpp"
#include "exec/timed_scheduler.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__senders_core.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/sequences.hpp>
#include <test_common/type_helpers.hpp>

#include <array>

namespace {

  template <class _A, class _B>
  concept __equivalent = __sequence_sndr::__all_contained_in<_A, _B>
                      && __sequence_sndr::__all_contained_in<_B, _A>
                      && ex::__v<ex::__mapply<ex::__msize, _A>>
                           == ex::__v<ex::__mapply<ex::__msize, _B>>;

  struct null_receiver {
    using __id = null_receiver;
    using __t = null_receiver;
    using receiver_concept = ex::receiver_t;

    template <class... _Values>
    void set_value(_Values&&...) noexcept {
    }

    template <class _Error>
    void set_error(_Error&&) noexcept {
    }

    void set_stopped() noexcept {
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env<> {
      return {};
    }

    struct ignore_values_fn_t {
      template <class... _Vs>
      void operator()(_Vs&&...) const noexcept {
      }
    };

    template <ex::sender _Item>
    [[nodiscard]]
    auto
      set_next(_Item&& __item) & noexcept(ex::__nothrow_decay_copyable<_Item>) -> next_sender auto {
      return stdexec::upon_error(
        stdexec::then(static_cast<_Item&&>(__item), ignore_values_fn_t{}), ignore_values_fn_t{});
    }
  };

  // a sequence adaptor that schedules each item to complete
  // on the specified scheduler
  [[maybe_unused]]
  static constexpr auto continues_each_on = [](auto sched) {
    return exec::transform_each(ex::continues_on(sched));
  };
  // a sequence adaptor that schedules each item to complete
  // on the specified scheduler after the specified duration
  [[maybe_unused]]
  static constexpr auto delays_each_on =
    []<class Sched>(Sched sched, duration_of_t<Sched> after) noexcept {
      auto delay_value = []<class Value>(Value&& value, Sched sched, duration_of_t<Sched> after) {
        return sequence(schedule_after(sched, after), static_cast<Value&&>(value));
      };
      auto delay_adaptor =
        stdexec::__binder_back<decltype(delay_value), Sched, duration_of_t<Sched>>{
          {sched, after},
          {},
          {}
      };
      return exec::transform_each(delay_adaptor);
    };
    
#if STDEXEC_HAS_STD_RANGES()

  TEST_CASE(
    "merge_each - merge two sequence senders of no elements",
    "[sequence_senders][merge_each][empty_sequence]") {
    using empty_sequence_t = stdexec::__call_result_t<empty_sequence_t>;

    [[maybe_unused]]
    std::array<empty_sequence_t, 2> array{empty_sequence(), empty_sequence()};

    [[maybe_unused]]
    auto sequences = iterate(std::views::all(array));
    using sequences_t = decltype(sequences);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<sequences_t>>);
    STATIC_REQUIRE(ex::__ok<stdexec::completion_signatures_of_t<sequences_t>>);

    [[maybe_unused]]
    auto merged = merge_each(sequences);
    using merged_t = decltype(merged);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<merged_t>>);
    STATIC_REQUIRE(ex::__ok<stdexec::completion_signatures_of_t<merged_t>>);

    STATIC_REQUIRE(ex::__callable<subscribe_t, merged_t, null_receiver>);

    int count = 0;

    auto v = ex::sync_wait(ignore_all_values(
      merged | then_each([&count](int x) {
        ++count;
        UNSCOPED_INFO("item: " << x << ", on thread id: " << std::this_thread::get_id());
      })));

    CHECK(count == 0);
    CHECK(v.has_value() == true);
  }

  TEST_CASE(
    "merge_each - merge two sequence senders of integers",
    "[sequence_senders][merge_each][empty_sequence]") {

    using range_sender_t = stdexec::__call_result_t<decltype(range), int, int>;

    [[maybe_unused]]
    std::array<range_sender_t, 2> array{range(100, 120), range(200, 220)};

    [[maybe_unused]]
    auto sequences = iterate(std::views::all(array));
    using sequences_t = decltype(sequences);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<sequences_t>>);
    STATIC_REQUIRE(ex::__ok<stdexec::completion_signatures_of_t<sequences_t>>);

    [[maybe_unused]]
    auto merged = merge_each(sequences);
    using merged_t = decltype(merged);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<merged_t>>);
    STATIC_REQUIRE(ex::__ok<stdexec::completion_signatures_of_t<merged_t>>);

    STATIC_REQUIRE(ex::__callable<subscribe_t, merged_t, null_receiver>);

    int count = 0;

    auto v = ex::sync_wait(ignore_all_values(
      merged | then_each([&count](int x) {
        ++count;
        UNSCOPED_INFO("item: " << x << ", on thread id: " << std::this_thread::get_id());
      })));

    CHECK(count == 40);
    CHECK(v.has_value() == true);
  }

  TEST_CASE(
    "merge_each - merge sequence of two sequence senders of integers and one empty sequence",
    "[sequence_senders][merge_each][merge][empty_sequence]") {

    using range_sequence_t = stdexec::__call_result_t<decltype(range), int, int>;
    STATIC_REQUIRE(__well_formed_sequence_sender<range_sequence_t>);
    STATIC_REQUIRE_FALSE(std::same_as<item_types_of_t<range_sequence_t>, item_types<>>);
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<range_sequence_t>,
        ex::completion_signatures<
          ex::set_error_t(std::exception_ptr),
          ex::set_stopped_t(),
          ex::set_value_t()
        >
      >);

    using just_range_sender_t = ex::__call_result_t<ex::just_t, range_sequence_t>;
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<just_range_sender_t>,
        ex::completion_signatures<ex::set_value_t(range_sequence_t)>
      >);

    using empty_sequence_t = stdexec::__call_result_t<empty_sequence_t>;
    STATIC_REQUIRE(__well_formed_sequence_sender<empty_sequence_t>);
    STATIC_REQUIRE(std::same_as<item_types_of_t<empty_sequence_t>, item_types<>>);
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<empty_sequence_t>,
        ex::completion_signatures<ex::set_value_t()>
      >);

    using just_empty_sender_t = ex::__call_result_t<ex::just_t, empty_sequence_t>;
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<just_empty_sender_t>,
        ex::completion_signatures<ex::set_value_t(empty_sequence_t)>
      >);

    auto sequences =
      merge(ex::just(range(100, 120)), ex::just(empty_sequence()), ex::just(range(200, 220)));
    using sequences_t = decltype(sequences);

    STATIC_REQUIRE(ex::__ok<__item_types_of_t<sequences_t>>);
    STATIC_REQUIRE(ex::__ok<ex::completion_signatures_of_t<sequences_t>>);

    STATIC_REQUIRE(
      __equivalent<
        __item_types_of_t<sequences_t>,
        item_types<just_range_sender_t, just_empty_sender_t>
      >);
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<sequences_t>,
        ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t()>
      >);

    auto merged = merge_each(sequences);
    using merged_t = decltype(merged);

    STATIC_REQUIRE(ex::__ok<__item_types_of_t<merged_t>>);
    STATIC_REQUIRE(
      __equivalent<
        ex::completion_signatures_of_t<merged_t>,
        ex::completion_signatures<
          ex::set_error_t(std::exception_ptr),
          ex::set_stopped_t(),
          ex::set_value_t()
        >
      >);

    int count = 0;

    auto v = ex::sync_wait(ignore_all_values(
      merged | then_each([&count](int x) {
        ++count;
        UNSCOPED_INFO("item: " << x << ", on thread id: " << std::this_thread::get_id());
      })));

    CHECK(count == 40);
    CHECK(v.has_value() == true);
  }


  TEST_CASE(
    "merge_each - merge_each of marble sequences",
    "[sequence_senders][merge_each][merge]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("  -1-3   -4|"_mstr);
    auto expected = get_marbles_from(__clock, "=^01-(23)-4|"_mstr);
    auto actual = __test.get_marbles_from(
      merge_each(merge(stdexec::just(__sequence0), stdexec::just(__sequence1))));
    CHECK(test_clock::time_point{6ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
    "merge_each - merge_each of marble sequences - concat",
    "[sequence_senders][merge_each][iterate]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("      -1-3-4|"_mstr);
    auto expected = get_marbles_from(__clock, "=^0--2-1-3-4|"_mstr);
    std::array<_tst_sched::__test_sequence, 2> __sequences{__sequence0, __sequence1};
    auto actual = __test.get_marbles_from(
      merge_each(iterate(__test.get_scheduler(), std::views::all(__sequences))));
    CHECK(test_clock::time_point{10ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
    "merge_each - merge_each of marble sequences with error",
    "[sequence_senders][merge_each][merge]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("  -1-3#-4|"_mstr);
    auto expected = get_marbles_from(
      __clock,
      // TODO FIX set_stopped issued instead of set_error
      "=^01-(23)#$"_mstr);
    auto actual = __test.get_marbles_from(
      merge_each(merge(stdexec::just(__sequence0), stdexec::just(__sequence1))));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }


  TEST_CASE(
    "merge_each - merge_each of marble sequences with error - concat",
    "[sequence_senders][merge_each][iterate]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("      -1-3#-4|"_mstr);
    auto expected = get_marbles_from(
      __clock,
      // TODO FIX set_stopped issued instead of set_error
      "=^0--2-1-3#$"_mstr);
    std::array<_tst_sched::__test_sequence, 2> __sequences{__sequence0, __sequence1};
    auto actual = __test.get_marbles_from(
      merge_each(iterate(__test.get_scheduler(), std::views::all(__sequences))));
    CHECK(test_clock::time_point{8ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
    "merge_each - merge_each of marble sequences with a value stopped",
    "[sequence_senders][merge_each][merge]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("  -1-3.-4|"_mstr);
    auto expected = get_marbles_from(
      __clock,
      // TODO FIX set_stopped issued instead of set_error
      "=^01-(23).$"_mstr);
    auto actual = __test.get_marbles_from(
      merge_each(merge(stdexec::just(__sequence0), stdexec::just(__sequence1))));
    CHECK(test_clock::time_point{4ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

  TEST_CASE(
    "merge_each - merge_each of marble sequences with a value stopped - concat",
    "[sequence_senders][merge_each][iterate]") {

    test_context __test{};
    auto __clock = __test.get_clock();
    CHECK(test_clock::time_point{0ms} == __clock.now());
    auto __sequence0 = __test.get_marble_sequence_from("  0--2|"_mstr);
    auto __sequence1 = __test.get_marble_sequence_from("      -1-3.-4|"_mstr);
    auto expected = get_marbles_from(
      __clock,
      // TODO FIX set_stopped issued instead of set_error
      "=^0--2-1-3.$"_mstr);
    std::array<_tst_sched::__test_sequence, 2> __sequences{__sequence0, __sequence1};
    auto actual = __test.get_marbles_from(
      merge_each(iterate(__test.get_scheduler(), std::views::all(__sequences))));
    CHECK(test_clock::time_point{8ms} == __clock.now());
    CAPTURE(__sequence0.__marbles_);
    CAPTURE(__sequence1.__marbles_);
    CHECK(expected == actual);
  }

// TODO - fix problem with stopping
#  if 0
  TEST_CASE(
    "merge_each - merge_each sender stops when a nested sequence fails",
    "[sequence_senders][merge_each][merge][iterate]") {

    auto sequences = merge(
      log_start(range(100, 120), "range 100-120"),
      ex::just(emits_error(std::runtime_error{"failed sequence "})),
      log_start(range(200, 220), "range 200-220")
      );

    [[maybe_unused]] auto merged = merge_each(std::move(sequences));

    int count = 0;

    auto v = ex::sync_wait(ignore_all_values(merged | then_each([&count](int x){
      ++count;
      UNSCOPED_INFO("item: " << x
        << ", on thread id: " << std::this_thread::get_id());
    })));

    CHECK(count == 20);
    CHECK(v.has_value() == false);
  }
#  endif // 0

#endif // STDEXEC_HAS_STD_RANGES()

} // namespace
