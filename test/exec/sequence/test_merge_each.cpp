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

#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/iterate.hpp"
#include "exec/sequence/merge.hpp"
#include "exec/sequence/merge_each.hpp"
#include "exec/sequence_senders.hpp"
#include "exec/static_thread_pool.hpp"     // IWYU pragma: keep
#include "exec/timed_thread_scheduler.hpp" // IWYU pragma: keep for duration_of_t
#include "exec/variant_sender.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__read_env.hpp"

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <array>

namespace {
  using namespace std::chrono_literals;
  using namespace exec;
  namespace ex = STDEXEC;

  template <class _A, class _B>
  concept __equivalent = __sequence_sndr::__all_contained_in<_A, _B>
                      && __sequence_sndr::__all_contained_in<_B, _A>
                      && ex::__mapply<ex::__msize, _A>::value
                           == ex::__mapply<ex::__msize, _B>::value;

  struct null_receiver {
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
      return STDEXEC::upon_error(
        STDEXEC::then(static_cast<_Item&&>(__item), ignore_values_fn_t{}), ignore_values_fn_t{});
    }
  };

  // a sequence adaptor that applies a function to each item
  [[maybe_unused]]
  static constexpr auto then_each = [](auto f) {
    return exec::transform_each(ex::then(f));
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
      auto delay_adaptor = STDEXEC::__closure<decltype(delay_value), Sched, duration_of_t<Sched>>{
        {sched, after},
        {},
        {}
      };
      return exec::transform_each(delay_adaptor);
    };
  // a sequence adaptor that applies a function to each item
  // the function must produce a sequence
  // all the sequences returned from the function are merged
  [[maybe_unused]]
  static constexpr auto flat_map = [](auto&& f) {
    auto map_merge = [](auto&& sequence, auto&& f) noexcept {
      return merge_each(
        exec::transform_each(
          static_cast<decltype(sequence)&&>(sequence), ex::then(static_cast<decltype(f)&&>(f))));
    };
    return STDEXEC::__closure<decltype(map_merge), decltype(f)>{
      {static_cast<decltype(f)&&>(f)}, {}, {}};
  };
  // when_all requires a successful completion
  // however stop_after_on has no successful completion
  // this uses variant_sender to add a successful completion
  // (the successful completion will never occur)
  [[maybe_unused]]
  static constexpr auto with_void = [](auto&& sender) noexcept
    -> variant_sender<STDEXEC::__call_result_t<ex::just_t>, decltype(sender)> {
    return {static_cast<decltype(sender)&&>(sender)};
  };
  // with_stop_token_from adds get_stop_token query, that returns the
  // token for the provided stop_source, to the receiver env
  [[maybe_unused]]
  static constexpr auto with_stop_token_from = [](auto& stop_source) noexcept {
    return ex::write_env(ex::prop{ex::get_stop_token, stop_source.get_token()});
  };
  // log_start completes with the provided sequence after printing provided string
  [[maybe_unused]]
  auto log_start = [](auto sequence, auto message) {
    return exec::sequence(
      ex::read_env(ex::get_stop_token) | STDEXEC::then([message](auto&& token) noexcept {
        UNSCOPED_INFO(
          message << (token.stop_requested() ? ", stop was requested" : ", stop not requested")
                  << ", on thread id: " << std::this_thread::get_id());
      }),
      ex::just(sequence));
  };
  // log_sequence prints the message when each value in the sequence is emitted
  [[maybe_unused]]
  auto log_sequence = [](auto sequence, auto message) {
    return sequence | then_each([message](auto&& value) mutable noexcept {
             UNSCOPED_INFO(message << ", on thread id: " << std::this_thread::get_id());
             return value;
           });
  };
  // emits_stopped completes with set_stopped after printing info
  [[maybe_unused]]
  auto emits_stopped = []() {
    return ex::just() | STDEXEC::let_value([]() noexcept {
             UNSCOPED_INFO("emitting stopped, on thread id: " << std::this_thread::get_id());
             return ex::just_stopped();
           });
  };
  // emits_error completes with set_error(error) after printing info
  [[maybe_unused]]
  auto emits_error = [](auto error) {
    return ex::just() | STDEXEC::let_value([error]() noexcept {
             UNSCOPED_INFO(error.what() << ", on thread id: " << std::this_thread::get_id());
             return ex::just_error(error);
           });
  };

#if STDEXEC_HAS_STD_RANGES()

  // a sequence of numbers from itoa()
  [[maybe_unused]]
  static constexpr auto range = [](auto from, auto to) {
    return exec::iterate(std::views::iota(from, to));
  };

  template <ex::sender Sender>
  struct as_sequence_t : Sender {
    using sender_concept = sequence_sender_t;
    using item_types = exec::item_types<Sender>;
    auto subscribe(auto receiver) {
      return connect(set_next(receiver, *static_cast<Sender*>(this)), receiver);
    }
  };

  TEST_CASE(
    "merge_each - merge two sequence senders of no elements",
    "[sequence_senders][merge_each][empty_sequence]") {
    using empty_sequence_t = STDEXEC::__call_result_t<empty_sequence_t>;

    [[maybe_unused]]
    std::array<empty_sequence_t, 2> array{empty_sequence(), empty_sequence()};

    [[maybe_unused]]
    auto sequences = iterate(std::views::all(array));
    using sequences_t = decltype(sequences);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<sequences_t>>);
    STATIC_REQUIRE(ex::__ok<STDEXEC::completion_signatures_of_t<sequences_t>>);

    [[maybe_unused]]
    auto merged = merge_each(sequences);
    using merged_t = decltype(merged);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<merged_t>>);
    STATIC_REQUIRE(ex::__ok<STDEXEC::completion_signatures_of_t<merged_t>>);

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

    using range_sender_t = STDEXEC::__call_result_t<decltype(range), int, int>;

    [[maybe_unused]]
    std::array<range_sender_t, 2> array{range(100, 120), range(200, 220)};

    [[maybe_unused]]
    auto sequences = iterate(std::views::all(array));
    using sequences_t = decltype(sequences);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<sequences_t>>);
    STATIC_REQUIRE(ex::__ok<STDEXEC::completion_signatures_of_t<sequences_t>>);

    [[maybe_unused]]
    auto merged = merge_each(sequences);
    using merged_t = decltype(merged);

    STATIC_REQUIRE(ex::__ok<item_types_of_t<merged_t>>);
    STATIC_REQUIRE(ex::__ok<STDEXEC::completion_signatures_of_t<merged_t>>);

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

    using range_sequence_t = STDEXEC::__call_result_t<decltype(range), int, int>;
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

    using empty_sequence_t = STDEXEC::__call_result_t<empty_sequence_t>;
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

// TODO - fix problem with stopping
#  if 0
  TEST_CASE(
    "merge_each - merge_each sender stops when a nested sequence fails",
    "[sequence_senders][static_thread_pool][merge_each][merge][iterate]") {

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
