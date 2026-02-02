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
#include "exec/single_thread_context.hpp"
#include "exec/timed_thread_scheduler.hpp"
#include "exec/variant_sender.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__read_env.hpp"

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <chrono>
#include <iomanip>
#include <stdexcept>

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
  static constexpr auto delays_each_on = []<class Sched>(Sched sched, auto after) noexcept {
    auto delay_value = []<class Value>(Value&& value, Sched sched, auto after) {
      return sequence(schedule_after(sched, after), static_cast<Value&&>(value));
    };
    auto delay_adaptor = STDEXEC::__closure(delay_value, sched, after);
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
    return STDEXEC::__closure(map_merge, static_cast<decltype(f)&&>(f));
  };

  // when_all requires a successful completion
  // however stop_after_on has no successful completion
  // this uses variant_sender to add a successful completion
  // (the successful completion will never occur)
  [[maybe_unused]]
  static constexpr auto with_void = [](auto&& sender) noexcept
    -> variant_sender<STDEXEC::__call_result_t<ex::just_t>, STDEXEC::__decay_t<decltype(sender)>> {
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
    template <ex::receiver Receiver>
    auto subscribe(Receiver receiver) {
      return connect(set_next(receiver, *static_cast<Sender*>(this)), receiver);
    }
  };

  TEST_CASE(
    "merge_each - merge_each sender merges all items from multiple threads",
    "[sequence_senders][single_thread_context][merge_each][merge][iterate]") {

    exec::single_thread_context ctx0;
    ex::scheduler auto sched0 = ctx0.get_scheduler();
    exec::single_thread_context ctx1;
    ex::scheduler auto sched1 = ctx1.get_scheduler();
    exec::single_thread_context ctx2;
    ex::scheduler auto sched2 = ctx2.get_scheduler();

    auto sequences = merge(
      ex::just(range(100, 120) | continues_each_on(sched1)),
      ex::just(empty_sequence()),
      ex::just(range(200, 220) | continues_each_on(sched2)));

    auto merged = merge_each(sequences);

    int count = 0;

    auto v = ex::sync_wait(ignore_all_values(
      merged | continues_each_on(sched0) | then_each([&count](int x) {
        ++count;
        UNSCOPED_INFO("item: " << x << ", on thread id: " << std::this_thread::get_id());
      })));

    CHECK(count == 40);
    CHECK(v.has_value() == true);
  }

  TEST_CASE(
    "merge_each - merge_each sender stops on failed item while merging all items from multiple "
    "threads",
    "[sequence_senders][single_thread_context][merge_each][merge][iterate]") {

    exec::single_thread_context ctx0;
    ex::scheduler auto sched0 = ctx0.get_scheduler();
    exec::timed_thread_context ctx1;
    ex::scheduler auto sched1 = ctx1.get_scheduler();
    auto origin = now(sched1);

    auto elapsed_ms = [&sched1, origin]() {
      using namespace std::chrono;
      return duration_cast<milliseconds>(now(sched1) - origin).count();
    };

    auto stop_after_on = [sched0, elapsed_ms](auto sched, auto after) {
      return schedule_after(sched, after)
           | STDEXEC::continues_on(sched0) // serializes output on the sched0 strand
           | STDEXEC::let_value([elapsed_ms]() noexcept {
               UNSCOPED_INFO(
                 "requesting stop  - at: " << std::setw(3) << elapsed_ms()
                                           << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_stopped();
             });
    };

    auto error_after_on = [sched0, elapsed_ms](auto sched, auto after, auto error) {
      return schedule_after(sched, after)
           | STDEXEC::continues_on(sched0) // serializes output on the sched0 strand
           | STDEXEC::let_value([elapsed_ms, error]() noexcept {
               UNSCOPED_INFO(
                 error.what() << " - at: " << std::setw(3) << elapsed_ms()
                              << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_error(error);
             });
    };

    // a sequence whose items are sequences
    auto sequences = merge(
      ex::just(stop_after_on(sched1, 10ms)),                                         // no items
      ex::just(range(100, 120)),                                                     // int items
      ex::just(empty_sequence()),                                                    // no items
      ex::just(range(200, 220)),                                                     // int items
      ex::just(error_after_on(sched1, 40ms, std::runtime_error{"failed sequence "})) // no items
    );

    // apply delays_each_on to every sequence item and
    // merge all the new sequences
    auto merged = sequences | flat_map([sched1](auto sequence) {
                    return sequence | delays_each_on(sched1, 10ms);
                  });

    int count = 0;

    auto v = ex::sync_wait(
      ex::when_all(
        with_void(stop_after_on(sched1, 50ms)),
        ignore_all_values(
          merged | continues_each_on(sched0) // serializes output on the sched0 strand
          | then_each([&count, elapsed_ms](int x) {
              ++count;
              UNSCOPED_INFO(
                "item: " << x << ", arrived at: " << std::setw(3) << elapsed_ms()
                         << "ms, on thread id: " << std::this_thread::get_id());
              return count;
            }))));

    CAPTURE(count < 40, count > 4);
    CHECK(v.has_value() == false);
  }

  TEST_CASE(
    "merge_each - merge_each sender stops while merging all items from multiple threads",
    "[sequence_senders][single_thread_context][merge_each][merge][iterate]") {

    exec::single_thread_context ctx0;
    ex::scheduler auto sched0 = ctx0.get_scheduler();
    exec::timed_thread_context ctx1;
    ex::scheduler auto sched1 = ctx1.get_scheduler();
    auto origin = now(sched1);

    auto elapsed_ms = [&sched1, origin]() {
      using namespace std::chrono;
      return duration_cast<milliseconds>(now(sched1) - origin).count();
    };

    auto stop_after_on = [sched0, elapsed_ms](auto sched, auto after) {
      return schedule_after(sched, after)
           | STDEXEC::continues_on(sched0) // serializes output on the sched0 strand
           | STDEXEC::let_value([elapsed_ms]() noexcept {
               UNSCOPED_INFO(
                 "requesting stop  - at: " << std::setw(3) << elapsed_ms()
                                           << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_stopped();
             });
    };

    auto error_after_on = [sched0, elapsed_ms](auto sched, auto after, auto error) {
      return schedule_after(sched, after)
           | STDEXEC::continues_on(sched0) // serializes output on the sched0 strand
           | STDEXEC::let_value([elapsed_ms, error]() noexcept {
               UNSCOPED_INFO(
                 error.what() << " - at: " << std::setw(3) << elapsed_ms()
                              << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_error(error);
             });
    };

    // a sequence whose items are sequences
    auto sequences = merge(
      ex::just(stop_after_on(sched1, 10ms)),                                         // no items
      ex::just(range(100, 120)),                                                     // int items
      ex::just(empty_sequence()),                                                    // no items
      ex::just(range(200, 220)),                                                     // int items
      ex::just(error_after_on(sched1, 50ms, std::runtime_error{"failed sequence "})) // no items
    );

    // apply delays_each_on to every sequence item and
    // merge all the new sequences
    auto merged = sequences | flat_map([sched1](auto sequence) {
                    return sequence | delays_each_on(sched1, 10ms);
                  });

    int count = 0;

    auto v = ex::sync_wait(
      ex::when_all(
        with_void(stop_after_on(sched1, 40ms)),
        ignore_all_values(
          merged | continues_each_on(sched0) // serializes output on the sched0 strand
          | then_each([&count, elapsed_ms](int x) {
              ++count;
              UNSCOPED_INFO(
                "item: " << x << ", arrived at: " << std::setw(3) << elapsed_ms()
                         << "ms, on thread id: " << std::this_thread::get_id());
              return count;
            }))));

    CAPTURE(count < 40, count > 4);
    CHECK(v.has_value() == false);
  }

#endif // STDEXEC_HAS_STD_RANGES()

} // namespace
