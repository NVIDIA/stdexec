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
#include "exec/sequence_senders.hpp"
#include "exec/variant_sender.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/timed_thread_scheduler.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__read_env.hpp"

#include <stdexcept>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/sequences.hpp>
#include <test_common/type_helpers.hpp>

#include <array>
#include <chrono>
#include <iomanip>

namespace {
  using namespace std::chrono_literals;
  using namespace exec;
  namespace ex = stdexec;

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
  static constexpr auto delays_each_on = []<class Sched>(Sched sched, auto after) noexcept {
    auto delay_value = []<class Value>(Value&& value, Sched sched, auto after) {
      return sequence(schedule_after(sched, after), static_cast<Value&&>(value));
    };
    auto delay_adaptor = stdexec::__binder_back<decltype(delay_value), Sched, decltype(after)>{
      {sched, after},
      {},
      {}
    };
    return exec::transform_each(delay_adaptor);
  };

#if STDEXEC_HAS_STD_RANGES()

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
           | stdexec::continues_on(sched0) // serializes output on the sched0 strand
           | stdexec::let_value([elapsed_ms]() noexcept {
               UNSCOPED_INFO(
                 "requesting stop  - at: " << std::setw(3) << elapsed_ms()
                                           << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_stopped();
             });
    };

    auto error_after_on = [sched0, elapsed_ms](auto sched, auto after, auto error) {
      return schedule_after(sched, after)
           | stdexec::continues_on(sched0) // serializes output on the sched0 strand
           | stdexec::let_value([elapsed_ms, error]() noexcept {
               UNSCOPED_INFO(
                 error.what() << " - at: " << std::setw(3) << elapsed_ms()
                              << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_error(error);
             });
    };

    // a sequence whose items are sequences
    auto sequences = merge(
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
           | stdexec::continues_on(sched0) // serializes output on the sched0 strand
           | stdexec::let_value([elapsed_ms]() noexcept {
               UNSCOPED_INFO(
                 "requesting stop  - at: " << std::setw(3) << elapsed_ms()
                                           << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_stopped();
             });
    };

    auto error_after_on = [sched0, elapsed_ms](auto sched, auto after, auto error) {
      return schedule_after(sched, after)
           | stdexec::continues_on(sched0) // serializes output on the sched0 strand
           | stdexec::let_value([elapsed_ms, error]() noexcept {
               UNSCOPED_INFO(
                 error.what() << " - at: " << std::setw(3) << elapsed_ms()
                              << "ms, on thread id: " << std::this_thread::get_id());
               return ex::just_error(error);
             });
    };

    // a sequence whose items are sequences
    auto sequences = merge(
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
