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

#include "exec/sequence/merge.hpp"

#include "exec/sequence.hpp"
#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/iterate.hpp"
#include "exec/sequence/transform_each.hpp"
#include "exec/sequence_senders.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/trampoline_scheduler.hpp"
#include "stdexec/__detail/__continues_on.hpp"
#include "stdexec/__detail/__just.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__upon_error.hpp"
#include <atomic>
#include <catch2/catch.hpp>

#include <mutex>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>
#include <thread>

namespace {

  struct null_receiver {
    using receiver_concept = ex::receiver_t;

    void set_value() noexcept {
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
    auto set_next(_Item&& __item) & noexcept(ex::__nothrow_decay_copyable<_Item>)
      -> STDEXEC::__call_result_t<
        STDEXEC::upon_error_t,
        STDEXEC::__call_result_t<STDEXEC::then_t, _Item, ignore_values_fn_t>,
        ignore_values_fn_t
      > {
      return STDEXEC::upon_error(
        STDEXEC::then(static_cast<_Item&&>(__item), ignore_values_fn_t{}), ignore_values_fn_t{});
    }
  };

  TEST_CASE(
    "merge - merge two sequence senders of no elements",
    "[sequence_senders][merge][empty_sequence]") {
    int counter = 0;
    auto merged = exec::merge(exec::empty_sequence(), exec::empty_sequence());
    auto op = exec::subscribe(merged, null_receiver{});
    ex::start(op);
    CHECK(counter == 0);
  }

  TEST_CASE(
    "merge - merge three sequence senders of no elements",
    "[sequence_senders][merge][empty_sequence]") {
    int counter = 0;
    auto merged =
      exec::merge(exec::empty_sequence(), exec::empty_sequence(), exec::empty_sequence());
    auto op = exec::subscribe(merged, null_receiver{});
    ex::start(op);
    CHECK(counter == 0);
  }

  TEST_CASE("merge - merge sender of 2 senders", "[sequence_senders][merge]") {
    int value = 0;
    int count = 0;
    auto merged = exec::merge(ex::just(84), ex::just(-42));
    auto transformed = exec::transform_each(merged, ex::then([&value, &count](int x) noexcept {
                                              value += x;
                                              ++count;
                                            }));
    auto op = exec::subscribe(transformed, null_receiver{});
    ex::start(op);
    CHECK(value == 42);
    CHECK(count == 2);
  }

  TEST_CASE(
    "merge - merge sender of 2 senders and ignores all values",
    "[sequence_senders][merge][ignore_all_values]") {
    int value = 0;
    int count = 0;
    auto merged = exec::merge(ex::just(84), ex::just(-42));
    auto transformed = exec::transform_each(merged, ex::then([&value, &count](int x) {
                                              value += x;
                                              ++count;
                                              return value;
                                            }))
                     | exec::ignore_all_values();
    ex::sync_wait(transformed);
    CHECK(value == 42);
    CHECK(count == 2);
  }

#if STDEXEC_HAS_STD_RANGES()
  TEST_CASE("merge - merge sender merges all items", "[sequence_senders][merge][iterate]") {
    auto range = [](auto from, auto to) {
      return exec::iterate(std::views::iota(from, to));
    };
    auto then_each = [](auto f) {
      return exec::transform_each(ex::then(f));
    };
    // this trampoline is used to interleave the merged iterate() sequences
    // the parameters set the max inline schedule recursion depth and max
    // inline schedule stack size
    exec::trampoline_scheduler sched{16, 512};
    int total = 0;
    int count = 0;
    std::ptrdiff_t max = 0;
    auto sum = exec::merge(range(100, 120), range(200, 220), range(300, 320))
             | then_each([&total, &count, &max](int x) noexcept {
                 std::ptrdiff_t current = 0;
                 current = std::abs(
                   reinterpret_cast<char*>(&current) - reinterpret_cast<char*>(&max));
                 max = current > max ? current : max;
                 UNSCOPED_INFO("item: " << x << ", stack size: " << current);
                 total += x;
                 ++count;
               });
    // this causes both iterate sequences to use the same trampoline.
    ex::sync_wait(exec::sequence(STDEXEC::schedule(sched), exec::ignore_all_values(sum)));
    UNSCOPED_INFO("max stack size: " << max);
    CHECK(total == 12570);
    CHECK(count == 60);
  }

  TEST_CASE(
    "merge - merge sender merges all items from multiple threads",
    "[sequence_senders][static_thread_pool][merge][iterate]") {

    exec::single_thread_context ctx0;
    ex::scheduler auto sched0 = ctx0.get_scheduler();
    exec::single_thread_context ctx1;
    ex::scheduler auto sched1 = ctx1.get_scheduler();
    exec::single_thread_context ctx2;
    ex::scheduler auto sched2 = ctx2.get_scheduler();
    exec::single_thread_context ctx3;
    ex::scheduler auto sched3 = ctx3.get_scheduler();

    auto range = [](auto from, auto to) {
      return exec::iterate(std::views::iota(from, to));
    };
    auto then_each = [](auto f) {
      return exec::transform_each(ex::then(f));
    };
    auto continues_each_on = [](auto sched) {
      return exec::transform_each(ex::continues_on(sched));
    };
    int total = 0;
    int count = 0;
    auto sum = exec::merge(
                 range(100, 120) | continues_each_on(sched0),
                 range(200, 220) | continues_each_on(sched1),
                 range(300, 320) | continues_each_on(sched2))
             | then_each([](int x) noexcept {
                 // runs on sched0 and sched1 and sched2 in parallel.
                 // access to shared data would need to be protected
                 return std::make_tuple(x, std::this_thread::get_id());
               })
             | continues_each_on(sched3) | then_each([&total, &count](auto v) {
                 // runs only on sched3, which is a strand (a static
                 // pool with one thread)
                 // it is safe to use shared data here
                 auto [x, id] = v;
                 total += x;
                 ++count;
                 UNSCOPED_INFO(
                   "item: " << x << ", from thread id: " << id
                            << ", on thread id: " << std::this_thread::get_id());
               });
    ex::sync_wait(exec::sequence(ex::schedule(sched3), exec::ignore_all_values(sum)));
    CHECK(total == 12570);
    CHECK(count == 60);
  }
#endif

  struct my_domain {
    template <ex::sender_expr_for<ex::then_t> Sender, class Env>
    static auto transform_sender(STDEXEC::set_value_t, Sender&&, const Env&) {
      return ex::just(int{21});
    }
  };

  TEST_CASE("merge - can be customized late", "[merge][ignore_all_values]") {
    // The customization will return a different value
    basic_inline_scheduler<my_domain> sched;
    int result = 0;
    int count = 0;
    auto start = ex::just(std::string{"hello"});
    auto with_scheduler = ex::write_env(ex::prop{ex::get_scheduler, inline_scheduler()});
    auto adaptor = ex::on(sched, ex::then([](std::string x) { return x + ", world"; }))
                 | with_scheduler;
    auto snd =
      exec::merge(start | exec::transform_each(adaptor), start | exec::transform_each(adaptor))
      | exec::transform_each(ex::then([&](int x) {
          result += x;
          ++count;
        }))
      | exec::ignore_all_values();
    ex::sync_wait(snd);
    CHECK(result == 42);
    CHECK(count == 2);
  }

} // namespace
