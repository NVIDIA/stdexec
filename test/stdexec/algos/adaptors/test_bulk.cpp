/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <exception>

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>

#include <numeric>
#include <vector>

namespace ex = stdexec;

namespace {
  template <class Shape, int N, int (&Counter)[N]>
  void function(Shape i) {
    Counter[i]++;
  }

  template <class Shape>
  struct function_object_t {
    int* Counter;

    void operator()(Shape i) {
      Counter[i]++;
    }
  };

  TEST_CASE("bulk returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk(ex::just(19), 8, [](int, int) { });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("bulk with environment returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk(ex::just(19), 8, [](int, int) { });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("bulk can be piped", "[adaptors][bulk]") {
    ex::sender auto snd = ex::just() | ex::bulk(42, [](int) { });
    (void) snd;
  }

  TEST_CASE("bulk keeps values_type from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    check_val_types<ex::__mset<pack<>>>(ex::just() | ex::bulk(n, [](int) { }));
    check_val_types<ex::__mset<pack<double>>>(ex::just(4.2) | ex::bulk(n, [](int, double) { }));
    check_val_types<ex::__mset<pack<double, std::string>>>(
      ex::just(4.2, std::string{}) | ex::bulk(n, [](int, double, std::string) { }));
  }

  TEST_CASE("bulk keeps error_types from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>( //
      ex::transfer_just(sched1) | ex::bulk(n, [](int) noexcept {}));
    check_err_types<ex::__mset<std::exception_ptr>>( //
      ex::transfer_just(sched2) | ex::bulk(n, [](int) noexcept {}));
    check_err_types<ex::__mset<int>>( //
      ex::just_error(n) | ex::bulk(n, [](int) noexcept {}));
    check_err_types<ex::__mset<int>>( //
      ex::transfer_just(sched3) | ex::bulk(n, [](int) noexcept {}));
    check_err_types<ex::__mset<std::exception_ptr, int>>( //
      ex::transfer_just(sched3) | ex::bulk(n, [](int) { throw std::logic_error{"err"}; }));
  }

  TEST_CASE("bulk can be used with a function", "[adaptors][bulk]") {
    constexpr int n = 9;
    static int counter[n]{};
    std::fill_n(counter, n, 0);

    ex::sender auto snd = ex::just() | ex::bulk(n, function<int, n, counter>);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk can be used with a function object", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};
    function_object_t<int> fn{counter};

    ex::sender auto snd = ex::just() | ex::bulk(n, fn);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk can be used with a lambda", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};

    ex::sender auto snd = ex::just() | ex::bulk(n, [&](int i) { counter[i]++; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk forwards values", "[adaptors][bulk]") {
    constexpr int n = 9;
    constexpr int magic_number = 42;
    int counter[n]{0};

    auto snd = ex::just(magic_number) //
             | ex::bulk(n, [&](int i, int val) {
                 if (val == magic_number) {
                   counter[i]++;
                 }
               });
    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk forwards values that can be taken by reference", "[adaptors][bulk]") {
    constexpr std::size_t n = 9;
    std::vector<int> vals(n, 0);
    std::vector<int> vals_expected(n);
    std::iota(vals_expected.begin(), vals_expected.end(), 0);

    auto snd =
      ex::just(std::move(vals)) //
      | ex::bulk(n, [&](std::size_t i, std::vector<int>& vals) { vals[i] = static_cast<int>(i); });
    auto op = ex::connect(std::move(snd), expect_value_receiver{vals_expected});
    ex::start(op);
  }

  TEST_CASE("bulk cannot be used to change the value type", "[adaptors][bulk]") {
    constexpr int magic_number = 42;
    constexpr int n = 2;

    auto snd = ex::just(magic_number)
             | ex::bulk(n, [](int, int) { return function_object_t<int>{nullptr}; });

    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);
  }

  TEST_CASE("bulk can throw, and set_error will be called", "[adaptors][bulk]") {
    constexpr int n = 2;

    auto snd = ex::just() //
             | ex::bulk(n, [](int) -> int { throw std::logic_error{"err"}; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk function is not called on error", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_error(std::string{"err"}) | ex::bulk(n, [&called](int) { called++; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
    ex::start(op);
  }

  TEST_CASE("bulk function in not called on stop", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_stopped() | ex::bulk(n, [&called](int) { called++; });
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk works with static thread pool", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      for (std::size_t n = 0; n < 9u; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch)
                 | ex::bulk(n, [&counter](std::size_t idx) { counter[idx] = 0; })
                 | ex::bulk(n, [&counter](std::size_t idx) { counter[idx]++; });
        stdexec::sync_wait(std::move(snd));

        const std::size_t actual =
          static_cast<std::size_t>(std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch, 42)
                 | ex::bulk(
                     n,
                     [&counter](std::size_t idx, int val) {
                       if (val == 42) {
                         counter[idx] = 0;
                       }
                     })
                 | ex::bulk(n, [&counter](std::size_t idx, int val) {
                     if (val == 42) {
                       counter[idx]++;
                     }
                   });
        auto [val] = stdexec::sync_wait(std::move(snd)).value();

        CHECK(val == 42);

        const std::size_t actual =
          static_cast<std::size_t>(std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel that can be taken by reference") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> vals(n, 0);
        std::vector<int> vals_expected(n);
        std::iota(vals_expected.begin(), vals_expected.end(), 1);

        auto snd =
          ex::transfer_just(sch, std::move(vals))
          | ex::bulk(
            n, [](std::size_t idx, std::vector<int>& vals) { vals[idx] = static_cast<int>(idx); })
          | ex::bulk(n, [](std::size_t idx, std::vector<int>& vals) { ++vals[idx]; });
        auto [vals_actual] = stdexec::sync_wait(std::move(snd)).value();

        CHECK(vals_actual == vals_expected);
      }
    }

    SECTION("With exception") {
      constexpr int n = 9;
      auto snd = ex::transfer_just(sch)
               | ex::bulk(n, [](int) { throw std::runtime_error("bulk"); });

      CHECK_THROWS_AS(stdexec::sync_wait(std::move(snd)), std::runtime_error);
    }

    SECTION("With concurrent enqueueing") {
      constexpr std::size_t n = 4;
      std::vector<int> counters_1(n, 0);
      std::vector<int> counters_2(n, 0);

      stdexec::sender auto snd = stdexec::when_all(
        stdexec::schedule(sch) | stdexec::bulk(n, [&](std::size_t id) { counters_1[id]++; }),
        stdexec::schedule(sch) | stdexec::bulk(n, [&](std::size_t id) { counters_2[id]++; }));

      stdexec::sync_wait(std::move(snd));

      CHECK(std::count(counters_1.begin(), counters_1.end(), 1) == static_cast<int>(n));
      CHECK(std::count(counters_2.begin(), counters_2.end(), 1) == static_cast<int>(n));
    }
  }

  TEST_CASE("eager customization of bulk works with static thread pool", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      std::vector<std::thread::id> tids(42);

      auto fun = [&tids](std::size_t idx) {
        tids[idx] = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      };

      auto snd = ex::just() //
               | ex::continues_on(sch) | ex::bulk(tids.size(), fun);
      CHECK(std::equal_to<void*>()(&snd.pool_, &pool));
      stdexec::sync_wait(std::move(snd));

      // All the work should not have run on the same thread
      const auto actual = static_cast<std::size_t>(std::count(tids.begin(), tids.end(), tids[0]));
      const std::size_t wrong = tids.size();

      CHECK(actual != wrong);
    }
  }

  TEST_CASE("lazy customization of bulk works with static thread pool", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      std::vector<std::thread::id> tids(42);

      auto fun = [&tids](std::size_t idx) {
        tids[idx] = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      };

      auto snd = ex::just() //
               | ex::bulk(tids.size(), fun);
      stdexec::sync_wait(stdexec::starts_on(sch, std::move(snd)));

      // All the work should not have run on the same thread
      const auto actual = static_cast<std::size_t>(std::count(tids.begin(), tids.end(), tids[0]));
      const std::size_t wrong = tids.size();

      CHECK(actual != wrong);
    }
  }

  TEST_CASE("default bulk works with non-default constructible types", "[adaptors][bulk]") {
    ex::sender auto s = ex::just(non_default_constructible{42}) | ex::bulk(1, [](int, auto&) { });
    ex::sync_wait(std::move(s));
  }

  TEST_CASE("static thread pool works with non-default constructible types", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    ex::sender auto s = ex::just(non_default_constructible{42}) | ex::continues_on(sch)
                      | ex::bulk(1, [](int, auto&) { });
    ex::sync_wait(std::move(s));
  }
} // namespace
