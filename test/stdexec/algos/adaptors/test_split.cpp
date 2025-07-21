/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;

using namespace std::chrono_literals;

namespace {

  TEST_CASE("split returns a sender", "[adaptors][split]") {
    auto snd = ex::split(ex::just(19));
    using Snd = decltype(snd);
    static_assert(ex::enable_sender<Snd>);
    static_assert(ex::sender<Snd>);
    static_assert(ex::same_as<ex::env_of_t<Snd>, ex::env<>>);
    (void) snd;
  }

  TEST_CASE("split with environment returns a sender", "[adaptors][split]") {
    auto snd = ex::split(ex::just(19));
    using Snd = decltype(snd);
    static_assert(ex::enable_sender<Snd>);
    static_assert(ex::sender_in<Snd, ex::env<>>);
    static_assert(ex::same_as<ex::env_of_t<Snd>, ex::env<>>);
    (void) snd;
  }

  TEST_CASE("split simple example", "[adaptors][split]") {
    auto snd = ex::split(ex::just(19));
    auto op1 = ex::connect(snd, expect_value_receiver{19});
    auto op2 = ex::connect(snd, expect_value_receiver{19});
    ex::start(op1);
    ex::start(op2);
    // The receiver will ensure that the right value is produced
  }

  TEST_CASE("split executes predecessor sender once", "[adaptors][split]") {
    SECTION("when parameters are passed") {
      int counter{};
      auto snd = ex::split(ex::just() | ex::then([&] {
                             counter++;
                             return counter;
                           }));
      auto op1 = ex::connect(snd, expect_value_receiver{1});
      auto op2 = ex::connect(snd, expect_value_receiver{1});
      REQUIRE(counter == 0);
      ex::start(op1);
      REQUIRE(counter == 1);
      ex::start(op2);
      // The receiver will ensure that the right value is produced
      REQUIRE(counter == 1);
    }

    SECTION("without parameters") {
      int counter{};
      auto snd = ex::split(ex::just() | ex::then([&] { counter++; }));
      ex::sync_wait(snd | ex::then([] { }));
      ex::sync_wait(snd | ex::then([] { }));
      REQUIRE(counter == 1);
    }
  }

  TEST_CASE("split passes lvalue references", "[adaptors][split]") {
    auto split = ex::split(ex::just(42));
    using split_t = decltype(split);
    using value_t = ex::value_types_of_t<split_t, ex::env<>, pack, ex::__mmake_set>;
    static_assert(ex::__mset_eq<value_t, ex::__mset<pack<const int&>>>);

    auto then = split | ex::then([](const int& cval) {
                  int& val = const_cast<int&>(cval);
                  const int prev_val = val;
                  val /= 2;
                  return prev_val;
                });

    auto op1 = ex::connect(std::move(then), expect_value_receiver{42});
    auto op2 = ex::connect(split, expect_value_receiver{21});
    ex::start(op1);
    ex::start(op2);
    // The receiver will ensure that the right value is produced
  }

  TEST_CASE("split forwards errors", "[adaptors][split]") {
    SECTION("of exception_ptr type") {
      auto split = ex::split(ex::just_error(std::exception_ptr{}));
      using split_t = decltype(split);
      using error_t = ex::error_types_of_t<split_t, ex::env<>, ex::__mmake_set>;
      static_assert(ex::__mset_eq<error_t, ex::__mset<const std::exception_ptr&>>);

      auto op = ex::connect(split, expect_error_receiver{});
      ex::start(op);
      // The receiver will ensure that the right value is produced
    }

    SECTION("of any type") {
      auto split = ex::split(ex::just_error(42));
      using split_t = decltype(split);
      using error_t = ex::error_types_of_t<split_t, ex::env<>, ex::__mmake_set>;
      static_assert(ex::__mset_eq<error_t, ex::__mset<const std::exception_ptr&, const int&>>);

      auto op = ex::connect(split, expect_error_receiver<int>{});
      ex::start(op);
    }
  }

  TEST_CASE("split forwards stop signal", "[adaptors][split]") {
    auto split = ex::split(ex::just_stopped());
    using split_t = decltype(split);
    static_assert(ex::sends_stopped<split_t, ex::env<>>);

    auto op = ex::connect(split, expect_stopped_receiver{});
    ex::start(op);
    // The receiver will ensure that the right value is produced
  }

  TEST_CASE("split forwards external stop signal (1)", "[adaptors][split]") {
    ex::inplace_stop_source ssource;
    bool called = false;
    int counter{};
    auto split = ex::split(ex::just() | ex::then([&] { called = true; }));
    auto sndr = ex::write_env(
      ex::upon_stopped(
        std::move(split),
        [&] {
          ++counter;
          return 42;
        }),
      ex::prop{ex::get_stop_token, ssource.get_token()});
    auto op1 = ex::connect(sndr, expect_value_receiver{42});
    auto op2 = ex::connect(std::move(sndr), expect_value_receiver{42});
    ssource.request_stop();
    CHECK(counter == 0);
    ex::start(op1);
    CHECK(counter == 1);
    CHECK(!called);
    ex::start(op2);
    CHECK(counter == 2);
    CHECK(!called);
  }

  TEST_CASE("split forwards external stop signal (2)", "[adaptors][split]") {
    ex::inplace_stop_source ssource;
    bool called = false;
    int counter{};
    auto split = ex::split(ex::just() | ex::then([&] {
                             called = true;
                             return 7;
                           }));
    auto sndr = ex::write_env(
      ex::upon_stopped(
        std::move(split),
        [&] {
          ++counter;
          return 42;
        }),
      ex::prop{ex::get_stop_token, ssource.get_token()});
    auto op1 = ex::connect(sndr, expect_value_receiver{7});
    auto op2 = ex::connect(sndr, expect_value_receiver{42});
    REQUIRE(counter == 0);
    ex::start(op1); // operation starts and finishes.
    REQUIRE(counter == 0);
    REQUIRE(called);
    ssource.request_stop();
    ex::start(op2); // operation completes immediately with stopped.
    REQUIRE(counter == 1);
  }

  TEST_CASE("split forwards external stop signal (3)", "[adaptors][split]") {
    impulse_scheduler sched;
    ex::inplace_stop_source ssource;
    bool called = false;
    int counter{};
    auto split = ex::split(ex::starts_on(sched, ex::just() | ex::then([&] {
                                                  called = true;
                                                  return 7;
                                                })));
    auto sndr = ex::write_env(
      ex::upon_stopped(
        std::move(split),
        [&] {
          ++counter;
          return 42;
        }),
      ex::prop{ex::get_stop_token, ssource.get_token()});
    auto op1 = ex::connect(sndr, expect_value_receiver{42});
    auto op2 = ex::connect(std::move(sndr), expect_value_receiver{42});
    REQUIRE(counter == 0);
    ex::start(op1); // puts a unit of work on the impulse_scheduler and
                    // op1 into the list of waiting operations.
    REQUIRE(counter == 0);
    REQUIRE(!called);
    ssource.request_stop(); // This executes op1's stop callback, which
                            // completes op1 with "stopped". counter == 1
    REQUIRE(counter == 1);
    ex::start(op2); // Should immediately call set_stopped without getting
                    // added to the list of waiting operations. counter == 2
    REQUIRE(counter == 2);
    REQUIRE(!called);
    sched.start_next(); // Impulse scheduler notices stop has been requested
                        // and completes the underlying operation (just | then(...))
                        // with "stopped".
    REQUIRE(counter == 2);
    REQUIRE(!called);
  }

  TEST_CASE("split forwards external stop signal (4)", "[adaptors][split]") {
    impulse_scheduler sched;
    ex::inplace_stop_source ssource;
    bool called = false;
    int counter{};
    auto split = ex::split(ex::just() | ex::then([&] {
                             called = true;
                             return 7;
                           }));
    auto sndr1 = ex::starts_on(
      sched,
      ex::upon_stopped(
        ex::write_env(split, ex::prop{ex::get_stop_token, ssource.get_token()}), [&] {
          ++counter;
          return 42;
        }));
    auto sndr2 = ex::write_env(
      ex::starts_on(
        sched,
        ex::upon_stopped(
          std::move(split),
          [&] {
            ++counter;
            return 42;
          })),
      ex::prop{ex::get_stop_token, ssource.get_token()});
    auto op1 = ex::connect(std::move(sndr1), expect_value_receiver{7});
    auto op2 = ex::connect(std::move(sndr2), expect_stopped_receiver{});
    REQUIRE(counter == 0);
    ex::start(op1); // puts a unit of work on the impulse_scheduler.
    REQUIRE(counter == 0);
    REQUIRE(!called);
    sched.start_next(); // Impulse scheduler starts split sender, which
                        // completes with 7.
    REQUIRE(counter == 0);
    REQUIRE(called);
    ex::start(op2); // puts another unit of work on the impulse_scheduler
    REQUIRE(counter == 0);
    ssource.request_stop();
    sched.start_next(); // Impulse scheduler notices stop has been
                        // requested and "stops" the work.
    REQUIRE(counter == 0);
  }

  TEST_CASE("split forwards results from a different thread", "[adaptors][split]") {
    exec::static_thread_pool pool{1};
    auto split = ex::schedule(pool.get_scheduler()) | ex::then([] {
                   std::this_thread::sleep_for(1ms);
                   return 42;
                 })
               | ex::split();

    auto [val] = ex::sync_wait(split).value();
    REQUIRE(val == 42);
  }

  TEST_CASE("split is thread-safe", "[adaptors][split]") {
    exec::static_thread_pool pool{1};

    std::mt19937_64 eng{std::random_device{}()}; // or seed however you want
    std::uniform_int_distribution<> dist{0, 1000};

    auto split = ex::transfer_just(pool.get_scheduler(), std::chrono::microseconds{dist(eng)})
               | ex::then([](std::chrono::microseconds delay) {
                   std::this_thread::sleep_for(delay);
                   return 42;
                 })
               | ex::split();

    const unsigned n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(n_threads);
    std::vector<int> thread_results(n_threads, 0);
    const std::vector<std::chrono::microseconds> delays = [&] {
      std::vector<std::chrono::microseconds> thread_delay(n_threads);
      for (unsigned tid = 0; tid < n_threads; tid++) {
        thread_delay[tid] = std::chrono::microseconds{dist(eng)};
      }
      return thread_delay;
    }();
    for (unsigned tid = 0; tid < n_threads; tid++) {
      threads[tid] = std::thread([&split, &delays, &thread_results, tid] {
        inline_scheduler scheduler{};

        std::this_thread::sleep_for(delays[tid]);
        auto [val] = ex::sync_wait(
                       split | ex::continues_on(scheduler) | ex::then([](int v) { return v; }))
                       .value();
        thread_results[tid] = val;
      });
    }
    for (unsigned tid = 0; tid < n_threads; tid++) {
      threads[tid].join();
      REQUIRE(thread_results[tid] == 42);
    }
  }

  TEST_CASE("split can be an rvalue", "[adaptors][split]") {
    auto [val] = ex::sync_wait(ex::just(42) | ex::split() | ex::then([](int v) { return v; }))
                   .value();

    REQUIRE(val == 42);
  }

  struct move_only_type {
    move_only_type()
      : val(0) {
    }

    move_only_type(int v)
      : val(v) {
    }

    move_only_type(move_only_type&&) = default;
    int val;
  };

  struct copy_and_movable_type {
    copy_and_movable_type(int v)
      : val(v) {
    }

    int val;
  };

  TEST_CASE("split into then", "[adaptors][split]") {
    SECTION("split with move only input sender of temporary") {
      auto snd = ex::split(ex::just(move_only_type{0})) | ex::then([](const move_only_type&) { });
      ex::sync_wait(snd);
    }

    SECTION("split with move only input sender by moving in") {
      auto snd0 = ex::just(move_only_type{});
      auto snd = ex::split(std::move(snd0)) | ex::then([](const move_only_type&) { });
      ex::sync_wait(snd);
    }

    SECTION("split with copyable rvalue input sender") {
      auto snd = ex::split(ex::just(copy_and_movable_type{0}))
               | ex::then([](const copy_and_movable_type&) { });
      ex::sync_wait(snd);
    }

    SECTION("split with copyable lvalue input sender") {
      auto snd0 = ex::just(copy_and_movable_type{0});
      auto snd = ex::split(snd0) | ex::then([](const copy_and_movable_type&) { });
      ex::sync_wait(snd);
    }

    SECTION("lvalue split move only sender") {
      auto multishot = ex::split(ex::just(move_only_type{0}));
      auto snd = multishot | ex::then([](const move_only_type&) { });

      REQUIRE(ex::sender_of<decltype(multishot), ex::set_value_t(const move_only_type&)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(move_only_type)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(move_only_type&)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(move_only_type&&)>);

      ex::sync_wait(snd);
    }

    SECTION("lvalue split copyable sender") {
      auto multishot = ex::split(ex::just(copy_and_movable_type{0}));
      ex::get_completion_signatures(multishot, ex::env<>{});
      auto snd = multishot | ex::then([](const copy_and_movable_type&) { });

      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(copy_and_movable_type)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(const copy_and_movable_type)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(copy_and_movable_type&)>);
      REQUIRE(ex::sender_of<decltype(multishot), ex::set_value_t(const copy_and_movable_type&)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(copy_and_movable_type&&)>);
      REQUIRE(!ex::sender_of<decltype(multishot), ex::set_value_t(const copy_and_movable_type&&)>);

      ex::sync_wait(snd);
    }
  }

  TEMPLATE_TEST_CASE(
    "split move-only and copyable senders",
    "[adaptors][split]",
    move_only_type,
    copy_and_movable_type) {
    int called = 0;
    auto multishot = ex::just(TestType(10)) | ex::then([&](TestType obj) {
                       ++called;
                       return TestType(obj.val + 1);
                     })
                   | ex::split();
    auto wa = ex::when_all(
      ex::then(multishot, [](const TestType& obj) { return obj.val; }),
      ex::then(multishot, [](const TestType& obj) { return obj.val * 2; }),
      ex::then(multishot, [](const TestType& obj) { return obj.val * 3; }));

    auto [v1, v2, v3] = ex::sync_wait(std::move(wa)).value();

    REQUIRE(called == 1);
    REQUIRE(v1 == 11);
    REQUIRE(v2 == 22);
    REQUIRE(v3 == 33);
  }

  template <class T>
  concept can_split_lvalue_of = requires(T t) { ex::split(t); };

  TEST_CASE("split can only accept copyable lvalue input senders", "[adaptors][split]") {
    static_assert(!can_split_lvalue_of<decltype(ex::just(move_only_type{0}))>);
    static_assert(can_split_lvalue_of<decltype(ex::just(copy_and_movable_type{0}))>);
  }

  TEST_CASE("split into when_all", "[adaptors][split]") {
    int counter{};
    auto snd = ex::split(ex::just() | ex::then([&] {
                           counter++;
                           return counter;
                         }));
    auto wa = ex::when_all(snd | ex::then([](auto) { return 10; }), snd | ex::then([](auto) {
                                                                      return 20;
                                                                    }));
    REQUIRE(counter == 0);
    auto [v1, v2] = ex::sync_wait(std::move(wa)).value();
    REQUIRE(counter == 1);
    REQUIRE(v1 == 10);
    REQUIRE(v2 == 20);
  }

  TEST_CASE("split can nest", "[adaptors][split]") {
    auto split_1 = ex::just(42) | ex::split();
    auto split_2 = split_1 | ex::split();

    auto [v1] = ex::sync_wait(split_1 | ex::then([](const int& cv) {
                                int& v = const_cast<int&>(cv);
                                return v = 1;
                              }))
                  .value();

    auto [v2] = ex::sync_wait(split_2 | ex::then([](const int& cv) {
                                int& v = const_cast<int&>(cv);
                                return v = 2;
                              }))
                  .value();

    auto [v3] = ex::sync_wait(split_1).value();

    REQUIRE(v1 == 1);
    REQUIRE(v2 == 2);
    REQUIRE(v3 == 1);
  }

  TEST_CASE("split doesn't advertise completion scheduler", "[adaptors][split]") {
    inline_scheduler sched;

    auto snd = ex::transfer_just(sched, 42) | ex::split();
    using snd_t = decltype(snd);
    static_assert(!ex::__callable<ex::get_completion_scheduler_t<ex::set_value_t>, snd_t>);
    static_assert(!ex::__callable<ex::get_completion_scheduler_t<ex::set_error_t>, snd_t>);
    static_assert(!ex::__callable<ex::get_completion_scheduler_t<ex::set_stopped_t>, snd_t>);
    (void) snd;
  }

  struct my_sender {
    using sender_concept = ex::sender_t;
    using is_sender = void;

    using completion_signatures = ex::completion_signatures_of_t<decltype(ex::just())>;

    template <class Recv>
    friend auto tag_invoke(ex::connect_t, my_sender&&, Recv&& recv) {
      return ex::connect(ex::just(), std::forward<Recv>(recv));
    }

    template <class Recv>
    friend auto tag_invoke(ex::connect_t, const my_sender&, Recv&& recv) {
      return ex::connect(ex::just(), std::forward<Recv>(recv));
    }
  };

  TEST_CASE("split accepts a custom sender", "[adaptors][split]") {
    auto snd1 = my_sender();
    auto snd2 = ex::split(std::move(snd1));
    static_assert(ex::__well_formed_sender<decltype(snd1)>);
    static_assert(ex::__well_formed_sender<decltype(snd2)>);
    using Snd = decltype(snd2);
    static_assert(ex::enable_sender<Snd>);
    static_assert(ex::sender<Snd>);
    static_assert(ex::same_as<ex::env_of_t<Snd>, ex::env<>>);
    (void) snd1;
    (void) snd2;
  }
} // namespace
