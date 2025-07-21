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

  template <class Shape, int N, int (&Counter)[N]>
  void function_range(Shape b, Shape e) {
    while (b != e) {
      Counter[b++]++;
    }
  }

  template <class Shape>
  struct function_object_t {
    int* Counter;

    void operator()(Shape i) {
      Counter[i]++;
    }
  };

  template <class Shape>
  struct function_object_range_t {
    int* Counter;

    void operator()(Shape b, Shape e) {
      while (b != e) {
        Counter[b++]++;
      }
    }
  };

  TEST_CASE("bulk returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk(ex::just(19), ex::par, 8, [](int, int) { });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("bulk_chunked returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk_chunked(ex::just(19), ex::par, 8, [](int, int, int) { });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("bulk_unchunked returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk_unchunked(ex::just(19), ex::par, 8, [](int, int) { });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("bulk with environment returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk(ex::just(19), ex::par, 8, [](int, int) { });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("bulk_chunked with environment returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk_chunked(ex::just(19), ex::par, 8, [](int, int, int) { });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("bulk_unchunked with environment returns a sender", "[adaptors][bulk]") {
    auto snd = ex::bulk_unchunked(ex::just(19), ex::par, 8, [](int, int) { });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("bulk can be piped", "[adaptors][bulk]") {
    ex::sender auto snd = ex::just() | ex::bulk(ex::par, 42, [](int) { });
    (void) snd;
  }

  TEST_CASE("bulk_chunked can be piped", "[adaptors][bulk]") {
    ex::sender auto snd = ex::just() | ex::bulk_chunked(ex::par, 42, [](int, int) { });
    (void) snd;
  }

  TEST_CASE("bulk_unchunked can be piped", "[adaptors][bulk]") {
    ex::sender auto snd = ex::just() | ex::bulk_unchunked(ex::par, 42, [](int) { });
    (void) snd;
  }

  TEST_CASE("bulk keeps values_type from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    check_val_types<ex::__mset<pack<>>>(ex::just() | ex::bulk(ex::par, n, [](int) { }));
    check_val_types<ex::__mset<pack<double>>>(
      ex::just(4.2) | ex::bulk(ex::par, n, [](int, double) { }));
    check_val_types<ex::__mset<pack<double, std::string>>>(
      ex::just(4.2, std::string{}) | ex::bulk(ex::par, n, [](int, double, std::string) { }));
  }

  TEST_CASE("bulk_chunked keeps values_type from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    check_val_types<ex::__mset<pack<>>>(
      ex::just() | ex::bulk_chunked(ex::par, n, [](int, int) { }));
    check_val_types<ex::__mset<pack<double>>>(
      ex::just(4.2) | ex::bulk_chunked(ex::par, n, [](int, int, double) { }));
    check_val_types<ex::__mset<pack<double, std::string>>>(
      ex::just(4.2, std::string{})
      | ex::bulk_chunked(ex::par, n, [](int, int, double, std::string) { }));
  }

  TEST_CASE("bulk_unchunked keeps values_type from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    check_val_types<ex::__mset<pack<>>>(ex::just() | ex::bulk_unchunked(ex::par, n, [](int) { }));
    check_val_types<ex::__mset<pack<double>>>(
      ex::just(4.2) | ex::bulk_unchunked(ex::par, n, [](int, double) { }));
    check_val_types<ex::__mset<pack<double, std::string>>>(
      ex::just(4.2, std::string{})
      | ex::bulk_unchunked(ex::par, n, [](int, double, std::string) { }));
  }

  TEST_CASE("bulk keeps error_types from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

#if !STDEXEC_MSVC()
    // MSVCBUG https://developercommunity.visualstudio.com/t/noexcept-expression-in-lambda-template-n/10718680
    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1) | ex::bulk(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2) | ex::bulk(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::just_error(n) | ex::bulk(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::transfer_just(sched3) | ex::bulk(ex::par, n, [](int) noexcept { }));
#  if !STDEXEC_STD_NO_EXCEPTIONS()
    check_err_types<ex::__mset<std::exception_ptr, int>>(
      ex::transfer_just(sched3) | ex::bulk(ex::par, n, [](int) { throw std::logic_error{"err"}; }));
#  endif
#endif
  }

  TEST_CASE("bulk_chunked keeps error_types from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1) | ex::bulk_chunked(ex::par, n, [](int, int) noexcept { }));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2) | ex::bulk_chunked(ex::par, n, [](int, int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::just_error(n) | ex::bulk_chunked(ex::par, n, [](int, int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::transfer_just(sched3) | ex::bulk_chunked(ex::par, n, [](int, int) noexcept { }));
#if !STDEXEC_STD_NO_EXCEPTIONS()
    check_err_types<ex::__mset<std::exception_ptr, int>>(
      ex::transfer_just(sched3)
      | ex::bulk_chunked(ex::par, n, [](int, int) { throw std::logic_error{"err"}; }));
#endif
  }

  TEST_CASE("bulk_unchunked keeps error_types from input sender", "[adaptors][bulk]") {
    constexpr int n = 42;
    inline_scheduler sched1{};
    error_scheduler sched2{};
    error_scheduler<int> sched3{43};

    check_err_types<ex::__mset<>>(
      ex::transfer_just(sched1) | ex::bulk_unchunked(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<std::exception_ptr>>(
      ex::transfer_just(sched2) | ex::bulk_unchunked(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::just_error(n) | ex::bulk_unchunked(ex::par, n, [](int) noexcept { }));
    check_err_types<ex::__mset<int>>(
      ex::transfer_just(sched3) | ex::bulk_unchunked(ex::par, n, [](int) noexcept { }));
#if !STDEXEC_STD_NO_EXCEPTIONS()
    check_err_types<ex::__mset<std::exception_ptr, int>>(
      ex::transfer_just(sched3)
      | ex::bulk_unchunked(ex::par, n, [](int) { throw std::logic_error{"err"}; }));
#endif
  }

  TEST_CASE("bulk can be used with a function", "[adaptors][bulk]") {
    constexpr int n = 9;
    static int counter1[n]{};
    std::fill_n(counter1, n, 0);

    ex::sender auto snd = ex::just() | ex::bulk(ex::par, n, function<int, n, counter1>);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter1) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk_chunked can be used with a function", "[adaptors][bulk]") {
    constexpr int n = 9;
    static int counter2[n]{};
    std::fill_n(counter2, n, 0);

    ex::sender auto snd = ex::just()
                        | ex::bulk_chunked(ex::par, n, function_range<int, n, counter2>);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter2[i] == 1);
    }
  }

  TEST_CASE("bulk_unchunked can be used with a function", "[adaptors][bulk]") {
    constexpr int n = 9;
    static int counter3[n]{};
    std::fill_n(counter3, n, 0);

    ex::sender auto snd = ex::just() | ex::bulk_unchunked(ex::par, n, function<int, n, counter3>);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter3[i] == 1);
    }
  }

  TEST_CASE("bulk can be used with a function object", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};
    function_object_t<int> fn{counter};

    ex::sender auto snd = ex::just() | ex::bulk(ex::par, n, fn);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk_chunked can be used with a function object", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};
    function_object_range_t<int> fn{counter};

    ex::sender auto snd = ex::just() | ex::bulk_chunked(ex::par, n, fn);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk_unchunked can be used with a function object", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};
    function_object_t<int> fn{counter};

    ex::sender auto snd = ex::just() | ex::bulk_unchunked(ex::par, n, fn);
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk can be used with a lambda", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};

    ex::sender auto snd = ex::just() | ex::bulk(ex::par, n, [&](int i) { counter[i]++; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i: counter) {
      CHECK(i == 1);
    }
  }

  TEST_CASE("bulk_chunked can be used with a lambda", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};

    ex::sender auto snd = ex::just() | ex::bulk_chunked(ex::par, n, [&](int b, int e) {
                            while (b < e)
                              counter[b++]++;
                          });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk_unchunked can be used with a lambda", "[adaptors][bulk]") {
    constexpr int n = 9;
    int counter[n]{0};

    ex::sender auto snd = ex::just() | ex::bulk_unchunked(ex::par, n, [&](int i) { counter[i]++; });
    auto op = ex::connect(std::move(snd), expect_void_receiver{});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk works with all standard execution policies", "[adaptors][bulk]") {
    ex::sender auto snd1 = ex::just() | ex::bulk(ex::seq, 9, [](int) { });
    ex::sender auto snd2 = ex::just() | ex::bulk(ex::par, 9, [](int) { });
    ex::sender auto snd3 = ex::just() | ex::bulk(ex::par_unseq, 9, [](int) { });
    ex::sender auto snd4 = ex::just() | ex::bulk(ex::unseq, 9, [](int) { });

    static_assert(ex::sender<decltype(snd1)>);
    static_assert(ex::sender<decltype(snd2)>);
    static_assert(ex::sender<decltype(snd3)>);
    static_assert(ex::sender<decltype(snd4)>);
    (void) snd1;
    (void) snd2;
    (void) snd3;
    (void) snd4;
  }

  TEST_CASE("bulk_chunked works with all standard execution policies", "[adaptors][bulk]") {
    ex::sender auto snd1 = ex::just() | ex::bulk_chunked(ex::seq, 9, [](int, int) { });
    ex::sender auto snd2 = ex::just() | ex::bulk_chunked(ex::par, 9, [](int, int) { });
    ex::sender auto snd3 = ex::just() | ex::bulk_chunked(ex::par_unseq, 9, [](int, int) { });
    ex::sender auto snd4 = ex::just() | ex::bulk_chunked(ex::unseq, 9, [](int, int) { });

    static_assert(ex::sender<decltype(snd1)>);
    static_assert(ex::sender<decltype(snd2)>);
    static_assert(ex::sender<decltype(snd3)>);
    static_assert(ex::sender<decltype(snd4)>);
    (void) snd1;
    (void) snd2;
    (void) snd3;
    (void) snd4;
  }

  TEST_CASE("bulk forwards values", "[adaptors][bulk]") {
    constexpr int n = 9;
    constexpr int magic_number = 42;
    int counter[n]{0};

    auto snd = ex::just(magic_number) | ex::bulk(ex::par, n, [&](int i, int val) {
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

  TEST_CASE("bulk_chunked forwards values", "[adaptors][bulk]") {
    constexpr int n = 9;
    constexpr int magic_number = 42;
    int counter[n]{0};

    auto snd = ex::just(magic_number) | ex::bulk_chunked(ex::par, n, [&](int b, int e, int val) {
                 if (val == magic_number) {
                   while (b < e) {
                     counter[b++]++;
                   }
                 }
               });
    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk_unchunked forwards values", "[adaptors][bulk]") {
    constexpr int n = 9;
    constexpr int magic_number = 42;
    int counter[n]{0};

    auto snd = ex::just(magic_number) | ex::bulk_unchunked(ex::par, n, [&](int i, int val) {
                 if (val == magic_number) {
                   counter[i]++;
                 }
               });
    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);

    for (int i = 0; i < n; i++) {
      CHECK(counter[i] == 1);
    }
  }

  TEST_CASE("bulk forwards values that can be taken by reference", "[adaptors][bulk]") {
    constexpr std::size_t n = 9;
    std::vector<int> vals(n, 0);
    std::vector<int> vals_expected(n);
    std::iota(vals_expected.begin(), vals_expected.end(), 0);

    auto snd = ex::just(std::move(vals))
             | ex::bulk(ex::par, n, [&](std::size_t i, std::vector<int>& vals) {
                 vals[i] = static_cast<int>(i);
               });
    auto op = ex::connect(std::move(snd), expect_value_receiver{vals_expected});
    ex::start(op);
  }

  TEST_CASE("bulk_chunked forwards values that can be taken by reference", "[adaptors][bulk]") {
    constexpr std::size_t n = 9;
    std::vector<int> vals(n, 0);
    std::vector<int> vals_expected(n);
    std::iota(vals_expected.begin(), vals_expected.end(), 0);

    auto snd =
      ex::just(std::move(vals))
      | ex::bulk_chunked(ex::par, n, [&](std::size_t b, std::size_t e, std::vector<int>& vals) {
          for (; b < e; ++b) {
            vals[b] = static_cast<int>(b);
          }
        });
    auto op = ex::connect(std::move(snd), expect_value_receiver{vals_expected});
    ex::start(op);
  }

  TEST_CASE("bulk_unchunked forwards values that can be taken by reference", "[adaptors][bulk]") {
    constexpr std::size_t n = 9;
    std::vector<int> vals(n, 0);
    std::vector<int> vals_expected(n);
    std::iota(vals_expected.begin(), vals_expected.end(), 0);

    auto snd = ex::just(std::move(vals))
             | ex::bulk_unchunked(ex::par, n, [&](std::size_t i, std::vector<int>& vals) {
                 vals[i] = static_cast<int>(i);
               });
    auto op = ex::connect(std::move(snd), expect_value_receiver{vals_expected});
    ex::start(op);
  }

  TEST_CASE("bulk cannot be used to change the value type", "[adaptors][bulk]") {
    constexpr int magic_number = 42;
    constexpr int n = 2;

    auto snd = ex::just(magic_number)
             | ex::bulk(ex::par, n, [](int, int) { return function_object_t<int>{nullptr}; });

    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);
  }

  TEST_CASE("bulk_chunked cannot be used to change the value type", "[adaptors][bulk]") {
    constexpr int magic_number = 42;
    constexpr int n = 2;

    auto snd = ex::just(magic_number) | ex::bulk_chunked(ex::par, n, [](int, int, int) {
                 return function_object_range_t<int>{nullptr};
               });

    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);
  }

  TEST_CASE("bulk_unchunked cannot be used to change the value type", "[adaptors][bulk]") {
    constexpr int magic_number = 42;
    constexpr int n = 2;

    auto snd = ex::just(magic_number) | ex::bulk_unchunked(ex::par, n, [](int, int) {
                 return function_object_t<int>{nullptr};
               });

    auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
    ex::start(op);
  }

#if !STDEXEC_STD_NO_EXCEPTIONS()
  TEST_CASE("bulk can throw, and set_error will be called", "[adaptors][bulk]") {
    constexpr int n = 2;

    auto snd = ex::just() | ex::bulk(ex::par, n, [](int) -> int { throw std::logic_error{"err"}; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk_chunked can throw, and set_error will be called", "[adaptors][bulk]") {
    constexpr int n = 2;

    auto snd = ex::just()
             | ex::bulk_chunked(ex::par, n, [](int, int) -> int { throw std::logic_error{"err"}; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk_unchunked can throw, and set_error will be called", "[adaptors][bulk]") {
    constexpr int n = 2;

    auto snd = ex::just()
             | ex::bulk_unchunked(ex::par, n, [](int) -> int { throw std::logic_error{"err"}; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{});
    ex::start(op);
  }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()

  TEST_CASE("bulk function is not called on error", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_error(std::string{"err"})
             | ex::bulk(ex::par, n, [&called](int) { called++; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
    ex::start(op);
  }

  TEST_CASE("bulk_chunked function is not called on error", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_error(std::string{"err"})
             | ex::bulk_chunked(ex::par, n, [&called](int, int) { called++; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
    ex::start(op);
  }

  TEST_CASE("bulk_unchunked function is not called on error", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_error(std::string{"err"})
             | ex::bulk_unchunked(ex::par, n, [&called](int) { called++; });
    auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
    ex::start(op);
  }

  TEST_CASE("bulk function in not called on stop", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_stopped() | ex::bulk(ex::par, n, [&called](int) { called++; });
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk_chunked function in not called on stop", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_stopped() | ex::bulk_chunked(ex::par, n, [&called](int, int) { called++; });
    auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
    ex::start(op);
  }

  TEST_CASE("bulk_unchunked function in not called on stop", "[adaptors][bulk]") {
    constexpr int n = 2;
    int called{};

    auto snd = ex::just_stopped() | ex::bulk_unchunked(ex::par, n, [&called](int) { called++; });
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
                 | ex::bulk(ex::par, n, [&counter](std::size_t idx) { counter[idx] = 0; })
                 | ex::bulk(ex::par, n, [&counter](std::size_t idx) { counter[idx]++; });
        ex::sync_wait(std::move(snd));

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch, 42)
                 | ex::bulk(
                     ex::par,
                     n,
                     [&counter](std::size_t idx, int val) {
                       if (val == 42) {
                         counter[idx] = 0;
                       }
                     })
                 | ex::bulk(ex::par, n, [&counter](std::size_t idx, int val) {
                     if (val == 42) {
                       counter[idx]++;
                     }
                   });
        auto [val] = ex::sync_wait(std::move(snd)).value();

        CHECK(val == 42);

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
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
            ex::par,
            n,
            [](std::size_t idx, std::vector<int>& vals) { vals[idx] = static_cast<int>(idx); })
          | ex::bulk(ex::par, n, [](std::size_t idx, std::vector<int>& vals) { ++vals[idx]; });
        auto [vals_actual] = ex::sync_wait(std::move(snd)).value();

        CHECK(vals_actual == vals_expected);
      }
    }

#if !STDEXEC_STD_NO_EXCEPTIONS()
    SECTION("With exception") {
      constexpr int n = 9;
      auto snd = ex::transfer_just(sch)
               | ex::bulk(ex::par, n, [](int) { throw std::runtime_error("bulk"); });

      CHECK_THROWS_AS(ex::sync_wait(std::move(snd)), std::runtime_error);
    }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()

    SECTION("With concurrent enqueueing") {
      constexpr std::size_t n = 4;
      std::vector<int> counters_1(n, 0);
      std::vector<int> counters_2(n, 0);

      ex::sender auto snd = ex::when_all(
        ex::schedule(sch) | ex::bulk(ex::par, n, [&](std::size_t id) { counters_1[id]++; }),
        ex::schedule(sch) | ex::bulk(ex::par, n, [&](std::size_t id) { counters_2[id]++; }));

      ex::sync_wait(std::move(snd));

      CHECK(std::count(counters_1.begin(), counters_1.end(), 1) == static_cast<int>(n));
      CHECK(std::count(counters_2.begin(), counters_2.end(), 1) == static_cast<int>(n));
    }
  }

  TEST_CASE("bulk_chunked works with static thread pool", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      for (std::size_t n = 0; n < 9u; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch)
                 | ex::bulk_chunked(
                     ex::par,
                     n,
                     [&counter](std::size_t b, std::size_t e) {
                       while (b < e)
                         counter[b++] = 0;
                     })
                 | ex::bulk_chunked(ex::par, n, [&counter](std::size_t b, std::size_t e) {
                     while (b < e)
                       counter[b++]++;
                   });
        ex::sync_wait(std::move(snd));

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch, 42)
                 | ex::bulk_chunked(
                     ex::par,
                     n,
                     [&counter](std::size_t b, std::size_t e, int val) {
                       if (val == 42) {
                         while (b < e)
                           counter[b++] = 0;
                       }
                     })
                 | ex::bulk_chunked(ex::par, n, [&counter](std::size_t b, std::size_t e, int val) {
                     if (val == 42) {
                       while (b < e)
                         counter[b++]++;
                     }
                   });
        auto [val] = ex::sync_wait(std::move(snd)).value();

        CHECK(val == 42);

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
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
          | ex::bulk_chunked(
            ex::par,
            n,
            [](std::size_t b, std::size_t e, std::vector<int>& vals) {
              while (b < e) {
                vals[b] = static_cast<int>(b);
                ++b;
              }
            })
          | ex::bulk_chunked(ex::par, n, [](std::size_t b, std::size_t e, std::vector<int>& vals) {
              while (b < e)
                ++vals[b++];
            });
        auto [vals_actual] = ex::sync_wait(std::move(snd)).value();

        CHECK(vals_actual == vals_expected);
      }
    }

#if !STDEXEC_STD_NO_EXCEPTIONS()
    SECTION("With exception") {
      constexpr int n = 9;
      auto snd = ex::transfer_just(sch) | ex::bulk_chunked(ex::par, n, [](int, int) {
                   throw std::runtime_error("bulk_chunked");
                 });

      CHECK_THROWS_AS(ex::sync_wait(std::move(snd)), std::runtime_error);
    }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()

    SECTION("With concurrent enqueueing") {
      constexpr std::size_t n = 4;
      std::vector<int> counters_1(n, 0);
      std::vector<int> counters_2(n, 0);

      ex::sender auto snd = ex::when_all(
        ex::schedule(sch)
          | ex::bulk_chunked(
            ex::par,
            n,
            [&](std::size_t b, std::size_t e) {
              while (b < e)
                counters_1[b++]++;
            }),
        ex::schedule(sch) | ex::bulk_chunked(ex::par, n, [&](std::size_t b, std::size_t e) {
          while (b < e)
            counters_2[b++]++;
        }));

      ex::sync_wait(std::move(snd));

      CHECK(std::count(counters_1.begin(), counters_1.end(), 1) == static_cast<int>(n));
      CHECK(std::count(counters_2.begin(), counters_2.end(), 1) == static_cast<int>(n));
    }
  }

  TEST_CASE("bulk_unchunked works with static thread pool", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      for (std::size_t n = 0; n < 9u; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch)
                 | ex::bulk_unchunked(ex::par, n, [&counter](std::size_t idx) { counter[idx] = 0; })
                 | ex::bulk_unchunked(ex::par, n, [&counter](std::size_t idx) { counter[idx]++; });
        ex::sync_wait(std::move(snd));

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> counter(n, 42);

        auto snd = ex::transfer_just(sch, 42)
                 | ex::bulk_unchunked(
                     ex::par,
                     n,
                     [&counter](std::size_t idx, int val) {
                       if (val == 42) {
                         counter[idx] = 0;
                       }
                     })
                 | ex::bulk_unchunked(ex::par, n, [&counter](std::size_t idx, int val) {
                     if (val == 42) {
                       counter[idx]++;
                     }
                   });
        auto [val] = ex::sync_wait(std::move(snd)).value();

        CHECK(val == 42);

        const std::size_t actual = static_cast<std::size_t>(
          std::count(counter.begin(), counter.end(), 1));
        const std::size_t expected = n;

        CHECK(expected == actual);
      }
    }

    SECTION("With values in the set_value channel that can be taken by reference") {
      for (std::size_t n = 0; n < 9; n++) {
        std::vector<int> vals(n, 0);
        std::vector<int> vals_expected(n);
        std::iota(vals_expected.begin(), vals_expected.end(), 1);

        auto snd = ex::transfer_just(sch, std::move(vals))
                 | ex::bulk_unchunked(
                     ex::par,
                     n,
                     [](std::size_t idx, std::vector<int>& vals) {
                       vals[idx] = static_cast<int>(idx);
                     })
                 | ex::bulk_unchunked(ex::par, n, [](std::size_t idx, std::vector<int>& vals) {
                     ++vals[idx];
                   });
        auto [vals_actual] = ex::sync_wait(std::move(snd)).value();

        CHECK(vals_actual == vals_expected);
      }
    }

#if !STDEXEC_STD_NO_EXCEPTIONS()
    SECTION("With exception") {
      constexpr int n = 9;
      auto snd = ex::transfer_just(sch) | ex::bulk_unchunked(ex::par, n, [](int) {
                   throw std::runtime_error("bulk_unchunked");
                 });

      CHECK_THROWS_AS(ex::sync_wait(std::move(snd)), std::runtime_error);
    }
#endif // !STDEXEC_STD_NO_EXCEPTIONS()

    SECTION("With concurrent enqueueing") {
      constexpr std::size_t n = 4;
      std::vector<int> counters_1(n, 0);
      std::vector<int> counters_2(n, 0);

      ex::sender auto snd = ex::when_all(
        ex::schedule(sch)
          | ex::bulk_unchunked(ex::par, n, [&](std::size_t id) { counters_1[id]++; }),
        ex::schedule(sch)
          | ex::bulk_unchunked(ex::par, n, [&](std::size_t id) { counters_2[id]++; }));

      ex::sync_wait(std::move(snd));

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

      auto snd = ex::just() | ex::continues_on(sch) | ex::bulk(ex::par, tids.size(), fun);
      ex::sync_wait(std::move(snd));

      // All the work should not have run on the same thread
      const auto actual = static_cast<std::size_t>(std::count(tids.begin(), tids.end(), tids[0]));
      const std::size_t wrong = tids.size();

      CHECK(actual != wrong);
    }
  }

  TEST_CASE(
    "eager customization of bulk_chunked works with static thread pool",
    "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      std::vector<std::thread::id> tids(42);

      auto fun = [&tids](std::size_t b, std::size_t e) {
        while (b < e) {
          tids[b++] = std::this_thread::get_id();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      };

      auto snd = ex::just() | ex::continues_on(sch) | ex::bulk_chunked(ex::par, tids.size(), fun);
      ex::sync_wait(std::move(snd));

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

      auto snd = ex::just() | ex::bulk(ex::par, tids.size(), fun);
      ex::sync_wait(ex::starts_on(sch, std::move(snd)));

      // All the work should not have run on the same thread
      const auto actual = static_cast<std::size_t>(std::count(tids.begin(), tids.end(), tids[0]));
      const std::size_t wrong = tids.size();

      CHECK(actual != wrong);
    }
  }

  TEST_CASE(
    "lazy customization of bulk_chunked works with static thread pool",
    "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("Without values in the set_value channel") {
      std::vector<std::thread::id> tids(42);

      auto fun = [&tids](std::size_t b, std::size_t e) {
        while (b < e) {
          tids[b++] = std::this_thread::get_id();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
      };

      auto snd = ex::just() | ex::bulk_chunked(ex::par, tids.size(), fun);
      ex::sync_wait(ex::starts_on(sch, std::move(snd)));

      // All the work should not have run on the same thread
      const auto actual = static_cast<std::size_t>(std::count(tids.begin(), tids.end(), tids[0]));
      const std::size_t wrong = tids.size();

      CHECK(actual != wrong);
    }
  }

  TEST_CASE("default bulk works with non-default constructible types", "[adaptors][bulk]") {
    ex::sender auto s = ex::just(non_default_constructible{42})
                      | ex::bulk(ex::par, 1, [](int, auto&) { });
    ex::sync_wait(std::move(s));
  }

  TEST_CASE("default bulk_chunked works with non-default constructible types", "[adaptors][bulk]") {
    ex::sender auto s = ex::just(non_default_constructible{42})
                      | ex::bulk_chunked(ex::par, 1, [](int, int, auto&) { });
    ex::sync_wait(std::move(s));
  }

  TEST_CASE(
    "default bulk_unchunked works with non-default constructible types",
    "[adaptors][bulk]") {
    ex::sender auto s = ex::just(non_default_constructible{42})
                      | ex::bulk_unchunked(ex::par, 1, [](int, auto&) { });
    ex::sync_wait(std::move(s));
  }

  TEST_CASE("static thread pool works with non-default constructible types", "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    ex::sender auto s = ex::just(non_default_constructible{42}) | ex::continues_on(sch)
                      | ex::bulk(ex::par, 1, [](int, auto&) { });
    ex::sync_wait(std::move(s));
  }

  template <class Sched, class Policy>
  int number_of_threads_in_bulk(Sched sch, const Policy& policy, int n) {
    std::vector<std::thread::id> tids(n);
    auto fun = [&tids](std::size_t idx) {
      tids[idx] = std::this_thread::get_id();
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    };

    auto snd = ex::just() | ex::continues_on(sch) | ex::bulk(policy, tids.size(), fun);
    ex::sync_wait(std::move(snd));

    std::sort(tids.begin(), tids.end());
    return static_cast<int>(std::unique(tids.begin(), tids.end()) - tids.begin());
  }

  TEST_CASE(
    "static thread pool execute bulk work in accordance with the execution policy",
    "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("seq execution policy") {
      REQUIRE(number_of_threads_in_bulk(sch, ex::seq, 42) == 1);
    }
    SECTION("unseq execution policy") {
      REQUIRE(number_of_threads_in_bulk(sch, ex::unseq, 42) == 1);
    }
    SECTION("par execution policy") {
      REQUIRE(number_of_threads_in_bulk(sch, ex::par, 42) > 1);
    }
    SECTION("par_unseq execution policy") {
      REQUIRE(number_of_threads_in_bulk(sch, ex::par_unseq, 42) > 1);
    }
  }

  template <class Sched, class Policy>
  int number_of_threads_in_bulk_chunked(Sched sch, const Policy& policy, int n) {
    std::vector<std::thread::id> tids(n);
    auto fun = [&tids](std::size_t b, std::size_t e) {
      while (b < e)
        tids[b++] = std::this_thread::get_id();
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    };

    auto snd = ex::just() | ex::continues_on(sch) | ex::bulk_chunked(policy, tids.size(), fun);
    ex::sync_wait(std::move(snd));

    std::sort(tids.begin(), tids.end());
    return static_cast<int>(std::unique(tids.begin(), tids.end()) - tids.begin());
  }

  TEST_CASE(
    "static thread pool execute bulk_chunked work in accordance with the execution policy",
    "[adaptors][bulk]") {
    exec::static_thread_pool pool{4};
    ex::scheduler auto sch = pool.get_scheduler();

    SECTION("seq execution policy") {
      REQUIRE(number_of_threads_in_bulk_chunked(sch, ex::seq, 42) == 1);
    }
    SECTION("unseq execution policy") {
      REQUIRE(number_of_threads_in_bulk_chunked(sch, ex::unseq, 42) == 1);
    }
    SECTION("par execution policy") {
      REQUIRE(number_of_threads_in_bulk_chunked(sch, ex::par, 42) > 1);
    }
    SECTION("par_unseq execution policy") {
      REQUIRE(number_of_threads_in_bulk_chunked(sch, ex::par_unseq, 42) > 1);
    }
  }

  struct my_domain {
    template <ex::sender_expr_for<ex::bulk_chunked_t> Sender, class... Env>
    static auto transform_sender(Sender, const Env&...) {
      return ex::just(std::string{"hijacked"});
    }
  };

  TEST_CASE("late customizing bulk_chunked also changes the behavior of bulk", "[adaptors][then]") {
    bool called{false};
    // The customization will return a different value
    basic_inline_scheduler<my_domain> sched;
    auto snd = ex::just(std::string{"hello"})
             | ex::on(sched, ex::bulk(ex::par, 1, [&called](int, std::string) { called = true; }));
    wait_for_value(std::move(snd), std::string{"hijacked"});
    REQUIRE_FALSE(called);
  }

  struct my_domain2 {
    template <ex::sender_expr_for<ex::bulk_t> Sender, class... Env>
    static auto transform_sender(Sender, const Env&...) {
      return ex::just(std::string{"hijacked"});
    }
  };

  TEST_CASE("bulk can be customized, independently of bulk_chunked", "[adaptors][then]") {
    bool called{false};
    // The customization will return a different value
    basic_inline_scheduler<my_domain2> sched;
    auto snd = ex::just(std::string{"hello"}) | ex::continues_on(sched)
             | ex::bulk(ex::par, 1, [&called](int, std::string) { called = true; });
    wait_for_value(std::move(snd), std::string{"hijacked"});
    REQUIRE_FALSE(called);

    // bulk_chunked will still use the default implementation
    auto snd2 = ex::just(std::string{"hello"}) | ex::continues_on(sched)
              | ex::bulk_chunked(ex::par, 1, [&called](int, int, std::string) { called = true; });
    wait_for_value(std::move(snd2), std::string{"hello"});
    REQUIRE(called);
  }
} // namespace
