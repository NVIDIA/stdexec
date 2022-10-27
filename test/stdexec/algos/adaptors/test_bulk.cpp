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
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>

#include <vector>

namespace ex = stdexec;

template <class Shape, int N, int (&Counter)[N]>
void function(Shape i) {
  Counter[i]++;
}

template <class Shape>
struct function_object_t {
  int *Counter;

  void operator()(Shape i) {
    Counter[i]++;
  }
};

TEST_CASE("bulk returns a sender", "[adaptors][bulk]") {
  auto snd = ex::bulk(ex::just(19), 8, [](int idx, int val) {});
  static_assert(ex::sender<decltype(snd)>);
  (void)snd;
}

TEST_CASE("bulk with environment returns a sender", "[adaptors][bulk]") {
  auto snd = ex::bulk(ex::just(19), 8, [](int idx, int val) {});
  static_assert(ex::sender<decltype(snd), empty_env>);
  (void)snd;
}

TEST_CASE("bulk can be piped", "[adaptors][bulk]") {
  ex::sender auto snd = ex::just() | ex::bulk(42, [](int i) {});
  (void)snd;
}

TEST_CASE("bulk keeps values_type from input sender", "[adaptors][bulk]") {
  constexpr int n = 42;
  check_val_types<type_array<type_array<>>>(ex::just() | ex::bulk(n, [](int) {}));
  check_val_types<type_array<type_array<double>>>(ex::just(4.2) | ex::bulk(n, [](int, double) {}));
  check_val_types<type_array<type_array<double, std::string>>>(
      ex::just(4.2, std::string{}) | ex::bulk(n, [](int, double, std::string) {}));
}

TEST_CASE("bulk keeps error_types from input sender", "[adaptors][bulk]") {
  constexpr int n = 42;
  inline_scheduler sched1{};
  error_scheduler sched2{};
  error_scheduler<int> sched3{43};

  check_err_types<type_array<>>( //
      ex::transfer_just(sched1) | ex::bulk(n, [](int) noexcept {}));
  check_err_types<type_array<std::exception_ptr>>( //
      ex::transfer_just(sched2) | ex::bulk(n, [](int) noexcept {}));
  check_err_types<type_array<int>>( //
      ex::just_error(n) | ex::bulk(n, [](int) noexcept {}));
  check_err_types<type_array<int>>( //
      ex::transfer_just(sched3) | ex::bulk(n, [](int) noexcept {}));
  check_err_types<type_array<std::exception_ptr, int>>( //
      ex::transfer_just(sched3) | ex::bulk(n, [](int) { throw std::logic_error{"err"}; }));
}

TEST_CASE("bulk can be used with a function", "[adaptors][bulk]") {
  constexpr int n = 9;
  static int counter[n]{};
  std::fill_n(counter, n, 0);

  ex::sender auto snd = ex::just() | ex::bulk(n, function<int, n, counter>);
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++) {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk can be used with a function object", "[adaptors][bulk]") {
  constexpr int n = 9;
  int counter[n]{0};
  function_object_t<int> fn{counter};

  ex::sender auto snd = ex::just() | ex::bulk(n, fn);
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++) {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk can be used with a lambda", "[adaptors][bulk]") {
  constexpr int n = 9;
  int counter[n]{0};

  ex::sender auto snd = ex::just() | ex::bulk(n, [&](int i) { counter[i]++; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++) {
    CHECK(counter[i] == 1);
  }
}

TEST_CASE("bulk forwards values", "[adaptors][bulk]") {
  constexpr int n = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto snd = ex::just(magic_number)
           | ex::bulk(n, [&](int i, int val) {
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

TEST_CASE("bulk cannot be used to change the value type", "[adaptors][bulk]") {
  constexpr int magic_number = 42;
  constexpr int n = 2;

  auto snd = ex::just(magic_number)
           | ex::bulk(n, [](int i, int) {
               return function_object_t<int>{nullptr};
             });

  auto op = ex::connect(std::move(snd), expect_value_receiver{magic_number});
  ex::start(op);
}

TEST_CASE("bulk can throw, and set_error will be called", "[adaptors][bulk]") {
  constexpr int n = 2;

  auto snd = ex::just() //
           | ex::bulk(n, [](int i) -> int {
               throw std::logic_error{"err"};
             });
  auto op = ex::connect(std::move(snd), expect_error_receiver{});
  ex::start(op);
}

TEST_CASE("bulk function is not called on error", "[adaptors][bulk]") {
  constexpr int n = 2;
  int called{};

  auto snd = ex::just_error(std::string{"err"})
           | ex::bulk(n, [&called](int) { called++; });
  auto op = ex::connect(std::move(snd), expect_error_receiver{std::string{"err"}});
  ex::start(op);
}

TEST_CASE("bulk function in not called on stop", "[adaptors][bulk]") {
  constexpr int n = 2;
  int called{};

  auto snd = ex::just_stopped()
           | ex::bulk(n, [&called](int) { called++; });
  auto op = ex::connect(std::move(snd), expect_stopped_receiver{});
  ex::start(op);
}

TEST_CASE("bulk works with static thread pool", "[adaptors][bulk]") {
  exec::static_thread_pool pool{4};
  ex::scheduler auto sch = pool.get_scheduler();

  SECTION("Without values in the set_value channel") {
    for (int n = 0; n < 9; n++) {
      std::vector<int> counter(n, 42);

      auto snd = ex::transfer_just(sch)
               | ex::bulk(n, [&counter](int idx) { counter[idx] = 0; })
               | ex::bulk(n, [&counter](int idx) { counter[idx]++; });
      stdexec::sync_wait(std::move(snd));

      const std::size_t actual = std::count(counter.begin(), counter.end(), 1);
      const std::size_t expected = n;

      CHECK(expected == actual);
    }
  }

  SECTION("With values in the set_value channel") {
    for (int n = 0; n < 9; n++) {
      std::vector<int> counter(n, 42);

      auto snd = ex::transfer_just(sch, 42)
               | ex::bulk(n, [&counter](int idx, int val) { if (val == 42) { counter[idx] = 0; } })
               | ex::bulk(n, [&counter](int idx, int val) { if (val == 42) { counter[idx]++; } });
      auto [val] = stdexec::sync_wait(std::move(snd)).value();

      CHECK(val == 42);

      const std::size_t actual = std::count(counter.begin(), counter.end(), 1);
      const std::size_t expected = n;

      CHECK(expected == actual);
    }
  }

  SECTION("With exception") {
    constexpr int n = 9;
    auto snd = ex::transfer_just(sch)
             | ex::bulk(n, [](int idx) {
                 throw std::runtime_error("bulk");
               });

    CHECK_THROWS_AS(stdexec::sync_wait(std::move(snd)), std::runtime_error);
  }
}

