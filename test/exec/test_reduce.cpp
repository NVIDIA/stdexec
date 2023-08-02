/*
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

#include <catch2/catch.hpp>
#include <exec/reduce.hpp>
#include <exec/inline_scheduler.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/static_thread_pool.hpp>

#include <algorithm>
#include <functional>
#include <span>
#include <numeric>
#include <vector>

TEST_CASE("exec reduce returns a sender with single input", "[adaptors][reduce]") {
  constexpr int N = 2048;
  int input[N] = {};
  std::fill_n(input, N, 1);

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(0);

  STATIC_REQUIRE(stdexec::sender_of<decltype(task), stdexec::set_value_t(int)>);

  (void) task;
}

TEST_CASE("exec reduce returns a sender with two inputs", "[adaptors][reduce]") {
  constexpr int N = 2048;
  int input[N] = {};
  std::fill_n(input, N, 1);

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input})
            | exec::reduce(0, std::minus<>{});

  STATIC_REQUIRE(stdexec::sender_of<decltype(task), stdexec::set_value_t(int)>);

  (void) task;
}

TEST_CASE("exec reduce returns init value when value range is empty", "[adaptors][reduce]") {
  constexpr int input[1]{};
  constexpr int init = 47;
  std::span<const int> range{input, 0};
  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), range) | exec::reduce(47);
  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == init);
}

TEST_CASE("exec reduce yields correct result for single value in range", "[adaptors][reduce]") {
  constexpr int single = 37;
  constexpr int init = 47;

  std::vector<int> input{single};

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == init + single);
}

TEST_CASE("exec reduce uses sum as default operation", "[adaptors][reduce]") {
  constexpr int N = 2048;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}

TEST_CASE("exec reduce uses the passed reduction operation", "[adaptors][reduce]") {
  constexpr int N = 2048;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 2);

  auto minFun = [](auto acc, auto value) {
    return std::min(acc, value);
  };

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input})
            | exec::reduce(init, minFun);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == 2);
}

TEST_CASE("exec reduce yields correct result with product as operation", "[adaptors][reduce]") {
  constexpr int N = 15;
  constexpr int init = 3;

  std::vector<int> input(N, 2);

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input})
            | exec::reduce(init, std::multiplies<>{});

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == 3 * std::pow(2, 15));
}

TEST_CASE("exec reduce works with custom value type", "[adaptors][reduce]") {
  constexpr int N = 2048;
  constexpr int init = 47;

  struct value_t {
    int x;
  };

  auto sum = [](value_t acc, value_t val) {
    return value_t{acc.x + val.x};
  };

  std::vector<value_t> input(N, value_t{1});

  exec::static_thread_pool pool{};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input})
            | exec::reduce(value_t{init}, sum);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  STATIC_REQUIRE(stdexec::sender_of<decltype(task), stdexec::set_value_t(value_t)>);

  REQUIRE(result.x == N + init);
}

TEST_CASE("exec reduce yields correct result if thread pool has 1 thread", "[adaptors][reduce]") {
  constexpr int N = 128;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::static_thread_pool pool{1};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}

TEST_CASE(
  "exec reduce correct if thread pool has more threads than values in range",
  "[adaptors][reduce]") {
  constexpr int N = 4;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::static_thread_pool pool{N + 2};
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}

TEST_CASE(
  "exec reduce correct if reduction threads are not balanced [adaptors][reduce]") {
  constexpr int N = 128;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::static_thread_pool pool{3};  // N mod 3 = 2
  auto task = stdexec::transfer_just(pool.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}

TEST_CASE("exec reduce runs on single_thread_context scheduler", "[adaptors][reduce]") {
  constexpr int N = 128;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::single_thread_context single{};
  auto task = stdexec::transfer_just(single.get_scheduler(), std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}

TEST_CASE("exec reduce runs on inline_scheduler", "[adaptors][reduce]") {
  constexpr int N = 128;
  constexpr int init = 47;

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 1);

  exec::inline_scheduler sched;
  auto task = stdexec::transfer_just(sched, std::span{input}) | exec::reduce(init);

  auto [result] = stdexec::sync_wait(std::move(task)).value();

  REQUIRE(result == (N * (N + 1) / 2) + init);
}
