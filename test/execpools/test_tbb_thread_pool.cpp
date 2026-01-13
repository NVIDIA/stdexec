/*
 * Copyright (c) 2023 Lucian Radu Teodorescu, Ben FrantzDale
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

#include <span>

#include <stdexec/execution.hpp>

#include <exec/inline_scheduler.hpp>
#include <test_common/schedulers.hpp>

#include <execpools/tbb/tbb_thread_pool.hpp>

namespace ex = STDEXEC;

namespace {
  // Example adapted from
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html#example-async-inclusive-scan
  [[nodiscard]]
  auto async_inclusive_scan(
    ex::scheduler auto sch,                    // 2
    std::span<const double> input,             // 1
    std::span<double> output,                  // 1
    double init,                               // 1
    std::size_t tile_count) -> ex::sender auto // 3
  {
    using namespace STDEXEC;
    std::size_t const tile_size = (input.size() + tile_count - 1) / tile_count;

    std::vector<double> partials(tile_count + 1);
    partials[0] = init;

    return transfer_just(sch, std::move(partials)) //
         | bulk(
             ex::par,
             tile_count,
             [=](std::size_t i, std::span<double> partials) {
               auto start = i * tile_size;
               auto end = (std::min) (input.size(), (i + 1) * tile_size);
               partials[i + 1] = *--std::inclusive_scan(
                 begin(input) + static_cast<long>(start),
                 begin(input) + static_cast<long>(end),
                 begin(output) + static_cast<long>(start));
             }) //
         | then([](std::vector<double>&& partials) {
             std::inclusive_scan(begin(partials), end(partials), begin(partials));
             return std::move(partials);
           }) //
         | bulk(
             ex::par,
             tile_count,
             [=](std::size_t i, std::span<const double> partials) {
               auto start = i * tile_size;
               auto end = (std::min) (input.size(), (i + 1) * tile_size);
               std::for_each(
                 begin(output) + static_cast<long>(start),
                 begin(output) + static_cast<long>(end),
                 [&](double& e) { e = partials[i] + e; });
             }) //
         | then([=](std::vector<double>&&) { return output; });
  }

  TEST_CASE(
    "execpools::tbb_thread_pool offers the parallel forward progress guarantee",
    "[tbb_thread_pool]") {
    execpools::tbb_thread_pool pool;
    auto pool_sched = pool.get_scheduler();
    CHECK(
      ex::get_forward_progress_guarantee(pool_sched) == ex::forward_progress_guarantee::parallel);
  }

  TEST_CASE(
    "ex::on works when changing threads with execpools::tbb_thread_pool",
    "[tbb_thread_pool]") {
    execpools::tbb_thread_pool pool;
    auto pool_sched = pool.get_scheduler();
    bool called{false};
    // launch some work on the thread pool
    ex::sender auto snd = ex::on(pool_sched, ex::just()) | ex::then([&] { called = true; });
    ex::sync_wait(std::move(snd));
    // the work should be executed
    REQUIRE(called);
  }

  TEST_CASE("more tbb_thread_pool", "[tbb_thread_pool]") {
    using namespace STDEXEC;

    execpools::tbb_thread_pool pool(1);
    exec::static_thread_pool other_pool(1);
    STDEXEC::inline_scheduler inline_sched;

    // Get a handle to the thread pool:
    auto other_sched = other_pool.get_scheduler();

    // Get a handle to the thread pool:
    auto tbb_sched = pool.get_scheduler();

    auto compute = [](int x) -> int {
      return x + 1;
    };

    // clang-format off
    auto work = when_all(
      starts_on(tbb_sched, just(1))    | then(compute) | then(compute),
      starts_on(other_sched, just(0))  | then(compute) | continues_on(tbb_sched)   | then(compute),
      starts_on(inline_sched, just(2)) | then(compute) | continues_on(other_sched) | then(compute) | continues_on(tbb_sched) | then(compute)
    );
    // clang-format on

    // Launch the work and wait for the result:
    auto [i, j, k] = ex::sync_wait(std::move(work)).value();
    CHECK(i == 3);
    CHECK(j == 2);
    CHECK(k == 5);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("tbb_thread_pool exceptions", "[tbb_thread_pool]") {
    // I know tbb::task_groups do cancellation with exceptions, which leaves them in a not-restartable
    // state. We'd better have it act normally here.
    using namespace STDEXEC;

    execpools::tbb_thread_pool tbb_pool;
    exec::static_thread_pool other_pool(1);
    {
      CHECK_THROWS(ex::sync_wait(starts_on(tbb_pool.get_scheduler(), just(0)) | then([](auto) {
                                   throw std::exception();
                                 })));
      CHECK_THROWS(ex::sync_wait(starts_on(other_pool.get_scheduler(), just(0)) | then([](auto) {
                                   throw std::exception();
                                 })));
    }
    // Ensure it still works normally after exceptions:
    {
      auto tbb_result = ex::sync_wait(
        starts_on(tbb_pool.get_scheduler(), just(0)) | then([](auto i) { return i + 1; }));
      CHECK(tbb_result.has_value());
      auto other_result = ex::sync_wait(
        starts_on(other_pool.get_scheduler(), just(0)) | then([](auto i) { return i + 1; }));
      CHECK(tbb_result == other_result);
    }
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE("tbb_thread_pool async_inclusive_scan", "[tbb_thread_pool]") {
    const auto input = std::array{1.0, 2.0, -1.0, -2.0};
    std::remove_const_t<decltype(input)> output;
    execpools::tbb_thread_pool pool{2};
    auto [value] = ex::sync_wait(async_inclusive_scan(pool.get_scheduler(), input, output, 0.0, 4))
                     .value();
    STATIC_REQUIRE(std::is_same_v<decltype(value), std::span<double>>);
    REQUIRE(value.data() == output.data());
    CHECK(output == std::array{1.0, 3.0, 2.0, 0.0});
  }
} // namespace
