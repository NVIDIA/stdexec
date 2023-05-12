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

#include <test_common/schedulers.hpp>
#include <exec/on.hpp>
#include <exec/inline_scheduler.hpp>

#include <tbbexec/tbb_thread_pool.hpp>

#include <tbb/global_control.h> // needed by TbbTestFixture below

namespace ex = stdexec;

template <ex::scheduler Sched = inline_scheduler>
inline auto _with_scheduler(Sched sched = {}) {
  return exec::write(exec::with(ex::get_scheduler, std::move(sched)));
}

namespace {

  // For use suppressing TSAN false positives
  // See https://stackoverflow.com/questions/72563202/tbbs-private-server-and-false-positive-threadsanitizer-data-races
  struct TbbTestFixture {
    ~TbbTestFixture() {
      // Expected to kill tbb::detail::r1::rml::private_server after each test,
      // which can otherwise trigger false positive tsan data race warnings.
      auto handle = tbb::task_scheduler_handle::get();
      tbb::finalize(handle, std::nothrow_t{});
    }
  };

  // Example adapted from
  // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html#example-async-inclusive-scan
  [[nodiscard]] stdexec::sender auto async_inclusive_scan(
    stdexec::scheduler auto sch,   // 2
    std::span<const double> input, // 1
    std::span<double> output,      // 1
    double init,                   // 1
    std::size_t tile_count)        // 3
  {
    using namespace stdexec;
    std::size_t const tile_size = (input.size() + tile_count - 1) / tile_count;

    std::vector<double> partials(tile_count + 1);
    partials[0] = init;

    // clang-format off
  return transfer_just(sch, std::move(partials))
       | bulk(tile_count,
             [=](std::size_t i, std::span<double> partials) {
               auto start = i * tile_size;
               auto end = std::min(input.size(), (i + 1) * tile_size);
               partials[i + 1] = *--std::inclusive_scan(
                   begin(input) + start, begin(input) + end, begin(output) + start);
             })
       | then([](std::vector<double>&& partials) {
           std::inclusive_scan(begin(partials), end(partials), begin(partials));
           return std::move(partials);
         })
       | bulk(tile_count,
             [=](std::size_t i, std::span<const double> partials) {
               auto start = i * tile_size;
               auto end = std::min(input.size(), (i + 1) * tile_size);
               std::for_each(begin(output) + start, begin(output) + end,
                   [&](double& e) { e = partials[i] + e; });
             })
       | then([=](std::vector<double>&& partials) { return output; });
    // clang-format on
  }

} // namespace

TEST_CASE_METHOD(
  TbbTestFixture,
  "exec::on works when changing threads with tbbexec::tbb_thread_pool",
  "[adaptors][exec::on]") {
  tbbexec::tbb_thread_pool pool;
  auto pool_sched = pool.get_scheduler();
  CHECK(
    stdexec::get_forward_progress_guarantee(pool_sched)
    == stdexec::forward_progress_guarantee::parallel);
  bool called{false};
  // launch some work on the thread pool
  ex::sender auto snd = exec::on(pool_sched, ex::just()) //
                      | ex::then([&] { called = true; }) | _with_scheduler();
  stdexec::sync_wait(std::move(snd));
  // the work should be executed
  REQUIRE(called);
}

TEST_CASE_METHOD(TbbTestFixture, "more tbb_thread_pool") {

  auto compute = [](int x) -> int {
    return x + 1;
  };

  tbbexec::tbb_thread_pool pool(1);

  exec::static_thread_pool other_pool(1);

  // Get a handle to the thread pool:
  auto other_sched = other_pool.get_scheduler();

  // Get a handle to the thread pool:
  auto tbb_sched = pool.get_scheduler();

  exec::inline_scheduler inline_sched;

  using namespace stdexec;

  // clang-format off
  auto work = when_all(
      on(tbb_sched, just(1))    | then(compute) | then(compute),
      on(other_sched, just(0))  | then(compute) | transfer(tbb_sched)   | then(compute),
      on(inline_sched, just(2)) | then(compute) | transfer(other_sched) | then(compute) | transfer(tbb_sched) | then(compute)
  );
  // clang-format on

  // Launch the work and wait for the result:
  auto [i, j, k] = stdexec::sync_wait(std::move(work)).value();
  CHECK(i == 3);
  CHECK(j == 2);
  CHECK(k == 5);
}

TEST_CASE_METHOD(TbbTestFixture, "tbb_thread_pool exceptions") {
  // I know tbb::task_groups do cancellation with exceptions, which leaves them in a not-restartable
  // state. We'd better have it act normally here.
  using namespace stdexec;

  tbbexec::tbb_thread_pool tbb_pool;
  exec::static_thread_pool other_pool(1);
  {
    CHECK_THROWS(stdexec::sync_wait(
      on(tbb_pool.get_scheduler(), just(0)) | then([](auto i) { throw std::exception(); })));
    CHECK_THROWS(stdexec::sync_wait(
      on(other_pool.get_scheduler(), just(0)) | then([](auto i) { throw std::exception(); })));
  }
  // Ensure it still works normally after exceptions:
  {
    auto tbb_result = stdexec::sync_wait(
      on(tbb_pool.get_scheduler(), just(0)) | then([](auto i) { return i + 1; }));
    CHECK(tbb_result.has_value());
    auto other_result = stdexec::sync_wait(
      on(other_pool.get_scheduler(), just(0)) | then([](auto i) { return i + 1; }));
    CHECK(tbb_result == other_result);
  }
}

TEST_CASE_METHOD(TbbTestFixture, "tbb_thread_pool async_inclusive_scan") {
  const auto input = std::array{1.0, 2.0, -1.0, -2.0};
  std::remove_const_t<decltype(input)> output;
  tbbexec::tbb_thread_pool pool{2};
  auto [value] =
    stdexec::sync_wait(async_inclusive_scan(pool.get_scheduler(), input, output, 0.0, 4)).value();
  STATIC_REQUIRE(std::is_same_v<decltype(value), std::span<double>>);
  REQUIRE(value.data() == output.data());
  CHECK(output == std::array{1.0, 3.0, 2.0, 0.0});
}
