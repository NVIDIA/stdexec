/*
 * Copyright (c) Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

#include <exec/when_all_range.hpp>

#include <catch2/catch.hpp>

namespace {

  TEST_CASE("when_all_range - sum up an array", "[when_all_range]") {
    std::array<int, 3> array{42, 43, 44};
    int sum = 0;
    auto sum_up = std::ranges::views::transform(std::ranges::views::all(array), [&sum](int x) {
      return stdexec::then(stdexec::just(), [&sum, x]() noexcept { sum += x; });
    });
    auto when_all = exec::when_all_range(sum_up);
    stdexec::sync_wait(std::move(when_all));
    CHECK(sum == (42 + 43 + 44));
  }

}