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

#include "exec/sequence/filter_each.hpp"

#include "exec/sequence/once.hpp"
#include "exec/sequence/then_each.hpp"
#include "exec/sequence/let_each.hpp"
#include "exec/sequence/ignore_all.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

TEST_CASE("sequence_senders - filter_each: fires once", "[sequence_senders][filter_each]") {
  auto always_true = [](auto&&...) {
    return true;
  };
  int count = 0;
  auto filtered = once(just(42))           //
                | filter_each(always_true) //
                | then_each([&](int value) {
                    CHECK(value == 42);
                    count += 1;
                  }) //
                | ignore_all();
  CHECK(count == 0);
  sync_wait(filtered);
  CHECK(count == 1);
}

TEST_CASE("sequence_senders - filter_each: fires none", "[sequence_senders][filter_each]") {
  auto always_false = [](auto&&...) {
    return false;
  };
  int count = 0;
  auto filtered = once(just(42))            //
                | filter_each(always_false) //
                | then_each([&](int value) {
                    count += 1;
                  }) //
                | ignore_all();
  CHECK(count == 0);
  sync_wait(filtered);
  CHECK(count == 0);
}