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

#include "exec/sequence/take_while.hpp"

#include "exec/sequence/enumerate_each.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/repeat.hpp"
#include "exec/sequence/then_each.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

TEST_CASE("sequence_senders - take_while stops a repeat", "[sequence_senders][take_while]") {
  int counter = 0;
  auto take_5 = repeat(just())   //
              | enumerate_each(1) //
              | then_each([&counter](int n) {
                  ++counter;
                  return n;
                })                                      //
              | take_while([](int n) { return n < 5; }) //
              | ignore_all();
  sync_wait(take_5);
  CHECK(counter == 5);
}
