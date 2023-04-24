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

#include "exec/sequence/merge_each.hpp"

#include "exec/sequence/once.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/then_each.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

TEST_CASE("Merge each", "[sequence_senders][merge_each]") {
  auto seq = once(just(once(just(42))));
  auto merged = merge_each_sequence(seq);
  auto ignored = ignore_all(merged | then_each([](int val) { 
    CHECK(val == 42);
  }));
  // __types<completion_signatures_of_t<__sndr>>{};
  // __debug_sender(ignored);
  sync_wait(ignored);
}