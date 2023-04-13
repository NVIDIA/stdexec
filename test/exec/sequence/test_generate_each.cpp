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

#include "exec/sequence/generate_each.hpp"

#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/let_each.hpp"

#include "exec/variant_sender.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

int generator() {
  static int counter = 0;
  return ++counter;
}

using just_t = decltype(just());
using just_stopped_t = decltype(just_stopped());

TEST_CASE("sequence_senders - generate_each", "[sequence_senders][generate_each]") {
  int counter = 0;
  auto take_5 = generate_each(&generator) //
              | let_value_each([&counter](int n) -> variant_sender<just_t, just_stopped_t> {
                  ++counter;
                  if (n == 5) {
                    return just_stopped();
                  }
                  return just();
                }) //
              | ignore_all();
  sync_wait(take_5);
  CHECK(counter == 5);
}
