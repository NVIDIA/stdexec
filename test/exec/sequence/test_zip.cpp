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

#include "exec/sequence/zip.hpp"
#include "exec/variant_sender.hpp"

#include "exec/sequence/enumerate_each.hpp"
#include "exec/sequence/let_each.hpp"
#include "exec/sequence/ignore_all.hpp"
#include "exec/sequence/repeat.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

using just_t = decltype(just());
using just_stopped_t = decltype(just_stopped());

TEST_CASE("sequence_senders - zip", "[sequence_senders][zip]") {
  // using env_t = __debug_env_t<empty_env>;
  // __types<__zip::__completions_t<env_t, repeat_t>>{};
  int called = 0;
  auto zip = exec::zip(repeat(just(0)), enumerate_each(repeat(just()))) //
           | let_value_each([&](int zero, int counter) -> variant_sender<just_t, just_stopped_t> {
                 CHECK(zero == 0);
                 CHECK(counter == called);
                 if (counter < 5) {
                   called += 1;
                   return just();
                 }
                 return just_stopped();
               });
  sync_wait(ignore_all(zip));
  CHECK(called == 5);
}