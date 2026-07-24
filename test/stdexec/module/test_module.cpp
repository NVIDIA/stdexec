/*
 * Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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
#include <catch2/catch_all.hpp>

import std;
import stdexec;

namespace ex = stdexec;

namespace
{
  TEST_CASE("I can utter ex::just_t in a modules build", "[modules]")
  {
    STATIC_REQUIRE(sizeof(ex::just_t) > 0);
  }

  TEST_CASE("I can invoke ex::just() in a modules build", "[modules]")
  {
    auto sender = ex::just();

    STATIC_REQUIRE(sizeof(decltype(sender)) > 0);
  }
}  // namespace
