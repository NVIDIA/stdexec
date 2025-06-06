/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "cpo_helpers.cuh"
#include <catch2/catch.hpp>

namespace {

  TEST_CASE("split is customizable", "[cpo][cpo_split]") {
    SECTION("by free standing sender") {
      cpo_test_sender_t<ex::split_t> snd{};

      {
        constexpr scope_t scope = decltype(ex::split(snd))::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }

      {
        constexpr scope_t scope = decltype(snd | ex::split())::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }
    }
  }
} // namespace
