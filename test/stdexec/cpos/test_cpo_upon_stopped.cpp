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

TEST_CASE("upon_stopped is customizable", "[cpo][cpo_upon_stopped]") {
  const auto f = []() {};

  SECTION("by free standing sender") {
    free_standing_sender_t<ex::upon_stopped_t> snd{};

    {
      constexpr scope_t scope = decltype(snd | ex::upon_stopped(f))::scope;
      STATIC_REQUIRE(scope == scope_t::free_standing);
    }

    {
      constexpr scope_t scope = decltype(ex::upon_stopped(snd, f))::scope;
      STATIC_REQUIRE(scope == scope_t::free_standing);
    }
  }

  SECTION("by completion scheduler") {
    scheduler_t<ex::upon_stopped_t, ex::set_stopped_t>::sender_t snd{};

    {
      constexpr scope_t scope = decltype(snd | ex::upon_stopped(f))::scope;
      STATIC_REQUIRE(scope == scope_t::scheduler);
    }

    {
      ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(snd));
      constexpr scope_t scope = decltype(ex::upon_stopped(snd, f))::scope;
      STATIC_REQUIRE(scope == scope_t::scheduler);
    }
  }
}
