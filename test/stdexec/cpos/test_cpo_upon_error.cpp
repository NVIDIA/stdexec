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

  TEST_CASE("upon_error is customizable", "[cpo][cpo_upon_error]") {
    const auto f = [](std::exception_ptr) {
    };

    SECTION("by sender domain") {
      cpo_test_sender_t<ex::upon_error_t> snd{};

      {
        constexpr scope_t scope = decltype(snd | ex::upon_error(f))::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }

      {
        constexpr scope_t scope = decltype(ex::upon_error(snd, f))::scope;
        STATIC_REQUIRE(scope == scope_t::free_standing);
      }
    }

    SECTION("by completion scheduler domain") {
      cpo_test_scheduler_t<ex::upon_error_t, ex::set_error_t>::sender_t snd{};

      {
        constexpr scope_t scope = decltype(snd | ex::upon_error(f))::scope;
        STATIC_REQUIRE(scope == scope_t::scheduler);
      }

      {
        ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(snd));
        constexpr scope_t scope = decltype(ex::upon_error(snd, f))::scope;
        STATIC_REQUIRE(scope == scope_t::scheduler);
      }
    }
  }
} // namespace
