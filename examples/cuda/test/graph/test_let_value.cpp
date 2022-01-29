/*
 * Copyright (c) NVIDIA
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
#include <catch2/catch.hpp>
#include <execution.hpp>

#include <schedulers/graph_scheduler.hpp>

#include <test_helpers.hpp>

namespace ex = std::execution;
namespace graph = example::cuda::graph;

TEST_CASE("graph let_value returns a sender", "[graph][adaptors][let_value]")
{
  graph::scheduler_t scheduler{};
  auto snd = ex::schedule(scheduler) | ex::let_value([&] {
               return ex::schedule(scheduler) | ex::then([] __device__ {});
             });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
}

TEST_CASE("graph let_value forwards values", "[graph][adaptors][let_value]")
{
  graph::scheduler_t scheduler{};

  SECTION("to graph sender")
  {
    auto snd = ex::schedule(scheduler) |                //
               ex::then([] __device__ { return 42; }) | //
               ex::let_value([&](int val) {
                 return ex::schedule(scheduler) | //
                        ex::then([=] __device__ {
                          return val / 2; });
               });

    auto [result] = std::this_thread::sync_wait(std::move(snd)).value();
    REQUIRE(result == 21);
  }

  SECTION("to any sender")
  {
    auto snd = ex::schedule(scheduler) |                //
               ex::then([] __device__ { return 42; }) | //
               ex::let_value([&](int val) {
                 return ex::just(val / 2);
               });

    auto [result] = std::this_thread::sync_wait(std::move(snd)).value();
    REQUIRE(result == 21);
  }
}
