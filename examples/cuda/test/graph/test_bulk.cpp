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

TEST_CASE("graph bulk returns a sender", "[graph][adaptors][then]")
{
  graph::scheduler_t scheduler{};
  auto snd = ex::schedule(scheduler) | ex::bulk(42, [] __device__ (int) {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
}

TEST_CASE("graph bulk works on GPU", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  const int n = 42;
  std::this_thread::sync_wait(ex::schedule(scheduler) |
                              ex::bulk(n, [=] __device__(int) {
                                if (is_on_gpu())
                                {
                                  flags.set();
                                }
                              }));

  REQUIRE(flags_storage.is_set_n_times(n));
}

TEST_CASE("graph bulk chain works on GPU", "[graph][adaptors][then]")
{
  flags_storage_t<3> flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  auto check = [=](int id) {
    return [=] __device__(int) {
      if (is_on_gpu())
      {
        flags.set(id);
      }
    };
  };

  const int n = 42;
  std::this_thread::sync_wait(ex::schedule(scheduler) |    //
                              ex::bulk(n, check(0)) |      //
                              ex::bulk(n, check(1)) |      //
                              ex::bulk(n, check(2)));

  REQUIRE(flags_storage.is_set_n_times(n));
}

TEST_CASE("graph bulk returns values", "[graph][adaptors][then]")
{
  graph::scheduler_t scheduler{};

  const int n = 42;

  auto [value] =
    std::this_thread::sync_wait(ex::schedule(scheduler) |
                                ex::then([=] __device__ { return 42; }) |
                                ex::bulk(n, [=] __device__ (int, int val) {

                                }))
      .value();

  REQUIRE(value == 42);
}
