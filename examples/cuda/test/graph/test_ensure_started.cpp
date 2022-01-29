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

TEST_CASE("graph ensure_started returns a sender",
          "[graph][adaptors][ensure_started]")
{
  graph::scheduler_t scheduler{};
  auto snd = ex::ensure_started(ex::schedule(scheduler) |
                                ex::then([] __device__ {}));
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
}

TEST_CASE("graph ensure_started starts work",
          "[graph][adaptors][ensure_started]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  ex::ensure_started(ex::schedule(scheduler) | //
                     ex::then([=] __device__ {
                       if (is_on_gpu())
                       {
                         flags.set();
                       }
                     }));
  cudaDeviceSynchronize();

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph ensure_started can be followed by a continuation",
          "[graph][adaptors][ensure_started]")
{
  flags_storage_t<2> flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  std::this_thread::sync_wait(ex::ensure_started(ex::schedule(scheduler) | //
                                                 ex::then([=] __device__ {
                                                   if (is_on_gpu())
                                                   {
                                                     flags.set(0);
                                                   }
                                                 })) | //
                              ex::then([=] __device__ {
                                if (is_on_gpu())
                                {
                                  flags.set(1);
                                }
                              }));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph ensure_started forwards values",
          "[graph][adaptors][ensure_started]")
{
  graph::scheduler_t scheduler{};

  auto snd = ex::ensure_started(ex::schedule(scheduler) |                  //
                                ex::then([=] __device__ { return 42; })) | //
             ex::then([=] __device__(int val) { return val; });

  auto [result] = std::this_thread::sync_wait(std::move(snd)).value();
  REQUIRE(result == 42);
}

TEST_CASE("graph ensure_started forwards errors",
          "[graph][adaptors][ensure_started]")
{
  graph::scheduler_t scheduler{};

  ex::sender auto snd =
    ex::ensure_started(ex::just_error(std::exception_ptr{}) | //
                       ex::transfer(scheduler) |              //
                       ex::then([=] __device__ { })) |
    ex::then([] __device__ {});

  receiver_tracer_t tracer{};
  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(tracer.num_nodes() == 0);
  REQUIRE(tracer.num_edges() == 0);
  REQUIRE(tracer.set_error_was_called_once());
  REQUIRE_FALSE(tracer.set_value_was_called());
  REQUIRE_FALSE(tracer.set_stopped_was_called());
}

TEST_CASE("graph ensure_started forwards stop signal",
          "[graph][adaptors][ensure_started]")
{
  graph::scheduler_t scheduler{};

  ex::sender auto snd = ex::ensure_started(ex::just_stopped() |      //
                                           ex::transfer(scheduler) | //
                                           ex::then([=] __device__ {})) |
                        ex::then([] __device__ {});

  receiver_tracer_t tracer{};
  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(tracer.num_nodes() == 0);
  REQUIRE(tracer.num_edges() == 0);
  REQUIRE(tracer.set_stopped_was_called_once());
  REQUIRE_FALSE(tracer.set_value_was_called());
  REQUIRE_FALSE(tracer.set_error_was_called());
}
