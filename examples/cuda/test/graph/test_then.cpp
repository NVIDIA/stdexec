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

TEST_CASE("graph then returns a sender", "[graph][adaptors][then]")
{
  graph::scheduler_t scheduler{};
  auto snd = ex::schedule(scheduler) | ex::then([] __device__ {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
}

TEST_CASE("graph then works on GPU", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  std::this_thread::sync_wait(ex::schedule(scheduler) |
                              ex::then([=] __device__ {
                                if (is_on_gpu())
                                {
                                  flags.set();
                                }
                              }));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph then chain works on GPU", "[graph][adaptors][then]")
{
  flags_storage_t<3> flags_storage{};
  auto flags = flags_storage.get();

  graph::scheduler_t scheduler{};

  auto check = [=](int id) {
    return [=] __device__() {
      if (is_on_gpu())
      {
        flags.set(id);
      }
    };
  };

  std::this_thread::sync_wait(ex::schedule(scheduler) | //
                              ex::then(check(0)) |      //
                              ex::then(check(1)) |      //
                              ex::then(check(2)));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph then returns values", "[graph][adaptors][then]")
{
  graph::scheduler_t scheduler{};

  auto [value] =
    std::this_thread::sync_wait(ex::schedule(scheduler) |
                                ex::then([=] __device__ { return 42; }))
      .value();

  REQUIRE(value == 42);
}

TEST_CASE("graph then accepts values", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();
  graph::scheduler_t scheduler{};

  std::this_thread::sync_wait(ex::schedule(scheduler) |
                              ex::then([=] __device__ { return 42; }) |
                              ex::then([=] __device__(int val) {
                                if (is_on_gpu() && val == 42)
                                {
                                  flags.set();
                                }
                              }));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph then accepts multiple values", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();
  graph::scheduler_t scheduler{};

  std::this_thread::sync_wait(ex::just(int{42}, double{2.4}) |
                              ex::transfer(scheduler) |
                              ex::then([=] __device__(int i, double d) {
                                if (is_on_gpu() && i == 42 && d == 2.4)
                                {
                                  flags.set();
                                }
                              }));

  REQUIRE(flags_storage.all_set_once());
}

TEST_CASE("graph then forwards stop signal", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  receiver_tracer_t tracer{};
  graph::scheduler_t scheduler{};

  auto snd = ex::just_stopped()       //
           | ex::transfer(scheduler)  //
           | ex::then([=] __device__ { flags.set(); });

  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(flags_storage.all_unset());
  REQUIRE(tracer.num_nodes() == 0);
  REQUIRE(tracer.num_edges() == 0);
  REQUIRE(tracer.set_stopped_was_called_once());
  REQUIRE_FALSE(tracer.set_value_was_called());
  REQUIRE_FALSE(tracer.set_error_was_called());
}

TEST_CASE("graph then forwards error signal", "[graph][adaptors][then]")
{
  flags_storage_t flags_storage{};
  auto flags = flags_storage.get();

  receiver_tracer_t tracer{};
  graph::scheduler_t scheduler{};

  auto snd = ex::just_error(std::exception_ptr{}) //
           | ex::transfer(scheduler)              //
           | ex::then([=] __device__ { flags.set(); });

  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(flags_storage.all_unset());
  REQUIRE(tracer.num_nodes() == 0);
  REQUIRE(tracer.num_edges() == 0);
  REQUIRE(tracer.set_error_was_called_once());
  REQUIRE_FALSE(tracer.set_value_was_called());
  REQUIRE_FALSE(tracer.set_stopped_was_called());
}

TEST_CASE("graph then constructs a node", "[graph][adaptors][then]")
{
  receiver_tracer_t tracer{};
  graph::scheduler_t scheduler{};

  auto snd = ex::schedule(scheduler) //
           | ex::then([=] __device__ {});

  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(tracer.num_nodes() == 1);
  REQUIRE(tracer.num_edges() == 0);
  REQUIRE(tracer.set_value_was_called_once());
  REQUIRE_FALSE(tracer.set_error_was_called());
  REQUIRE_FALSE(tracer.set_stopped_was_called());
}

TEST_CASE("graph then respects dependencies", "[graph][adaptors][then]")
{
  receiver_tracer_t tracer{};
  graph::scheduler_t scheduler{};

  auto snd = ex::schedule(scheduler) //
           | ex::then([=] __device__ {}) //
           | ex::then([=] __device__ {}) //
           | ex::then([=] __device__ {});

  auto op_state = ex::connect(std::move(snd), tracer.get());
  ex::start(op_state);

  REQUIRE(tracer.num_nodes() == 3);
  REQUIRE(tracer.num_edges() == 2);
  REQUIRE(tracer.set_value_was_called_once());
  REQUIRE_FALSE(tracer.set_error_was_called());
  REQUIRE_FALSE(tracer.set_stopped_was_called());
}
