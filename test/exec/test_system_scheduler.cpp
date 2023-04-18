/*
 * Copyright (c) 2023 Lee Howes
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

#include <thread>
#include <iostream>
#include <chrono>

#include <catch2/catch.hpp>
#include <exec/system_scheduler.hpp>
#include <stdexec/execution.hpp>

#include <exec/inline_scheduler.hpp>
#include <test_common/receivers.hpp>


namespace ex = stdexec;


TEST_CASE("trivial schedule task on system context", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  ex::sync_wait(ex::schedule(sched));
}

TEST_CASE("simple schedule task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched),
    [&] {
      pool_id = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(snd));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id!=pool_id);
  (void) snd;
}

TEST_CASE("simple schedule forward progress guarantee", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  REQUIRE(ex::get_forward_progress_guarantee(sched) == ex::forward_progress_guarantee::parallel);
}

TEST_CASE("get_completion_scheduler", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  REQUIRE(
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(ex::schedule(sched))) == sched);
  REQUIRE(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(ex::schedule(sched))) == sched);
}

TEST_CASE("simple chain task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  std::thread::id pool_id2{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched),
    [&] {
      pool_id = std::this_thread::get_id();
    });
  auto snd2 = ex::then(std::move(snd),
    [&] {
      pool_id2 = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(snd2));

  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id!=pool_id);
  REQUIRE(pool_id==pool_id2);
  (void) snd;
  (void) snd2;
}

TEST_CASE("simple bulk task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_ids[num_tasks];
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto bulk_snd = ex::bulk(
    ex::schedule(sched),
    num_tasks,
    [&](long id) {
      pool_ids[id] = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(bulk_snd));

  for(size_t i = 0; i < num_tasks; ++i) {
    REQUIRE(pool_ids[i] != std::thread::id{});
    REQUIRE(this_id!=pool_ids[i]);
  }
  (void) bulk_snd;
}


TEST_CASE("simple bulk chaining on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  constexpr size_t num_tasks = 16;
  std::thread::id pool_id{};
  std::thread::id propagated_pool_ids[num_tasks];
  std::thread::id pool_ids[num_tasks];
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();

  auto snd = ex::then(ex::schedule(sched),
    [&] {
      pool_id = std::this_thread::get_id();
      return pool_id;
    });

  auto bulk_snd = ex::bulk(std::move(snd),
    num_tasks,
    [&](long id, std::thread::id propagated_pool_id) {
      propagated_pool_ids[id] = propagated_pool_id;
      pool_ids[id] = std::this_thread::get_id();
    });

  ex::sync_wait(std::move(bulk_snd));


  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id!=pool_id);
  for(size_t i = 0; i < num_tasks; ++i) {
    REQUIRE(pool_ids[i] != std::thread::id{});
    REQUIRE(propagated_pool_ids[i] == pool_id);
    REQUIRE(this_id!=pool_ids[i]);
  }
  (void) snd;
  (void) bulk_snd;
}
