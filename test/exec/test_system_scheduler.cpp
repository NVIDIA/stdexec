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

#include <catch2/catch.hpp>
#include <exec/system_scheduler.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

TEST_CASE("simple schedule task on system context", "[types][system_scheduler]") {
  std::thread::id this_id = std::this_thread::get_id();
  std::thread::id pool_id{};
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  auto snd = ex::then(ex::schedule(sched), [&] {pool_id = std::this_thread::get_id();});
  ex::sync_wait(snd);
  REQUIRE(pool_id != std::thread::id{});
  REQUIRE(this_id!=pool_id);
  std::cout << "This: " << this_id << "; pool: " << pool_id << "\n";
  (void) snd;
}

TEST_CASE("simple schedule forward progress guarantee", "[types][system_scheduler]") {
  exec::system_context ctx;
  exec::system_scheduler sched = ctx.get_scheduler();
  REQUIRE(ex::get_forward_progress_guarantee(sched) == ex::forward_progress_guarantee::parallel);
}
