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

#include <catch2/catch.hpp>

#include <execution.hpp>
#include <async_scope.hpp>

#include "../examples/schedulers/static_thread_pool.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = std::execution;

void expect_empty(_P2519::execution::async_scope& scope) {
  ex::run_loop loop;
  ex::scheduler auto sch = loop.get_scheduler();
  CHECK_FALSE(std::this_thread::execute_may_block_caller(sch));
  auto op = ex::connect(
    ex::then(scope.empty(), [&](){  loop.finish(); }),
    expect_void_receiver{ex::make_env(ex::with(ex::get_scheduler, sch))});
  ex::start(op);
  loop.run();
}

TEST_CASE("async_scope will complete", "[types][type_async_scope]") {
  example::static_thread_pool ctx{1};


  ex::scheduler auto sch = ctx.get_scheduler();

  SECTION("after construction") {
    _P2519::execution::async_scope scope;
    expect_empty(scope);
  }

  SECTION("after spawn") {
    _P2519::execution::async_scope scope;
    ex::sender auto begin = ex::schedule(sch);
    scope.spawn(begin);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(scope);
  }

  SECTION("after nest result discarded") {
    _P2519::execution::async_scope scope;
    ex::sender auto begin = ex::schedule(sch);
    {ex::sender auto nst = scope.nest(begin); (void)nst;}
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(scope);
  }

  SECTION("after nest result started") {
    _P2519::execution::async_scope scope;
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto nst = scope.nest(begin);
    auto op = ex::connect(std::move(nst), expect_void_receiver{});
    ex::start(op);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(scope);
  }

  SECTION("after spawn_future result discarded") {
    example::static_thread_pool ctx{1};
    _P2519::execution::async_scope scope;
    std::atomic_bool produced{false};
    ex::sender auto begin = ex::schedule(sch);
    {ex::sender auto ftr = scope.spawn_future(begin | _P2300::execution::then([&](){produced = true;})); (void)ftr;}
    _P2300::this_thread::sync_wait(scope.empty() | _P2300::execution::then([&](){if(!produced.load()){std::terminate();}}));
    expect_empty(scope);
  }
  
  SECTION("after spawn_future result started") {
    example::static_thread_pool ctx{1};
    _P2519::execution::async_scope scope;
    std::atomic_bool produced{false};
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto ftr = scope.spawn_future(begin | _P2300::execution::then([&](){produced = true;}));
    _P2300::this_thread::sync_wait(scope.empty() | _P2300::execution::then([&](){if(!produced.load()){std::terminate();}}));
    auto op = ex::connect(std::move(ftr), expect_void_receiver{});
    ex::start(op);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(scope);
  }
}

