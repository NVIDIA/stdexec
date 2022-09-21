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
#include <async_scope.hpp>

#include "../examples/schedulers/static_thread_pool.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = std::execution;

template<class _Scheduler>
void expect_empty(_Scheduler&, _P2519::execution::async_scope& scope) {
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
  _P2519::execution::async_scope scope;

  ex::scheduler auto sch = ctx.get_scheduler();

  SECTION("after construction") {
    expect_empty(sch, scope);
  }

  SECTION("after spawn") {
    ex::sender auto begin = ex::schedule(sch);
    scope.spawn(begin);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(sch, scope);
  }

  SECTION("after nest result discarded") {
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto nst = scope.nest(begin);
    (void)nst;
    expect_empty(sch, scope);
  }

  SECTION("after nest result started") {
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto nst = scope.nest(begin);
    auto op = ex::connect(std::move(nst), expect_void_receiver{});
    ex::start(op);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(sch, scope);
  }

  SECTION("after spawn_future result discarded") {
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto ftr = scope.spawn_future(begin);
    expect_empty(sch, scope);
  }

  SECTION("after spawn_future result started") {
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto ftr = scope.spawn_future(begin);
    auto op = ex::connect(std::move(ftr), expect_void_receiver{});
    ex::start(op);
    _P2300::this_thread::sync_wait(scope.empty());
    expect_empty(sch, scope);
  }
}

