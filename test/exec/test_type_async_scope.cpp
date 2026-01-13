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

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace {
  void expect_empty(exec::async_scope& scope) {
    ex::run_loop loop;
    ex::scheduler auto sch = loop.get_scheduler();
    CHECK_FALSE(ex::execute_may_block_caller(sch));
    auto op = ex::connect(
      ex::then(
        scope.on_empty(),
        [&]() {
          loop.finish();
    }),
      expect_void_receiver{ex::prop{ex::get_scheduler, sch}});
    ex::start(op);
    loop.run();
  }

  TEST_CASE("async_scope will complete", "[types][type_async_scope]") {
    exec::static_thread_pool ctx{1};

    ex::scheduler auto sch = ctx.get_scheduler();

    SECTION("after construction") {
      exec::async_scope scope;
      expect_empty(scope);
    }

    SECTION("after spawn") {
      exec::async_scope scope;
      ex::sender auto begin = ex::schedule(sch);
      scope.spawn(begin);
      ex::sync_wait(scope.on_empty());
      expect_empty(scope);
    }

    SECTION("after nest result discarded") {
      exec::async_scope scope;
      ex::sender auto begin = ex::schedule(sch);
      {
        ex::sender auto nst = scope.nest(begin);
        (void) nst;
      }
      ex::sync_wait(scope.on_empty());
      expect_empty(scope);
    }

    SECTION("after nest result started") {
      exec::async_scope scope;
      ex::sender auto begin = ex::schedule(sch);
      ex::sender auto nst = scope.nest(begin);
      auto op = ex::connect(std::move(nst), expect_void_receiver{});
      ex::start(op);
      ex::sync_wait(scope.on_empty());
      expect_empty(scope);
    }

    SECTION("after spawn_future result discarded") {
      exec::static_thread_pool ctx{1};
      exec::async_scope scope;
      std::atomic_bool produced{false};
      ex::sender auto begin = ex::schedule(sch);
      {
        ex::sender auto ftr = scope.spawn_future(begin | ex::then([&]() { produced = true; }));
        (void) ftr;
      }
      ex::sync_wait(scope.on_empty() | ex::then([&]() { STDEXEC_ASSERT(produced.load()); }));
      expect_empty(scope);
    }

    SECTION("after spawn_future result started") {
      exec::static_thread_pool ctx{1};
      exec::async_scope scope;
      std::atomic_bool produced{false};
      ex::sender auto begin = ex::schedule(sch);
      ex::sender auto ftr = scope.spawn_future(begin | ex::then([&]() { produced = true; }));
      ex::sync_wait(scope.on_empty() | ex::then([&]() { STDEXEC_ASSERT(produced.load()); }));
      auto op = ex::connect(std::move(ftr), expect_void_receiver{});
      ex::start(op);
      ex::sync_wait(scope.on_empty());
      expect_empty(scope);
    }

    SECTION("request_stop nested spawn_future") {
      exec::static_thread_pool ctx{1};
      exec::async_scope scope;
      ex::sender auto begin = ex::schedule(sch);
      ex::sender auto ftr = scope.spawn_future(scope.spawn_future(begin));
      scope.request_stop();
      ex::sync_wait(ex::when_all(scope.on_empty(), std::move(ftr)));
      // Verify the program finishes without crashing
    }
  }
} // namespace
