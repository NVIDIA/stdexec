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

#include <stdexec/execution.hpp>
#include <exec/async_scope.hpp>

#include "exec/env.hpp"
#include "exec/static_thread_pool.hpp"

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

namespace {
void expect_empty(exec::async_scope& context) {
  ex::run_loop loop;
  ex::scheduler auto sch = loop.get_scheduler();
  CHECK_FALSE(stdexec::execute_may_block_caller(sch));
  auto op = ex::connect(
    ex::then(context.on_empty(), [&](){ loop.finish(); }),
    expect_void_receiver{exec::make_env(exec::with(ex::get_scheduler, sch))});
  ex::start(op);
  loop.run();
}
}

TEST_CASE("async_scope will complete", "[types][type_async_scope]") {
  exec::static_thread_pool ctx{1};


  ex::scheduler auto sch = ctx.get_scheduler();

  SECTION("after construction") {
    exec::async_scope context;
    expect_empty(context);
  }

  SECTION("after spawn") {
    exec::async_scope context;
    auto scope = context.get_nester();
    ex::sender auto begin = ex::schedule(sch);
    exec::spawn(scope, begin);
    stdexec::sync_wait(context.on_empty());
    expect_empty(context);
  }

  SECTION("after nest result discarded") {
    exec::async_scope context;
    auto scope = context.get_nester();
    ex::sender auto begin = ex::schedule(sch);
    {ex::sender auto nst = exec::nest(scope, begin); (void)nst;}
    stdexec::sync_wait(context.on_empty());
    expect_empty(context);
  }

  SECTION("after nest result started") {
    exec::async_scope context;
    auto scope = context.get_nester();
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto nst = exec::nest(scope, begin);
    auto op = ex::connect(std::move(nst), expect_void_receiver{});
    ex::start(op);
    stdexec::sync_wait(context.on_empty());
    expect_empty(context);
  }

  SECTION("after spawn_future result discarded") {
    exec::static_thread_pool ctx{1};
    exec::async_scope context;
    auto scope = context.get_nester();
    std::atomic_bool produced{false};
    ex::sender auto begin = ex::schedule(sch);
    {ex::sender auto ftr = exec::spawn_future(scope, begin | stdexec::then([&](){produced = true;})); (void)ftr;}
    stdexec::sync_wait(context.on_empty() | stdexec::then([&](){STDEXEC_ASSERT(produced.load());}));
    expect_empty(context);
  }

  SECTION("after spawn_future result started") {
    exec::static_thread_pool ctx{1};
    exec::async_scope context;
    auto scope = context.get_nester();
    std::atomic_bool produced{false};
    ex::sender auto begin = ex::schedule(sch);
    ex::sender auto ftr = exec::spawn_future(scope, begin | stdexec::then([&](){produced = true;}));
    stdexec::sync_wait(context.on_empty() | stdexec::then([&](){STDEXEC_ASSERT(produced.load());}));
    auto op = ex::connect(std::move(ftr), expect_void_receiver{});
    ex::start(op);
    stdexec::sync_wait(context.on_empty());
    expect_empty(context);
  }
}

TEST_CASE("enter_async_scope will complete", "[types][type_async_scope]") {
  exec::static_thread_pool ctx{2};

  ex::scheduler auto sch = ctx.get_scheduler();

  SECTION("after nest result started") {
    ex::sender auto scoped = exec::enter_async_scope([sch](auto scope) {
      ex::sender auto delay = ex::schedule(sch)
      | ex::then([](){std::this_thread::sleep_for(std::chrono::milliseconds(200));});
      for (int i = 5; i >= 0; --i) {
        exec::spawn(scope, delay);
      }

      ex::sender auto answer = ex::schedule(sch)
      | ex::then([](){return 42;});
      return ex::when_all(
        exec::nest(scope, answer),
        exec::nest(scope, answer),
        exec::nest(scope, answer));

    })
    | ex::then([](int a, int b, int c){ return a-b+c; });
    auto res = stdexec::sync_wait(scoped);
    REQUIRE(res.has_value());
    auto [v] = res.value();
    CHECK(v == 42);
  }
}

