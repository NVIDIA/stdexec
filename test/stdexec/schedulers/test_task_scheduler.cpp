/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch.hpp>
#include <exec/single_thread_context.hpp>
#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

namespace ex = STDEXEC;

namespace
{
  TEST_CASE("simple task_scheduler test", "[scheduler][task_scheduler]")
  {
    ex::task_scheduler sched{dummy_scheduler{}};
    STATIC_REQUIRE(ex::scheduler<decltype(sched)>);
    auto sndr = sched.schedule();
    STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    auto op = ex::connect(std::move(sndr), expect_value_receiver{});
    ex::start(op);
    // The receiver checks that it's called
  }

  TEST_CASE("task_scheduler starts work on the correct execution context",
            "[scheduler][task_scheduler]")
  {
    exec::single_thread_context ctx;
    ex::task_scheduler          sched{ctx.get_scheduler()};
    auto                        sndr = ex::starts_on(sched,
                              ex::just() | ex::then([] { return ::std::this_thread::get_id(); }));
    auto [tid]                       = ex::sync_wait(std::move(sndr)).value();
    CHECK(tid == ctx.get_thread_id());
  }

  static bool g_called = false;

  template <class Sndr>
  struct opaque_sender : private Sndr
  {
    using sender_concept = ex::sender_t;
    explicit opaque_sender(Sndr sndr)
      : Sndr{std::move(sndr)}
    {}
    using Sndr::connect;
    using Sndr::get_completion_signatures;
    using Sndr::get_env;
  };

  struct test_domain
  {
    template <ex::sender_expr_for<ex::bulk_chunked_t> Sndr, class Env>
    auto transform_sender(ex::set_value_t, Sndr sndr, Env const &) const
    {
      return ex::then(opaque_sender{std::move(sndr)}, []() noexcept { g_called = true; });
    }
  };

  TEST_CASE("bulk_unchunked dispatches correctly through task_scheduler",
            "[scheduler][task_scheduler]")
  {
    ex::task_scheduler sched{dummy_scheduler<test_domain>{}};
    auto               sndr = ex::on(sched,
                       ex::just(-1)
                         | ex::bulk_chunked(ex::par_unseq, 100, [](size_t, size_t, int &) {}));
    g_called                = false;
    auto [val]              = ex::sync_wait(std::move(sndr)).value();
    CHECK(val == -1);
    CHECK(g_called);
  }

  TEST_CASE("bulk dispatches correctly through task_scheduler", "[scheduler][task_scheduler]")
  {
    ex::task_scheduler sched{dummy_scheduler<test_domain>{}};
    auto sndr  = ex::on(sched, ex::just(-1) | ex::bulk(ex::par_unseq, 100, [](size_t, int &) {}));
    g_called   = false;
    auto [val] = ex::sync_wait(std::move(sndr)).value();
    CHECK(val == -1);
    CHECK(g_called);
  }

  TEST_CASE("task_scheduler cannot be constructed from a parallel_scheduler",
            "[scheduler][task_scheduler]")
  {
    STATIC_REQUIRE(!std::is_constructible_v<ex::task_scheduler, ex::parallel_scheduler>);
  }

  TEST_CASE("task_scheduler can deliver a stopped signal", "[scheduler][task_scheduler]")
  {
    ex::inplace_stop_source stop_source;
    stop_source.request_stop();

    exec::single_thread_context ctx;
    ex::task_scheduler          sched{ctx.get_scheduler()};

    auto env    = ex::prop{ex::get_stop_token, stop_source.get_token()};
    auto sndr   = ex::schedule(sched) | ex::write_env(env);
    auto result = ex::sync_wait(std::move(sndr));
    CHECK(!result.has_value());
  }
}  // namespace
