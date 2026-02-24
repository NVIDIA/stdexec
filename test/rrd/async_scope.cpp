/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <stdexec_relacy.hpp>

#include <exec/async_scope.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;
using exec::async_scope;

struct drop_async_scope_future : rl::test_suite<drop_async_scope_future, 1>
{
  static size_t const dynamic_thread_count = 1;

  void thread(unsigned)
  {
    exec::single_thread_context ctx;
    ex::scheduler auto          sch = ctx.get_scheduler();

    exec::async_scope scope;
    std::atomic_bool  produced{false};
    ex::sender auto   begin = ex::schedule(sch);
    {
      ex::sender auto ftr = scope.spawn_future(begin | ex::then([&]() { produced.store(true); }));
      (void) ftr;
    }
    ex::sync_wait(scope.on_empty() | ex::then([&]() { RL_ASSERT(produced.load()); }));
  }
};

struct attach_async_scope_future : rl::test_suite<attach_async_scope_future, 1>
{
  static size_t const dynamic_thread_count = 1;

  void thread(unsigned)
  {
    exec::single_thread_context ctx;
    ex::scheduler auto          sch = ctx.get_scheduler();

    exec::async_scope scope;
    std::atomic_bool  produced{false};
    ex::sender auto   begin = ex::schedule(sch);
    ex::sender auto   ftr   = scope.spawn_future(begin | ex::then([&]() { produced.store(true); }));
    ex::sender auto   ftr_then = std::move(ftr) | ex::then([&] { RL_ASSERT(produced.load()); });
    ex::sync_wait(ex::when_all(scope.on_empty(), std::move(ftr_then)));
  }
};

struct async_scope_future_set_result : rl::test_suite<async_scope_future_set_result, 1>
{
  static size_t const dynamic_thread_count = 1;

  void thread(unsigned)
  {
    struct throwing_copy
    {
      throwing_copy() = default;

      throwing_copy(throwing_copy const &)
      {
        throw std::logic_error("");
      }
    };

    exec::single_thread_context ctx;
    ex::scheduler auto          sch = ctx.get_scheduler();

    exec::async_scope scope;
    ex::sender auto   begin = ex::schedule(sch);
    ex::sender auto   ftr   = scope.spawn_future(begin | ex::then([] { return throwing_copy(); }));
    bool              threw = false;
    STDEXEC_TRY
    {
      ex::sync_wait(std::move(ftr));
      RL_ASSERT(false);
    }
    STDEXEC_CATCH(std::logic_error const &)
    {
      threw = true;
    }
    STDEXEC_CATCH_ALL
    {
      RL_ASSERT(false);
    }
    RL_ASSERT(threw);
    ex::sync_wait(scope.on_empty());
  }
};

template <int test_case>
struct async_scope_request_stop : rl::test_suite<async_scope_request_stop<test_case>, 1>
{
  static size_t const dynamic_thread_count = 1;

  void thread(unsigned)
  {
    exec::single_thread_context ctx;
    ex::scheduler auto          sch = ctx.get_scheduler();

    if constexpr (test_case == 0)
    {
      exec::async_scope scope;
      ex::sender auto   begin = ex::schedule(sch);
      ex::sender auto   ftr   = scope.spawn_future(scope.spawn_future(begin));
      scope.request_stop();
      ex::sync_wait(ex::when_all(scope.on_empty(), std::move(ftr)));
    }
    else
    {
      exec::async_scope scope;
      ex::sender auto   begin = ex::schedule(sch);
      {
        // Drop the future on the floor
        ex::sender auto ftr = scope.spawn_future(scope.spawn_future(begin));
      }
      scope.request_stop();
      ex::sync_wait(scope.on_empty());
    }
  }
};

auto main() -> int
{
  rl::test_params p;
  p.iteration_count       = 100000;
  p.execution_depth_limit = 10000;
  rl::simulate<drop_async_scope_future>(p);
  rl::simulate<attach_async_scope_future>(p);
  rl::simulate<async_scope_future_set_result>(p);
  rl::simulate<async_scope_request_stop<0>>(p);
  rl::simulate<async_scope_request_stop<1>>(p);
  return 0;
}
