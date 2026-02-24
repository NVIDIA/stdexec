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

#include <stdexec/coroutine.hpp>

#if !STDEXEC_NO_STDCPP_COROUTINES()

#  include <stdexec/execution.hpp>

#  include <exec/single_thread_context.hpp>
#  include <stdexec/__detail/__task.hpp>

#  include <atomic>

#  include <catch2/catch.hpp>

#  include <test_common/allocators.hpp>
#  include <test_common/senders.hpp>

namespace ex = STDEXEC;

STDEXEC_PRAGMA_IGNORE_GNU("-Wmismatched-new-delete")

namespace
{
  constinit std::atomic<int> g_thread_id = 0;
  thread_local int const     thread_id   = g_thread_id++;

  // This is a work-around for apple clang bugs in Release mode
  [[maybe_unused]] STDEXEC_PP_WHEN(STDEXEC_APPLE_CLANG(), [[clang::optnone]]) auto get_id() -> int
  {
    return thread_id;
  }

  TEST_CASE("task is a sender", "[types][task]")
  {
    STATIC_REQUIRE(ex::sender<ex::task<void>>);
  }

  auto test_task_void() -> ex::task<void>
  {
    CHECK(get_id() == 0);
    co_await ex::schedule(ex::inline_scheduler{});
    CHECK(get_id() == 0);
  }

  TEST_CASE("test task<void>", "[types][task]")
  {
    auto t = test_task_void();
    ex::sync_wait(std::move(t));
  }

  auto test_task_int() -> ex::task<int>
  {
    CHECK(get_id() == 0);
    co_await ex::schedule(ex::inline_scheduler{});
    CHECK(get_id() == 0);
    co_return 42;
  }

  TEST_CASE("test task<int>", "[types][task]")
  {
    auto t   = test_task_int();
    auto [i] = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  auto test_task_int_ref(int& i) -> ex::task<int&>
  {
    CHECK(get_id() == 0);
    co_await ex::schedule(ex::inline_scheduler{});
    CHECK(get_id() == 0);
    co_return i;
  }

  TEST_CASE("test task<int&>", "[types][task]")
  {
    int  value = 42;
    auto t     = test_task_int_ref(value) | ex::then([](int& i) { return std::ref(i); });
    auto [i]   = ex::sync_wait(std::move(t)).value();
    STATIC_REQUIRE(std::same_as<decltype(i), std::reference_wrapper<int>>);
    CHECK(&i.get() == &value);
  }

  struct test_env : ex::env<>
  {
    using allocator_type = test_allocator<std::byte>;
  };

  template <class Alloc>
  auto test_task_allocator(std::allocator_arg_t, [[maybe_unused]] Alloc alloc)
    -> ex::task<void, test_env>
  {
    auto alloc2 = co_await ex::read_env(ex::get_allocator);
    STATIC_REQUIRE(std::same_as<decltype(alloc2), test_allocator<std::byte>>);
    CHECK(alloc == alloc2);
    co_return;
  }

  TEST_CASE("test task with allocator", "[types][task]")
  {
    size_t   bytes = 0;
    ex::prop env{ex::get_allocator, test_allocator<std::byte>{&bytes}};
    auto     t = test_task_allocator(std::allocator_arg, ex::get_allocator(env));
    CHECK(bytes > 0);
    ex::sync_wait(std::move(t) | ex::write_env(env));
    CHECK(bytes == 0);
  }

  auto test_task_awaits_just_sender() -> ex::task<int>
  {
    co_return co_await ex::just(42);
  }

  TEST_CASE("test task can await a just sender", "[types][task]")
  {
    auto t   = test_task_awaits_just_sender();
    auto [i] = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  auto test_task_awaits_just_error_sender() -> ex::task<int>
  {
    co_await ex::just_error(std::runtime_error("error"));
    co_return 42;
  }

  TEST_CASE("test task can await a just error sender", "[types][task]")
  {
    auto t = test_task_awaits_just_error_sender();
    REQUIRE_THROWS_AS(ex::sync_wait(std::move(t)), std::runtime_error);
  }

  auto test_task_awaits_just_stopped_sender() -> ex::task<int>
  {
    co_await ex::just_stopped();
    FAIL("Expected co_awaiting just_stopped to stop the task");
    co_return 42;
  }

  TEST_CASE("test task can await a just stopped sender", "[types][task]")
  {
    auto t   = test_task_awaits_just_stopped_sender();
    auto res = ex::sync_wait(std::move(t));
    CHECK(!res.has_value());
  }

  // A sender type that does not claim to complete inline:
  struct just_int : ex::__result_of<ex::just, int>
  {
    explicit just_int(int i)
      : ex::__result_of<ex::just, int>(ex::just(i))
    {}

    [[nodiscard]]
    auto get_env() const noexcept
    {
      return ex::env{};
    }
  };

  template <ex::scheduler Worker>
  auto test_task_awaits_task_scheduler(Worker worker) -> ex::task<int>
  {
    CHECK(get_id() == 0);
    int i = co_await ex::starts_on(worker,
                                   just_int(42)
                                     | ex::then(
                                       [](int i)
                                       {
                                         CHECK(get_id() != 0);
                                         return i;
                                       }));
    CHECK(get_id() == 0);
    co_return i;
  }

  TEST_CASE("test task can await a just_int sender with affinity to task_scheduler",
            "[types][task]")
  {
    exec::single_thread_context ctx;
    auto                        t = test_task_awaits_task_scheduler(ctx.get_scheduler());
    auto [i]                      = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  // Test affinity with a run_loop scheduler, which is infallible but not inline:
  struct test_env2
  {
    using scheduler_type = ex::run_loop::scheduler;
    struct environment_type
    {};

    template <ex::__not_same_as<environment_type> _Env>
      requires ex::__callable<ex::get_scheduler_t, _Env const &>
    explicit test_env2(_Env const & other) noexcept
      : sch(ex::get_scheduler(other))
    {}

    [[nodiscard]]
    auto query(ex::get_scheduler_t) const noexcept
    {
      return sch;
    }

    ex::run_loop::scheduler sch;
  };

  template <ex::scheduler Worker>
  auto test_task_awaits_run_loop_scheduler(Worker worker) -> ex::task<int, test_env2>
  {
    CHECK(get_id() == 0);
    int i = co_await ex::starts_on(worker,
                                   just_int(42)
                                     | ex::then(
                                       [](int i)
                                       {
                                         CHECK(get_id() != 0);
                                         return i;
                                       }));
    CHECK(get_id() == 0);
    co_return i;
  }

  TEST_CASE("test task can await a just_int sender with affinity to run_loop", "[types][task]")
  {
    exec::single_thread_context ctx;
    auto                        t = test_task_awaits_run_loop_scheduler(ctx.get_scheduler());
    auto [i]                      = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  // FUTURE TODO: add support so that `co_await sndr` can return a reference.
  // auto test_task_awaits_just_ref_sender() -> ex::task<void> {
  //   int value = 42;
  //   decltype(auto) value_ref = co_await just_ref(value);
  //   STATIC_REQUIRE(std::same_as<decltype(value_ref), int&>);
  //   CHECK(&value_ref == &value);
  //   co_return;
  // }

  // TEST_CASE("test task can await a just_ref sender", "[types][task]") {
  //   auto t = test_task_awaits_just_ref_sender();
  //   ex::sync_wait(std::move(t));
  // }

  // TODO: add tests for stop token support in task

}  // anonymous namespace

#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
