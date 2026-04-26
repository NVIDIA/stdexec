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

#  include <exec/just_from.hpp>
#  include <exec/single_thread_context.hpp>
#  include <exec/static_thread_pool.hpp>

#  include <atomic>

#  include <catch2/catch_all.hpp>

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

  auto test_task_int_ref(int &i) -> ex::task<int &>
  {
    CHECK(get_id() == 0);
    co_await ex::schedule(ex::inline_scheduler{});
    CHECK(get_id() == 0);
    co_return i;
  }

  TEST_CASE("test task<int&>", "[types][task]")
  {
    int  value = 42;
    auto t     = test_task_int_ref(value) | ex::then([](int &i) { return std::ref(i); });
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
    explicit test_env2(_Env const &other) noexcept
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

  // In debug GCC builds, this test can cause a stack overflow due to
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94794, results in a symmetric
  // transfer failing to be a tail call.
#  if !STDEXEC_GCC()                                                                               \
    || (defined(__OPTIMIZE__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__))
  auto sync() -> ex::task<int>
  {
    co_return 42;
  }

  auto nested() -> ex::task<int>
  {
    auto sched = co_await ex::read_env(ex::get_scheduler);
    static_assert(std::same_as<decltype(sched), ex::task_scheduler>);
    co_await ex::schedule(sched);
    co_return 42;
  }

  auto test_task_awaits_inline_sndr_without_stack_overflow() -> ex::task<int>
  {
    int result = co_await nested();
    for (int i = 0; i < 1'000'000; ++i)
    {
      result += co_await sync();
    }
    for (int i = 0; i < 1'000'000; ++i)
    {
      result += co_await ex::just(42);
    }
    co_return result;
  }

  TEST_CASE("test task can await a just_int sender without stack overflow", "[types][task]")
  {
    auto t   = test_task_awaits_inline_sndr_without_stack_overflow();
    auto [i] = ex::sync_wait(std::move(t)).value();
    CHECK(i == 84'000'042);
  }
#  endif

  struct my_env
  {
    template <class>
    using env_type = my_env;

    template <class Env>
      requires std::invocable<ex::get_delegation_scheduler_t, Env const &>
            && std::same_as<std::invoke_result_t<ex::get_delegation_scheduler_t, Env const &>,
                            ex::run_loop::scheduler>
    explicit my_env(Env const &env) noexcept
      : delegation_scheduler_(ex::get_delegation_scheduler(env))
    {}

    [[nodiscard]]
    auto query(ex::get_delegation_scheduler_t) const noexcept
    {
      return delegation_scheduler_;
    }

    ex::run_loop::scheduler delegation_scheduler_;
  };

  auto
  test_task_provides_additional_queries_with_a_custom_env(ex::run_loop::scheduler sync_wt_dlgtn_sch)
    -> ex::task<int, my_env>
  {
    // Fetch sync_wait's run_loop scheduler from the environment.
    ex::run_loop::scheduler tsk_dlgtn_sch = co_await ex::read_env(ex::get_delegation_scheduler);
    CHECK(tsk_dlgtn_sch == sync_wt_dlgtn_sch);
    co_return 13;
  }

  TEST_CASE("task can provide additional queries through a custom environment", "[types][task]")
  {
    ex::sync_wait(ex::let_value(ex::read_env(ex::get_delegation_scheduler),
                                [](ex::run_loop::scheduler sync_wt_dlgtn_sch)
                                {
                                  return test_task_provides_additional_queries_with_a_custom_env(
                                    sync_wt_dlgtn_sch);
                                }));
  }

  // FUTURE TODO: add support so that `co_await sndr` can return a reference.

  constinit int global_int = 0;

  constexpr auto wrap_ref = ex::then([](auto &i) noexcept { return std::ref(i); });

  auto test_task_of_reference_type() -> ex::task<int &>
  {
    int &i = co_await []() -> ex::task<int &>
    {
      co_return global_int;
    }();
    CHECK(&i == &global_int);
    co_return i;
  }

  TEST_CASE("task supports reference types", "[types][task]")
  {
    global_int = 42;
    auto t     = test_task_of_reference_type();
    auto [i]   = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  TEST_CASE("task can co_await a sender of reference type", "[types][task]")
  {
    global_int = 42;
    auto t     = []() -> ex::task<int &>
    {
      int &i = co_await wrap_ref(
        exec::just_from([](auto sink) noexcept { return sink(global_int); }));
      CHECK(&i == &global_int);
      co_return i;
    }();
    auto [i] = ex::sync_wait(std::move(t)).value();
    CHECK(i == 42);
  }

  struct inline_affine_stopped_sender
  {
    using sender_concept        = ex::sender_tag;
    using completion_signatures = ex::completion_signatures<ex::set_stopped_t()>;

    template <class Receiver>
    struct operation
    {
      Receiver rcvr_;

      void start() & noexcept
      {
        ex::set_stopped(std::move(rcvr_));
      }
    };

    template <class Receiver>
    auto connect(Receiver rcvr) && -> operation<Receiver>
    {
      return {std::move(rcvr)};
    }

    struct attrs
    {
      [[nodiscard]]
      static constexpr auto query(ex::__get_completion_behavior_t<ex::set_stopped_t>) noexcept
      {
        return ex::__completion_behavior::__inline_completion
             | ex::__completion_behavior::__asynchronous_affine;
      }
    };

    [[nodiscard]]
    auto get_env() const noexcept -> attrs
    {
      return {};
    }
  };

  TEST_CASE("task co_awaiting inline|async_affine stopped sender does not deadlock",
            "[types][task]")
  {
    auto res = ex::sync_wait(
      []() -> ex::task<int>
      {
        co_await inline_affine_stopped_sender{};
        FAIL("Expected co_awaiting inline_affine_stopped_sender to stop the task");
        co_return 42;
      }());
    CHECK(!res.has_value());
  }

  TEST_CASE("repro for NVIDIA/stdexec#2041", "[types][task]")
  {
    auto task = []() -> ex::task<void>
    {
      co_return;
    };
    auto pool  = exec::static_thread_pool(1);
    auto scope = ex::counting_scope();
    for (int i = 0; i < 1000; ++i)
    {
      ex::spawn(ex::starts_on(pool.get_scheduler(), task()) | ex::upon_error([](auto) noexcept {}),
                scope.get_token());
    }
    ex::sync_wait(scope.join());
  }

  struct sink
  {
    using receiver_concept = ex::receiver_tag;
    void set_value() noexcept {}
    void set_error(std::exception_ptr) noexcept {}
    void set_stopped() noexcept {}
  };

  static_assert(!ex::sender_in<ex::task<void>, ex::env<>>);
  static_assert(!ex::sender_to<ex::task<void>, sink>);
  static_assert(ex::sender_in<ex::task<void>, ex::__sync_wait::__env>);

  // TODO: add tests for stop token support in task

}  // anonymous namespace

#endif  // !STDEXEC_NO_STDCPP_COROUTINES()
