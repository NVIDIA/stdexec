/*
 * Copyright (c) 2026 University of Liège
 * Copyright (c) 2026 NVIDIA
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
#include <catch2/catch_all.hpp>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")

namespace ex = STDEXEC;

namespace
{
  template <bool MayThrow = false>
  auto then_store_thread_id(std::thread::id &id) noexcept
  {
    return ex::then([&id]() noexcept(!MayThrow) { id = std::this_thread::get_id(); });
  }

  //! @test Check that @c ex::get_start_scheduler is a forwarding query.
  TEST_CASE("get_start_scheduler is a forwarding query", "[sched_queries][get_start_scheduler]")
  {
    STATIC_CHECK(ex::forwarding_query(ex::get_start_scheduler));
  }

  //! @test Check that the start scheduler that @c ex::sync_wait sets in the receiver
  //! environment is a @c run_loop scheduler on the thread on which it starts the
  //! operation state.
  TEST_CASE("get_start_scheduler with sync_wait", "[sched_queries][get_start_scheduler]")
  {
    std::thread::id tid;

    auto sndr = ex::read_env(ex::get_start_scheduler)
              | ex::let_value(
                  [&](auto schd)
                  {
                    STATIC_CHECK(std::same_as<decltype(schd), ex::run_loop::scheduler>);
                    return ex::schedule(schd) | ::then_store_thread_id(tid);
                  });

    ex::sync_wait(std::move(sndr));

    CHECK(tid == std::this_thread::get_id());
  }

  //! @test Check that the start scheduler that @c ex::let_value sets in the receiver
  //! environment of the sender returned by the closure is the completion scheduler of the
  //! predecessor.
  //!
  //! Indeed, @c ex::let_value starts the successor from the completion of the
  //! predecessor.
  TEST_CASE("get_start_scheduler with let_value", "[sched_queries][get_start_scheduler]")
  {
    std::thread::id          pool_tid, tid;
    exec::static_thread_pool pool{1};

    auto sndr = ex::schedule(pool.get_scheduler())  //
              | ::then_store_thread_id(pool_tid)    //
              | ex::let_value(
                  [&]() noexcept
                  {
                    return ex::read_env(ex::get_start_scheduler)
                         | ex::let_value(
                             [&](auto schd)
                             {
                               STATIC_CHECK(
                                 std::same_as<decltype(schd), decltype(pool.get_scheduler())>);
                               return ex::schedule(schd) | ::then_store_thread_id(tid);
                             });
                  });

    ex::sync_wait(std::move(sndr));

    CHECK(tid == pool_tid);
    CHECK(tid != std::this_thread::get_id());
  }

  //! @test Check that the start scheduler that @c ex::continues_on sets in the receiver
  //! environment of the schedule sender is the completion scheduler of the predecessor.
  //!
  //! Indeed, @c ex::continues_on starts an hop operation state from the completion of the
  //! predecessor.
  TEST_CASE("get_start_scheduler with continues_on and inline_scheduler and only set_value_t",
            "[sched_queries][get_start_scheduler]")
  {
    std::thread::id          pool_tid, tid;
    exec::static_thread_pool pool{1};

    auto sndr = ex::schedule(pool.get_scheduler())        //
              | ::then_store_thread_id(pool_tid)          //
              | ex::continues_on(ex::inline_scheduler{})  //
              | ::then_store_thread_id(tid);

    using sndr_t = decltype(sndr);
    using env_t  = ex::prop<ex::get_start_scheduler_t, ex::run_loop::scheduler>;

    static_assert(ex::get_completion_signatures<sndr_t, env_t>()
                  == ex::completion_signatures<ex::set_value_t()>{});

    STATIC_CHECK(
      std::same_as<ex::__completion_scheduler_of_t<ex::set_value_t, decltype(sndr), env_t>,
                   decltype(pool.get_scheduler())>);

    ex::sync_wait(std::move(sndr));

    CHECK(tid == pool_tid);
    CHECK(tid != std::this_thread::get_id());
  }

  TEST_CASE("get_start_scheduler with continues_on and inline_scheduler and both set_value_t and "
            "set_error_t",
            "[sched_queries][get_start_scheduler]")
  {
    std::thread::id          pool_tid, tid;
    exec::static_thread_pool pool{1};

    using env_t = ex::prop<ex::get_start_scheduler_t, ex::run_loop::scheduler>;

    auto sndr_pre = ex::schedule(pool.get_scheduler()) | ::then_store_thread_id<true>(pool_tid);

    using sndr_pre_t = decltype(sndr_pre);

    static_assert(
      ex::get_completion_signatures<sndr_pre_t, env_t>()
      == ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>{});

    STATIC_CHECK(std::same_as<ex::__completion_scheduler_of_t<ex::set_value_t, sndr_pre_t, env_t>,
                              decltype(pool.get_scheduler())>);

    STATIC_CHECK(std::same_as<ex::__completion_scheduler_of_t<ex::set_error_t, sndr_pre_t, env_t>,
                              decltype(pool.get_scheduler())>);

    auto sndr = std::move(sndr_pre) | ex::continues_on(ex::inline_scheduler{})
              | ::then_store_thread_id<true>(tid);

    using sndr_t = decltype(sndr);

    static_assert(
      ex::get_completion_signatures<sndr_t, env_t>()
      == ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>{});

    // fails !
    STATIC_CHECK(std::same_as<ex::__completion_scheduler_of_t<ex::set_value_t, sndr_t, env_t>,
                              decltype(pool.get_scheduler())>);

    // fails !
    STATIC_CHECK(std::same_as<ex::__completion_scheduler_of_t<ex::set_error_t, sndr_t, env_t>,
                              decltype(pool.get_scheduler())>);

    ex::sync_wait(std::move(sndr));

    CHECK(tid == pool_tid);
    CHECK(tid != std::this_thread::get_id());
  }
}  // namespace

STDEXEC_PRAGMA_POP()
