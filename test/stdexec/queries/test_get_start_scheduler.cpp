/*
 * Copyright (c) 2022 ETH Zurich
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
  consteval bool test_is_forwarding_query()
  {
    return ex::forwarding_query(ex::get_start_scheduler);
  }
  static_assert(test_is_forwarding_query());

  auto then_store_thread_id(std::thread::id &id) noexcept
  {
    return ex::then([&id]() noexcept { id = std::this_thread::get_id(); });
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
  //!
  //! See also:
  //! - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__let.hpp#L180
  //! - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__schedulers.hpp#L639-L655
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

  //! @test Show that @c ex::continues_on onto an inline scheduler behaves as
  //! expected at run time. However, the compile-time scheduler queries are not correct:
  //! they indicate a hop onto the start scheduler set in the outer receiver environment
  //! rather than continuing on the completion scheduler of the predecessor.
  //!
  //! Indeed, although @c ex::continues_on starts an operation state from the
  //! completion of the predecessor, it does not set the start scheduler accordingly
  //! through a secondary environment.
  //!
  //! See also:
  //! - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__continues_on.hpp#L193
  //! - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__continues_on.hpp#L127
  TEST_CASE("get_start_scheduler with continues_on and inline_scheduler",
            "[sched_queries][get_start_scheduler]")
  {
    std::thread::id          pool_tid, tid;
    exec::static_thread_pool pool{1};

    auto sndr = ex::schedule(pool.get_scheduler())        //
              | ::then_store_thread_id(pool_tid)          //
              | ex::continues_on(ex::inline_scheduler{})  //
              | ::then_store_thread_id(tid);

    STATIC_CHECK(std::same_as<ex::__completion_scheduler_of_t<
                                ex::set_value_t,
                                decltype(sndr),
                                ex::prop<ex::get_start_scheduler_t, ex::run_loop::scheduler>>,
                              decltype(pool.get_scheduler())>);

    ex::sync_wait(std::move(sndr));

    CHECK(tid == pool_tid);  // run-time behavior is as expected
    CHECK(tid != std::this_thread::get_id());
  }
}  // namespace

STDEXEC_PRAGMA_POP()
