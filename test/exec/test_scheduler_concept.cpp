/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <test_common/catch2.hpp>

#include <stdexec/execution.hpp>

#include <exec/reschedule.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/thread_pool_base.hpp>
#include <exec/timed_thread_scheduler.hpp>
#include <exec/trampoline_scheduler.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <concepts>
#  include <type_traits>
#endif

// Regression tests for issue #2134: per [exec.sched], a scheduler type must
// define a nested `scheduler_concept` alias derived from `scheduler_tag`.
// These tests fail to compile if a scheduler in stdexec forgets it.

namespace ex = STDEXEC;

namespace
{
  template <class Sch>
  concept has_scheduler_concept = requires {
    typename Sch::scheduler_concept;
    requires std::derived_from<typename Sch::scheduler_concept, ex::scheduler_tag>;
  };

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  class inline_test_thread_pool : public exec::thread_pool_base<inline_test_thread_pool>
  {
   public:
    [[nodiscard]]
    auto available_parallelism() const noexcept -> std::uint32_t
    {
      return 1;
    }

    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() noexcept -> ex::forward_progress_guarantee
    {
      return ex::forward_progress_guarantee::parallel;
    }

    void enqueue(exec::_pool_::task_base* task, std::uint32_t tid = 0) noexcept
    {
      ++enqueued_;
      task->execute_(task, tid);
    }

    std::uint32_t enqueued_ = 0;
  };
#endif

  TEST_CASE("schedulers provide the scheduler_concept nested alias",
            "[types][schedulers][scheduler_concept]")
  {
    STATIC_REQUIRE(has_scheduler_concept<ex::inline_scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<ex::run_loop::scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<ex::task_scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<ex::parallel_scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<exec::static_thread_pool::scheduler>);
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
    STATIC_REQUIRE(has_scheduler_concept<inline_test_thread_pool::scheduler>);
#endif
    STATIC_REQUIRE(has_scheduler_concept<exec::timed_thread_scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<exec::trampoline_scheduler>);
    STATIC_REQUIRE(has_scheduler_concept<exec::__resched::__scheduler>);
  }

  TEST_CASE("schedulers satisfy the stdexec scheduler concept",
            "[types][schedulers][scheduler_concept]")
  {
    STATIC_REQUIRE(ex::scheduler<ex::inline_scheduler>);
    STATIC_REQUIRE(ex::scheduler<ex::run_loop::scheduler>);
    STATIC_REQUIRE(ex::scheduler<ex::task_scheduler>);
    STATIC_REQUIRE(ex::scheduler<ex::parallel_scheduler>);
    STATIC_REQUIRE(ex::scheduler<exec::static_thread_pool::scheduler>);
    STATIC_REQUIRE(ex::scheduler<exec::trampoline_scheduler>);
  }
}  // namespace
