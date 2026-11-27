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

#include <stdexec_relacy.hpp>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

// A task on a pool of two workers waits for a task that it schedules on the
// same pool (https://github.com/NVIDIA/stdexec/issues/1305). With blocks of one
// task, the worker that schedules the task grants it to the thieves right away,
// and it wakes the other worker if that one is not stealing. Relacy reports a
// deadlock if the other worker can go to sleep without seeing the task while
// the worker that granted it does not see it go to sleep.
struct nested_sync_wait : rl::test_suite<nested_sync_wait, 1>
{
  static size_t const dynamic_thread_count = 2;

  void thread(unsigned)
  {
    exec::static_thread_pool pool{
      2,
      exec::bwos_params{.numBlocks = 2, .blockSize = 1}
    };
    auto sched = pool.get_scheduler();
    auto inner = [&]
    {
      auto [value] = ex::sync_wait(ex::starts_on(sched, ex::just(42))).value();
      return value;
    };
    auto [value] = ex::sync_wait(ex::starts_on(sched, ex::just() | ex::then(inner))).value();
    RL_ASSERT(value == 42);
  }
};

auto main() -> int
{
  rl::test_params p;
  p.iteration_count       = 50000;
  p.execution_depth_limit = 20000;
  p.search_type           = rl::random_scheduler_type;
  // rl::simulate reports failures by returning false; it does not abort.
  return rl::simulate<nested_sync_wait>(p) ? 0 : 1;
}
