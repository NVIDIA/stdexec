/*
 * Copyright (c) 2025 NVIDIA Corporation
 * Copyright (c) 2025 Chris Cotter
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

#include "../../relacy/relacy_std.hpp"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

struct sync_wait_bg_thread : rl::test_suite<sync_wait_bg_thread, 1> {
  static size_t const dynamic_thread_count = 1;

  void thread(unsigned) {
    exec::static_thread_pool pool{1};
    auto sender = ex::schedule(pool.get_scheduler()) | ex::then([] { return 42; });

    auto [val] = ex::sync_wait(sender).value();
    RL_ASSERT(val == 42);
  }
};

auto main() -> int {
  rl::test_params p;
  p.iteration_count = 50000;
  p.execution_depth_limit = 10000;
  p.search_type = rl::random_scheduler_type;
  rl::simulate<sync_wait_bg_thread>(p);
  return 0;
}
