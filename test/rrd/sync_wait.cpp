#include "../../relacy/relacy_std.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;

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
