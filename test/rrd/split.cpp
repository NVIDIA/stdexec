#include "../../relacy/relacy_std.hpp"
#include "../../relacy/relacy_cli.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

using rl::nvar;
using rl::nvolatile;
using rl::mutex;

namespace ex = stdexec;

struct split_bug : rl::test_suite<split_bug, 1> {
  static size_t const dynamic_thread_count = 2;

  void thread(unsigned) {
    exec::static_thread_pool pool{1};
    auto split = ex::schedule(pool.get_scheduler()) | ex::then([] { return 42; }) | ex::split();

    auto [val] = ex::sync_wait(split).value();
    RL_ASSERT(val == 42);
  }
};

auto main() -> int {
  rl::test_params p;
  p.iteration_count = 50000;
  p.execution_depth_limit = 10000;
  rl::simulate<split_bug>(p);
  return 0;
}
