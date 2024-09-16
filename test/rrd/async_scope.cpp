#include "../../relacy/relacy_std.hpp"
#include "../../relacy/relacy_cli.hpp"

#include <stdexec/execution.hpp>
#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>

#include <chrono>
#include <random>
#include <iostream>

using rl::nvar;
using rl::nvolatile;
using rl::mutex;

namespace ex = stdexec;
using exec::async_scope;

struct async_scope_bug : rl::test_suite<async_scope_bug, 1>
{
    static size_t const dynamic_thread_count = 2;

    void thread(unsigned)
    {
        exec::static_thread_pool ctx{1};

        ex::scheduler auto sch = ctx.get_scheduler();

        exec::async_scope scope;
        std::atomic_bool produced{false};
        ex::sender auto begin = ex::schedule(sch);
        {
            ex::sender auto ftr = scope.spawn_future(begin | stdexec::then([&]() { produced.store(true); }));
            (void) ftr;
        }
        stdexec::sync_wait(scope.on_empty() | stdexec::then([&]() {
            RL_ASSERT(produced.load());
        }));
    }
};

int main()
{
    rl::test_params p;
    p.iteration_count = 50000;
    p.execution_depth_limit = 10000;
    rl::simulate<async_scope_bug>(p);
    return 0;
}
