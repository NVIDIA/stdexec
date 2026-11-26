/*
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 */

#include <stdexec_relacy.hpp>

// Public schedule_all pipeline, original header, tracked independent outputs.
#include <cstdlib>
#include <exec/sequence/ignore_all_values.hpp>
#include <exec/sequence/transform_each.hpp>
#include <exec/static_thread_pool.hpp>
#include <ranges>
#include <stdexec/execution.hpp>

struct schedule_all_publication : rl::test_suite<schedule_all_publication, 1>
{
  static constexpr std::size_t dynamic_thread_count = 2;
  rl::var<int>                 outputs[4];
  void                         before()
  {
    for (auto& output: outputs)
      output($) = 0;
  }
  void thread(unsigned)
  {
    exec::static_thread_pool pool{2};
    auto                     sender = exec::schedule_all(pool, std::views::iota(0, 4))
                | exec::transform_each(
                    stdexec::then([&](int i) noexcept { outputs[i]($) = i + 1; }))
                | exec::ignore_all_values()
                | stdexec::then(
                    [&]() noexcept
                    {
                      for (int i = 0; i < 4; ++i)
                        RL_ASSERT(outputs[i]($) == i + 1);
                    });
    RL_ASSERT(stdexec::sync_wait(std::move(sender)).has_value());
  }
};
int main(int argc, char** argv)
{
  rl::test_params p;
  p.iteration_count       = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100000;
  p.execution_depth_limit = 20000;
  p.search_type           = rl::random_scheduler_type;
  return rl::simulate<schedule_all_publication>(p) ? 0 : 1;
}
