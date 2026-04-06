#include <stdexec/execution.hpp>

#ifndef STDEXEC_TEST_PARALLEL_SCHEDULER
#  include <exec/static_thread_pool.hpp>
#endif

#include <cstdlib>
#include <utility>

int main()
{
#ifdef STDEXEC_TEST_PARALLEL_SCHEDULER
  auto x = stdexec::starts_on(stdexec::get_parallel_scheduler(), stdexec::just(42));
#else
  exec::static_thread_pool pool{1};
  auto                     x = stdexec::starts_on(pool.get_scheduler(), stdexec::just(42));
#endif
  auto [a] = stdexec::sync_wait(std::move(x)).value();
  return a == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
