#ifdef STDEXEC_TEST_SYSTEM_CONTEXT
#  include <exec/system_context.hpp>
#else
#  include <exec/static_thread_pool.hpp>
#endif
#include <stdexec/execution.hpp>

#include <cstdlib>

int main()
{
#ifdef STDEXEC_TEST_SYSTEM_CONTEXT
  auto x = stdexec::starts_on(exec::get_parallel_scheduler(), stdexec::just(42));
#else
  exec::static_thread_pool pool{1};
  auto                     x = stdexec::starts_on(pool.get_scheduler(), stdexec::just(42));
#endif
  auto [a] = stdexec::sync_wait(std::move(x)).value();
  return a == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
