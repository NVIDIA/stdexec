#include <exec/system_context.hpp>
#include <stdexec/execution.hpp>

#include <cstdlib>

int main() {
  auto x = stdexec::starts_on(exec::get_parallel_scheduler(), stdexec::just(42));
  auto [a] = stdexec::sync_wait(std::move(x)).value();
  return a == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
