#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

int main() {
  std::vector<int> input(2048, 1);

  auto snd = ex::transfer_just(nvexec::stream_context{}.get_scheduler(), input)
           | nvexec::reduce();

  auto [result] = ex::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
