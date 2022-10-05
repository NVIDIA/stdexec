#include <range/v3/range/concepts.hpp>
#include <range/v3/view/subrange.hpp>

#include <thrust/device_vector.h>

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

int main() {
  const int n = 2 * 1024;
  thrust::device_vector<int> input(n, 1);
  auto rng = ranges::subrange(thrust::raw_pointer_cast(input.data()),
                              thrust::raw_pointer_cast(input.data()) + input.size());

  nvexec::stream_context stream{};

  auto snd = ex::transfer_just(stream.get_scheduler(), rng)
           | nvexec::reduce();

  auto [result] = ex::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
