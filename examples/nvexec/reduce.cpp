#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>
#include <span>

namespace ex = stdexec;

int main() {
  const int n = 2 * 1024;
  thrust::device_vector<float> input(n, 1.0f);
  float* first = thrust::raw_pointer_cast(input.data());
  float* last = thrust::raw_pointer_cast(input.data()) + input.size();

  nvexec::stream_context stream_ctx{};

  auto snd = ex::transfer_just(stream_ctx.get_scheduler(), std::span{first, last})
           | nvexec::reduce(42.0f);

  auto [result] = stdexec::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
}
