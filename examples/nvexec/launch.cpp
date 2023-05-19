#define STDEXEC_THROW_ON_CUDA_ERROR

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <numeric>
#include <cub/cub.cuh>

int main() {
  std::vector<int> input(2048, 0);
  std::iota(input.begin(), input.end(), 1);

  nvexec::stream_context stream{};

  auto snd = stdexec::transfer_just(stream.get_scheduler(), input)
           | nvexec::launch(
               [] (cudaStream_t stm, std::vector<int>& v) {
               });

  auto result = stdexec::sync_wait(std::move(snd)).value();
}
