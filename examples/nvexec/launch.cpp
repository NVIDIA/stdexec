/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <numeric>
#include <cub/cub.cuh>

#include <thrust/device_vector.h>

constexpr std::size_t N = 2 * 1024;
constexpr std::size_t THREAD_BLOCK_SIZE = 128u;
constexpr std::size_t NUM_BLOCKS = (N + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

enum {
  scaling = 2
};

auto bench() -> int {
  std::vector<int> input(N, 0);
  std::iota(input.begin(), input.end(), 1);
  std::ranges::transform(input, input.begin(), [](int i) { return i * scaling; });
  return std::accumulate(input.begin(), input.end(), 0);
}

auto main() -> int {
  thrust::device_vector<int> input(N, 0);
  std::iota(input.begin(), input.end(), 1);
  int* first = thrust::raw_pointer_cast(input.data());
  int* last = first + input.size();

  nvexec::stream_context stream{};

  auto snd = stdexec::transfer_just(stream.get_scheduler(), first, last)
           | nvexec::launch(
               {.grid_size = NUM_BLOCKS, .block_size = THREAD_BLOCK_SIZE},
               [](cudaStream_t, int* first, int* last) {
                 assert(nvexec::is_on_gpu());
                 int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 if (idx < (last - first)) {
                   first[idx] *= scaling;
                 }
               })
           | stdexec::then([](int* first, int* last) {
               assert(nvexec::is_on_gpu());
               return std::accumulate(first, last, 0);
             });

  auto [result] = stdexec::sync_wait(std::move(snd)).value();

  std::cout << "result: " << result << std::endl;
  std::cout << "benchmark: " << bench() << std::endl;
}
