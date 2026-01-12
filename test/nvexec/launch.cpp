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
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "common.cuh"
#include "nvexec/stream_context.cuh"

#include <algorithm>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace ex = STDEXEC;

using nvexec::is_on_gpu;

namespace { namespace {
  constexpr std::size_t N = 2ul * 1024;
  constexpr std::size_t THREAD_BLOCK_SIZE = 128u;
  constexpr std::size_t NUM_BLOCKS = (N + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

#define scaling 2

  auto bench() -> int {
    std::vector<int> input(N, 0);
    std::iota(input.begin(), input.end(), 1);
    std::ranges::transform(input, input.begin(), [](int i) { return i * scaling; });
    return std::accumulate(input.begin(), input.end(), 0);
  }

  TEST_CASE("nvexec launch executes on GPU", "[cuda][stream][adaptors][launch]") {
    thrust::device_vector<int> input(N, 0);
    std::iota(input.begin(), input.end(), 1);

    int* first = thrust::raw_pointer_cast(input.data());
    int* last = first + input.size();

    nvexec::stream_context stream{};

    flags_storage_t<2> flags_storage{};
    auto flags = flags_storage.get();

    auto snd = STDEXEC::transfer_just(stream.get_scheduler(), first, last)
             | nvexec::launch(
                 {.grid_size = NUM_BLOCKS, .block_size = THREAD_BLOCK_SIZE},
                 [flags](cudaStream_t, int* first, int* last) -> void {
                   const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                   const ptrdiff_t size = last - first;

                   // this should be executing on the GPU
                   if (idx == 0 && nvexec::is_on_gpu()) {
                     flags.set(0);
                   }

                   if (idx < size) {
                     first[idx] *= scaling;
                   }
                 })
             | STDEXEC::then([flags](int* first, int* last) -> int {
                 printf("In then() size=%d\n", (int) (last - first));
                 if (nvexec::is_on_gpu()) {
                   flags.set(1);
                 }
                 return std::accumulate(first, last, 0);
               });

    auto [result] = STDEXEC::sync_wait(std::move(snd)).value();

    REQUIRE(result == bench());
    REQUIRE(flags_storage.all_set_once());
  }

}} // namespace
