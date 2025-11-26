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

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>
#include <span>

namespace ex = stdexec;

auto main() -> int {
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
