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

// clang-format Language: Cpp

#pragma once

#include "common.cuh"

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
#  include "nvexec/stream/common.cuh"
#  include "nvexec/detail/throw_on_cuda_error.cuh"
#endif

#include <ranges>
#include <algorithm>
#include <execution>

#include <thrust/iterator/counting_iterator.h>

template <class Policy>
auto is_gpu_policy([[maybe_unused]] Policy&& policy) -> bool {
#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
  bool* flag{};
  STDEXEC_TRY_CUDA_API(cudaMallocHost(&flag, sizeof(bool)));
  std::for_each(policy, flag, flag + 1, [](bool& f) { f = nvexec::is_on_gpu(); });

  bool h_flag = *flag;
  STDEXEC_ASSERT_CUDA_API(cudaFreeHost(flag));

  return h_flag;
#else
  return false;
#endif
}

template <class Policy>
void run_stdpar(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t& grid,
  Policy&& policy,
  std::string_view method) {
  fields_accessor accessor = grid.accessor();
  time_storage_t time{is_gpu_policy(policy)};

  auto begin = thrust::make_counting_iterator(std::size_t{0});
  auto end = thrust::make_counting_iterator(accessor.cells);

  std::for_each(policy, begin, end, grid_initializer(dt, accessor));

  report_performance(grid.cells, n_iterations, method, [&]() {
    auto writer = dump_vtk(write_vtk, accessor);
    for (std::size_t compute_step = 0; compute_step < n_iterations; compute_step++) {

      std::for_each(policy, begin, end, update_h(accessor));
      std::for_each(policy, begin, end, update_e(time.get(), dt, accessor));
    }

    writer();
  });
}
