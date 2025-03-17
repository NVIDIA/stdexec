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

void run_cpp(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t &grid,
  std::string_view method) {
  fields_accessor accessor = grid.accessor();

  auto initializer = grid_initializer(dt, accessor);
  for (std::size_t i = 0; i < accessor.cells; i++) {
    initializer(i);
  }

  auto action = [&]() {
    time_storage_t time{false};
    auto h_updater = update_h(accessor);
    auto e_updater = update_e(time.get(), dt, accessor);

    auto writer = dump_vtk(write_vtk, accessor);

    for (std::size_t compute_step = 0; compute_step < n_iterations; compute_step++) {
      for (std::size_t i = 0; i < accessor.cells; i++) {
        h_updater(i);
      }
      for (std::size_t i = 0; i < accessor.cells; i++) {
        e_updater(i);
      }
    }

    writer();
  };

  report_performance(grid.cells, n_iterations, method, action);
}
