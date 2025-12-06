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

#include "maxwell/snr.cuh"
#include "maxwell/cpp.cuh"

auto main(int argc, char *argv[]) -> int {
  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--iterations\n"
              << "\t--run-cpp\n"
              << "\t--run-inline-scheduler\n"
              << "\t--N\n"
              << std::endl;
    return 0;
  }

  const bool write_vtk = value(params, "write-vtk");
  const std::size_t n_iterations = value(params, "iterations", 100);
  const std::size_t N = value(params, "N", 512);

  auto run_snr_on = [&](std::string_view scheduler_name, stdexec::scheduler auto &&scheduler) {
    grid_t grid{N, is_gpu_scheduler(scheduler)};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_snr(
      dt,
      write_vtk,
      n_iterations,
      grid,
      scheduler_name,
      std::forward<decltype(scheduler)>(scheduler));
  };

  report_header();

  if (value(params, "run-inline-scheduler")) {
    run_snr_on("CPU (snr inline)", stdexec::inline_scheduler{});
  }

  if (value(params, "run-cpp")) {
    grid_t grid{N, false /* !gpu */};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_cpp(dt, write_vtk, n_iterations, grid, "CPU (cpp)");
  }
}
