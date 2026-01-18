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

// This file causes clangd to crash during parsing
#if !defined(STDEXEC_CLANGD_INVOKED)

#  include "maxwell/cuda.cuh"
#  include "maxwell/snr.cuh"
#  include "maxwell/stdpar.cuh" // IWYU pragma: keep

auto main(int argc, char *argv[]) -> int {
  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--iterations\n"
              << (STDEXEC_HAS_PARALLEL_ALGORITHMS() ? "\t--run-stdpar\n" : "") //
              << "\t--run-cuda\n"
              << "\t--run-stream-scheduler\n"
              << "\t--N\n"
              << std::endl;
    return 0;
  }

  const bool write_vtk = value(params, "write-vtk");
  const std::size_t n_iterations = value(params, "iterations", 1000);
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

  if (value(params, "run-cuda")) {
    grid_t grid{N, true /* gpu */};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_cuda(dt, write_vtk, n_iterations, grid, "GPU (cuda)");
  }

  if (value(params, "run-stream-scheduler")) {
    nvexec::stream_context stream_ctx{};
    run_snr_on("GPU (snr cuda stream)", stream_ctx.get_scheduler());
  }

#  if STDEXEC_HAS_PARALLEL_ALGORITHMS()
  if (value(params, "run-stdpar")) {
    const bool gpu = is_gpu_policy(stdexec::par_unseq);
    std::string_view method = gpu ? "GPU (stdpar)" : "CPU (stdpar)";
    grid_t grid{N, gpu};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_stdpar(dt, write_vtk, n_iterations, grid, stdexec::par_unseq, method);
  }
#  endif // STDEXEC_HAS_PARALLEL_ALGORITHMS()
}

#endif // !defined(STDEXEC_CLANGD_INVOKED)
