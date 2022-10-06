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
#include "maxwell/std.cuh"
#include "maxwell/stdpar.cuh"
#include "maxwell/cpp.cuh"
#include "maxwell/cuda.cuh"

int main(int argc, char *argv[]) {
  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--write-results\n"
              << "\t--inner-iterations\n"
              << "\t--run-std\n"
              << "\t--run-stdpar\n"
              << "\t--run-cpp\n"
              << "\t--run-cuda\n"
              << "\t--run-inline-scheduler\n"
              << "\t--run-thread-pool-scheduler\n"
              << "\t--run-stream-scheduler\n"
              << "\t--N\n"
              << std::endl;
    return 0;
  }

  const bool write_vtk = value(params, "write-vtk");
  const bool write_results = value(params, "write-results");
  const std::size_t n_inner_iterations = value(params, "inner-iterations", 100);
  const std::size_t n_outer_iterations = value(params, "outer-iterations", 10);
  const std::size_t N = value(params, "N", 512);

  auto run_snr_on = [&](std::string_view scheduler_name,
                        std::execution::scheduler auto &&scheduler) {
    grid_t grid{N, is_gpu_scheduler(scheduler)};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_snr(dt, write_vtk, n_inner_iterations, n_outer_iterations, grid, 
            scheduler_name, std::forward<decltype(scheduler)>(scheduler));

    if (write_results) {
      store_results(accessor);
    }
  };

  report_header();

  // S&R
  if (value(params, "run-inline-scheduler")) {
    run_snr_on("CPU (snr inline)", exec::inline_scheduler{});
  }
  if (value(params, "run-thread-pool-scheduler")) {
    exec::static_thread_pool pool{std::thread::hardware_concurrency()};
    run_snr_on("CPU (snr thread pool)", pool.get_scheduler());
  }

  if (value(params, "run-cuda")) {
    grid_t grid{N, true /* gpu */};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_cuda(dt, write_vtk, n_inner_iterations, n_outer_iterations, grid, "GPU (cuda)");

    if (write_results) {
      store_results(accessor);
    }
  }

  if (value(params, "run-stream-scheduler")) {
    nvexec::stream_context stream_ctx{};
    run_snr_on("GPU (snr cuda stream)", stream_ctx.get_scheduler());
  }

  // Naive
  if (value(params, "run-std")) {
    grid_t grid{N, false /* !gpu */};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_std(dt, write_vtk, n_inner_iterations, n_outer_iterations, grid, "CPU (std)");

    if (write_results) {
      store_results(accessor);
    }
  }

  #ifdef _NVHPC_CUDA
  if (value(params, "run-stdpar")) {
    const bool gpu = is_gpu_policy(std::execution::par_unseq);
    std::string_view method = gpu ? "GPU (stdpar)" : "CPU (stdpar)";
    grid_t grid{N, gpu};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_stdpar(dt, write_vtk, n_inner_iterations, n_outer_iterations, grid, std::execution::par_unseq, method);

    if (write_results) {
      store_results(accessor);
    }
  }
  #endif

  if (value(params, "run-cpp")) {
    grid_t grid{N, false /* !gpu */};

    auto accessor = grid.accessor();
    auto dt = calculate_dt(accessor.dx, accessor.dy);

    run_cpp(dt, write_vtk, n_inner_iterations, n_outer_iterations, grid, "CPU (cpp)");

    if (write_results) {
      store_results(accessor);
    }
  }
}

