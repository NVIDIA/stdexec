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

#include <stdexec/execution.hpp> // IWYU pragma: export

#include <exec/inline_scheduler.hpp>
#include <exec/repeat_n.hpp>

#include "./common.cuh"

#if STDEXEC_CUDA_COMPILATION()
#  include <nvexec/multi_gpu_context.cuh> // IWYU pragma: export
#  include <nvexec/stream_context.cuh>    // IWYU pragma: export
#else
namespace nvexec {
  inline constexpr bool is_on_gpu() noexcept {
    return false;
  }
} // namespace nvexec
#endif

namespace ex = stdexec;

template <class SchedulerT>
[[nodiscard]]
auto is_gpu_scheduler([[maybe_unused]] SchedulerT&& scheduler) -> bool {
  auto snd = ex::just() | ex::on(scheduler, ex::then([] { return nvexec::is_on_gpu(); }));
  auto [on_gpu] = ex::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(
  float dt,
  float* time,
  bool write_results,
  std::size_t n_iterations,
  fields_accessor accessor,
  ex::scheduler auto&& computer) {
  return ex::on(
           computer,
           ex::just() //
             | ex::bulk(ex::par, accessor.cells, update_h(accessor))
             | ex::bulk(ex::par, accessor.cells, update_e(time, dt, accessor))
             | exec::repeat_n(n_iterations))
       | ex::then(dump_vtk(write_results, accessor));
}

void run_snr(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t& grid,
  std::string_view scheduler_name,
  ex::scheduler auto&& computer) {
  time_storage_t time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  auto init =
    ex::on(computer, ex::just() | ex::bulk(ex::par, grid.cells, grid_initializer(dt, accessor)));
  ex::sync_wait(init);

  auto snd = maxwell_eqs_snr(dt, time.get(), write_vtk, n_iterations, accessor, computer);

  report_performance(grid.cells, n_iterations, scheduler_name, [&snd] {
    ex::sync_wait(std::move(snd));
  });
}
