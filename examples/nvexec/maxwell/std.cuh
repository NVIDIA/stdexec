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

#include <chrono>
#include <thread>
#include <vector>
#include <barrier>

template <class Shape>
auto even_share(Shape n, std::size_t rank, std::size_t size) noexcept -> std::pair<Shape, Shape> {
  const auto avg_per_thread = n / size;
  const auto n_big_share = avg_per_thread + 1;
  const auto big_shares = n % size;
  const auto is_big_share = rank < big_shares;
  const auto begin = is_big_share ? n_big_share * rank
                                  : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
  const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

  return std::make_pair(begin, end);
}

void run_std(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t &grid,
  std::string_view method) {
  fields_accessor accessor = grid.accessor();

  const std::size_t n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(n_threads);
  std::barrier barrier(static_cast<std::ptrdiff_t>(n_threads));

  std::vector<std::chrono::system_clock::time_point> begins(n_threads);
  std::vector<std::chrono::system_clock::time_point> ends(n_threads);

  for (std::size_t tid = 0; tid < n_threads; tid++) {
    threads[tid] = std::thread([=, &barrier, &begins, &ends] {
      time_storage_t time{false};
      auto h_updater = update_h(accessor);
      auto e_updater = update_e(time.get(), dt, accessor);
      auto initializer = grid_initializer(dt, accessor);

      auto [begin, end] = even_share(accessor.cells, tid, n_threads);

      for (std::size_t i = begin; i < end; i++) {
        initializer(i);
      }

      const bool writer_thread = write_vtk && tid == 0;
      auto writer = dump_vtk(writer_thread, accessor);

      barrier.arrive_and_wait();
      begins[tid] = std::chrono::system_clock::now();

      for (std::size_t compute_step = 0; compute_step < n_iterations; compute_step++) {
        for (std::size_t i = begin; i < end; i++) {
          h_updater(i);
        }
        barrier.arrive_and_wait();
        for (std::size_t i = begin; i < end; i++) {
          e_updater(i);
        }
        barrier.arrive_and_wait();
      }

      writer();
      if (write_vtk) {
        barrier.arrive_and_wait();
      }
      ends[tid] = std::chrono::system_clock::now();
    });
  }

  for (std::size_t tid = 0; tid < n_threads; tid++) {
    threads[tid].join();
  }

  std::chrono::system_clock::time_point begin = begins[0];
  std::chrono::system_clock::time_point end = ends[0];

  for (std::size_t tid = 1; tid < n_threads; tid++) {
    begin = std::min(begins[tid], begin);
    end = std::max(ends[tid], end);
  }

  report_performance(
    grid.cells, n_iterations, method, std::chrono::duration<double>(end - begin).count());
}
