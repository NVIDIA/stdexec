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

#include "stdexec/__detail/__config.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if STDEXEC_CUDA_COMPILATION()
#  define STDEXEC_STDERR
#  include "nvexec/detail/throw_on_cuda_error.cuh"
#endif

struct deleter_t {
  bool on_gpu{};

  template <class T>
  void operator()(T *ptr) {
#if STDEXEC_CUDA_COMPILATION()
    if (on_gpu) {
      STDEXEC_ASSERT_CUDA_API(cudaFree(ptr));
    } else
#endif
    {
      free(ptr);
    }
  }
};

template <class T>
inline auto allocate_on(bool gpu, std::size_t elements = 1) -> std::unique_ptr<T, deleter_t> {
  T *ptr{};

#if STDEXEC_CUDA_COMPILATION()
  if (gpu) {
    STDEXEC_TRY_CUDA_API(cudaMallocManaged(&ptr, elements * sizeof(T)));
  } else
#endif
  {
    ptr = reinterpret_cast<T *>(malloc(elements * sizeof(T)));
  }

  std::memset(ptr, 0, elements * sizeof(T));
  return std::unique_ptr<T, deleter_t>(ptr, deleter_t{gpu});
}

enum class field_id : int {
  er,
  hr,
  mh,
  hx,
  hy,
  ez,
  dz,
  fields_count
};

struct fields_accessor {
  float dx;
  float dy;

  float width;
  float height;

  std::size_t n;
  std::size_t cells;

  float *base_ptr;

  STDEXEC_ATTRIBUTE(nodiscard, host, device) auto get(field_id id) const -> float * {
    return base_ptr + static_cast<int>(id) * cells;
  }
};

struct grid_t {
  float width = 160;
  float height = 160;

  std::size_t n{};
  std::size_t cells{};

  std::unique_ptr<float, deleter_t> fields_{};

  grid_t(grid_t &&) = delete;
  grid_t(const grid_t &) = delete;

  grid_t(std::size_t n, bool gpu)
    : n(n)
    , cells(n * n)
    , fields_(
        allocate_on<float>(
          gpu,
          static_cast<std::size_t>(cells) * static_cast<int>(field_id::fields_count))) {
  }

  [[nodiscard]]
  auto accessor() const -> fields_accessor {
    return {
      .dx = height / static_cast<float>(n),
      .dy = width / static_cast<float>(n),
      .width = width,
      .height = height,
      .n = n,
      .cells = cells,
      .base_ptr = fields_.get()};
  }
};

constexpr float C0 = 299792458.0f; // Speed of light [metres per second]

STDEXEC_ATTRIBUTE(host, device)
inline auto
  is_circle_part(float x, float y, float object_x, float object_y, float object_size) -> bool {
  const float os2 = object_size * object_size;
  return ((x - object_x) * (x - object_x) + (y - object_y) * (y - object_y) <= os2);
}

inline auto calculate_dt(float dx, float dy) -> float {
  const float cfl = 0.3;
  return cfl * (std::min) (dx, dy) / C0;
}

struct grid_initializer_t {
  float dt;
  fields_accessor accessor;

  STDEXEC_ATTRIBUTE(host, device) void operator()(std::size_t cell_id) const {
    const std::size_t row = cell_id / accessor.n;
    const std::size_t column = cell_id % accessor.n;

    float er = 1.0f;
    float hr = 1.0f;

    const float x = static_cast<float>(column) * accessor.dx;
    const float y = static_cast<float>(row) * accessor.dy;

    const float soil_y = accessor.width / 2.2f;
    const float object_y = soil_y - 22.0f;
    const float object_size = 3.0;
    const float soil_er_hr = 1.3;

    if (y < soil_y) {
      const float middle_x = accessor.width / 2;
      const float object_x = middle_x;

      if (is_circle_part(x, y, object_x, object_y, object_size)) {
        er = hr = 200000; /// Relative permeabuliti of Iron
      } else {
        er = hr = soil_er_hr;
      }
    }

    accessor.get(field_id::er)[cell_id] = er;
    accessor.get(field_id::hr)[cell_id] = hr;

    accessor.get(field_id::hx)[cell_id] = {};
    accessor.get(field_id::hy)[cell_id] = {};

    accessor.get(field_id::ez)[cell_id] = {};
    accessor.get(field_id::dz)[cell_id] = {};

    accessor.get(field_id::mh)[cell_id] = C0 * dt / hr;
  }
};

inline auto grid_initializer(float dt, fields_accessor accessor) -> grid_initializer_t {
  return {.dt = dt, .accessor = accessor};
}

STDEXEC_ATTRIBUTE(host, device)
inline auto right_nid(std::size_t cell_id, std::size_t col, std::size_t N) -> std::size_t {
  return col == N - 1 ? cell_id - (N - 1) : cell_id + 1;
}

STDEXEC_ATTRIBUTE(host, device)
inline auto left_nid(std::size_t cell_id, std::size_t col, std::size_t N) -> std::size_t {
  return col == 0 ? cell_id + N - 1 : cell_id - 1;
}

STDEXEC_ATTRIBUTE(host, device)
inline auto bottom_nid(std::size_t cell_id, std::size_t row, std::size_t N) -> std::size_t {
  return row == 0 ? cell_id + N * (N - 1) : cell_id - N;
}

STDEXEC_ATTRIBUTE(host, device)
inline auto top_nid(std::size_t cell_id, std::size_t row, std::size_t N) -> std::size_t {
  return row == N - 1 ? cell_id - N * (N - 1) : cell_id + N;
}

struct h_field_calculator_t {
  fields_accessor accessor;

  STDEXEC_ATTRIBUTE(always_inline, host, device) void operator()(std::size_t cell_id) const {
    const std::size_t N = accessor.n;
    const std::size_t column = cell_id % N;
    const std::size_t row = cell_id / N;
    const float *ez = accessor.get(field_id::ez);
    const float cell_ez = ez[cell_id];
    const float neighbour_ex = ez[top_nid(cell_id, row, N)];
    const float neighbour_ez = ez[right_nid(cell_id, column, N)];
    const float mh = accessor.get(field_id::mh)[cell_id];
    const float cex = (neighbour_ex - cell_ez) / accessor.dy;
    const float cey = (cell_ez - neighbour_ez) / accessor.dx;
    accessor.get(field_id::hx)[cell_id] -= mh * cex;
    accessor.get(field_id::hy)[cell_id] -= mh * cey;
  }
};

inline auto update_h(fields_accessor accessor) -> h_field_calculator_t {
  return {accessor};
}

struct e_field_calculator_t {
  float dt;
  float *time;
  fields_accessor accessor;
  std::size_t source_position;

  STDEXEC_ATTRIBUTE(nodiscard, host, device)
  auto gaussian_pulse(float t, float t_0, float tau) const -> float {
    return static_cast<float>(std::exp(-(((t - t_0) / tau) * (t - t_0) / tau)));
  }

  STDEXEC_ATTRIBUTE(nodiscard, host, device)
  auto calculate_source(float t, float frequency) const -> float {
    const float tau = 0.5f / frequency;
    const float t_0 = 6.0f * tau;
    return gaussian_pulse(t, t_0, tau);
  }

  STDEXEC_ATTRIBUTE(always_inline, host, device) void operator()(std::size_t cell_id) const {
    const std::size_t N = accessor.n;
    const std::size_t column = cell_id % N;
    const std::size_t row = cell_id / N;
    const bool source_owner = cell_id == source_position;
    const float er = accessor.get(field_id::er)[cell_id];
    const float *hx = accessor.get(field_id::hx);
    const float *hy = accessor.get(field_id::hy);
    const float cell_hy = hy[cell_id];
    const float neighbour_hy = hy[left_nid(cell_id, column, N)];
    const float hy_diff = cell_hy - neighbour_hy;
    const float cell_hx = hx[cell_id];
    const float neighbour_hx = hx[bottom_nid(cell_id, row, N)];
    const float hx_diff = neighbour_hx - cell_hx;
    float cell_dz = accessor.get(field_id::dz)[cell_id];

    cell_dz += C0 * dt * (hy_diff / accessor.dx + hx_diff / accessor.dy);

    if (source_owner) {
      cell_dz += calculate_source(*time, 5E+7);
      *time += dt;
    }

    accessor.get(field_id::ez)[cell_id] = cell_dz / er;
    accessor.get(field_id::dz)[cell_id] = cell_dz;
  }
};

inline auto update_e(float *time, float dt, fields_accessor accessor) -> e_field_calculator_t {
  std::size_t source_position = accessor.n / 2 + (accessor.n * (accessor.n / 2));
  return {.dt = dt, .time = time, .accessor = accessor, .source_position = source_position};
}

class result_dumper_t {
  bool write_results_{};
  fields_accessor accessor_;

  void write_vtk(const std::string &filename) const {
    if (!write_results_) {
      return;
    }

    float *ez = accessor_.get(field_id::ez);

    int rank_ = 0;
    if (rank_ == 0) {
      printf("\twriting vtk");
      fflush(stdout);
    }

    FILE *f = fopen(filename.c_str(), "w");

    const std::size_t nx = accessor_.n;
    const float dx = accessor_.dx;
    const float dy = accessor_.dy;

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "vtk output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(f, "POINTS %d double\n", (int) (accessor_.cells * 4));

    for (std::size_t cell_id = 0; cell_id < accessor_.cells; cell_id++) {
      const std::size_t i = cell_id % nx;
      const std::size_t j = cell_id / nx;

      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 1));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 1));
    }

    fprintf(f, "CELLS %d %d\n", (int) accessor_.cells, (int) accessor_.cells * 5);

    for (std::size_t cell_id = 0; cell_id < accessor_.cells; cell_id++) {
      const std::size_t point_offset = cell_id * 4;
      fprintf(
        f,
        "4 %d %d %d %d\n",
        (int) (point_offset + 0),
        (int) (point_offset + 1),
        (int) (point_offset + 2),
        (int) (point_offset + 3));
    }

    fprintf(f, "CELL_TYPES %d\n", (int) accessor_.cells);

    for (std::size_t cell_id = 0; cell_id < accessor_.cells; cell_id++) {
      fprintf(f, "9\n");
    }

    fprintf(f, "CELL_DATA %d\n", (int) accessor_.cells);
    fprintf(f, "SCALARS Ez double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    for (std::size_t cell_id = 0; cell_id < accessor_.cells; cell_id++) {
      fprintf(f, "%lf\n", ez[cell_id]);
    }

    fclose(f);

    if (rank_ == 0) {
      printf(".\n");
      fflush(stdout);
    }
  }

 public:
  result_dumper_t(bool write_results, fields_accessor accessor)
    : write_results_(write_results)
    , accessor_(accessor) {
  }

  void operator()([[maybe_unused]] bool update_time = true) const {
    int rank_ = 0;
    const std::string filename = std::string("output_") + std::to_string(rank_) + "_"
                               + std::to_string(0) + ".vtk";

    write_vtk(filename);
  }
};

inline auto dump_vtk(bool write_results, fields_accessor accessor) -> result_dumper_t {
  return {write_results, accessor};
}

class time_storage_t {
  std::unique_ptr<float, deleter_t> time_{};

 public:
  explicit time_storage_t(bool gpu)
    : time_(allocate_on<float>(gpu)) {
  }

  [[nodiscard]]
  auto get() const -> float * {
    return time_.get();
  }
};

auto bin_name(int node_id) -> std::string {
  return "out_" + std::to_string(node_id) + ".bin";
}

inline void report_header() {
  std::cout << std::fixed << std::showpoint << std::setw(24) << "method"
            << ", " << std::setw(11) << "elapsed [s]"
            << ", " << std::setw(11) << "BW [GB/s]"
            << "\n";
}

void report_performance(
  std::size_t cells,
  std::size_t iterations,
  std::string_view method,
  double elapsed) {
  // Assume perfect locality
  const std::size_t memory_accesses_per_cell = 6ul * 2; // 8 + 9;
  const std::size_t memory_accesses = iterations * cells * memory_accesses_per_cell;
  const std::size_t bytes_accessed = memory_accesses * sizeof(float);

  const double bytes_per_second = static_cast<double>(bytes_accessed) / elapsed;
  const double gbytes_per_second = bytes_per_second / 1024 / 1024 / 1024;

  std::cout << std::setw(24) << method << ", " << std::setw(11) << std::setprecision(3) << elapsed
            << ", " << std::setw(11) << std::setprecision(3) << gbytes_per_second << std::endl;
}

template <class Action>
void report_performance(
  std::size_t cells,
  std::size_t iterations,
  std::string_view method,
  Action action) {
  auto begin = std::chrono::high_resolution_clock::now();
  action();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - begin).count();

  report_performance(cells, iterations, method, elapsed);
}

auto contains(std::string_view str, char c) -> bool {
  return str.find(c) != std::string_view::npos;
}

auto split(std::string_view str, char by = '=') -> std::pair<std::string_view, std::string_view> {
  auto it = str.find(by);
  return std::make_pair(str.substr(0, it), str.substr(it + 1, str.size() - it - 1));
}

[[nodiscard]]
auto parse_cmd(int argc, char *argv[]) -> std::map<std::string_view, std::size_t> {
  std::map<std::string_view, std::size_t> params;
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  for (auto arg: args) {
    if (arg.starts_with("--")) {
      arg = arg.substr(2, arg.size() - 2);
    }

    if (arg.starts_with("-")) {
      arg = arg.substr(1, arg.size() - 1);
    }

    if (contains(arg, '=')) {
      auto [name, value] = split(arg);
      std::from_chars(value.begin(), value.end(), params[name]);
    } else {
      params[arg] = 1;
    }
  }

  return params;
}

[[nodiscard]]
auto value(
  const std::map<std::string_view, std::size_t> &params,
  std::string_view name,
  std::size_t default_value = 0) -> std::size_t {
  if (params.count(name)) {
    return params.at(name);
  }
  return default_value;
}
