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
#include "nvexec/stream_context.cuh"

#include <mpi.h>
#include <vector>

static auto even_share(std::size_t n, std::size_t rank, std::size_t size) noexcept
  -> std::pair<std::size_t, std::size_t> {
  const auto avg_per_thread = n / size;
  const auto n_big_share = avg_per_thread + 1;
  const auto big_shares = n % size;
  const auto is_big_share = rank < big_shares;
  const auto begin = is_big_share ? n_big_share * rank
                                  : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
  const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

  return std::make_pair(begin, end);
}

template <class T>
auto device_alloc(std::size_t elements = 1) -> std::unique_ptr<T, deleter_t> {
  T *ptr{};
  STDEXEC_TRY_CUDA_API(cudaMalloc(reinterpret_cast<void **>(&ptr), elements * sizeof(T)));
  return std::unique_ptr<T, deleter_t>(ptr, deleter_t{true});
}

namespace distributed {
  struct fields_accessor {
    float dx;
    float dy;

    float width;
    float height;

    std::size_t n;
    std::size_t cells;
    std::size_t begin;
    std::size_t end;

    float *base_ptr;

    [[nodiscard]]
    __host__ __device__ auto own_cells() const -> std::size_t {
      return end - begin;
    }

    [[nodiscard]]
    __host__ __device__ auto get(field_id id) const -> float * {
      return base_ptr + static_cast<int>(id) * (own_cells() + 2 * n) + n;
    }
  };

  struct grid_t {
    float width = 160;
    float height = 160;

    std::size_t n{};
    std::size_t cells{};

    std::size_t begin{};
    std::size_t end{};
    std::size_t own_cells{};

    std::unique_ptr<float, deleter_t> fields_{};

    grid_t(grid_t &&) = delete;
    grid_t(const grid_t &) = delete;

    grid_t(std::size_t n, std::size_t grid_begin, std::size_t grid_end)
      : n(n)
      , cells(n * n)
      , begin(grid_begin)
      , end(grid_end)
      , own_cells(end - begin)
      , fields_(
          device_alloc<float>(
            static_cast<std::size_t>(own_cells + n * 2)
            * static_cast<int>(field_id::fields_count))) {
    }

    [[nodiscard]]
    auto accessor() const -> fields_accessor {
      return {height / n, width / n, width, height, n, cells, begin, end, fields_.get()};
    }
  };

  class result_dumper_t {
    bool write_results_{};
    std::size_t rank_{};
    std::size_t &report_step_;
    fields_accessor accessor_;

    bool with_halo_{};

    void write_vtk(const std::string &filename) const {
      if (!write_results_) {
        return;
      }

      float *ez = accessor_.get(field_id::ez);
      std::vector<float> h_ez(accessor_.own_cells() + 2 * accessor_.n);

      cudaMemcpy(
        h_ez.data(), accessor_.get(field_id::ez), sizeof(float) * h_ez.size(), cudaMemcpyDefault);
      ez = h_ez.data();

      if (rank_ == 0) {
        printf("\twriting report #%d", (int) report_step_);
        fflush(stdout);
      }

      FILE *f = fopen(filename.c_str(), "w");

      const std::size_t nx = accessor_.n;
      const float dx = accessor_.dx;
      const float dy = accessor_.dy;

      const std::size_t own_cells = accessor_.own_cells() + (with_halo_ ? 2 * accessor_.n : 0);

      fprintf(f, "# vtk DataFile Version 3.0\n");
      fprintf(f, "vtk output\n");
      fprintf(f, "ASCII\n");
      fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
      fprintf(f, "POINTS %d double\n", (int) (own_cells * 4));

      const float y_offset = with_halo_ ? dy : 0.0f;
      for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
        const std::size_t cell_id = own_cell_id + accessor_.begin;
        const std::size_t i = cell_id % nx;
        const std::size_t j = cell_id / nx;

        fprintf(
          f,
          "%lf %lf 0.0\n",
          dx * static_cast<float>(i + 0),
          dy * static_cast<float>(j + 0) - y_offset);
        fprintf(
          f,
          "%lf %lf 0.0\n",
          dx * static_cast<float>(i + 1),
          dy * static_cast<float>(j + 0) - y_offset);
        fprintf(
          f,
          "%lf %lf 0.0\n",
          dx * static_cast<float>(i + 1),
          dy * static_cast<float>(j + 1) - y_offset);
        fprintf(
          f,
          "%lf %lf 0.0\n",
          dx * static_cast<float>(i + 0),
          dy * static_cast<float>(j + 1) - y_offset);
      }

      fprintf(f, "CELLS %d %d\n", (int) own_cells, (int) own_cells * 5);

      for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
        const std::size_t point_offset = own_cell_id * 4;
        fprintf(
          f,
          "4 %d %d %d %d\n",
          (int) (point_offset + 0),
          (int) (point_offset + 1),
          (int) (point_offset + 2),
          (int) (point_offset + 3));
      }

      fprintf(f, "CELL_TYPES %d\n", (int) own_cells);

      for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
        fprintf(f, "9\n");
      }

      fprintf(f, "CELL_DATA %d\n", (int) own_cells);
      fprintf(f, "SCALARS Ez double 1\n");
      fprintf(f, "LOOKUP_TABLE default\n");

      for (std::size_t own_cell_id = 0; own_cell_id < own_cells; own_cell_id++) {
        fprintf(f, "%lf\n", ez[own_cell_id - (with_halo_ ? accessor_.n : 0)]);
      }

      fclose(f);

      if (rank_ == 0) {
        printf(".\n");
        fflush(stdout);
      }
    }

   public:
    result_dumper_t(
      bool write_results,
      int rank,
      std::size_t &report_step,
      fields_accessor accessor)
      : write_results_(write_results)
      , rank_(rank)
      , report_step_(report_step)
      , accessor_(accessor) {
    }

    void operator()() const {
      const std::string filename = std::string("output_") + std::to_string(rank_) + "_"
                                 + std::to_string(report_step_) + ".vtk";

      write_vtk(filename);
    }
  };

  __host__ auto
    dump_vtk(bool write_results, int rank, std::size_t &report_step, fields_accessor accessor)
      -> result_dumper_t {
    return {write_results, rank, report_step, accessor};
  }

  template <class AccessorT>
  struct grid_initializer_t {
    float dt;
    AccessorT accessor;

    __host__ __device__ void operator()(std::size_t cell_id) const {
      const std::size_t row = (accessor.begin + cell_id) / accessor.n;
      const std::size_t column = (accessor.begin + cell_id) % accessor.n;

      float er = 1.0f;
      float hr = 1.0f;

      const float x = static_cast<float>(column) * accessor.dx;
      const float y = static_cast<float>(row) * accessor.dy;

      const float soil_y = accessor.width / 2.2;
      const float object_y = soil_y - 22.0;
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

      accessor.get(field_id::hx)[cell_id] = {};
      accessor.get(field_id::hy)[cell_id] = {};

      accessor.get(field_id::ez)[cell_id] = {};
      accessor.get(field_id::dz)[cell_id] = {};

      accessor.get(field_id::mh)[cell_id] = C0 * dt / hr;
    }
  };

  template <class AccessorT>
  __host__ __device__ inline auto
    grid_initializer(float dt, AccessorT accessor) -> grid_initializer_t<AccessorT> {
    return {dt, accessor};
  }

  __host__ __device__ inline auto
    right_nid(std::size_t cell_id, std::size_t col, std::size_t N) -> std::size_t {
    return col == N - 1 ? cell_id - (N - 1) : cell_id + 1;
  }

  __host__ __device__ inline auto
    left_nid(std::size_t cell_id, std::size_t col, std::size_t N) -> std::size_t {
    return col == 0 ? cell_id + N - 1 : cell_id - 1;
  }

  __host__ __device__ inline auto
    bottom_nid(std::size_t cell_id, std::size_t row, std::size_t N) -> std::size_t {
    return cell_id - N;
  }

  __host__ __device__ inline auto
    top_nid(std::size_t cell_id, std::size_t row, std::size_t N) -> std::size_t {
    return cell_id + N;
  }

  template <class AccessorT>
  struct h_field_calculator_t {
    AccessorT accessor;

    __host__ __device__ void operator()(std::size_t cell_id) const __attribute__((always_inline)) {
      const std::size_t N = accessor.n;
      const std::size_t column = (accessor.begin + cell_id) % N;
      const std::size_t row = (accessor.begin + cell_id) / N;
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

  template <class AccessorT>
  __host__ __device__ inline auto update_h(AccessorT accessor) -> h_field_calculator_t<AccessorT> {
    return {accessor};
  }

  template <class AccessorT>
  struct e_field_calculator_t {
    float dt;
    float *time;
    AccessorT accessor;
    std::size_t source_position;

    [[nodiscard]]
    __host__ __device__ auto gaussian_pulse(float t, float t_0, float tau) const -> float {
      return exp(-(((t - t_0) / tau) * (t - t_0) / tau));
    }

    [[nodiscard]]
    __host__ __device__ auto calculate_source(float t, float frequency) const -> float {
      const float tau = 0.5f / frequency;
      const float t_0 = 6.0f * tau;
      return gaussian_pulse(t, t_0, tau);
    }

    __host__ __device__ void operator()(std::size_t cell_id) const __attribute__((always_inline)) {
      const std::size_t N = accessor.n;
      const std::size_t column = (accessor.begin + cell_id) % N;
      const std::size_t row = (accessor.begin + cell_id) / N;
      const bool source_owner = (accessor.begin + cell_id) == source_position;
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

  template <class AccessorT>
  __host__ __device__ inline auto
    update_e(float *time, float dt, AccessorT accessor) -> e_field_calculator_t<AccessorT> {
    std::size_t source_position = accessor.n / 2 + (accessor.n * (accessor.n / 2));
    return {dt, time, accessor, source_position};
  }
} // namespace distributed

// TODO Combine hz/hy in a float2 type to pass in a single MPI copy
auto main(int argc, char *argv[]) -> int {
  int rank{};
  int size{1};

  // Initialize MPI
  {
    int prov{};

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  std::cout << rank << " / " << size << std::endl;

  auto params = parse_cmd(argc, argv);

  if (value(params, "help") || value(params, "h")) {
    std::cout << "Usage: " << argv[0] << " [OPTION]...\n"
              << "\t--write-vtk\n"
              << "\t--iterations\n"
              << "\t--N\n"
              << std::endl;
    return 0;
  }

  const bool write_wtk = value(params, "write-vtk");
  const std::size_t n_iterations = value(params, "iterations", 1000);
  const std::size_t N = value(params, "N", 512);
  const auto [row_begin, row_end] = even_share(N, rank, size);
  const std::size_t rank_begin = row_begin * N;
  const std::size_t rank_end = row_end * N;
  distributed::grid_t grid{N, rank_begin, rank_end};

  auto accessor = grid.accessor();
  auto dt = calculate_dt(accessor.dx, accessor.dy);

  nvexec::stream_context stream_context{};
  nvexec::stream_scheduler gpu = stream_context.get_scheduler(nvexec::stream_priority::low);
  nvexec::stream_scheduler gpu_with_priority = stream_context
                                                 .get_scheduler(nvexec::stream_priority::high);

  time_storage_t time{true /* on gpu */};

  auto shift = [](std::size_t shift, auto action) {
    return [=](std::size_t cell_id) {
      action(shift + cell_id);
    };
  };

  ex::sync_wait(
    ex::schedule(gpu)
    | ex::bulk(ex::par, accessor.own_cells(), distributed::grid_initializer(dt, accessor)));

  const int prev_rank = rank == 0 ? size - 1 : rank - 1;
  const int next_rank = rank == (size - 1) ? 0 : rank + 1;

  auto exchange_hx = [&] {
    MPI_Request requests[2];
    MPI_Irecv(
      accessor.get(field_id::hx) - N, N, MPI_FLOAT, prev_rank, 0, MPI_COMM_WORLD, requests + 0);
    MPI_Isend(
      accessor.get(field_id::hx) + accessor.own_cells() - N,
      N,
      MPI_FLOAT,
      next_rank,
      0,
      MPI_COMM_WORLD,
      requests + 1);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  };

  auto exchange_ez = [&] {
    MPI_Request requests[2];
    MPI_Irecv(
      accessor.get(field_id::ez) + accessor.own_cells(),
      N,
      MPI_FLOAT,
      next_rank,
      0,
      MPI_COMM_WORLD,
      requests + 0);
    MPI_Isend(accessor.get(field_id::ez), N, MPI_FLOAT, prev_rank, 0, MPI_COMM_WORLD, requests + 1);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  };

  exchange_hx();
  exchange_ez();

  std::size_t report_step = 0;
  auto write = distributed::dump_vtk(write_wtk, rank, report_step, accessor);

  MPI_Barrier(MPI_COMM_WORLD);
  const auto begin = std::chrono::system_clock::now();

#if defined(OVERLAP)
  const std::size_t border_cells = N;
  const std::size_t bulk_cells = accessor.own_cells() - border_cells;

  auto border_h_update = distributed::update_h(accessor);
  auto bulk_h_update = shift(border_cells, distributed::update_h(accessor));

  auto border_e_update = shift(bulk_cells, distributed::update_e(time.get(), dt, accessor));
  auto bulk_e_update = distributed::update_e(time.get(), dt, accessor);

  for (std::size_t compute_step = 0; compute_step < n_iterations; compute_step++) {
    auto compute_h = ex::when_all(
      ex::just() | ex::on(gpu, ex::bulk(ex::par, bulk_cells, bulk_h_update)),
      ex::just() | ex::on(gpu_with_priority, ex::bulk(ex::par, border_cells, border_h_update))
        | ex::then(exchange_hx));

    auto compute_e = ex::when_all(
      ex::just() | ex::on(gpu, ex::bulk(ex::par, bulk_cells, bulk_e_update)),
      ex::just() | ex::on(gpu_with_priority, ex::bulk(ex::par, border_cells, border_e_update))
        | ex::then(exchange_ez));

    ex::sync_wait(std::move(compute_h));
    ex::sync_wait(std::move(compute_e));
  }

  write();
#else
  for (std::size_t compute_step = 0; compute_step < n_iterations; compute_step++) {
    auto compute_h =
      ex::just()
      | ex::on(gpu, ex::bulk(ex::par, accessor.own_cells(), distributed::update_h(accessor)))
      | ex::then(exchange_hx);

    auto compute_e =
      ex::just()
      | ex::on(
        gpu,
        ex::bulk(ex::par, accessor.own_cells(), distributed::update_e(time.get(), dt, accessor)))
      | ex::then(exchange_ez);

    ex::sync_wait(std::move(compute_h));
    ex::sync_wait(std::move(compute_e));
  }

  write();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  const auto end = std::chrono::system_clock::now();

  if (rank == 0) {
    const double elapsed = std::chrono::duration<double>(end - begin).count();

    report_header();
    report_performance(
      grid.cells,
      n_iterations,
      "GPU (distributed)",
      std::chrono::duration<double>(end - begin).count());
  }

  MPI_Finalize();
}
