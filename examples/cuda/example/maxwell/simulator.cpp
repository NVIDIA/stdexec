/*
 * Copyright (c) NVIDIA
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

#include <schedulers/graph_scheduler.hpp>
#include <schedulers/inline_scheduler.hpp>
#include <schedulers/openmp_scheduler.hpp>

#include <chrono>

namespace graph = example::cuda::graph;
namespace ex = std::execution;

namespace detail
{

template <class Sender, class Receiver>
struct operation_state_t
{
  Sender sender_;
  Receiver receiver_;
  std::size_t n_{};

  friend void
  tag_invoke(std::execution::start_t, operation_state_t &self) noexcept
  {
    for (std::size_t i = 0; i < self.n_; i++)
    {
      // Temporary solution
      std::this_thread::sync_wait(std::move(self.sender_));
    }

    std::execution::set_value(std::move(self.receiver_));
  }
};

template <class S>
struct repeat_n_sender_t
{
  using completion_signatures = std::execution::completion_signatures<
    std::execution::set_value_t(),
    std::execution::set_error_t(std::exception_ptr)>;

  S sender_;
  std::size_t n_{};

  template <std::__decays_to<repeat_n_sender_t> Self, class Receiver>
    requires std::tag_invocable<std::execution::connect_t, S, Receiver> friend auto
  tag_invoke(std::execution::connect_t, Self &&self, Receiver &&r)
  {
    return operation_state_t<S, Receiver>{
      std::move(self.sender_),
      std::forward<Receiver>(r),
      self.n_};
  }

  template <std::__none_of<std::execution::connect_t> Tag, class... Ts>
    requires std::tag_invocable<Tag, S, Ts...> friend decltype(auto)
  tag_invoke(Tag tag, const repeat_n_sender_t &s, Ts &&...ts) noexcept
  {
    return tag(s.sender_, std::forward<Ts>(ts)...);
  }
};

struct repeat_n_t
{
  template <graph::graph_sender _Sender, class _Counter>
  auto operator()(_Counter n, _Sender &&__sndr) const noexcept
  {
    return graph::repeat_n(n, std::forward<_Sender>(__sndr));
  }

  template <class _Sender>
  auto operator()(std::size_t n, _Sender &&__sndr) const noexcept
  {
    return repeat_n_sender_t<_Sender>{std::forward<_Sender>(__sndr), n};
  }
};

} // namespace __repeat_n_receiver

inline constexpr detail::repeat_n_t repeat_n{};

#define USE_ONLY_CPU 0

template <class T>
__host__ void custom_allocate(T **ptr, std::size_t size)
{
  printf("Started allocation of %zu bytes\n", size);
  fflush(stdout);

  if (USE_ONLY_CPU)
  {
    *ptr = reinterpret_cast<T *>(malloc(size));
  }
  else
  {
    cudaMallocManaged(ptr, size);
  }

  printf("Completed allocation of %zu bytes\n", size);
  fflush(stdout);
}

__host__ void custom_free(void *ptr)
{
  if (USE_ONLY_CPU)
  {
    free(ptr);
  }
  else
  {
    cudaFree(ptr);
  }
}

enum class field_id : int
{
  er,
  hr,
  mh,
  hx,
  hy,
  ez,
  dz,
  fields_count
};

template <int N>
struct column_accessor
{
  float *row_ptr;

  __host__ __device__ float &operator[](int column) const
  {
    if (column < 0)
    {
      return *(row_ptr + (N - 1));
    }
    else if (column >= N)
    {
      return *(row_ptr + 0);
    }

    return *(row_ptr + column);
  }
};

template <int N>
struct row_accessor
{
  float *field_ptr;

  __host__ __device__ column_accessor<N> operator[](int row) const
  {
    if (row < 0)
    {
      return {field_ptr + (N - 1) * N};
    }
    else if (row >= N)
    {
      return {field_ptr + 0 * N};
    }

    return {field_ptr + row * N};
  }
};

template <int N>
struct fields_accessor
{
  constexpr static float width = 160;
  constexpr static float height = 160;

  constexpr static float dx = width / N;
  constexpr static float dy = height / N;

  constexpr static int nx = N;
  constexpr static int ny = N;
  constexpr static int cells = nx * ny;

  float *base_ptr;

  [[nodiscard]] __host__ __device__ float &get(field_id id,
                                               int row,
                                               int column) const
  {
    return row_accessor<N>{get(id)}[row][column];
  }

  [[nodiscard]] __host__ __device__ float *get(field_id id) const
  {
    return base_ptr + static_cast<int>(id) * cells;
  }
};

template <int N>
struct grid_t
{
  constexpr static int nx = N;
  constexpr static int ny = N;
  constexpr static int cells = nx * ny;

  float *fields_{};

  grid_t(grid_t &&) = delete;
  grid_t(const grid_t &) = delete;

  grid_t()
  {
    custom_allocate(&fields_,
                    sizeof(float) * cells *
                      static_cast<int>(field_id::fields_count));
  }

  ~grid_t()
  {
    custom_free(fields_);
    fields_ = nullptr;
  }

  [[nodiscard]] fields_accessor<N> accessor() const { return {fields_}; }
};

constexpr float C0 = 299792458.0f; // Speed of light [metres per second]

__host__ __device__ bool is_circle_part(float x,
                                        float y,
                                        float object_x,
                                        float object_y,
                                        float object_size)
{
  return ((x - object_x) * (x - object_x) + (y - object_y) * (y - object_y) <=
          object_size * object_size);
}

__host__ __device__ float calculate_dt(float dx, float dy)
{
  const float cfl = 0.3;
  return cfl * std::min(dx, dy) / C0;
}

template <int N>
struct grid_initializer_t
{
  float dt;
  fields_accessor<N> accessor;

  __host__ __device__ void operator()(int cell_id) const
  {
    float er = 1.0f;
    float hr = 1.0f;

    const int row = cell_id / N;
    const int column = cell_id % N;

    const float x = static_cast<float>(column) * accessor.dx;
    const float y = static_cast<float>(row) * accessor.dy;

    const float soil_y = accessor.width / 2.2;
    const float object_y = soil_y - 22.0;
    const float object_size = 3.0;
    const float soil_er_hr = 1.3;

    if (y < soil_y)
    {
      const float middle_x = accessor.width / 2;
      const float object_x = middle_x;

      if (is_circle_part(x, y, object_x, object_y, object_size))
      {
        er = hr = 200000; /// Relative permeabuliti of Iron
      }
      else
      {
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

template <int N>
__host__ __device__ grid_initializer_t<N>
grid_initializer(float dt, fields_accessor<N> accessor)
{
  return {dt, accessor};
}

template <int N>
__host__ __device__ int right_nid(int cell_id, int col)
{
  return col == N - 1 ? 0 : cell_id + 1;
}

template <int N>
__host__ __device__ int top_nid(int cell_id, int row, int col)
{
  return row == N - 1 ? col : cell_id + N;
}

template <int N>
__host__ __device__ int left_nid(int cell_id, int col)
{
  return col == 0 ? cell_id + N - 1 : cell_id - 1;
}

template <int N>
__host__ __device__ int bottom_nid(int cell_id, int row, int col)
{
  return row == 0 ? (N - 1) * N + col : cell_id - N;
}

template <int N>
struct h_field_calculator_t
{
  fields_accessor<N> accessor;

  // read: ez, mh, hx, hy
  // write: hx, hy
  // total: 6
  __host__ __device__ void operator()(int cell_id) const
  {
    const float *ez = accessor.get(field_id::ez);
    float *hx = accessor.get(field_id::hx);
    float *hy = accessor.get(field_id::hy);

    const int row = cell_id / N;
    const int column = cell_id % N;
    const float cell_ez = ez[cell_id];
    const float neighbour_ex = ez[top_nid<N>(cell_id, row, column)];
    const float cex = (neighbour_ex - cell_ez) / accessor.dy;
    const float neighbour_ez = ez[right_nid<N>(cell_id, column)];
    const float cey = -(neighbour_ez - cell_ez) / accessor.dx;
    const float mh = accessor.get(field_id::mh)[cell_id];

    hx[cell_id] -= mh * cex;
    hy[cell_id] -= mh * cey;
  }
};

template <int N>
__host__ __device__ h_field_calculator_t<N>
update_h(fields_accessor<N> accessor)
{
  return {accessor};
}

template <int N>
struct e_field_calculator_t
{
  float dt;
  float *time;
  fields_accessor<N> accessor;
  int source_position;

  [[nodiscard]] __host__ __device__ float gaussian_pulse(float t,
                                                         float t_0,
                                                         float tau) const
  {
    return exp(-(((t - t_0) / tau) * (t - t_0) / tau));
  }

  [[nodiscard]] __host__ __device__ float
  calculate_source(float t, float frequency) const
  {
    const float tau = 0.5f / frequency;
    const float t_0 = 6.0f * tau;
    return gaussian_pulse(t, t_0, tau);
  }

  // reads: hx, hy, dz, er: 4
  // writes: dz, ez: 2
  // total: 6
  //
  // read: 7; write: 2; 9 memory accesses
  __host__ __device__ void operator()(int cell_id) const
  {
    float *dz = accessor.get(field_id::dz);
    float *ez = accessor.get(field_id::ez);
    float *er = accessor.get(field_id::er);
    float *hx = accessor.get(field_id::hx);
    float *hy = accessor.get(field_id::hy);

    const int row = cell_id / N;
    const int column = cell_id % N;

    const float chz =
      (hy[cell_id] - hy[left_nid<N>(cell_id, column)]) / accessor.dx -
      (hx[cell_id] - hx[bottom_nid<N>(cell_id, row, column)]) / accessor.dy;

    float cell_dz = dz[cell_id] + C0 * dt * chz;

    if (cell_id == source_position)
    {
      cell_dz += calculate_source(*time, 5E+7);
      *time += dt;
    }

    // read 2 values, write 1 value
    ez[cell_id] = cell_dz / er[cell_id];
    dz[cell_id] = cell_dz;
  }
};

template <int N>
__host__ __device__ e_field_calculator_t<N>
update_e(float *time,
         float dt,
         fields_accessor<N> accessor,
         int source_position = N / 2 + (N * (N / 2)))
{
  return {dt, time, accessor, source_position};
}

template <int N>
struct result_dumper_t
{
  bool write_results;
  int &report_step;
  fields_accessor<N> accessor;

  void write_vtk(const std::string &filename) const
  {
    if (!write_results)
      return;

    FILE *f = fopen(filename.c_str(), "w");

    const int nx = accessor.nx;
    const int ny = accessor.ny;

    const float dx = accessor.dx;
    const float dy = accessor.dy;

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "vtk output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(f, "POINTS %d double\n", nx * ny * 4);

    for (int j = 0; j < ny; j++)
    {
      for (int i = 0; i < nx; i++)
      {
        fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 0));
        fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 0));
        fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 1));
        fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 1));
      }
    }

    fprintf(f, "CELLS %d %d\n", nx * ny, nx * ny * 5);

    for (int j = 0; j < ny; j++)
    {
      for (int i = 0; i < nx; i++)
      {
        const int point_offset = (j * nx + i) * 4;
        fprintf(f,
                "4 %d %d %d %d\n",
                point_offset + 0,
                point_offset + 1,
                point_offset + 2,
                point_offset + 3);
      }
    }

    fprintf(f, "CELL_TYPES %d\n", nx * ny);
    for (int i = 0; i < nx * ny; i++)
    {
      fprintf(f, "9\n");
    }

    fprintf(f, "CELL_DATA %d\n", nx * ny);
    fprintf(f, "SCALARS Ez double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    const float *e = &accessor.get(field_id::ez, 0, 0);
    for (unsigned int i = 0; i < nx * ny; i++)
    {
      fprintf(f, "%lf\n", e[i]);
    }

    fclose(f);
  }

  void operator()() const
  {
    const std::string filename = std::string("output_") +
                                 std::to_string(report_step++) + ".vtk";

    write_vtk(filename);
  }
};

template <int N>
__host__ __device__ result_dumper_t<N>
dump_results(bool write_results, int &report_step, fields_accessor<N> accessor)
{
  return {write_results, report_step, accessor};
}

class time_storage_t
{
  float *time_{};

public:
  time_storage_t() { custom_allocate(&time_, sizeof(float)); }

  ~time_storage_t() { custom_free(time_); }

  [[nodiscard]] float *get() const { return time_; }
};

template <class GridAccessorT,
          std::execution::scheduler ComputeSchedulerT,
          std::execution::scheduler WriterSchedulerT>
auto maxwell_eqs(float dt,
                 float *time,
                 bool write_results,
                 int &report_step,
                 int n_inner_iterations,
                 int n_outer_iterations,
                 GridAccessorT &&accessor,
                 ComputeSchedulerT &&computer,
                 WriterSchedulerT &&writer)
{
  return repeat_n(                                                        //
           n_outer_iterations,                                            //
           repeat_n(                                                      //
             n_inner_iterations,                                          //
             ex::schedule(computer) |                                     //
               ex::bulk(accessor.cells, update_h(accessor)) |             //
               ex::bulk(accessor.cells, update_e(time, dt, accessor))     //
             ) |                                                          //
             ex::transfer(writer) |                                       //
             ex::then([] { printf("\tcomplete report step\n"); }) |       //
             ex::then(dump_results(write_results, report_step, accessor)) //
           ) |
         ex::then([] { printf("\tsimulation complete\n"); });
}

int main()
{
  graph::scheduler_t gpu_scheduler;
  example::openmp_scheduler cpu_scheduler{};

  int n_inner_iterations = 100;
  int n_outer_iterations = 10;

  constexpr int N = 512;

  time_storage_t time{};
  grid_t<N> grid{};

  auto accessor = grid.accessor();
  auto dt = calculate_dt(accessor.dx, accessor.dy);

  std::this_thread::sync_wait(
    ex::schedule(gpu_scheduler)
    | ex::bulk(grid.cells, grid_initializer(dt, accessor)));

  int report_step = 0;

  auto snd = maxwell_eqs(dt,
                         time.get(),
                         false,
                         report_step,
                         n_inner_iterations,
                         n_outer_iterations,
                         accessor,
                         gpu_scheduler,
                         cpu_scheduler);

  auto begin = std::chrono::high_resolution_clock::now();
  std::this_thread::sync_wait(std::move(snd));
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - begin).count();

  // Assume perfect locality
  const std::size_t memory_accesses_per_cell = 6 * 2; // 8 + 9;
  const std::size_t memory_accesses = grid.cells * memory_accesses_per_cell;
  const std::size_t bytes_accessed  = n_inner_iterations * n_outer_iterations
                                      * memory_accesses * sizeof(float);

  const double bytes_per_second = static_cast<double>(bytes_accessed) / elapsed;
  const double gbytes_per_second = bytes_per_second / 1024 / 1024 / 1024;

  printf(
    "\ncomputed %u cells in %gs (%g GB/s)\n",
    grid.cells,
    elapsed,
    gbytes_per_second);
}
