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
#include <schedulers/distributed_scheduler.hpp>
#include <schedulers/inline_scheduler.hpp>
#include <schedulers/openmp_scheduler.hpp>

#include <chrono>

namespace distributed = example::cuda::distributed;
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

#ifdef _NVHPC_CUDA
#include <nv/target>

__host__ __device__ inline device_type is_gpu()
{
  if target (nv::target::is_host)
  {
    return false;
  }
  else
  {
    return true;
  }
}
#elif defined(__clang__) && defined(__CUDA__)
__host__ inline bool is_gpu() { return false; }
__device__ inline bool is_gpu() { return true; }
#endif

template <class SchedulerT>
[[nodiscard]] bool is_gpu_scheduler(SchedulerT &&scheduler)
{
  auto [on_gpu] = std::this_thread::sync_wait(
                    ex::schedule(scheduler) |
                    ex::then([] __host__ __device__ { return is_gpu(); }))
                    .value();
  return on_gpu;
}

struct deleter_t
{
  bool on_gpu{};

  template <class T>
  void operator()(T *ptr)
  {
    if (on_gpu)
    {
      cudaFree(ptr);
    }
    else
    {
      free(ptr);
    }
  }
};

template <class T, class SchedulerT>
__host__ std::unique_ptr<T, deleter_t> allocate_on(SchedulerT &&scheduler,
                                                     std::size_t elements = 1)
{
  const bool gpu = is_gpu_scheduler(scheduler);

  T *ptr{};
  if (gpu)
  {
    cudaMallocManaged(&ptr, elements * sizeof(T));
  }
  else
  {
    ptr = reinterpret_cast<T *>(malloc(elements * sizeof(T)));
  }

  return std::unique_ptr<T, deleter_t>(ptr, deleter_t{gpu});
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
struct fields_accessor
{
  constexpr static float width = 160;
  constexpr static float height = 160;

  constexpr static float dx = width / N;
  constexpr static float dy = height / N;

  constexpr static int n = N;
  constexpr static int cells = N * N;

  int begin{};
  int end{};
  float *base_ptr;

  [[nodiscard]] __host__ __device__ int n_own_cells() const
  {
    return end - begin;
  }

  [[nodiscard]] __host__ __device__ float *get(field_id id) const
  {
    return base_ptr + static_cast<int>(id) * (n_own_cells() + 2 * N) + N;
  }
};

template <int N>
struct grid_t
{
  constexpr static int n = N;
  constexpr static int cells = N * N;

  int begin_{};
  int end_{};
  int n_own_cells_{};
  std::unique_ptr<float, deleter_t> fields_{};

  grid_t(grid_t &&) = delete;
  grid_t(const grid_t &) = delete;

  template <class SchedulerT>
  explicit grid_t(int begin,
                  int end,
                  SchedulerT &&scheduler)
      : begin_{begin}
      , end_{end}
      , n_own_cells_{end - begin}
      , fields_(allocate_on<float>(std::forward<SchedulerT>(scheduler),
                                   (n_own_cells_ + N * 2) *
                                     static_cast<int>(field_id::fields_count)))
  {}

  [[nodiscard]] fields_accessor<N> accessor() const
  {
    return {begin_, end_, fields_.get()};
  }
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
    const int row = cell_id / N;
    const int column = cell_id % N;
    cell_id -= accessor.begin;

    float er = 1.0f;
    float hr = 1.0f;

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
  return col == N - 1 ? cell_id - (N - 1) : cell_id + 1;
}

template <int N>
__host__ __device__ int left_nid(int cell_id, int col)
{
  return col == 0 ? cell_id + N - 1 : cell_id - 1;
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
    const int column = cell_id % N;
    cell_id -= accessor.begin;

    const float *ez = accessor.get(field_id::ez);
    float *hx = accessor.get(field_id::hx);
    float *hy = accessor.get(field_id::hy);

    const float cell_ez = ez[cell_id];
    const float neighbour_ex = ez[cell_id + N];
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
    const int column = cell_id % N;
    const bool source_owner = cell_id == source_position;
    cell_id -= accessor.begin;

    float *dz = accessor.get(field_id::dz);
    float *ez = accessor.get(field_id::ez);
    float *er = accessor.get(field_id::er);
    float *hx = accessor.get(field_id::hx);
    float *hy = accessor.get(field_id::hy);

    const float chz =
      (hy[cell_id] - hy[left_nid<N>(cell_id, column)]) / accessor.dx -
      (hx[cell_id] - hx[cell_id - N]) / accessor.dy;

    float cell_dz = dz[cell_id] + C0 * dt * chz;

    if (source_owner)
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
  bool write_results{};
  int rank{};
  int &report_step;
  fields_accessor<N> accessor;

  bool with_halo{};

  void write_vtk(const std::string &filename) const
  {
    if (!write_results)
      return;

    if (rank == 0)
    {
      printf("\twriting report #%d", report_step);
      fflush(stdout);
    }

    FILE *f = fopen(filename.c_str(), "w");

    const int nx = accessor.n;
    const float dx = accessor.dx;
    const float dy = accessor.dy;

    const int own_cells = accessor.n_own_cells() + (with_halo ? 2 * N : 0);

    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "vtk output\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(f, "POINTS %d double\n", own_cells * 4);

    const float y_offset = with_halo ? dy : 0.0f;
    for (int own_cell_id = 0; own_cell_id < own_cells; own_cell_id++)
    {
      const int cell_id = own_cell_id + accessor.begin;
      const int i = cell_id % nx;
      const int j = cell_id / nx;

      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 0) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 0) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 1), dy * static_cast<float>(j + 1) - y_offset);
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(i + 0), dy * static_cast<float>(j + 1) - y_offset);
    }

    fprintf(f, "CELLS %d %d\n", own_cells, own_cells * 5);

    for (int own_cell_id = 0; own_cell_id < own_cells; own_cell_id++)
    {
      const int point_offset = own_cell_id * 4;
      fprintf(f,
              "4 %d %d %d %d\n",
              point_offset + 0,
              point_offset + 1,
              point_offset + 2,
              point_offset + 3);
    }

    fprintf(f, "CELL_TYPES %d\n", own_cells);

    for (int own_cell_id = 0; own_cell_id < own_cells; own_cell_id++)
    {
      fprintf(f, "9\n");
    }

    fprintf(f, "CELL_DATA %d\n", own_cells);
    fprintf(f, "SCALARS Ez double 1\n");
    fprintf(f, "LOOKUP_TABLE default\n");

    const float *e = accessor.get(field_id::ez);
    for (int own_cell_id = 0; own_cell_id < own_cells; own_cell_id++)
    {
      fprintf(f, "%lf\n", e[own_cell_id - (with_halo ? N : 0)]);
    }

    fclose(f);

    if (rank == 0)
    {
      printf(".\n");
      fflush(stdout);
    }
  }

  void operator()() const
  {
    const std::string filename = std::string("output_") +
                                 std::to_string(rank) + "_" +
                                 std::to_string(report_step++) + ".vtk";

    write_vtk(filename);
  }
};

template <int N>
__host__ __device__ result_dumper_t<N> dump_results(bool write_results,
                                                    int rank,
                                                    int &report_step,
                                                    fields_accessor<N> accessor)
{
  return {write_results, rank, report_step, accessor};
}

class time_storage_t
{
  std::unique_ptr<float, deleter_t> time_{};

public:

  template <class SchedulerT>
  explicit time_storage_t(SchedulerT&& scheduler)
    : time_(allocate_on<float>(std::forward<SchedulerT>(scheduler)))
  {

  }

  [[nodiscard]] float *get() const { return time_.get(); }
};

namespace detail
{

struct halo_exchange_t
{
  template <distributed::distributed_sender _Sender,
            class Sh1,
            class Sh2,
            class... Ts,
            std::size_t... I>
  auto distributed_helper(_Sender &&sndr,
                          Sh1 border_size,
                          Sh2 field_size,
                          std::tuple<Ts...> tpl,
                          std::index_sequence<I...>) const noexcept
  {
    std::tuple<Ts...> prev_recv = std::make_tuple( (std::get<I>(tpl) - border_size)... );
    std::tuple<Ts...> prev_send = std::make_tuple( std::get<I>(tpl)... );
    std::tuple<Ts...> next_recv = std::make_tuple( (std::get<I>(tpl) + field_size)... );
    std::tuple<Ts...> next_send = std::make_tuple( (std::get<I>(tpl) + field_size - border_size) ... );

    return std::forward<_Sender>(sndr) |
           distributed::rotate_exchange(static_cast<int>(border_size),
                                        prev_recv,
                                        prev_send,
                                        next_recv,
                                        next_send);
  }

  template <distributed::distributed_sender _Sender, class Sh1, class Sh2, class... Ts>
  auto operator()(_Sender &&sndr,
                  Sh1 border_size,
                  Sh2 field_size,
                  std::tuple<Ts...> tpl) const noexcept
  {
    return distributed_helper(std::forward<_Sender>(sndr),
                              border_size,
                              field_size,
                              tpl,
                              std::make_index_sequence<sizeof...(Ts)>{});
  }

  template <class _Sender, class Sh1, class Sh2, class... Ts>
  auto operator()(_Sender &&sndr,
                  Sh1 border_size,
                  Sh2 field_size,
                  std::tuple<Ts...> tpl) const noexcept
  {
    return std::apply(
      [&](auto... pointers) {
        return ex::bulk(std::forward<_Sender>(sndr),
                        border_size,
                        [=] __host__ __device__(int i) {
                          ((pointers[i - border_size] = pointers[field_size - border_size + i]), ...);
                          ((pointers[field_size + i] = pointers[i]), ...);
                        });
      },
      tpl);
  }

  template <class Shape, class... Ts>
  std::execution::__binder_back<halo_exchange_t, Shape, Shape, std::tuple<Ts*...>>
  operator()(Shape border_size, Shape field_size, Ts*... fields) const
  {
    return {{}, {}, {border_size, field_size, std::make_tuple(fields...)}};
  }
};

}

inline constexpr detail::halo_exchange_t halo_exchange{};


template <class GridAccessorT,
          std::execution::scheduler ComputeSchedulerT,
          std::execution::scheduler WriterSchedulerT>
auto maxwell_eqs(float dt,
                 float *time,
                 bool write_results,
                 int node_id,
                 int &report_step,
                 int n_inner_iterations,
                 int n_outer_iterations,
                 GridAccessorT &&accessor,
                 ComputeSchedulerT &&computer,
                 WriterSchedulerT &&writer)
{
  const int border_size = accessor.n;
  const int own_cells = accessor.n_own_cells();

  auto write = dump_results(write_results, node_id, report_step, accessor);

  return repeat_n(                                                      //
           n_outer_iterations,                                          //
           repeat_n(                                                    //
             n_inner_iterations,                                        //
             ex::schedule(computer) |                                   //
               ex::bulk(accessor.cells, update_h(accessor)) |           //
               halo_exchange(border_size,                               //
                             own_cells,                                 //
                             accessor.get(field_id::hx),                //
                             accessor.get(field_id::hy)) |              //
               ex::bulk(accessor.cells, update_e(time, dt, accessor)) | //
               halo_exchange(border_size,                               //
                             own_cells,                                 //
                             accessor.get(field_id::ez),                //
                             accessor.get(field_id::dz))                //
             ) |                                                        //
             ex::transfer(writer) |                                     //
             ex::then(std::move(write))) |                              //
         ex::then([node_id] {
           if (node_id == 0) {
             printf("simulation complete");
           }
         });
}

template <class SenderT>
void report_performance(
  int cells,
  int iterations,
  int node_id,
  SenderT &&snd)
{
  auto begin = std::chrono::high_resolution_clock::now();
  std::this_thread::sync_wait(std::move(snd));
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration<double>(end - begin).count();

  // Assume perfect locality
  const std::size_t memory_accesses_per_cell = 6 * 2; // 8 + 9;
  const std::size_t memory_accesses = cells * memory_accesses_per_cell;
  const std::size_t bytes_accessed  = iterations * memory_accesses * sizeof(float);

  const double bytes_per_second = static_cast<double>(bytes_accessed) / elapsed;
  const double gbytes_per_second = bytes_per_second / 1024 / 1024 / 1024;

  if (node_id == 0)
  {
    printf(" in %gs (%g GB/s)\n", elapsed, gbytes_per_second);
  }
}

int main(int argc, char *argv[])
{
  const bool write_results = true;

  // graph::scheduler_t gpu_scheduler;
  distributed::scheduler_t gpu_scheduler(&argc, &argv);
  example::openmp_scheduler cpu_scheduler{};

  const int node_id = gpu_scheduler.node_id();

  int n_inner_iterations = 100;
  int n_outer_iterations = 10;

  constexpr int N = 512;
  auto [grid_begin, grid_end] = gpu_scheduler.bulk_range(N * N);

  time_storage_t time{gpu_scheduler};
  grid_t<N> grid{grid_begin, grid_end, gpu_scheduler};

  auto accessor = grid.accessor();
  auto dt = calculate_dt(accessor.dx, accessor.dy);

  std::this_thread::sync_wait(
    ex::schedule(gpu_scheduler) |
    ex::bulk(grid.cells, grid_initializer(dt, accessor)) |
    halo_exchange(accessor.n,
                  accessor.n_own_cells(),
                  accessor.get(field_id::er),
                  accessor.get(field_id::hr),
                  accessor.get(field_id::mh),
                  accessor.get(field_id::hx),
                  accessor.get(field_id::hy),
                  accessor.get(field_id::ez),
                  accessor.get(field_id::dz)));

  int report_step = 0;
  auto snd = maxwell_eqs(dt,
                         time.get(),
                         write_results,
                         node_id,
                         report_step,
                         n_inner_iterations,
                         n_outer_iterations,
                         accessor,
                         gpu_scheduler,
                         cpu_scheduler);

  report_performance(grid.cells,
                     n_inner_iterations * n_outer_iterations,
                     node_id,
                     std::move(snd));
}
