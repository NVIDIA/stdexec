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
#pragma once

#include <execution.hpp>
#include <numeric>
#include <vector>
#include <span>

#include <mpi.h>

#include <schedulers/detail/distributed/concepts.hpp>
#include <schedulers/detail/distributed/environment.hpp>
#include <schedulers/detail/distributed/pipeline_end.hpp>
#include <schedulers/detail/distributed/schedule_from.hpp>
#include <schedulers/detail/distributed/rotate_exchange.hpp>
#include <schedulers/detail/distributed/then.hpp>
#include <schedulers/detail/distributed/bulk.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

namespace example::cuda::distributed
{

class scheduler_t
{
  detail::context_t context_;

  class mpi_context_sanitizer_t
  {
    bool was_initialized_{};
  public:
    void set_initialized()
    {
      was_initialized_ = true;
    }

    ~mpi_context_sanitizer_t()
    {
      if (was_initialized_)
      {
        int finalized{};
        MPI_Finalized(&finalized);

        if (!finalized)
        {
          MPI_Finalize();
        }
      }
    }
  };

  mpi_context_sanitizer_t &get_sanitizer()
  {
    static mpi_context_sanitizer_t sanitizer{};
    return sanitizer;
  }

  explicit scheduler_t(detail::context_t context)
    : context_(std::move(context))
  {
  }

  detail::context_t make_context(int *argc, char ***argv, cudaStream_t stream)
  {
    int initialized {};
    MPI_Initialized(&initialized);

    if (initialized == 0)
    {
      MPI_Init(argc, argv);
      get_sanitizer().set_initialized();
    }

    int rank{};
    int size{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> devices_prefix(size + 1, 0);

    int local_devices{};
    cudaGetDeviceCount(&local_devices);

    int node_has_gpu = local_devices > 0;
    int each_node_has_gpu {};

    MPI_Allreduce(&node_has_gpu, &each_node_has_gpu, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!each_node_has_gpu)
    {
      throw std::runtime_error("Each node has to have a GPU");
    }

    return detail::context_t{rank, size, stream};
  }

public:
  template <class Receiver>
  struct operation_state_t
  {
    Receiver receiver_;

    friend auto tag_invoke(std::execution::start_t,
                           operation_state_t &self) noexcept
    {
      std::execution::set_value(std::move(self.receiver_));
    }
  };

  struct sender_t
  {
    const detail::context_t &context_;

    [[nodiscard]] scheduler_t construct_scheduler() const
    {
      return scheduler_t{context_};
    }

  public:
    using completion_signatures = std::execution::completion_signatures<
      std::execution::set_value_t(),
      std::execution::set_error_t(std::exception_ptr)>;

    // Root operation state
    template <class Receiver>
    friend operation_state_t<Receiver> tag_invoke(std::execution::connect_t,
                                                  const sender_t &,
                                                  Receiver &&r)
    {
      return {std::forward<Receiver>(r)};
    }

    sender_t() = delete;
    explicit sender_t(const detail::context_t &context)
        : context_(context)
    {}

    [[nodiscard]] cudaStream_t stream() const { return context_.stream_; }

    template <class CPO>
    friend scheduler_t
    tag_invoke(std::execution::get_completion_scheduler_t<CPO>,
               const sender_t &self) noexcept
    {
      return self.construct_scheduler();
    }

    friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                     const sender_t &s) noexcept
    {
      return cuda::storage_description_t{};
    }

    using value_t = cuda::variant<cuda::tuple<>>;

    static constexpr bool is_cuda_distributed_api = true;
  };

  scheduler_t(int *argc, char ***argv, cudaStream_t stream = 0) noexcept
      : context_{make_context(argc, argv, stream)}
  {}

  bool operator==(const scheduler_t &other) const
  {
    return context_.rank_ == other.context_.rank_ &&
           context_.size_ == other.context_.size_ &&
           context_.stream_ == other.context_.stream_;
  }

  [[nodiscard]] cudaStream_t stream() const { return context_.stream_; };
  [[nodiscard]] const detail::context_t &context() const { return context_; };

  template <class S, class F>
  friend detail::then::sender_t<S, F> tag_invoke(std::execution::then_t,
                                                 const scheduler_t &,
                                                 S &&self,
                                                 F f) noexcept
  {
    return detail::then::sender_t<S, F>{std::forward<S>(self), f};
  }

  template <class S, std::integral Shape, class F>
  friend detail::bulk::sender_t<S, Shape, F> tag_invoke(std::execution::bulk_t,
                                                        const scheduler_t &,
                                                        S &&self,
                                                        Shape &&shape,
                                                        F f) noexcept
  {
    return detail::bulk::sender_t{std::forward<S>(self),
                                  std::forward<Shape>(shape),
                                  f};
  }

  template <class S, class Scheduler>
  friend auto tag_invoke(std::execution::transfer_t,
                         const scheduler_t &self,
                         S &&sndr,
                         Scheduler &&sched)
  {
    return std::execution::schedule_from(
      std::forward<Scheduler>(sched),
      detail::pipeline_end::sender_t<S>{std::forward<S>(sndr), self.context_});
  }

  template <class S>
  friend detail::schedule_from::sender_t<scheduler_t, S>
  tag_invoke(std::execution::schedule_from_t,
             const scheduler_t &self,
             S &&sndr) noexcept
  {
    return {{std::forward<S>(sndr)}, {self}};
  }

  template <distributed_sender S>
  friend auto tag_invoke(std::this_thread::sync_wait_t,
                         const scheduler_t &sched,
                         S &&self)
  {
    return std::this_thread::sync_wait(
      detail::pipeline_end::sender_t<S>{std::forward<S>(self), sched.context_});
  }

  friend std::execution::forward_progress_guarantee
  tag_invoke(std::execution::get_forward_progress_guarantee_t,
             const scheduler_t &) noexcept
  {
    return std::execution::forward_progress_guarantee::parallel;
  }

  [[nodiscard]] sender_t schedule() const noexcept
  {
    return sender_t{context_};
  }

  friend inline sender_t tag_invoke(std::execution::schedule_t,
                                    const scheduler_t &scheduler) noexcept
  {
    return scheduler.schedule();
  }

  [[nodiscard]] int node_id() const
  {
    return context_.rank_;
  }

  [[nodiscard]] int n_nodes() const
  {
    return context_.size_;
  }

  template <class T>
  [[nodiscard]] std::pair<T, T> bulk_range(T n) const
  {
    return even_share(n, context_.rank_, context_.size_);
  }

  static constexpr bool is_cuda_distributed_api = true;
};

} // namespace example::cuda::distributed
