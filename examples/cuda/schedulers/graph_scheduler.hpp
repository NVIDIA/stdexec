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
#include <span>

#include <schedulers/detail/graph/consumer.hpp>
#include <schedulers/detail/graph/pipeline_end.hpp>
#include <schedulers/detail/graph/schedule_from.hpp>
#include <schedulers/detail/graph/ensure_started.hpp>
#include <schedulers/detail/graph/bulk.hpp>
#include <schedulers/detail/graph/then.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

namespace example::cuda::graph
{

class scheduler_t
{
  cudaStream_t stream_{};

public:
  template <class Receiver>
  struct operation_state_t
  {
    Receiver receiver_;

    friend auto tag_invoke(std::execution::start_t,
                           operation_state_t &self) noexcept
    {
      if constexpr(graph_receiver<std::decay_t<Receiver>>)
      {
        auto consumer = self.receiver_.get_consumer();
        consumer(detail::thread_id_t{}, detail::block_id_t{});

        std::execution::set_value(std::move(self.receiver_),
                                  std::span<cudaGraphNode_t>{});
      }
      else
      {
        std::execution::set_value(std::move(self.receiver_));
      }
    }
  };

  struct sender_t
  {
    cudaStream_t stream_{};

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
    explicit sender_t(cudaStream_t stream)
        : stream_(stream)
    {}

    [[nodiscard]] cudaStream_t stream() const { return stream_; }

    template <class CPO>
    friend scheduler_t
    tag_invoke(std::execution::get_completion_scheduler_t<CPO>,
               const sender_t &self) noexcept
    {
      return scheduler_t{self.stream()};
    }

    friend constexpr auto tag_invoke(cuda::storage_requirements_t,
                                     const sender_t &s) noexcept
    {
      return cuda::storage_description_t{};
    }

    using value_t = cuda::variant<cuda::tuple<>>;

    static constexpr bool is_gpu_pipeline = true;
    static constexpr bool is_cuda_graph_api = true;
  };

  scheduler_t(cudaStream_t stream = 0) noexcept
      : stream_(stream)
  {}

  bool operator==(const scheduler_t &) const = default;

  [[nodiscard]] cudaStream_t stream() const { return stream_; };

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

  template <class S>
  friend detail::schedule_from::sender_t<scheduler_t, S>
  tag_invoke(std::execution::schedule_from_t,
             const scheduler_t &self,
             S &&sndr) noexcept
  {
    return {{std::forward<S>(sndr)}, {self}};
  }

  template <class S, class Scheduler>
  friend auto tag_invoke(std::execution::transfer_t,
                         const scheduler_t &self,
                         S &&sndr,
                         Scheduler &&sched)
  {
    return std::execution::schedule_from(
      std::forward<Scheduler>(sched),
      detail::pipeline_end::sender_t<S>{std::forward<S>(sndr), self.stream()});
  }

  template <graph_sender S>
  friend detail::ensure_started::sender_t<S>
  tag_invoke(std::execution::ensure_started_t,
             const scheduler_t &sched,
             S &&sndr) noexcept
  {
    return detail::ensure_started::sender_t<S>{sched.stream(),
                                               std::forward<S>(sndr)};
  }

  template <graph_sender S>
  friend auto tag_invoke(std::this_thread::sync_wait_t,
                         const scheduler_t &sched,
                         S &&self)
  {
    return std::this_thread::sync_wait(
      detail::pipeline_end::sender_t<S>{std::forward<S>(self), sched.stream()});
  }

  friend std::execution::forward_progress_guarantee
  tag_invoke(std::execution::get_forward_progress_guarantee_t,
             const scheduler_t &) noexcept
  {
    return std::execution::forward_progress_guarantee::parallel;
  }

  [[nodiscard]] sender_t schedule() const noexcept
  {
    return sender_t{stream()};
  }

  friend inline sender_t tag_invoke(std::execution::schedule_t,
                                    const scheduler_t &scheduler) noexcept
  {
    return scheduler.schedule();
  }

  static constexpr bool is_cuda_graph_api = true;
};

} // namespace example::cuda::graph
