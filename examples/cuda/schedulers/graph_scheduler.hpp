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
// #include <schedulers/detail/graph/repeat_n.hpp>
#include <schedulers/detail/graph/pipeline_end.hpp>
#include <schedulers/detail/graph/schedule_from.hpp>
// #include <schedulers/detail/graph/ensure_started.hpp>
#include <schedulers/detail/graph/kernel.hpp>
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
  scheduler_t(cudaStream_t stream = 0) noexcept
      : stream_(stream)
  {}

  bool operator==(const scheduler_t &) const = default;

  [[nodiscard]] cudaStream_t stream() const { return stream_; };

  template <class S, class F>
  friend detail::kernel::then_sender_t<S, int, F> tag_invoke(std::execution::then_t,
                                                             const scheduler_t &,
                                                             S &&self,
                                                             F f) noexcept
  {
    return detail::kernel::then_sender_t<S, int, F>(std::forward<S>(self), 1, f);
  }

  template <class S, std::integral Shape, class F>
  friend detail::kernel::bulk_sender_t<S, Shape, F> tag_invoke(std::execution::bulk_t,
                                                               const scheduler_t &,
                                                               S &&self,
                                                               Shape &&shape,
                                                               F f) noexcept
  {
    return detail::kernel::bulk_sender_t<S, Shape, F>{std::forward<S>(self),
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

  // TODO: think on this
  template <class S>
  friend void // detail::ensure_started::sender_t<S>
  tag_invoke(std::execution::ensure_started_t,
             const scheduler_t &sched,
             S &&sndr) noexcept = delete;
  //{
    //return detail::ensure_started::sender_t<S>{sched.stream(),
                                               //std::forward<S>(sndr)};
  //}

  // TODO: figure this one out
  //template <std::__one_of<std::execution::let_value_t,
                          //std::execution::let_error_t,
                          //std::execution::let_stopped_t> Tag,
            //graph_sender S,
            //class F>
  //friend auto tag_invoke(Tag tag,
                         //const scheduler_t &sched,
                         //S &&sndr,
                         //F fun) noexcept
  //{
    //return tag(
      //detail::pipeline_end::sender_t<S>{std::forward<S>(sndr), sched.stream()},
      //[&](auto &&...args) -> detail::pipeline_end::sender_t<
                            //std::invoke_result_t<F, decltype(args)...>> {
        //auto s = fun(args...);
        //return detail::pipeline_end::sender_t<std::decay_t<decltype(s)>>{
          //std::move(s),
          //sched.stream()};
      //});
  //}

  // TODO: this should reuse bits of pipeline_end's connect and directly cudaEventSynchronize
  template <class S>
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

  [[nodiscard]] detail::schedule_from::sender_t<scheduler_t, decltype(std::execution::just())> schedule() const noexcept
  {
    return {std::execution::just(), {*this}};
  }

  friend inline auto tag_invoke(std::execution::schedule_t,
                                    const scheduler_t &scheduler) noexcept
  {
    return scheduler.schedule();
  }

  static constexpr bool is_cuda_graph_api = true;
};

} // namespace example::cuda::graph
