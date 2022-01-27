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

#include <schedulers/detail/graph/concepts.hpp>
#include <schedulers/detail/graph/consumer.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>
#include <schedulers/detail/graph/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <execution.hpp>
#include <type_traits>
#include <span>

namespace example::cuda::graph::detail::schedule_from
{

template <class R>
struct receiver_t
{
  R receiver_;

  template <class... Ts>
  friend void
  tag_invoke(std::execution::set_value_t, receiver_t &&self, Ts&&... ts) noexcept
  {
    auto consumer = self.receiver_.get_consumer();
    consumer(
      thread_id_t{},
      block_id_t{},
      std::forward<Ts>(ts)...);

    std::execution::set_value(
      std::move(self.receiver_),
      std::span<cudaGraphNode_t>{});
  }

  friend void tag_invoke(
    std::execution::set_error_t,
    receiver_t &&r,
    std::exception_ptr ex_ptr) noexcept
  {
    std::execution::set_error(std::move(r.receiver_), ex_ptr);
  }

  friend void tag_invoke(std::execution::set_stopped_t, receiver_t &&r) noexcept
  {
    std::execution::set_stopped(std::move(r.receiver_));
  }

  graph_info_t graph() noexcept { return receiver_.graph(); }

  template <std::__none_of<std::execution::set_value_t> Tag, class... Ts>
  requires std::invocable<Tag, const R &, Ts...> friend decltype(auto)
  tag_invoke(Tag tag, const receiver_t &self, Ts &&...ts) noexcept
  {
    return ((Tag &&) tag)(std::as_const(self.receiver_), (Ts &&) ts...);
  }

  static constexpr bool is_cuda_graph_api = true;
};

template <class Scheduler, class S>
struct sender_t : sender_base_t<sender_t<Scheduler, S>, S>
{
  Scheduler scheduler_;

  template <class Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<Receiver>{std::forward<Receiver>(receiver)});
  }

  template <class CPO>
  friend Scheduler tag_invoke(
    std::execution::get_completion_scheduler_t<CPO>,
    const sender_t &self) noexcept
  {
    return self.scheduler_;
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto
  tag_invoke(std::execution::get_completion_signatures_t, Self &&, _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>, _Env>;

  using value_t = std::execution::
    value_types_of_t<S, std::execution::no_env, cuda::tuple, cuda::variant>;
};

} // namespace graph::schedule_from
