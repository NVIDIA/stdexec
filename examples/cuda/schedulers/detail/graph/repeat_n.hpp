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
#include <schedulers/detail/graph/environment.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>
#include <schedulers/detail/graph/sender_base.hpp>
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/storage.hpp>
#include <schedulers/detail/tuple.hpp>
#include <schedulers/detail/variant.hpp>

#include <execution.hpp>
#include <span>

namespace example::cuda::graph
{

namespace detail::repeat_n
{

template <class Receiver>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<Receiver>, Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<Receiver>, Receiver>;
  friend super_t;

  graph_t graph_;
  std::size_t n_;

  void set_value(std::span<cudaGraphNode_t> predecessors) &&noexcept
  try
  {
    auto instance = graph_.instantiate();

    for (std::size_t i = 0; i < n_; i++)
    {
      instance.launch();
    }

    auto consumer = this->base().get_consumer();
    consumer(thread_id_t{}, block_id_t{});

    std::execution::set_value(std::move(this->base()),
                              std::span<cudaGraphNode_t>{});
  }
  catch (...)
  {
    std::execution::set_error(std::move(this->base()),
                              std::current_exception());
  }

  void set_error(std::exception_ptr ex_ptr) &&noexcept
  {
    std::execution::set_error(std::move(this->base()), ex_ptr);
  }

  void set_stopped() &&noexcept
  {
    std::execution::set_stopped(std::move(this->base()));
  }

public:
  receiver_t(Receiver receiver, cudaStream_t stream, std::size_t n)
      : super_t{std::move(receiver)}
      , graph_(stream)
      , n_(n)
  {}

  friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
    -> detail::env_t<std::execution::env_of_t<Receiver>>
  {
    auto base_env = std::execution::get_env(self.base());
    return detail::env_t<std::execution::env_of_t<Receiver>>{
      base_env,
      cuda::get_storage(base_env),
      self.graph_.graph()};
  }

  [[nodiscard]] sink_consumer_t get_consumer() const noexcept { return {}; }

  static constexpr bool is_cuda_graph_api = true;
};

template <graph_sender S, class Shape>
struct sender_t : sender_base_t<sender_t<S, Shape>, S>
{
  using completion_signatures = std::execution::completion_signatures<
    std::execution::set_value_t(),
    std::execution::set_error_t(std::exception_ptr)>;

  using value_t = value_of_t<S>;
  using super_t = sender_base_t<sender_t<S, Shape>, S>;
  friend super_t;

  cudaStream_t stream_;
  Shape shape_;

  template <graph_receiver Receiver>
  auto connect(Receiver &&receiver) &&noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<Receiver>{std::forward<Receiver>(receiver),
                           stream_,
                           static_cast<std::size_t>(shape_)});
  }

  explicit sender_t(S sender, cudaStream_t stream, Shape shape)
      : super_t{std::forward<S>(sender)}
      , stream_{stream}
      , shape_{shape}
  {}
};


struct repeat_n_t
{
  template <class Shape, graph_sender Sender>
  auto operator()(Shape n, Sender &&sndr) const noexcept
  {
    auto sched =
      std::execution::get_completion_scheduler<std::execution::set_value_t>(
        std::as_const(sndr));

    return sender_t<Sender, Shape>{std::forward<Sender>(sndr),
                                   sched.stream(),
                                   n};
  }
};

} // namespace repeat_n

inline constexpr detail::repeat_n::repeat_n_t repeat_n{};

} // namespace example::cuda::graph
