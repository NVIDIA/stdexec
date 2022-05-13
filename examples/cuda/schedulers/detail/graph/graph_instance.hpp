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
#include <schedulers/detail/helpers.hpp>
#include <schedulers/detail/tuple.hpp>

#include <cstdlib>

namespace example::cuda::graph::detail
{

struct graph_instance_t
{
  cudaGraph_t graph_;
  cudaStream_t stream_;
  cudaGraphExec_t instance_;

  explicit graph_instance_t(cudaGraph_t graph, cudaStream_t stream)
      : graph_{graph}
      , stream_{stream}
      , instance_{nullptr}
  {
    check(cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0));
  }

  graph_instance_t(graph_instance_t &&other)
      : graph_{other.graph_}
      , stream_{other.stream_}
      , instance_{other.instance_}
  {
    other.instance_ = nullptr;
  }

  ~graph_instance_t()
  {
    if (instance_)
    {
      cudaGraphExecDestroy(instance_);
      instance_ = nullptr;
    }
  }

  void launch() const
  {
    check(cudaGraphLaunch(instance_, stream_));
  }
};

class graph_info_t
{
  cudaGraph_t graph_{};
  cudaStream_t stream_{};

public:
  graph_info_t() = default;

  graph_info_t(cudaGraph_t graph, cudaStream_t stream)
      : graph_{graph}
      , stream_{stream}
  {}

  [[nodiscard]] graph_instance_t instantiate() const
  {
    return graph_instance_t{graph_, stream_};
  }

  [[nodiscard]] cudaGraph_t get() const { return graph_; }
};

struct graph_t
{
  cudaStream_t stream_;
  cudaGraph_t graph_;

  explicit graph_t(cudaStream_t stream)
      : stream_(stream)
  {
    cudaGraphCreate(&graph_, 0);
  }

  graph_t(graph_t &&other)
      : stream_(other.stream_)
      , graph_(other.graph_)
  {
    other.graph_ = nullptr;
  }

  ~graph_t()
  {
    if (graph_)
    {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }

  void sync() const
  {
    check(cudaStreamSynchronize(stream_));
  }

  [[nodiscard]] graph_instance_t instantiate() const
  {
    return graph_instance_t{graph_, stream_};
  }

  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
};

inline constexpr struct get_predecessors_t
{
  template <std::execution::operation_state OpState>
    requires std::tag_invocable<get_predecessors_t, OpState>
  constexpr auto operator()(const OpState &op_state) const noexcept
  {
    return std::tag_invoke(get_predecessors_t{}, op_state);
  }

  template <std::execution::operation_state OpState>
  constexpr std::span<cudaGraphNode_t> operator()(const OpState &op_state) const noexcept
  {
    return {};
  }
} get_predecessors{};

inline constexpr struct get_graph_t
{
  template <class EnvT>
    requires std::tag_invocable<get_graph_t, EnvT>
  constexpr auto operator()(EnvT &&env) const noexcept
  {
    return std::tag_invoke(get_graph_t{}, std::forward<EnvT>(env));
  }
} get_graph{};

} // namespace graph
