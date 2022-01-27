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

  static constexpr bool is_cuda_graph_api = true;
};

class graph_info_t
{
  bool is_launched_multiple_times_{};
  cudaGraph_t graph_{};

public:
  graph_info_t() = default;

  graph_info_t(cudaGraph_t graph, bool is_launched_multiple_times = false)
      : is_launched_multiple_times_{is_launched_multiple_times}
      , graph_{graph}
  {}

  [[nodiscard]] cudaGraph_t get() const { return graph_; }

  [[nodiscard]] bool is_multi_launch() const
  {
    return is_launched_multiple_times_;
  }
};

struct graph_t
{
  cudaStream_t stream_;
  cudaGraph_t graph_;
  cudaGraphNode_t node_;

  explicit graph_t(cudaStream_t stream)
      : stream_(stream)
  {
    cudaGraphCreate(&graph_, 0);
  }

  graph_t(graph_t &&other)
      : stream_(other.stream_)
      , graph_(other.graph_)
      , node_(other.node_)
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

  [[nodiscard]] graph_instance_t instantiate() const
  {
    return graph_instance_t{graph_, stream_};
  }

  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }

  static constexpr bool is_cuda_graph_api = true;
};

} // namespace graph
