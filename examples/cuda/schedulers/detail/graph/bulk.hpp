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
#include <span>

namespace example::cuda::graph::detail::bulk
{

template <class Shape,
          class F,
          class ConsumerT,
          class ArgsTuple,
          unsigned int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void bulk_kernel(Shape shape,
                                                            F f,
                                                            ConsumerT consumer,
                                                            ArgsTuple *args)
{
  detail::invoke(
    [=] __device__(auto... args) {
      const unsigned int i = blockIdx.x * BlockThreads + threadIdx.x;

      if (i < static_cast<unsigned int>(shape))
      {
        f(static_cast<Shape>(i), args...);
      }

      consumer(thread_id_t{threadIdx.x}, block_id_t{blockIdx.x}, args...);
    },
    *args);
}

template <class InTuple, class Receiver, class Shape, class F>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, Shape, F>,
                                       Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, Shape, F>,
                                     Receiver>;
  friend super_t;

  Shape shape_;
  F function_;

  void set_value(std::span<cudaGraphNode_t> predecessors) && noexcept try
  {
    graph_env auto env = std::execution::get_env(this->base());

    std::byte *storage_ptr = cuda::get_storage(env);

    auto consumer = this->base().get_consumer();
    using consumer_t = std::decay_t<decltype(consumer)>;

    void *kernel_args[4] = {&shape_, &function_, &consumer, &storage_ptr};

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size =
      (static_cast<unsigned int>(shape_) + block_size - 1) / block_size;

    cudaKernelNodeParams kernel_node_params{};
    kernel_node_params.func =
      (void *)bulk_kernel<Shape, F, consumer_t, InTuple, block_size>;
    kernel_node_params.gridDim = dim3(grid_size, 1, 1);
    kernel_node_params.blockDim = dim3(block_size, 1, 1);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = kernel_args;
    kernel_node_params.extra = nullptr;

    cudaGraph_t graph = env.graph().get();

    std::array<cudaGraphNode_t, 1> node{};
    check(cudaGraphAddKernelNode(node.data(),
                                 graph,
                                 predecessors.data(),
                                 predecessors.size(),
                                 &kernel_node_params));

    std::execution::set_value(std::move(this->base()),
                              std::span<cudaGraphNode_t>{node});
  } catch(...) {
    std::execution::set_error(std::move(this->base()),
                              std::current_exception());
  }

  void set_error(std::exception_ptr ex_ptr) && noexcept
  {
    std::execution::set_error(std::move(this->base()), ex_ptr);
  }

  void set_stopped() && noexcept
  {
    std::execution::set_stopped(std::move(this->base()));
  }

public:
  explicit receiver_t(Receiver receiver, Shape shape, F function)
    : super_t(std::move(receiver))
    , shape_(shape)
    , function_(function)
  {}

  [[nodiscard]] consumer_t get_consumer() const noexcept
  {
    return {cuda::get_storage(std::execution::get_env(this->base()))};
  }

  static constexpr bool is_cuda_graph_api = true;
};

template <graph_sender S, std::integral Shape, class F>
struct sender_t : sender_base_t<sender_t<S, Shape, F>, S>
{
  using value_t = value_of_t<S>;

  Shape shape_;
  F function_;

  template <graph_receiver Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<value_t, Receiver, Shape, std::decay_t<F>>{std::forward<Receiver>(
                                                       receiver),
                                                     shape_,
                                                     function_});
  }

  template <std::__decays_to<sender_t> Self, class _Env>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  explicit sender_t(S sender, Shape shape, F function)
    : sender_base_t<sender_t<S, Shape, F>, S>{std::forward<S>(sender)}
    , shape_{shape}
    , function_{function}
  {}
};
} // namespace example::cuda::graph::detail::bulk
