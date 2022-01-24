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

namespace example::cuda::graph::detail::then
{

template <class F, class ConsumerT, class ArgsTuple>
__global__ __launch_bounds__(1) void then_kernel(F f,
                                                 ConsumerT consumer,
                                                 ArgsTuple *args)
{
  detail::invoke(
    [=] __device__(auto... args) {
      if constexpr (std::is_void_v<std::invoke_result_t<F, decltype(args)...>>)
      {
        f(args...);
        consumer(thread_id_t{}, block_id_t{});
      }
      else
      {
        consumer(thread_id_t{}, block_id_t{}, f(args...));
      }
    },
    *args);
}

template <class InTuple, class Receiver, class F>
class receiver_t
    : std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, F>,
                                       Receiver>
{
  using super_t =
    std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, F>, Receiver>;
  friend super_t;

  F function_;

  void set_value(std::span<cudaGraphNode_t> predecessors) && noexcept try
  {
    std::byte *storage_ptr = cuda::get_storage(this->base());
    auto consumer = this->base().get_consumer();
    using consumer_t = std::decay_t<decltype(consumer)>;

    void *kernel_args[3] = {&function_, &consumer, &storage_ptr};

    cudaKernelNodeParams kernel_node_params{};
    kernel_node_params.func = (void *)then_kernel<F, consumer_t, InTuple>;
    kernel_node_params.gridDim = dim3(1, 1, 1);
    kernel_node_params.blockDim = dim3(1, 1, 1);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = kernel_args;
    kernel_node_params.extra = nullptr;

    cudaGraph_t graph = this->base().graph().get();

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
  explicit receiver_t(Receiver receiver, F function)
    : std::execution::receiver_adaptor<receiver_t<InTuple, Receiver, F>,
    Receiver>(std::move(receiver))
    , function_(function)
  {}

  [[nodiscard]] graph_info_t graph() noexcept
  {
    return this->base().graph();
  }

  [[nodiscard]] consumer_t get_consumer() const noexcept
  {
    return {cuda::get_storage(this->base())};
    }

  static constexpr bool is_cuda_graph_api = true;
};

template <graph_sender S, class F>
struct sender_t : sender_base_t<sender_t<S, F>, S>
{
  using arguments_t = value_of_t<S>;
  using value_t = cuda::apply_t<F, arguments_t>;
  using super_t = sender_base_t<sender_t<S, F>, S>;
  friend super_t;

  F function_;

  template <graph_receiver Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    return std::execution::connect(
      std::move(this->sender_),
      receiver_t<arguments_t, Receiver, F>{std::forward<Receiver>(receiver),
                                           function_});
  }

  template <class Result>
  using set_value_ = std::__minvoke1<
    std::__uncurry<std::__qf<std::execution::set_value_t>>,
    std::__if<std::is_void<Result>, std::__types<>, std::__types<Result>>>;
  template <class... Args>
  requires std::invocable<F, Args...> using result =
    set_value_<std::invoke_result_t<F, Args...>>;

  template <class EnvT>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         const sender_t &,
                         EnvT)
    -> std::execution::make_completion_signatures<
      S, EnvT, std::execution::__with_exception_ptr, result>;

  explicit sender_t(S sender, F function)
    : super_t{std::forward<S>(sender)}
    , function_{function}
  {}

  auto storage_requirements() const noexcept
  {
    auto predecessor_requirements = cuda::storage_requirements(this->sender_);
    using self_requirement_t = cuda::static_storage_from<value_t>;
    cuda::storage_description_t self_requirement{self_requirement_t::alignment,
                                                 self_requirement_t::size};

    const std::size_t alignment = std::max(self_requirement.alignment,
                                           predecessor_requirements.alignment);

    const std::size_t size = std::max(self_requirement.size,
                                      predecessor_requirements.size);

    return cuda::storage_description_t{alignment, size};
  }
};
} // namespace example::cuda::graph::detail::then
