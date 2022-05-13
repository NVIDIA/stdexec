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

namespace example::cuda::graph::detail::kernel
{

struct bulk_kernel_t
{
  template <class Shape,
            class F,
            class Receiver,
            class ArgsInput,
            unsigned int BlockThreads>
  __global__ __launch_bounds__(BlockThreads) static void kernel(Shape shape,
                                                                F f,
                                                                Receiver * receiver,
                                                                ArgsInput * args)
  {
    cuda::visit([=] __device__ (auto signature) {
      args->~ArgsInput();

      cuda::apply(
        [=] __device__(auto tag, auto &&... args) {
          using Tag = decltype(tag);
          const unsigned int i = blockIdx.x * BlockThreads + threadIdx.x;

          if constexpr (std::is_same_v<Tag, std::execution::set_value_t>)
          {
            if (i < static_cast<unsigned int>(shape))
            {
              f(static_cast<Shape>(i), args...);
            }
          }

          __syncthreads();
          if (i == 0)
          {
            tag(std::move(*receiver), std::forward<decltype(args)>(args)...);
          }
        },
        std::move(signature));
    }, std::move(*args));
  }
};

struct then_kernel_t
{
  template <class Shape,
            class F,
            class Receiver,
            class ArgsInput,
            unsigned int BlockThreads>
  __global__ __launch_bounds__(BlockThreads) static void kernel(Shape shape,
                                                                F f,
                                                                Receiver * receiver,
                                                                ArgsInput * args)
  {
    cuda::visit([=] __device__ (auto signature) {
      args->~ArgsInput();

      cuda::apply(
        [=] __device__(auto tag, auto &&... args) {
          using Tag = decltype(tag);

          if constexpr (std::is_same_v<Tag, std::execution::set_value_t>)
          {
            if constexpr (std::is_void_v<decltype(f(std::forward<decltype(args)>(args)...))>)
            {
              f(std::forward<decltype(args)>(args)...);
              std::execution::set_value(std::move(*receiver));
            }
            else
            {
              std::execution::set_value(std::move(*receiver), f(std::forward<decltype(args)>(args)...));
            }
          }
          else
          {
            tag(std::move(*receiver), std::forward<decltype(args)>(args)...);
          }
        },
        std::move(signature));
    }, std::move(*args));
  }
};

template <class KernelT, std::execution::sender S, std::integral Shape, class F>
struct sender_t;

template <class KernelT,
          std::execution::sender Sender,
          std::execution::receiver Receiver,
          std::execution::operation_state OpState,
          class Env>
struct operation_state_t
  : operation_state_base_t<Sender, operation_state_t<KernelT, Sender, Receiver, OpState, Env>, OpState, Env>
{
private:
  Receiver receiver_;
  using Base = operation_state_base_t<Sender, operation_state_t<KernelT, Sender, Receiver, OpState, Env>, OpState, Env>;

public:
  template <class Shape, class F>
  operation_state_t(OpState && state, Receiver && rcvr, Shape shape, F function, const Env & env)
    : Base{std::move(state)}, receiver_(std::forward<Receiver>(rcvr))
  {
    using storage_type = storage_type_for_t<Sender, Env>;
    std::byte *storage_ptr = cuda::get_storage(env);

    auto receiver_ptr = &receiver_;
    void *kernel_args[4] = {&shape, &function, &receiver_ptr, &storage_ptr};

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size =
      (static_cast<unsigned int>(shape) + block_size - 1) / block_size;

    auto predecessors = get_predecessors(this->op_state_);

    cudaKernelNodeParams kernel_node_params{};
    kernel_node_params.func =
      (void *)KernelT::template kernel<Shape, F, Receiver, storage_type, block_size>;
    kernel_node_params.gridDim = dim3(grid_size, 1, 1);
    kernel_node_params.blockDim = dim3(block_size, 1, 1);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = kernel_args;
    kernel_node_params.extra = nullptr;

    cudaGraph_t graph = get_graph(env).get();

    check(cudaGraphAddKernelNode(&this->node_,
                                 graph,
                                 predecessors.data(),
                                 predecessors.size(),
                                 &kernel_node_params));
  }

  friend void tag_invoke(std::execution::start_t, operation_state_t & self) noexcept
  {
    std::execution::start(self.op_state_);
  }
};

template <class KernelT, std::execution::sender S, std::integral Shape, class F>
struct sender_t : sender_base_t<sender_t<KernelT, S, Shape, F>, S>
{
  Shape shape_;
  F function_;

  template <std::execution::receiver Receiver>
    requires std::is_trivially_copyable_v<Receiver>
  auto connect(Receiver &&receiver) && noexcept
  {
    auto env = std::execution::get_env(receiver);

    using storage_type = storage_type_for_t<S, decltype(env)>;
    std::byte *storage_ptr = cuda::get_storage(env);

    using invented_receiver_t = consumer_receiver_t<storage_type, decltype(env)>;
    invented_receiver_t invented_receiver(storage_ptr, env);

    return operation_state_t<KernelT, S, Receiver, std::execution::connect_result_t<S, invented_receiver_t>, decltype(env)>(
      std::execution::connect(std::move(this->sender_), std::move(invented_receiver)),
      std::forward<Receiver>(receiver),
      shape_, function_, env
    );
  }

  template <std::__decays_to<sender_t> Self, class _Env>
    requires std::same_as<KernelT, bulk_kernel_t>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         Self &&,
                         _Env)
    -> std::execution::completion_signatures_of_t<std::__member_t<Self, S>,
                                                  _Env>;

  template <class Result>
  using set_value_ = std::__minvoke1<
    std::__uncurry<std::__qf<std::execution::set_value_t>>,
    std::__if<std::is_void<Result>, std::__types<>, std::__types<Result>>>;
  template <class... Args>
  requires std::invocable<F, Args...> using result =
    set_value_<std::invoke_result_t<F, Args...>>;

  template <class EnvT>
    requires std::same_as<KernelT, then_kernel_t>
  friend auto tag_invoke(std::execution::get_completion_signatures_t,
                         const sender_t &,
                         EnvT)
    -> std::execution::make_completion_signatures<
      S, EnvT, std::execution::__with_exception_ptr, result>;

  explicit sender_t(S sender, Shape shape, F function)
    : sender_base_t<sender_t<KernelT, S, Shape, F>, S>{std::forward<S>(sender)}
    , shape_{shape}
    , function_{function}
  {}
};

template <class... Args>
using bulk_sender_t = sender_t<bulk_kernel_t, Args...>;

template <class... Args>
using then_sender_t = sender_t<then_kernel_t, Args...>;

} // namespace example::cuda::graph::detail::kernel
